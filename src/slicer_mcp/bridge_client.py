"""HTTP client for the Slicer MCP bridge (POST /bridge/<name>). No exec, no stock WebServer endpoints.

Every request body carries `slicer_mcp` = {token, launch, episode, prev_episode (reset only), request}
(the bridge side is SlicerMCPBridgeLib/guard.py):
  token     the bridge token: `token`, else $SLICER_MCP_BRIDGE_TOKEN, else the launch state file's. Without it the
            bridge answers only a liveness ping.
  launch    the bridge's launch id: `expected_launch` ($SLICER_MCP_LAUNCH_NONCE, the job's own Slicer) or else learned
            from the first ping; a ping that reports another id raises BridgeError("wrong_launch")
  episode   reset_episode binds the bridge to a new episode id (the caller's, else a random one); later calls carry it
  request   one random id per call, kept across retries: the bridge answers a retry of a call it already ran from its
            cache, so a connection error after the bridge ran a mutation never runs it twice
"""

import base64
import os
import struct
import time
import uuid
from dataclasses import dataclass, field

import requests

from .config import TOKEN_ENV, bridge_url, launch_token
from .finite import nonfinite_paths

VIEWS = ("3d", "axial", "sagittal", "coronal")
LAUNCH_ENV = "SLICER_MCP_LAUNCH_NONCE"
REFUSALS = ("unauthorized", "wrong_launch", "episode_owned", "episode_mismatch", "episode_required", "duplicate_in_flight",
            "invalid_envelope", "busy")


class BridgeError(RuntimeError):
    def __init__(self, name, error):
        self.name = name
        self.type = (error or {}).get("type", "unknown")
        self.message = (error or {}).get("message", "")
        self.traceback = (error or {}).get("traceback")
        super().__init__(f"bridge {name}: {self.type}: {self.message}")


@dataclass
class BridgeResponse:
    name: str
    result: dict
    state: dict
    images: dict
    warnings: list
    latency_ms: dict
    wall_ms: float
    raw: dict = field(repr=False, default_factory=dict)
    identity: dict | None = None
    render: dict | None = None


def png_size(png: bytes) -> tuple[int, int]:
    if png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", png[16:24])


class BridgeClient:
    def __init__(self, url=None, obs_size=None, timeout=300.0, connect_retries=3, expected_launch=None, token=None):
        self.url = (url or bridge_url()).rstrip("/")
        self.obs_size = obs_size
        self.timeout = timeout
        self.connect_retries = connect_retries
        self.session = requests.Session()
        self.session.trust_env = False
        self.launch_id = expected_launch or os.environ.get(LAUNCH_ENV) or None
        self.token = token or launch_token()
        self.episode_id = None

    def _envelope(self, name, episode_id):
        env = {"request": uuid.uuid4().hex}
        if self.token:
            env["token"] = self.token
        if self.launch_id:
            env["launch"] = self.launch_id
        if name == "reset_episode":
            env["episode"] = episode_id or uuid.uuid4().hex
            if self.episode_id:
                env["prev_episode"] = self.episode_id
        elif self.episode_id:
            env["episode"] = self.episode_id
        return env

    def call(self, name, args=None, capture=(), episode_id=None):
        """POST /bridge/<name>. reset_episode binds the bridge to `episode_id` (default: a random id)."""
        if name != "ping" and not self.launch_id:
            self.ping()
        env = self._envelope(name, episode_id)
        body = {"args": args or {}, "capture": list(capture), "slicer_mcp": env}
        t0 = time.perf_counter()
        for attempt in range(self.connect_retries + 1):
            try:
                r = self.session.post(f"{self.url}/bridge/{name}", json=body, timeout=self.timeout)
                break
            except requests.ConnectionError:
                if attempt == self.connect_retries:
                    raise
                time.sleep(0.5 * 2 ** attempt)
        wall = (time.perf_counter() - t0) * 1000
        if r.status_code == 404:
            raise BridgeError(name, {"type": "not_found", "message": f"no bridge handler {name!r}"})
        r.raise_for_status()
        d = r.json()
        err_type = (d.get("error") or {}).get("type")
        if name == "reset_episode" and err_type not in REFUSALS:
            self.episode_id = env["episode"]
        bad = nonfinite_paths(d)
        if bad:
            raise BridgeError(name, {"type": "protocol",
                                     "message": f"the bridge sent inf or nan at {', '.join(bad[:5])}"})
        if name == "ping" and d.get("ok"):
            lid = (d.get("result") or {}).get("launch_id")
            if lid is None and self.launch_id:
                raise BridgeError(name, {"type": "unauthorized", "message": (
                    f"the bridge at {self.url} answered only a liveness ping: this client has no valid bridge token "
                    f"(set ${TOKEN_ENV} or start the bridge with `slicer-mcp launch`)")})
            if self.launch_id and lid != self.launch_id:
                raise BridgeError(name, {"type": "wrong_launch", "message": (
                    f"the bridge at {self.url} is launch {str(lid)[:8]}..., not the expected {self.launch_id[:8]}... "
                    "(another job's Slicer on this port?)")})
            if lid:
                self.launch_id = lid
        if not d.get("ok"):
            raise BridgeError(name, d.get("error"))
        images = {k: base64.b64decode(v) for k, v in (d.get("images") or {}).items()}
        missing = [v for v in capture if v not in images]
        if missing:
            raise BridgeError(name, {"type": "protocol", "message": f"missing frames {missing}"})
        if self.obs_size:
            for k, png in images.items():
                if k in VIEWS and png_size(png) != (self.obs_size, self.obs_size):
                    raise BridgeError(name, {"type": "protocol",
                                             "message": f"frame {k} is {png_size(png)}, expected {self.obs_size}^2"})
        return BridgeResponse(name, d.get("result") or {}, d.get("state") or {}, images, d.get("warnings") or [],
                              d.get("latency_ms") or {}, round(wall, 1), d, d.get("identity"), d.get("render"))

    def ping(self):
        return self.call("ping").result

    def release(self):
        """Unbind the bridge from this client's episode, so another client may reset it. Never raises."""
        if not self.episode_id:
            return
        try:
            self.call("release_episode")
        except Exception:  # noqa: BLE001
            pass
        self.episode_id = None
