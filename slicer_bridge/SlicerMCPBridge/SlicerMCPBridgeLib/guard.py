"""Request admission on the bridge: the bridge token, the launch id, the episode binding and the request-id cache.
Stdlib only.

Every request body may carry an envelope next to args / capture:

    {"args": {...}, "capture": [...],
     "slicer_mcp": {"token": "<bridge token>", "launch": "<launch id>", "episode": "<episode id>",
                    "prev_episode": "<episode id>", "request": "<id>"}}

Slicer's SlicerHTTPServer does not pass the HTTP headers to a request handler, so these values travel in the body.

  token     a per-launch secret ($SLICER_MCP_BRIDGE_TOKEN, else random). `slicer-mcp launch` writes it to its state
            file, which only the local user can read. Every request except ping must carry it; a request without it
            is refused before any handler runs. Ping without the token only reports that the bridge is alive.
  launch    the random id this Slicer made at startup (or $SLICER_MCP_LAUNCH_NONCE), reported by an authenticated
            ping as `launch_id`. Every request except ping must carry it, so a client that reaches another job's
            Slicer on its port gets an error, never an answer.
  episode   reset_episode binds the bridge to the episode id it carries. Later requests must carry the bound id. A
            reset while an episode is bound must name it as prev_episode, unless the bound episode has been idle for
            longer than the lease. release_episode {episode} unbinds.
  request   a client-chosen id per call, the same on a retry. The last responses are kept by id: a retry of a call
            the bridge already ran gets the first response again (replayed: true) instead of a second run. A response
            is replayed only to the same call (handler name and the whole body); another request reusing the id is
            refused.

Each admitted request gets a sequence number; responses carry `identity` = {episode_id, case_id, volume_sha256, seq}.
"""

import collections
import hashlib
import json
import os
import re
import secrets
import time

LAUNCH_ENV = "SLICER_MCP_LAUNCH_NONCE"
TOKEN_ENV = "SLICER_MCP_BRIDGE_TOKEN"
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
FREE = ("ping",)
DEFAULT_LEASE_S = 900.0
CACHE_SIZE = 8


def new_launch_id(environ=None) -> str:
    v = (environ if environ is not None else os.environ).get(LAUNCH_ENV)
    if v:
        if not ID_RE.fullmatch(v):
            raise ValueError(f"${LAUNCH_ENV} must match {ID_RE.pattern}")
        return v
    return secrets.token_hex(16)


def new_token(environ=None) -> str:
    """The bridge token: $SLICER_MCP_BRIDGE_TOKEN, else a random secret."""
    v = (environ if environ is not None else os.environ).get(TOKEN_ENV)
    if v:
        if not TOKEN_RE.fullmatch(v):
            raise ValueError(f"${TOKEN_ENV} must match {TOKEN_RE.pattern}")
        return v
    return secrets.token_hex(32)


def fingerprint(name: str, req: dict) -> str:
    """The call a cached response answers: the handler name and the whole request body (args, capture, envelope)."""
    return hashlib.sha256(json.dumps([name, req], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Refused(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind
        self.message = message


def envelope(req: dict) -> dict:
    """The request's `slicer_mcp` envelope, checked (Refused 'invalid_envelope')."""
    env = req.get("slicer_mcp") or {}
    if not isinstance(env, dict):
        raise Refused("invalid_envelope", "slicer_mcp must be an object")
    out = {}
    for k in ("token", "launch", "episode", "prev_episode", "request"):
        v = env.get(k)
        if v is None:
            continue
        if not isinstance(v, str) or not ID_RE.fullmatch(v):
            raise Refused("invalid_envelope", f"slicer_mcp.{k} must match {ID_RE.pattern}")
        out[k] = v
    return out


class RequestGuard:
    def __init__(self, launch_id: str, lease_s: float = DEFAULT_LEASE_S, cache_size: int = CACHE_SIZE,
                 clock=time.monotonic, token: str | None = None):
        self.launch_id = launch_id
        self.token = token or secrets.token_hex(32)
        self.lease_s = float(lease_s)
        self.clock = clock
        self.episode_id = None
        self.case_id = None
        self.volume_sha256 = None
        self.last_active = None
        self.seq = 0
        self._cache = collections.OrderedDict()
        self._cache_size = int(cache_size)
        self._in_flight = set()

    def authorized(self, env: dict) -> bool:
        """The envelope carries this bridge's token."""
        tok = env.get("token")
        return isinstance(tok, str) and secrets.compare_digest(tok.encode(), self.token.encode())

    def check_token(self, name: str, env: dict):
        """Refused 'unauthorized' unless the request carries the bridge token (ping is free)."""
        if name not in FREE and not self.authorized(env):
            raise Refused("unauthorized", f"{name}: missing or wrong bridge token; not run")

    def check_launch(self, name: str, env: dict):
        """Refused unless the request carries the bridge token and this Slicer's launch id (ping is free)."""
        self.check_token(name, env)
        if name not in FREE and env.get("launch") != self.launch_id:
            raise Refused("wrong_launch", f"{name} carries launch {env.get('launch')!r}, but this Slicer is launch "
                                          f"{self.launch_id[:8]}...; not run (another job's Slicer on this port?)")

    def admit(self, name: str, env: dict) -> int | None:
        """Refused (kind, message) unless this request may run; else its sequence number (None for ping)."""
        if name in FREE:
            return None
        self.check_launch(name, env)
        now = self.clock()
        if name == "reset_episode":
            ep = env.get("episode")
            if not ep:
                raise Refused("episode_required", "reset_episode needs slicer_mcp.episode (the new episode id)")
            if self.episode_id is not None and env.get("prev_episode") != self.episode_id:
                idle = now - (now if self.last_active is None else self.last_active)
                if idle <= self.lease_s:
                    raise Refused("episode_owned", f"the bridge belongs to another episode (active {idle:.0f} s ago, "
                                                   f"lease {self.lease_s:.0f} s); reset_episode not run")
            self.episode_id, self.case_id, self.volume_sha256 = ep, None, None
        elif name == "release_episode":
            if self.episode_id is not None and env.get("episode") != self.episode_id:
                raise Refused("episode_mismatch", "release_episode: not the bound episode")
        elif self.episode_id is not None and env.get("episode") != self.episode_id:
            raise Refused("episode_mismatch", f"{name} carries episode {env.get('episode')!r}, not the episode this "
                                              f"bridge is bound to; not run")
        self.last_active = now
        self.seq += 1
        return self.seq

    def release(self):
        self.episode_id = self.case_id = self.volume_sha256 = self.last_active = None

    def loaded(self, case_id, volume_sha256):
        """A reset loaded this case (its voxels hash to volume_sha256)."""
        self.case_id, self.volume_sha256 = case_id, volume_sha256

    def identity(self, seq) -> dict:
        return {"episode_id": self.episode_id, "case_id": self.case_id, "volume_sha256": self.volume_sha256,
                "seq": seq}

    def replay(self, name: str, env: dict, req: dict):
        """The response already sent for this request id (marked replayed), or None. Refused while it is running, and
        when the id was used for another call (name or body differ)."""
        rid = env.get("request")
        if rid is None:
            return None
        if rid in self._in_flight:
            raise Refused("duplicate_in_flight", f"request {rid} is still running; not run twice")
        hit = self._cache.get(rid)
        if hit is None:
            return None
        if hit[0] != fingerprint(name, req):
            raise Refused("request_id_reused", f"request id {rid} was used for another call; {name} not run")
        return {**hit[1], "replayed": True}

    def start(self, env: dict):
        if env.get("request") is not None:
            self._in_flight.add(env["request"])

    def finish(self, name: str, env: dict, req: dict, response: dict):
        rid = env.get("request")
        if rid is None:
            return
        self._in_flight.discard(rid)
        self._cache[rid] = (fingerprint(name, req), response)
        self._cache.move_to_end(rid)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
