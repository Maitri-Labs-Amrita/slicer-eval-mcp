"""Slicer MCP bridge HTTP server: its own SlicerHTTPServer on 127.0.0.1, one handler, POST /bridge/<name> only.

There is no exec route: the only reachable code is the functions registered in registry.HANDLERS.

Request body (JSON):  {"args": {...}, "capture": ["3d", "axial", ...],
                       "slicer_mcp": {"launch": .., "episode": .., "prev_episode": .., "request": ..}}  (guard.py)
Response (JSON):      {"ok": true, "result": {...}, "state": {...}, "images": {view: base64 PNG},
                       "warnings": [...], "latency_ms": {"handler": .., "capture": .., "total": ..},
                       "bridge_version": "..", "identity": {episode_id, case_id, volume_sha256, seq},
                       "render": {view: {"before": n, "frame": m}}}
               or     {"ok": false, "error": {"type": "..", "message": ".."}, ...}

`render`: each observation view counts the renders VTK finished (its render window's EndEvent); `before` is
the count when the request was admitted, `frame` the count right after that view's capture. frame > before proves the
frame was drawn after the call. A request refused by the guard (wrong launch, another episode, a duplicate) runs no
handler and returns no state.
"""

import base64
import json
import math
import os
import re
import time
import traceback

import slicer

from WebServer import SlicerHTTPServer
from WebServerLib.BaseRequestHandler import BaseRequestHandler

from . import limits
from .env import Env
from .guard import FREE, RequestGuard, Refused, envelope, new_launch_id, new_token
from .obswin import ObservationWindow, app_slice_logics
from .presets import PresetStore
from .registry import HANDLERS, BridgeArgError

from . import env as _env  # noqa: F401
from . import handlers_2d as _h2d  # noqa: F401
from . import handlers_3d as _h3d  # noqa: F401
from . import handlers_seg as _hseg  # noqa: F401
from . import handlers_slices as _hsl  # noqa: F401

_PATH = re.compile(rb"^/bridge/([a-z_0-9]{1,64})$")
CAPTURABLE = ("3d", "axial", "sagittal", "coronal")
NESTED_OK = ("ping", "capture")
RECENTRE_AFTER = ("point_at", "jump_to_point")


class BridgeRequestHandler(BaseRequestHandler):
    def __init__(self, bridge, logMessage=None, **kwargs):
        self.bridge = bridge
        self.logMessage = logMessage or (lambda *a: None)

    def canHandleRequest(self, method, uri, requestBody):
        if method != "POST":
            return 0.0
        path = uri.split(b"?", 1)[0]
        m = _PATH.fullmatch(path)
        return 1.0 if m and m.group(1).decode() in HANDLERS else 0.0

    def handleRequest(self, method, uri, requestBody):
        name = _PATH.fullmatch(uri.split(b"?", 1)[0]).group(1).decode()
        return b"application/json", encode(self.bridge.dispatch(name, requestBody))


def first_nonfinite(obj, path=""):
    """The path of the first inf or nan in obj (e.g. "state.camera.zoom", "result.points[2]"), or None."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return path or "value"
    if isinstance(obj, dict):
        items = ((f"{path}.{k}" if path else str(k), v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        items = ((f"{path}[{i}]", v) for i, v in enumerate(obj))
    else:
        return None
    for p, v in items:
        bad = first_nonfinite(v, p)
        if bad is not None:
            return bad
    return None


def encode(out: dict) -> bytes:
    """The response as strict JSON. A response that holds inf or nan is replaced by an internal error that names
    the first such value."""
    try:
        return json.dumps(out, allow_nan=False).encode()
    except ValueError:
        bad = first_nonfinite(out)
        err = {"ok": False, "bridge_version": out.get("bridge_version"), "warnings": [],
               "error": {"type": "internal", "message": f"the response holds a non-finite number at {bad}; not sent"}}
        if out.get("identity") is not None:
            err["identity"] = out["identity"]
        return json.dumps(err, allow_nan=False).encode()


class Bridge:
    def __init__(self, port=None, obs_size=None, cases_dir=None, config_path=None, runtime_path=None, launch="unknown"):
        """Protocol settings come from the config JSON; host/port/paths from runtime.json unless overridden.
        `launch` labels how this Slicer was started ("eval": `slicer-mcp launch`); ping reports it so tests and the
        launch check can tell it from any other."""
        self.launch = launch
        limits.load(config_path)
        rt = limits.load_runtime(runtime_path)
        self.host = rt["bridge_host"]
        p = port if port is not None else rt["bridge_port"]
        self.port = 0 if p in (0, "0", "auto") else int(p)
        self.guard = RequestGuard(new_launch_id(), lease_s=float(rt.get("episode_lease_s", 900)), token=new_token())
        cases_dir = cases_dir or rt["cases_dir"]
        obs_size = obs_size or limits.DEFAULT_OBS_SIZE
        self.presets = PresetStore()
        self.obs = ObservationWindow(obs_size)
        self.env = Env(self.obs, self.presets, cases_dir)
        self.server = None
        self._depth = 0

    def start(self):
        self.server = SlicerHTTPServer(server_address=(self.host, self.port),
                                       requestHandlers=[BridgeRequestHandler(self)],
                                       logMessage=lambda *a: None)
        self.port = int(self.server.server_address[1])
        self.server.start()
        return self.info()

    def stop(self):
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self.obs is not None:
            self.obs.close()
            self.obs = None

    def info(self):
        return {
            "bridge_version": limits.BRIDGE_VERSION,
            "slicer_version": slicer.app.applicationVersion,
            "slicer_revision": slicer.app.revision,
            "port": self.port,
            "obs_size": self.obs.size_px,
            "device_pixel_ratio": self.obs.dpr,
            "screen": self.obs.screen_info(),
            "render_sizes": self.obs.render_sizes(),
            "views_adopted_by_main_layout": [t for t in (limits.THREED_VIEW, *(v[0] for v in limits.SLICE_VIEWS.values()))
                                             if self.obs.adopted_by_layout_manager(t)],
            "presets_sha256": self.presets.sha256,
            "config_version": limits.CONFIG["config_version"],
            "config_sha256": limits.CONFIG_SHA256,
            "handlers": sorted(HANDLERS),
            "window_title": self.obs.widget.windowTitle,
            "main_window_title": slicer.util.mainWindow().windowTitle if slicer.util.mainWindow() else None,
            "launch": self.launch,
            "launch_id": self.guard.launch_id,
            "pid": os.getpid(),
            "main_window_visible": bool(slicer.util.mainWindow() is not None and slicer.util.mainWindow().isVisible()),
            "layout_manager": slicer.app.layoutManager() is not None,
            "slice_logic_count": app_slice_logics().GetNumberOfItems(),
            "obs_window": self.obs.visibility(),
            **self.obs.render_stack(),
        }

    def _reply(self, t0, kind, message):
        return {"ok": False, "bridge_version": limits.BRIDGE_VERSION, "warnings": [],
                "error": {"type": kind, "message": message},
                "latency_ms": {"total": round((time.perf_counter() - t0) * 1000, 1)}}

    def dispatch(self, name, body):
        t0 = time.perf_counter()
        try:
            req = json.loads(body.decode() or "{}") if body else {}
            if not isinstance(req, dict):
                raise BridgeArgError("request body must be a JSON object")
            env = envelope(req)
            authed = self.guard.authorized(env)
            self.guard.check_launch(name, env)
            if not authed:
                env.pop("request", None)
            hit = self.guard.replay(name, env, req)
            if hit is not None:
                return hit
            if self._depth and name not in NESTED_OK:
                return self._reply(t0, "busy", f"the bridge is still handling another request; {name} was not run")
            seq = self.guard.admit(name, env)
        except Refused as e:
            return self._reply(t0, e.kind, e.message)
        except (BridgeArgError, ValueError) as e:
            return self._reply(t0, "invalid_argument", str(e))
        self._depth += 1
        self.guard.start(env)
        out = None
        try:
            out = self._dispatch(name, req, seq, authed)
            return out
        finally:
            self._depth -= 1
            self.guard.finish(name, env, req, out if out is not None else self._reply(t0, "internal", "no response"))

    def public_info(self):
        """What ping reports without the bridge token: only that the bridge is alive."""
        return {"alive": True, "bridge_version": limits.BRIDGE_VERSION}

    def _dispatch(self, name, req, seq, authed=True):
        t0 = time.perf_counter()
        warnings = []
        out = {"ok": False, "bridge_version": limits.BRIDGE_VERSION, "warnings": warnings}
        if seq is not None:
            self.env.identity = self.guard.identity(seq)
        try:
            args = req.get("args") or {}
            capture = req.get("capture") or []
            if not isinstance(args, dict) or not isinstance(capture, list):
                raise BridgeArgError("args must be an object and capture a list")
            bad = [c for c in capture if c not in CAPTURABLE]
            if bad:
                raise BridgeArgError(f"unknown capture views {bad}; allowed {list(CAPTURABLE)}")
            if name in FREE:
                out.update(ok=True, result=self.info() if authed else self.public_info())
                return out
            if name == "release_episode":
                self.guard.release()
                out.update(ok=True, result={}, identity=self.guard.identity(seq))
                return out
            before = dict(self.obs.render_counts)
            repairs = self.obs.ensure_visible() if name == "reset_episode" else None
            result = HANDLERS[name](self.env, args, warnings)
            if name == "reset_episode":
                self.guard.loaded(self.env.volume.GetName(), self.env.volume_sha256)
                self.env.identity = self.guard.identity(seq)
            if name in RECENTRE_AFTER:
                self.env.sync_slice_views()
            if repairs is not None:
                result = {**(result or {}), "obs_window": {"repairs": repairs}}
            t1 = time.perf_counter()
            images, render = {}, {}
            for view in capture:
                images[view] = base64.b64encode(self.obs.capture(view)).decode()
                render[view] = {"before": before.get(view, 0), "frame": self.obs.render_counts.get(view, 0)}
            t2 = time.perf_counter()
            out.update(ok=True, result=result or {}, state=self.env.state(), images=images, render=render,
                       latency_ms={"handler": round((t1 - t0) * 1000, 1), "capture": round((t2 - t1) * 1000, 1)})
        except BridgeArgError as e:
            out["error"] = {"type": "invalid_argument", "message": str(e)}
        except Exception as e:
            out["error"] = {"type": "internal", "message": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()}
        if seq is not None:
            out["identity"] = self.guard.identity(seq)
        out.setdefault("latency_ms", {})["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return out


HANDLERS.setdefault("ping", lambda env, args, warnings: {})
HANDLERS.setdefault("release_episode", lambda env, args, warnings: {})
