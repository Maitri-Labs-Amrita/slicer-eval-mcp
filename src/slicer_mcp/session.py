"""The environment session: tool call -> bridge call / episode bookkeeping -> trace step.

The MCP server is a thin layer over this: it validates args (Pydantic), lists the profile's tools and turns a
StepOutcome into MCP content. Logging happens here, so the trace is identical whoever drives it. Step 0 of every
trace is the runner's reset_episode (the initial observation: all four views).

The profile's allowlist is enforced here too (defence in depth): a tool outside it is refused and logged.

Stale-render guard: every frame passes through StaleFrameGuard (stale.py). A frozen frame (byte-identical to the
view's previous one while that view's state changed) raises BridgeError type "stale_render", which the server
reports as an `environment` error, so the runner aborts the episode.
"""

import json
import time
from dataclasses import dataclass, field

from slicer_mcp.bridge_client import VIEWS, BridgeError
from slicer_mcp.episode import EpisodeError, EpisodeSession
from slicer_mcp.finite import finite_copy, nonfinite_paths
from slicer_mcp.reports import check_files
from slicer_mcp.stale import StaleFrameGuard
from slicer_mcp.tracing.logger import TraceLogger, git_info, new_episode_id, utc_now

TOOL_SCHEMA_VERSION = "slicer-mcp-tools-v1"
SLICES = ("axial", "sagittal", "coronal")
RUNNER_TOOLS = ("reset_episode", "end_episode", "observe")


def _slice_of(args):
    v = args.get("view")
    return (v,) if v in SLICES else ()


BRIDGE_TOOLS = {
    "observe": ("capture", lambda a: VIEWS),
    "reset_view": ("reset_view", lambda a: VIEWS),
    "set_slice": ("set_slice", _slice_of),
    "scroll_slice": ("scroll_slice", _slice_of),
    "set_window_preset": ("set_window_preset", lambda a: SLICES),
    "jump_to_point": ("jump_to_point", lambda a: SLICES),
    "point_at": ("point_at", lambda a: SLICES),
    "orbit": ("orbit", lambda a: ("3d",)),
    "set_view_direction": ("set_view_direction", lambda a: ("3d",)),
    "zoom": ("zoom", lambda a: SLICES if a.get("view") == "slices" else ("3d",)),
    "pan_to": ("pan_to", lambda a: ("3d",)),
    "set_render_preset": ("set_render_preset", lambda a: ("3d",)),
    "set_clip_box": ("set_clip_box", lambda a: ("3d",)),
    "set_clip_plane": ("set_clip_plane", lambda a: ("3d",)),
    "clear_clip": ("clear_clip", lambda a: ("3d",)),
    "set_segment_visibility": ("set_segment_visibility", lambda a: VIEWS),
    "set_segment_opacity": ("set_segment_opacity", lambda a: ("3d",)),
    "set_3d_content": ("set_3d_content", lambda a: ("3d",)),
    "set_slice_outlines": ("set_slice_outlines", lambda a: SLICES),
    "center_on_segments": ("center_on_segments", lambda a: ("3d",)),
    "measure": ("measure", _slice_of),
}
EPISODE_TOOLS = ("bookmark_view", "submit_report")
CASE_INFO_RUNNER_ONLY = ("volume_sha256", "volume_dtype")
SESSION_ARGS = ("composite",)


@dataclass
class StepOutcome:
    step: int
    tool: str
    ok: bool
    state: dict
    frames: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    result: dict = field(default_factory=dict)
    error: dict | None = None
    view_ref: str | None = None
    done: bool = False
    identity: dict | None = None
    case_info: dict | None = None


class EnvSession:
    def __init__(self, client, profile, traces_dir, ping_info: dict | None = None,
                 stale_guard: dict | None = None, host_info: dict | None = None):
        self.client = client
        self.profile = profile
        self.protocol = profile.protocol
        self.traces_dir = traces_dir
        self.tools = RUNNER_TOOLS + profile.tools
        self.views = VIEWS
        self.ping_info = ping_info or client.ping()
        self.logger: TraceLogger | None = None
        self.episode: EpisodeSession | None = None
        self.step_index = 0
        self.last_state: dict = {}
        self.last_trace_dir = None
        self.last_status = None
        self.guard = StaleFrameGuard(stale_guard)
        self.host_info = host_info
        self.episode_id = None
        self.loaded = None

    def _check_identity(self, name, r):
        """The bridge answered for this session's episode and the case it loaded at reset."""
        ident = getattr(r, "identity", None)
        if not ident or self.episode_id is None:
            return
        want = {"episode_id": self.episode_id, **({k: self.loaded[k] for k in ("case_id", "volume_sha256")}
                                                  if self.loaded else {})}
        bad = {k: (ident.get(k), v) for k, v in want.items() if ident.get(k) != v}
        if bad:
            raise BridgeError(name, {"type": "identity_mismatch", "message": (
                f"the bridge answered for another episode or volume: {bad}; abort the episode")})

    def _check_fresh(self, name, r):
        self._check_identity(name, r)
        stale = self.guard.check(r.images, r.state, getattr(r, "render", None))
        if stale:
            raise BridgeError(name, {"type": "stale_render", "message": (
                f"frozen frame: {'; '.join(stale)} (a changed view with a byte-identical frame, or no render after "
                "the call). Slicer is not drawing (observation window closed or covered, or the display asleep); abort the episode")})

    def reset(self, case_id: str, client_meta: dict | None = None) -> StepOutcome:
        """Start a new episode (runner only). Any open episode is closed as abandoned."""
        bad = nonfinite_paths(client_meta or {})
        if bad:
            raise ValueError(f"client_meta must hold finite numbers only (inf or nan at {', '.join(bad[:5])})")
        self.close("abandoned")
        cm = dict(client_meta or {})
        t0 = time.perf_counter()
        reset_args = {"case_id": case_id, **({"with_segments": True} if self.profile.segmentation else {})}
        episode_id = new_episode_id(case_id, self.profile.name)
        self.episode_id, self.loaded = episode_id, None
        r = self.client.call("reset_episode", reset_args, capture=self.views, episode_id=episode_id)
        ident = getattr(r, "identity", None)
        if ident:
            self.loaded = {"case_id": ident.get("case_id"), "volume_sha256": ident.get("volume_sha256")}
        self._check_fresh("reset_episode", r)
        info = r.result.pop("case_info")
        if ident and info.get("volume_sha256") not in (None, ident.get("volume_sha256")):
            raise BridgeError("reset_episode", {"type": "identity_mismatch",
                                                "message": "case_info volume_sha256 differs from the reset identity"})
        result = {"case_info": {k: v for k, v in info.items() if k not in CASE_INFO_RUNNER_ONLY}}
        wall = (time.perf_counter() - t0) * 1000
        self.logger = TraceLogger(self.traces_dir, episode_id)
        report = self.profile.report
        self.episode = EpisodeSession(case_id, report_model=report.Model if report is not None else None)
        self.step_index = 0
        self.last_state = r.state
        p = self.ping_info
        self.logger.write_meta({
            "case_id": case_id,
            "profile": self.profile.name,
            "model": {"id": cm.pop("model_id", None), "version": cm.pop("model_version", None)},
            "system_prompt_sha256": cm.pop("system_prompt_sha256", None),
            "seed": cm.pop("seed", None),
            "client_meta": cm,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "tools": list(self.profile.tools),
            "state_format": "full",
            "report_schema_version": report.SCHEMA_VERSION if report is not None else None,
            "report_plugin_sha256": self.profile.report_sha256,
            "protocol": {"config_version": self.protocol.version, "config_sha256": self.protocol.sha256},
            "presets_sha256": p.get("presets_sha256"),
            "bridge": {"version": p.get("bridge_version"), "config_sha256": p.get("config_sha256")},
            "slicer": {"version": p.get("slicer_version"), "revision": p.get("slicer_revision")},
            **{k: p[k] for k in ("vtk", "opengl", "mesa") if k in p},
            "server_git": git_info(),
            "image": {"size_px": p.get("obs_size"), "device_pixel_ratio": p.get("device_pixel_ratio"),
                      "views": list(self.views)},
            "start_time": utc_now(),
            "end_time": None,
            "status": "running",
            "reset_latency_ms": {"bridge": r.latency_ms, "bridge_wall_ms": r.wall_ms, "server_ms": round(wall, 1)},
            "node_inventory": r.result.get("node_inventory"),
            **({"segments_manifest_sha256": r.result.get("segments_manifest_sha256")}
               if self.profile.segmentation else {}),
            "obs_window": r.result.get("obs_window"),
            "launch": p.get("launch"),
            "launch_id": p.get("launch_id"),
            "identity": ident,
            "host": self.host_info,
        })
        self.logger.log_step(0, "reset_episode", args={"case_id": case_id}, args_after_validation=reset_args,
                             state_after=r.state, frames=r.images, result=result or None, latency_ms=self._lat(wall, r),
                             warnings=r.warnings, extra={**self._extra(r), "case_info": info})
        return StepOutcome(0, "reset_episode", True, r.state, r.images, r.warnings, result, identity=ident,
                           case_info=info)

    def call(self, tool: str, args: dict | None = None, *, args_after_validation: dict | None = None,
             server_warnings: list | None = None, model_reasoning: str | None = None,
             tokens: dict | None = None, rejected: str | None = None) -> StepOutcome:
        """Run one tool call and log it as the next step. `args` = what the model sent; `args_after_validation`
        = the server-validated args, else the raw args. The bridge re-clamps and reports its own final
        args; the logged args_after_validation is the bridge's when it has one (that is what replay re-issues).
        `rejected` = the server's validation message for args it refused: the call is logged as an
        invalid_argument error step and nothing is executed."""
        if self.logger is None:
            raise RuntimeError("no episode: call reset() first")
        logger = self.logger
        args = dict(args or {})
        sent = dict(args_after_validation if args_after_validation is not None else args)
        bad = nonfinite_paths(args) or nonfinite_paths(sent)
        if bad and rejected is None:
            rejected = f"arguments must be finite numbers (inf or nan at {', '.join(bad[:5])})"
        if bad:
            args, sent = finite_copy(args), finite_copy(sent)
        self.step_index += 1
        step = self.step_index
        warnings = list(server_warnings or [])
        t0 = time.perf_counter()
        out = StepOutcome(step, tool, False, self.last_state, warnings=warnings)
        bridge_lat = None
        final_args = sent
        try:
            if self.episode.done:
                raise EpisodeError("episode already ended (report submitted)")
            if rejected is not None:
                raise EpisodeError(rejected)
            if tool not in self.tools or tool == "reset_episode":
                raise EpisodeError(f"tool {tool!r} is not available")
            if tool in BRIDGE_TOOLS:
                handler, frames_for = BRIDGE_TOOLS[tool]
                wanted = self.views if sent.get("composite") else frames_for(sent)
                capture = tuple(v for v in wanted if v in self.views)
                bridge_args = {k: v for k, v in sent.items() if k not in SESSION_ARGS}
                r = self.client.call(handler, bridge_args, capture=capture)
                bridge_lat = r
                self._check_fresh(handler, r)
                final_args = r.result.pop("args_after_validation", sent)
                if "composite" in sent:
                    final_args = {**final_args, "composite": sent["composite"]}
                out.ok, out.state, out.frames, out.result = True, r.state, r.images, r.result
                out.identity = getattr(r, "identity", None)
                warnings += r.warnings
            elif tool == "bookmark_view":
                r = self.client.call("capture", {}, capture=self.views)
                bridge_lat = r
                self._check_fresh("capture", r)
                out.identity = getattr(r, "identity", None)
                b = self.episode.bookmark(sent.get("label"), step, r.state, r.images)
                out.ok, out.state, out.frames, out.view_ref = True, r.state, r.images, b.view_ref
                out.result = {"view_ref": b.view_ref}
            elif tool == "submit_report":
                rep = self.episode.validate_report(sent.get("report"))
                result, files = self.profile.report.accept(rep)
                json.dumps(result, allow_nan=False)
                logger.write_report_files(check_files(files))
                self.episode.end(rep)
                out.ok, out.done, out.result = True, True, result
            else:
                raise EpisodeError(f"unknown tool {tool!r}")
        except BridgeError as e:
            if e.type not in ("invalid_argument", "not_found"):
                self._log(logger, step, tool, args, None, out, t0, bridge_lat, model_reasoning, tokens,
                          {"type": e.type, "message": e.message}, self._extra(bridge_lat))
                raise
            out.error = {"type": e.type, "message": e.message}
        except EpisodeError as e:
            out.error = {"type": "invalid_argument", "message": str(e)}
        except Exception as e:
            self._log(logger, step, tool, args, None, out, t0, bridge_lat, model_reasoning, tokens,
                      {"type": "environment", "message": f"{type(e).__name__}: {e}"}, self._extra(bridge_lat))
            raise
        self.last_state = out.state
        self._log(logger, step, tool, args, final_args if out.ok else None, out, t0, bridge_lat, model_reasoning,
                  tokens, out.error, self._extra(bridge_lat))
        if out.done:
            self.close("submitted")
        return out

    def _log(self, logger, step, tool, args, final_args, out, t0, r, model_reasoning, tokens, error, extra=None):
        logger.log_step(step, tool, args=args, args_after_validation=final_args, state_after=out.state,
                        frames=out.frames, view_ref=out.view_ref, result=out.result or None,
                        latency_ms=self._lat((time.perf_counter() - t0) * 1000, r),
                        model_reasoning=model_reasoning, tokens=tokens, warnings=out.warnings, error=error,
                        extra=extra)

    @staticmethod
    def _extra(r) -> dict:
        """Trace-only fields of a step (never sent to the model): the bridge's identity block and render counts."""
        out = {}
        if r is not None and getattr(r, "identity", None):
            out["identity"] = r.identity
        if r is not None and getattr(r, "render", None):
            out["render"] = r.render
        return out

    @staticmethod
    def _lat(server_ms, r):
        d = {"server_ms": round(server_ms, 1)}
        if r is not None:
            d["bridge"] = r.latency_ms
            d["bridge_wall_ms"] = r.wall_ms
        return d

    def close(self, status: str):
        if self.logger is None:
            return
        self.logger.update_meta(end_time=utc_now(), status=status, n_steps=self.step_index,
                                n_bookmarks=len(self.episode.bookmarks))
        self.logger.close()
        self.last_trace_dir = self.logger.dir
        self.last_status = status
        self.logger = None

    @property
    def trace_dir(self):
        """The current (or, after it ended, the last) episode's trace directory."""
        return self.logger.dir if self.logger else self.last_trace_dir
