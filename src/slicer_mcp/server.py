"""Slicer MCP server (stdio): a profile's tools over one Slicer bridge.

    slicer-mcp serve [--profile DIR] [--slicer-url URL] [--runtime runtime.json]

A thin layer over EnvSession (which does the logging): it lists the profile's tools with Pydantic-derived schemas,
validates and clamps arguments, and turns each StepOutcome into MCP content:

    content = [TextContent(JSON {step_index, state, result, warnings, error, frames}), ImageContent per frame]

The images follow in the order named by `frames`. `state` is the bridge state as is (the trace keeps the same). The
result's `_meta` (runner only, never in the model's text) carries `slicer_mcp/identity` = the bridge's {episode_id,
case_id, volume_sha256, seq} for the call and, on reset_episode, `slicer_mcp/episode_id` and
`slicer_mcp/case_info` (the bridge's case_info with volume_sha256). Errors the model caused are MCP tool errors
(is_error=true) with the same JSON; the episode continues. reset_episode, end_episode and observe are runner-only
(tool _meta "slicer_mcp/runner_only": true): the client must hide them from the model. The client may send
`slicer_mcp/model_reasoning` (str) and `slicer_mcp/tokens` ({"in", "out"}) in the request _meta; they are logged in
the step. Built on the official `mcp` SDK's low-level Server. stdout is the MCP channel: all
logging goes to stderr.
"""

import argparse
import base64
import json
import logging
import os
import sys

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from slicer_mcp import __version__, display
from slicer_mcp.bridge_client import BridgeClient, BridgeError
from slicer_mcp.config import bridge_url, launch_token, load_profile, load_runtime
from slicer_mcp.finite import nonfinite_paths
from slicer_mcp.schemas.tools import ArgsRejected, validate_args
from slicer_mcp.session import BRIDGE_TOOLS, TOOL_SCHEMA_VERSION, EnvSession
from slicer_mcp.tools.registry import RUNNER_ONLY_META, instructions, tool_specs

log = logging.getLogger("slicer_mcp.server")
META_REASONING = "slicer_mcp/model_reasoning"
META_TOKENS = "slicer_mcp/tokens"
META_EPISODE = "slicer_mcp/episode_id"
META_IDENTITY = "slicer_mcp/identity"
META_CASE_INFO = "slicer_mcp/case_info"


class StartupError(RuntimeError):
    pass


def check_bridge(ping: dict, profile, expected_launch=None, require_eval=False):
    """Refuse to serve a bridge that runs another protocol config or frame size, or lacks a handler we need; that has
    no launch id or another one than the job's own (SLICER_MCP_LAUNCH_NONCE / runtime launch_nonce); or, with runtime
    require_eval_launch, that is not an eval launch."""
    protocol = profile.protocol
    lid = ping.get("launch_id")
    if not lid:
        raise StartupError("the bridge reports no launch_id: no valid bridge token (start the bridge with "
                           "`slicer-mcp launch` and serve with the same runtime config, or set "
                           "SLICER_MCP_BRIDGE_TOKEN)")
    if expected_launch and lid != expected_launch:
        raise StartupError(f"the bridge is launch {lid[:8]}..., not this job's {expected_launch[:8]}... (another "
                           "job's Slicer on this port?)")
    if require_eval and ping.get("launch") != "eval":
        raise StartupError(f"bridge launch is {ping.get('launch')!r}, expected 'eval' (runtime require_eval_launch)")
    if ping.get("config_sha256") != protocol.sha256:
        raise StartupError(f"bridge protocol config sha256 {str(ping.get('config_sha256'))[:12]} != server "
                           f"{protocol.sha256[:12]} ({protocol.path}); launch the bridge with the same profile")
    size = protocol["observation"]["size_px"]
    if ping.get("obs_size") != size:
        raise StartupError(f"bridge frame size {ping.get('obs_size')} != config {size}")
    needed = {"reset_episode", "capture"} | {BRIDGE_TOOLS[t][0] for t in profile.tools if t in BRIDGE_TOOLS}
    missing = sorted(needed - set(ping.get("handlers") or ()))
    if missing:
        raise StartupError(f"bridge lacks handlers {missing}")


def _meta_str(meta, key):
    v = (meta or {}).get(key)
    return v if isinstance(v, str) else None


def _meta_tokens(meta):
    v = (meta or {}).get(META_TOKENS)
    if isinstance(v, dict) and all(isinstance(x, int) and not isinstance(x, bool) for x in v.values()):
        return dict(v)
    return None


class EnvServer:
    """Owns one EnvSession and the MCP handlers. Calls are serialised (one Slicer, one episode)."""

    def __init__(self, session: EnvSession, profile, display_check: str = "off"):
        self.session = session
        self.display_check = display_check
        self.profile = profile
        self.protocol = profile.protocol
        self.specs = {s.name: s for s in tool_specs(profile)}
        self._lock = anyio.Lock()
        self.server = Server("slicer-mcp", version=f"{__version__}+{TOOL_SCHEMA_VERSION}",
                             instructions=instructions(profile),
                             on_list_tools=self._list_tools, on_call_tool=self._call_tool)

    def tools(self) -> list[types.Tool]:
        return [types.Tool(
            name=s.name, title=s.title, description=s.description, input_schema=s.input_schema,
            annotations=types.ToolAnnotations(title=s.title, read_only_hint=s.read_only,
                                              destructive_hint=False, open_world_hint=False),
            meta={RUNNER_ONLY_META: s.runner_only}) for s in self.specs.values()]

    async def _list_tools(self, ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools())

    async def _call_tool(self, ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        meta = params.meta or ctx.meta or {}
        async with self._lock:
            return await anyio.to_thread.run_sync(self.handle, params.name, params.arguments, dict(meta))

    def handle(self, tool: str, raw_args, meta: dict | None = None) -> types.CallToolResult:
        reasoning, tokens = _meta_str(meta, META_REASONING), _meta_tokens(meta)
        try:
            if tool == "reset_episode":
                return self._reset(raw_args)
            if tool == "end_episode":
                return self._end()
            run = self.session.call
            if self.session.logger is None:
                if self.session.episode is not None and self.session.episode.done:
                    return self._error("episode_closed", "the episode has ended (report submitted)")
                return self._error("no_episode", "no episode is running (the runner must call reset_episode)")
            spec = self.specs.get(tool)
            if spec is None:
                out = run(tool, raw_args if isinstance(raw_args, dict) else {"_raw": raw_args},
                          model_reasoning=reasoning, tokens=tokens)
                return self._result(out)
            try:
                args, warnings = validate_args(spec.model, raw_args)
            except ArgsRejected as e:
                out = run(tool, raw_args if isinstance(raw_args, dict) else {"_raw": raw_args},
                          rejected=f"{tool}: {e}", model_reasoning=reasoning, tokens=tokens)
                return self._result(out)
            out = run(tool, raw_args or {}, args_after_validation=args, server_warnings=warnings,
                      model_reasoning=reasoning, tokens=tokens)
            return self._result(out)
        except BridgeError as e:
            log.error("bridge error in %s: %s", tool, e)
            return self._error("environment", f"{e.type}: {e.message}")
        except RuntimeError as e:
            return self._error("episode_closed", str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected error in %s", tool)
            return self._error("environment", f"{type(e).__name__}: {e}")

    def _reset(self, raw_args) -> types.CallToolResult:
        try:
            args, _ = validate_args(self.specs["reset_episode"].model, raw_args)
        except ArgsRejected as e:
            return self._error("invalid_argument", f"reset_episode: {e}")
        bad = nonfinite_paths(args)
        if bad:
            return self._error("invalid_argument", f"reset_episode: inf or nan at {', '.join(bad[:5])}")
        errors = [] if self.display_check == "off" else \
            display.evaluate(display.display_status(asleep_only=True), self.display_check)[0]
        if errors:
            self.session.close("abandoned")
            return self._error("environment", f"display: {errors[0]}")
        out = self.session.reset(args["case_id"], args.get("client_meta") or {})
        res = self._result(out)
        res.meta = {**(res.meta or {}), META_EPISODE: self.session.logger.episode_id}
        return res

    def _end(self) -> types.CallToolResult:
        """Close the open episode (abandoned); idempotent."""
        self.session.close("abandoned")
        d = self.session.last_trace_dir
        meta = {}
        if d is not None:
            try:
                meta = json.loads((d / "meta.json").read_text())
            except (OSError, ValueError):
                pass
        result = {"trace_dir": str(d) if d else None, "episode_id": meta.get("episode_id"),
                  "status": meta.get("status")}
        body = {"step_index": None, "state": None, "result": result, "warnings": [], "error": None, "frames": []}
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(body))], is_error=False)

    @staticmethod
    def payload(out) -> dict:
        return {"step_index": out.step, "state": out.state, "result": out.result or {}, "warnings": out.warnings,
                "error": out.error, "frames": list(out.frames)}

    def _result(self, out) -> types.CallToolResult:
        content = [types.TextContent(text=json.dumps(self.payload(out), allow_nan=False))]
        for png in out.frames.values():
            content.append(types.ImageContent(data=base64.b64encode(png).decode(), mime_type="image/png"))
        meta = {**({META_IDENTITY: out.identity} if getattr(out, "identity", None) else {}),
                **({META_CASE_INFO: out.case_info} if getattr(out, "case_info", None) else {})} or None
        return types.CallToolResult(content=content, is_error=out.error is not None, meta=meta)

    def _error(self, kind: str, message: str) -> types.CallToolResult:
        step = self.session.step_index if self.session.logger is not None else None
        body = {"step_index": step, "state": self.session.last_state or None, "result": {}, "warnings": [],
                "error": {"type": kind, "message": message}, "frames": []}
        return types.CallToolResult(content=[types.TextContent(text=json.dumps(body))], is_error=True)

    async def run_stdio(self):
        async with stdio_server() as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())


def build(profile=None, slicer_url=None, runtime=None, client=None, report=None) -> EnvServer:
    """Load the profile and the runtime config, ping the bridge and refuse on a mismatch, and return the server (not
    yet running)."""
    try:
        prof = load_profile(profile, report=report)
        rt = load_runtime(runtime)
    except (OSError, ValueError, KeyError) as e:
        raise StartupError(f"cannot load the profile or runtime config: {type(e).__name__}: {e}") from None
    injected = client
    expected = os.environ.get("SLICER_MCP_LAUNCH_NONCE") or rt.get("launch_nonce") or None
    if client is None:
        client = BridgeClient(slicer_url or bridge_url(rt), obs_size=prof.protocol["observation"]["size_px"],
                              timeout=rt["client_timeout_s"], expected_launch=expected, token=launch_token(rt))
    try:
        ping = client.ping()
    except Exception as e:  # noqa: BLE001
        raise StartupError(f"bridge not reachable at {getattr(client, 'url', '?')}: {e}") from None
    check_bridge(ping, prof, expected_launch=expected, require_eval=bool(rt.get("require_eval_launch", False)))
    mode = rt.get("display_check", "require") if injected is None else "off"
    status = display.display_status() if mode != "off" else None
    if status is not None:
        errors, warnings = display.evaluate(status, mode)
        for w in warnings:
            log.warning("display: %s", w)
        if errors:
            raise StartupError(f"display: {errors[0]}")
    session = EnvSession(client, prof, rt["traces_dir"], ping_info=ping, stale_guard=rt.get("stale_guard"),
                         host_info={"display": status})
    return EnvServer(session, prof, display_check=mode)


def add_args(ap):
    ap.add_argument("--profile", help="the task profile: a folder with protocol.json, or a name in profiles/ "
                                      "(default: $SLICER_MCP_PROFILE, else ct_basic)")
    ap.add_argument("--slicer-url", help="bridge URL (default: SLICER_URL, else config/runtime.json)")
    ap.add_argument("--runtime", help="runtime config (default: $SLICER_MCP_RUNTIME, else config/runtime.json)")
    ap.add_argument("--report", help="the task's report plugin, a Python file (default: the profile's mcp.report, else "
                                     "the built-in JSON report; docs: src/slicer_mcp/reports.py)")


def run(a):
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        srv = build(a.profile, a.slicer_url, a.runtime, report=a.report)
    except StartupError as e:
        print(f"slicer-mcp: refusing to start: {e}", file=sys.stderr)
        sys.exit(2)
    log.info("serving profile %s (%d tools, %s, protocol %s) on stdio", srv.profile.name, len(srv.specs),
             TOOL_SCHEMA_VERSION, srv.protocol.sha256[:12])
    try:
        anyio.run(srv.run_stdio)
    finally:
        srv.session.close("abandoned")
        release = getattr(srv.session.client, "release", None)
        if release is not None:
            release()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slicer-mcp serve", description="Slicer MCP server (stdio)")
    add_args(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
