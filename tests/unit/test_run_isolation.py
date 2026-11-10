"""Run isolation without Slicer: the bridge's guard and its real Bridge.dispatch (slicer, vtk and qt
stubbed) behind a small HTTP server, driven by the real BridgeClient; and EnvSession / the MCP server over the fake
bridge.

- a request without the bridge token, or with another launch id, is refused before any handler runs; ping without
  the token reports only that the bridge is alive
- reset binds the episode; a foreign reset or call is refused; identity on every response
- reset returns case_info (volume_sha256 only in the runner's _meta, never in the model's text)
- a dropped connection after the bridge ran a mutation: the retry is answered from the cache, not run again
- observe: runner-only capture, logged as a step
- a frame without a render after the call is stale
"""

import importlib
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from slicer_mcp.bridge_client import BridgeClient, BridgeError
from slicer_mcp.config import REPO
from slicer_mcp.session import EnvSession
from slicer_mcp.stale import StaleFrameGuard
from tests.unit.fakes import FakeClient, png

STUBS = ("slicer", "qt", "vtk", "vtk.util", "vtk.util.numpy_support")
TOKEN = "t0k" * 16
T = {"token": TOKEN}


@pytest.fixture
def lib(monkeypatch):
    """SlicerMCPBridgeLib.server and .guard, imported with Slicer stubbed."""
    for m in STUBS:
        monkeypatch.setitem(sys.modules, m, mock.MagicMock())
    ws = types.ModuleType("WebServer")
    ws.SlicerHTTPServer = object
    wsl = types.ModuleType("WebServerLib")
    brh = types.ModuleType("WebServerLib.BaseRequestHandler")
    brh.BaseRequestHandler = object
    for name, mod in (("WebServer", ws), ("WebServerLib", wsl), ("WebServerLib.BaseRequestHandler", brh)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.syspath_prepend(str(REPO / "slicer_bridge" / "SlicerMCPBridge"))
    before = set(sys.modules)
    server = importlib.import_module("SlicerMCPBridgeLib.server")
    guard = importlib.import_module("SlicerMCPBridgeLib.guard")
    yield types.SimpleNamespace(server=server, guard=guard)
    for name in set(sys.modules) - before:
        if name.startswith("SlicerMCPBridgeLib"):
            del sys.modules[name]


class FakeObs:
    size_px = 8

    def __init__(self, draws=True):
        self.render_counts = {"3d": 0, "axial": 0, "sagittal": 0, "coronal": 0}
        self.draws = draws

    def capture(self, view):
        if self.draws:
            self.render_counts[view] += 1
        return png(self.render_counts[view], 8)

    def ensure_visible(self):
        return []


class FakeVolume:
    def GetName(self):
        return "case_x"


class FakeEnv:
    def __init__(self):
        self.extra_images, self.identity, self.volume, self.volume_sha256 = {}, None, FakeVolume(), None
        self.runs = []

    def state(self):
        return {"case_loaded": True, "n": len(self.runs)}


def make_bridge(lib, launch_id="L1", lease_s=900.0, clock=None, draws=True):
    """A server.Bridge without Slicer: the real dispatch, guard and capture loop; handlers count their runs."""
    b = object.__new__(lib.server.Bridge)
    b.launch = "eval"
    b.guard = lib.guard.RequestGuard(launch_id, lease_s=lease_s, token=TOKEN, **({"clock": clock} if clock else {}))
    b.obs, b.env, b._depth = FakeObs(draws), FakeEnv(), 0
    b.info = lambda: {"launch": "eval", "launch_id": b.guard.launch_id}

    def reset(env, args, warnings):
        env.runs.append("reset_episode")
        env.volume_sha256 = "h" * 64
        return {"case_info": {"dims": [1, 2, 3], "volume_sha256": env.volume_sha256}}

    def orbit(env, args, warnings):
        env.runs.append("orbit")
        return {}
    handlers = dict(lib.server.HANDLERS, reset_episode=reset, orbit=orbit)
    lib.server.HANDLERS.clear()
    lib.server.HANDLERS.update(handlers)
    return b


@pytest.fixture
def http(lib):
    """Serve a Bridge over HTTP; `drop` = names whose next response is lost after the bridge ran the call."""
    servers = []

    def serve(bridge, drop=()):
        pending = set(drop)

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                name = self.path.rsplit("/", 1)[-1]
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                out = json.dumps(bridge.dispatch(name, body)).encode()
                if name in pending:
                    pending.discard(name)
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"
    yield serve
    for s in servers:
        s.shutdown()


def test_launch_id_from_env_or_random(lib):
    g = lib.guard
    assert g.new_launch_id({"SLICER_MCP_LAUNCH_NONCE": "job42-abc"}) == "job42-abc"
    a, b = g.new_launch_id({}), g.new_launch_id({})
    assert a != b and len(a) == 32
    with pytest.raises(ValueError):
        g.new_launch_id({"SLICER_MCP_LAUNCH_NONCE": "bad id with spaces"})


def test_guard_rules(lib):
    t = [0.0]
    g = lib.guard.RequestGuard("L", lease_s=10, clock=lambda: t[0], token=TOKEN)
    R = lib.guard.Refused
    assert g.admit("ping", {}) is None
    with pytest.raises(R, match="launch"):
        g.admit("orbit", {**T, "launch": "other"})
    assert g.admit("orbit", {**T, "launch": "L"}) == 1
    with pytest.raises(R) as e:
        g.admit("reset_episode", {**T, "launch": "L"})
    assert e.value.kind == "episode_required"
    assert g.admit("reset_episode", {**T, "launch": "L", "episode": "e1"}) == 2
    with pytest.raises(R) as e:
        g.admit("reset_episode", {**T, "launch": "L", "episode": "e2"})
    assert e.value.kind == "episode_owned"
    with pytest.raises(R) as e:
        g.admit("orbit", {**T, "launch": "L", "episode": "e2"})
    assert e.value.kind == "episode_mismatch"
    assert g.admit("reset_episode", {**T, "launch": "L", "episode": "e2", "prev_episode": "e1"}) == 3
    t[0] = 11.0
    assert g.admit("reset_episode", {**T, "launch": "L", "episode": "e3"}) == 4 and g.episode_id == "e3"
    g.admit("release_episode", {**T, "launch": "L", "episode": "e3"})
    g.release()
    assert g.episode_id is None


def test_token_is_required_for_every_call_but_ping(lib):
    g = lib.guard.RequestGuard("L", token=TOKEN)
    R = lib.guard.Refused
    assert g.admit("ping", {}) is None
    for env in ({"launch": "L"}, {"launch": "L", "token": "x" * 48}):
        with pytest.raises(R) as e:
            g.admit("orbit", env)
        assert e.value.kind == "unauthorized" and "launch" not in e.value.message
    assert g.admit("orbit", {**T, "launch": "L"}) == 1
    assert lib.guard.new_token({"SLICER_MCP_BRIDGE_TOKEN": TOKEN}) == TOKEN and len(lib.guard.new_token({})) == 64
    with pytest.raises(ValueError):
        lib.guard.new_token({"SLICER_MCP_BRIDGE_TOKEN": "short"})


def test_without_the_token_nothing_runs_and_ping_hides_the_launch_id(lib, http):
    b = make_bridge(lib, "L1")
    url = http(b)
    anon = BridgeClient(url, connect_retries=0, token=None)
    anon.token = None
    assert anon.ping() == {"alive": True, "bridge_version": lib.server.limits.BRIDGE_VERSION}
    for name in ("reset_episode", "orbit", "capture", "release_episode"):
        out = b.dispatch(name, json.dumps({"args": {"case_id": "case_x"}, "capture": ["3d"],
                                           "slicer_mcp": {"launch": "L1", "episode": "e", "request": name}}).encode())
        assert not out["ok"] and out["error"]["type"] == "unauthorized" and "images" not in out, name
        assert "L1" not in json.dumps(out)
    assert b.env.runs == [] and b.guard.episode_id is None
    with pytest.raises(BridgeError) as e:
        anon.call("orbit", {})
    assert e.value.type == "unauthorized"
    assert BridgeClient(url, connect_retries=0, token=TOKEN).ping()["launch_id"] == "L1"


def test_bridge_response_with_inf_or_nan_becomes_an_internal_error(lib):
    b = make_bridge(lib)
    lib.server.HANDLERS["orbit"] = lambda env, args, warnings: {"pts": [1.0, float("inf")], "ok": 2.0}
    h = lib.server.BridgeRequestHandler(b)
    env = {**T, "launch": "L1", "episode": "ep1"}
    h.handleRequest("POST", b"/bridge/reset_episode", json.dumps(
        {"args": {"case_id": "case_x"}, "slicer_mcp": {**env, "request": "r0"}}).encode())
    kind, body = h.handleRequest("POST", b"/bridge/orbit", json.dumps(
        {"args": {}, "slicer_mcp": {**env, "request": "r1"}}).encode())
    out = json.loads(body)
    assert kind == b"application/json" and out["ok"] is False and out["error"]["type"] == "internal"
    assert "result.pts[1]" in out["error"]["message"] and out["identity"]["seq"] == 2
    assert lib.server.first_nonfinite({"a": [{"b": float("nan")}], "c": float("inf")}) == "a[0].b"
    assert lib.server.first_nonfinite({"a": 1.0}) is None


def test_wrong_launch_is_refused_before_the_handler(lib, http):
    b = make_bridge(lib, "L1")
    url = http(b)
    with pytest.raises(BridgeError) as e:
        BridgeClient(url, connect_retries=0, expected_launch="L0", token=TOKEN).ping()
    assert e.value.type == "wrong_launch"
    c = BridgeClient(url, connect_retries=0, expected_launch="L0", token=TOKEN)
    with pytest.raises(BridgeError) as e:
        c.call("orbit", {})
    assert e.value.type == "wrong_launch" and b.env.runs == []


def test_reset_binds_and_a_foreign_client_is_refused(lib, http):
    b = make_bridge(lib)
    url = http(b)
    owner, other = BridgeClient(url, connect_retries=0, token=TOKEN), BridgeClient(url, connect_retries=0, token=TOKEN)
    r = owner.call("reset_episode", {"case_id": "case_x"}, capture=("3d",), episode_id="ep1")
    assert r.identity == {"episode_id": "ep1", "case_id": "case_x", "volume_sha256": "h" * 64, "seq": 1}
    assert r.result["case_info"]["volume_sha256"] == "h" * 64
    assert r.render == {"3d": {"before": 0, "frame": 1}}
    for name in ("reset_episode", "orbit"):
        with pytest.raises(BridgeError) as e:
            other.call(name, {"case_id": "case_x"})
        assert e.value.type in ("episode_owned", "episode_mismatch")
    assert b.env.runs == ["reset_episode"]
    r = owner.call("orbit", {}, capture=("3d",))
    assert r.identity["episode_id"] == "ep1" and r.identity["seq"] == 2
    r2 = owner.call("reset_episode", {"case_id": "case_x"}, episode_id="ep2")
    assert r2.identity["episode_id"] == "ep2" and owner.episode_id == "ep2"
    owner.release()
    other.call("reset_episode", {"case_id": "case_x"}, episode_id="ep3")
    assert b.guard.episode_id == "ep3"


def test_a_lost_response_is_not_run_twice(lib, http):
    b = make_bridge(lib)
    url = http(b, drop=("orbit",))
    c = BridgeClient(url, connect_retries=2, token=TOKEN)
    c.call("reset_episode", {"case_id": "case_x"}, episode_id="ep1")
    r = c.call("orbit", {})
    assert b.env.runs == ["reset_episode", "orbit"]
    assert r.raw.get("replayed") is True and r.identity["seq"] == 2


def test_a_cached_response_goes_only_to_the_same_call(lib):
    """Ping needs no launch id, so a ping (or any other call) reusing a request id must not get that request's
    cached response; the exact retry still does."""
    b = make_bridge(lib, "L1")
    env = {**T, "launch": "L1", "episode": "ep1", "request": "r1"}
    body = {"args": {"case_id": "case_x"}, "capture": ["3d"], "slicer_mcp": env}
    first = b.dispatch("reset_episode", json.dumps(body).encode())
    assert first["ok"] and first.get("images")
    anon = b.dispatch("ping", json.dumps({"slicer_mcp": {"request": "r1"}}).encode())
    assert anon["ok"] and anon["result"] == b.public_info() and "images" not in anon
    for name, other in (("ping", {"slicer_mcp": {**T, "request": "r1"}}),
                        ("orbit", {"args": {}, "slicer_mcp": {**env, "episode": "ep1"}}),
                        ("reset_episode", {**body, "capture": []})):
        out = b.dispatch(name, json.dumps(other).encode())
        assert not out["ok"] and out["error"]["type"] == "request_id_reused" and "images" not in out, name
    retry = b.dispatch("reset_episode", json.dumps(body).encode())
    assert retry["replayed"] is True and retry["images"] == first["images"]
    assert b.env.runs == ["reset_episode"]


def test_a_bridge_response_with_inf_or_nan_is_refused(lib, http):
    """A response with NaN / Infinity is refused by the client too (it would end up in the session's state and the
    strict-JSON trace)."""
    b = make_bridge(lib)
    lib.server.HANDLERS["orbit"] = lambda env, args, warnings: {"distance": float("nan")}
    c = BridgeClient(http(b), connect_retries=0, token=TOKEN)
    c.call("reset_episode", {"case_id": "case_x"}, episode_id="ep1")
    with pytest.raises(BridgeError) as e:
        c.call("orbit", {})
    assert e.value.type == "protocol" and "result.distance" in e.value.message


def test_ids_with_a_trailing_newline_are_refused(lib):
    """`$` in re.match accepts a trailing newline; ids are checked with fullmatch."""
    with pytest.raises(lib.guard.Refused):
        lib.guard.envelope({"slicer_mcp": {"request": "r1\n"}})
    with pytest.raises(ValueError):
        lib.guard.new_launch_id({"SLICER_MCP_LAUNCH_NONCE": "abc\n"})


def test_a_bridge_that_did_not_render_gives_a_stale_frame(lib, http):
    b = make_bridge(lib, draws=False)
    c = BridgeClient(http(b), connect_retries=0, token=TOKEN)
    r = c.call("reset_episode", {"case_id": "case_x"}, capture=("3d",), episode_id="ep1")
    assert r.render == {"3d": {"before": 0, "frame": 0}}
    stale = StaleFrameGuard().check(r.images, r.state, r.render)
    assert stale and "not rendered" in stale[0]


def test_render_counts_flag_any_frame_not_drawn():
    g = StaleFrameGuard()
    f = {"3d": png(1), "axial": png(2)}
    assert g.check(f, {}, {"3d": {"before": 3, "frame": 4}, "axial": {"before": 3, "frame": 5}}) == []
    out = g.check(f, {}, {"3d": {"before": 4, "frame": 4}, "axial": {"before": 5, "frame": 6}})
    assert len(out) == 1 and out[0].startswith("3d")
    assert g.check({"montage": png(3)}, {}, {}) == []


def test_session_records_identity_and_refuses_another_episodes_answer(protocol, tmp_path, seg_profile):
    fake = FakeClient(protocol)
    sess = EnvSession(fake, seg_profile, tmp_path)
    out = sess.reset("case_test")
    assert out.identity["episode_id"] == sess.trace_dir.name == fake.episode_id
    assert "volume_sha256" not in out.result["case_info"] and out.case_info["volume_sha256"] == "v" * 64
    step = sess.call("orbit", {"azimuth_deg": 10})
    assert step.identity["episode_id"] == sess.trace_dir.name
    fake.episode_id = "someone_else"
    with pytest.raises(BridgeError) as e:
        sess.call("orbit", {"azimuth_deg": 20})
    assert e.value.type == "identity_mismatch"
    lines = [json.loads(x) for x in (sess.trace_dir / "steps.jsonl").read_text().splitlines()]
    assert lines[0]["identity"]["volume_sha256"] == "v" * 64 and lines[0]["case_info"]["volume_sha256"] == "v" * 64
    assert lines[1]["identity"]["seq"] > lines[0]["identity"]["seq"]
    assert "volume_sha256" not in lines[0]["result"]["case_info"]


def test_mcp_meta_carries_identity_and_case_info_not_the_text(make_server):
    srv = make_server()
    reset = srv.handle("reset_episode", {"case_id": "case_test"})
    text = reset.content[0].text
    assert "volume_sha256" not in text and "identity" not in text
    assert reset.meta["slicer_mcp/identity"]["volume_sha256"] == "v" * 64
    assert reset.meta["slicer_mcp/case_info"]["volume_sha256"] == "v" * 64 and reset.meta["slicer_mcp/case_info"]["dims"]
    obs = srv.handle("observe", {})
    p = json.loads(obs.content[0].text)
    assert not obs.is_error and p["step_index"] == 1 and p["frames"] == ["3d", "axial", "sagittal", "coronal"]
    assert obs.meta["slicer_mcp/identity"]["episode_id"] == reset.meta["slicer_mcp/episode_id"]
    assert "observe" in {t.name for t in srv.tools() if (t.meta or {}).get("slicer_mcp/runner_only")}


def test_server_refuses_a_bridge_without_or_with_another_launch_id(protocol, seg_profile):
    from slicer_mcp.server import StartupError, check_bridge
    ping = FakeClient(protocol).ping()
    check_bridge(ping, seg_profile, expected_launch="fakelaunch")
    with pytest.raises(StartupError, match="launch_id"):
        check_bridge({**ping, "launch_id": None}, seg_profile)
    with pytest.raises(StartupError, match="not this job"):
        check_bridge(ping, seg_profile, expected_launch="other")
    with pytest.raises(StartupError, match="eval"):
        check_bridge({**ping, "launch": "dev"}, seg_profile, require_eval=True)


def test_render_stack_is_parsed_from_the_gl_report(monkeypatch):
    """VTK / OpenGL / Mesa versions for provenance, from vtkRenderWindow.ReportCapabilities()."""
    monkeypatch.syspath_prepend(str(REPO / "slicer_bridge" / "SlicerMCPBridge"))
    ri = importlib.import_module("SlicerMCPBridgeLib.renderinfo")
    report = ("OpenGL vendor string:  Mesa\nOpenGL renderer string:  llvmpipe (LLVM 19.1.1, 256 bits)\n"
              "OpenGL version string:  4.5 (Core Profile) Mesa 24.2.8-1ubuntu1~24.04.1\nOpenGL extensions:  \n")
    gl = ri.parse_gl_report(report)
    assert gl == {"vendor": "Mesa", "renderer": "llvmpipe (LLVM 19.1.1, 256 bits)",
                  "version": "4.5 (Core Profile) Mesa 24.2.8-1ubuntu1~24.04.1"}
    assert ri.mesa_version(gl) == "24.2.8"
    mac = ri.parse_gl_report("OpenGL vendor string:  Apple\nOpenGL version string:  4.1 Metal - 89.3\n")
    assert mac["renderer"] is None and ri.mesa_version(mac) is None
    assert ri.parse_gl_report("") == {"vendor": None, "renderer": None, "version": None}


def test_trace_meta_records_the_render_stack(protocol, tmp_path, seg_profile):
    """The bridge's vtk / opengl / mesa reach trace/meta.json (the client's provenance reads them there)."""
    sess = EnvSession(FakeClient(protocol), seg_profile, tmp_path)
    sess.reset("case_test")
    meta = json.loads((sess.trace_dir / "meta.json").read_text())
    assert meta["vtk"] == "9.4.20250513" and meta["mesa"] == "24.2.8"
    assert meta["opengl"]["renderer"].startswith("llvmpipe")
