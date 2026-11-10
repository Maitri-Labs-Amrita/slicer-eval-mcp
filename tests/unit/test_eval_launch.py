"""slicer-mcp launch (slicer_mcp.launch) without Slicer: the command line, the port refusal, and the checks against a
fake bridge."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from slicer_mcp import launch as es
from slicer_mcp.config import REPO, load_protocol


def test_command_line_has_no_webserver_and_honours_the_window_flag():
    cmd = es.slicer_command("/x/Slicer", 2017)
    assert cmd[0] == "/x/Slicer" and "--no-main-window" not in cmd
    assert {"--no-splash", "--ignore-slicerrc", "--disable-settings"} <= set(cmd)
    i = cmd.index("--python-script")
    assert cmd[i + 1] == str(REPO / "slicer_bridge" / "SlicerMCPBridge" / "start_bridge.py")
    assert cmd[cmd.index("--") + 1:] == ["--port", "2017"]
    assert not any("WebServer" in c or "2016" in c for c in cmd)
    head = es.slicer_command("/x/Slicer", 2018, main_window=False, config="c.json", runtime="r.json")
    assert "--no-main-window" in head and head[head.index("--") + 1:] == ["--port", "2018", "--config", "c.json",
                                                                         "--runtime", "r.json"]


def test_slicer_path_from_flag_env_or_runtime(monkeypatch, tmp_path):
    exe = tmp_path / "Slicer"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.delenv("SLICER_MCP_SLICER_EXECUTABLE", raising=False)
    assert es.resolve_slicer(None, {"slicer_executable": str(exe)}) == str(exe)
    monkeypatch.setenv("SLICER_MCP_SLICER_EXECUTABLE", str(exe))
    assert es.resolve_slicer(None, {"slicer_executable": "/nope"}) == str(exe)
    with pytest.raises(SystemExit):
        es.resolve_slicer("/does/not/exist", {})


def _serve(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def fake_bridge(protocol, launch="eval", exec_status=404):
    info = {"launch": launch, "launch_id": "L" * 32, "config_sha256": protocol.sha256, "bridge_version": "0.3.0",
            "main_window_visible": True, "layout_manager": True, "slice_logic_count": 9,
            "obs_window": {"visible": True, "minimized": False},
            "render_sizes": {v: [512, 512] for v in ("3d", "axial", "sagittal", "coronal")}}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body=b""):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/bridge/ping":
                return self._send(200, json.dumps({"ok": True, "result": info}).encode())
            self._send(exec_status if self.path == "/slicer/exec" else 404)

        def do_GET(self):
            self._send(404)
    return _serve(H)


@pytest.fixture()
def no_dev_port(monkeypatch):
    real = es.port_open
    monkeypatch.setattr(es, "port_open", lambda host, port, timeout=0.5: False if port == es.DEV_PORT
                        else real(host, port, timeout))


def test_check_passes_on_an_eval_bridge(no_dev_port):
    protocol = load_protocol()
    srv = fake_bridge(protocol)
    try:
        problems, facts = es.check_launch("127.0.0.1", srv.server_port, protocol)
    finally:
        srv.shutdown()
    assert problems == [], problems
    assert facts["POST /slicer/exec"] == 404 and facts["port_2016_listening"] is False


@pytest.mark.parametrize("kw,needle", [({"launch": "dev"}, "launch is 'dev'"),
                                       ({"exec_status": 200}, "/slicer/exec on the bridge port returned 200")])
def test_check_fails_on_dev_bridge_or_exec_route(no_dev_port, kw, needle):
    protocol = load_protocol()
    srv = fake_bridge(protocol, **kw)
    try:
        problems, _ = es.check_launch("127.0.0.1", srv.server_port, protocol)
    finally:
        srv.shutdown()
    assert any(needle in p for p in problems), problems


def test_check_fails_when_2016_listens(monkeypatch):
    protocol = load_protocol()
    srv = fake_bridge(protocol)
    dev = socket.socket()
    dev.bind(("127.0.0.1", 0))
    dev.listen()
    monkeypatch.setattr(es, "DEV_PORT", dev.getsockname()[1])
    try:
        problems, facts = es.check_launch("127.0.0.1", srv.server_port, protocol)
    finally:
        srv.shutdown()
        dev.close()
    assert facts["port_2016_listening"] is True and any("listens on" in p for p in problems)


def test_start_refuses_when_a_port_is_busy(monkeypatch):
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen()
    port = busy.getsockname()[1]
    try:
        with pytest.raises(SystemExit, match="ports .* in use"):
            es.main(["--port", str(port), "--slicer", "/does/not/matter"])
    finally:
        busy.close()


FAKE_SLICER = '''"""A fake Slicer: serves a bridge ping on --port (0 = a free one), writes --port-file, reports $SLICER_MCP_LAUNCH_NONCE."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[sys.argv.index("--") + 1:]
port = int(argv[argv.index("--port") + 1])
pf = argv[argv.index("--port-file") + 1] if "--port-file" in argv else None
open(os.environ["FAKE_ENV_OUT"], "w").write(json.dumps({{k: os.environ.get(k) for k in ("HOME", "TMPDIR",
                                                          "XDG_CONFIG_HOME", "SLICER_MCP_LAUNCH_NONCE",
                                                          "SLICER_MCP_BRIDGE_TOKEN")}}))
info = {{"launch": "eval", "launch_id": os.environ["SLICER_MCP_LAUNCH_NONCE"], "config_sha256": "{sha}",
        "obs_window": {{"visible": True, "minimized": False}},
        "render_sizes": {{v: [512, 512] for v in ("3d", "axial", "sagittal", "coronal")}}}}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, body=b""):
        self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{{}}")
        if self.path == "/bridge/ping":
            authed = (body.get("slicer_mcp") or {{}}).get("token") == os.environ["SLICER_MCP_BRIDGE_TOKEN"]
            return self._send(200, json.dumps({{"ok": True, "result": info if authed else {{"alive": True}}}}).encode())
        self._send(404)
    def do_GET(self):
        self._send(404)
srv = HTTPServer(("127.0.0.1", port), H)
if pf:
    json.dump({{"port": srv.server_address[1], "launch_id": info["launch_id"], "pid": os.getpid()}}, open(pf + ".t", "w"))
    os.replace(pf + ".t", pf)
srv.serve_forever()
'''


def _runtime(tmp_path, port, **el):
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    rt["bridge_port"] = port
    rt["eval_launch"] = {"startup_timeout_s": 30, "caffeinate": False, "dev_port": None,
                         "log": str(tmp_path / "job" / "logs" / "slicer.log"),
                         "state_file": str(tmp_path / "job" / "state" / "eval.json"), **el}
    p = tmp_path / "runtime.json"
    p.write_text(json.dumps(rt))
    return p


def test_auto_port_launch_outside_the_repo_writes_the_port_back(tmp_path, monkeypatch):
    """Port auto, log and state file outside the repo (parents created), the job's own HOME / TMPDIR, the
    nonce handed to Slicer, and the bound port and nonce written into the job's runtime file and state file."""
    protocol = load_protocol()
    exe = tmp_path / "Slicer"
    (tmp_path / "py").symlink_to(sys.executable)
    exe.write_text(f"#!{tmp_path / 'py'}\n" + FAKE_SLICER.format(sha=protocol.sha256))
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_ENV_OUT", str(tmp_path / "env.json"))
    monkeypatch.delenv("SLICER_MCP_LAUNCH_NONCE", raising=False)
    monkeypatch.delenv("SLICER_MCP_BRIDGE_TOKEN", raising=False)
    rtp = _runtime(tmp_path, "auto")
    try:
        es.main(["--runtime", str(rtp), "--slicer", str(exe), "--no-main-window"])
        rt = json.loads(rtp.read_text())
        state = json.loads((tmp_path / "job" / "state" / "eval.json").read_text())
        assert isinstance(rt["bridge_port"], int) and rt["bridge_port"] > 0 and rt["bridge_port"] == state["port"]
        assert rt["launch_nonce"] == state["launch_id"] and len(rt["launch_nonce"]) == 32
        env = json.loads((tmp_path / "env.json").read_text())
        assert env["SLICER_MCP_LAUNCH_NONCE"] == rt["launch_nonce"]
        assert env["HOME"].startswith(str(tmp_path / "job" / "state")) and env["TMPDIR"] != os.environ.get("TMPDIR")
        assert (tmp_path / "job" / "logs" / "slicer.log").exists()
        url = f"http://127.0.0.1:{rt['bridge_port']}"
        assert es.ping(url, token=state["token"])["launch_id"] == rt["launch_nonce"]
        assert es.ping(url) == {"alive": True}
    finally:
        es.main(["--runtime", str(rtp), "--stop"])
    assert not (tmp_path / "job" / "state" / "eval.json").exists()
    assert not Path(env["HOME"]).parent.exists()


def _fake_slicer(tmp_path, protocol, monkeypatch):
    exe = tmp_path / "Slicer"
    (tmp_path / "py").symlink_to(sys.executable)
    exe.write_text(f"#!{tmp_path / 'py'}\n" + FAKE_SLICER.format(sha=protocol.sha256))
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_ENV_OUT", str(tmp_path / "env.json"))
    monkeypatch.delenv("SLICER_MCP_LAUNCH_NONCE", raising=False)
    monkeypatch.delenv("SLICER_MCP_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("SLICER_MCP_PROFILE", raising=False)
    return exe


def test_launch_state_holds_token_and_profile_and_check_uses_them(tmp_path, monkeypatch, capsys):
    """The state file is 0600 and records the bridge token and the profile; `--check` without --profile checks the
    recorded profile and authenticates with the recorded token."""
    protocol = load_protocol("ct_segments")
    assert protocol.sha256 != load_protocol("ct_basic").sha256
    exe = _fake_slicer(tmp_path, protocol, monkeypatch)
    rtp = _runtime(tmp_path, "auto")
    sp = tmp_path / "job" / "state" / "eval.json"
    try:
        es.main(["--runtime", str(rtp), "--slicer", str(exe), "--profile", "ct_segments"])
        state = json.loads(sp.read_text())
        assert sp.stat().st_mode & 0o777 == 0o600
        assert Path(state["profile"]) == protocol.path.parent and len(state["token"]) == 64
        env = json.loads((tmp_path / "env.json").read_text())
        assert env["SLICER_MCP_BRIDGE_TOKEN"] == state["token"]
        capsys.readouterr()
        es.main(["--runtime", str(rtp), "--check"])
        assert "OK: eval launch" in capsys.readouterr().out
        state["token"] = "0" * 64
        es.write_state(sp, state)
        with pytest.raises(SystemExit):
            es.main(["--runtime", str(rtp), "--check"])
        assert "liveness ping" in capsys.readouterr().out
    finally:
        es.main(["--runtime", str(rtp), "--stop"])


class _Proc:
    def __init__(self, rc=None):
        self.returncode = rc

    def poll(self):
        return self.returncode


def test_wait_accepts_only_its_own_launch_and_only_while_alive():
    """Another job's bridge answering on the port is ignored; a dead launcher process ends the wait."""
    seen = iter([{"launch_id": "other"}, {"launch_id": "other"}, {"launch_id": "mine"}])
    got = es.wait_for_bridge(_Proc(), "127.0.0.1", 1, "mine", 10, poll_s=0, ping_fn=lambda u, timeout: next(seen))
    assert got == (1, {"launch_id": "mine"})
    assert es.wait_for_bridge(_Proc(3), "127.0.0.1", 1, "mine", 10, poll_s=0,
                              ping_fn=lambda u, timeout: {"launch_id": "mine"}) == ("exited", 3)
    assert es.wait_for_bridge(_Proc(), "127.0.0.1", 1, "mine", 0, poll_s=0,
                              ping_fn=lambda u, timeout: {"launch_id": "other"}) == ("timeout", None)


def test_check_reports_another_launch_id_and_can_skip_the_dev_port(no_dev_port):
    protocol = load_protocol()
    srv = fake_bridge(protocol)
    try:
        problems, facts = es.check_launch("127.0.0.1", srv.server_port, protocol, dev_port=None, launch_id="x" * 32)
    finally:
        srv.shutdown()
    assert any("launch_id" in p for p in problems) and facts["dev_port"] is None
    assert es.dev_port_of({"eval_launch": {"dev_port": None}}) is None
    assert es.dev_port_of({"eval_launch": {}}) == es.DEV_PORT and es.dev_port_of({}, "none") is None
    assert es.dev_port_of({}, "2099") == 2099


def test_stop_checks_process_identity(tmp_path):
    """A recorded pid that now belongs to another process (other start time) is not killed."""
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        for _ in range(100):
            ident = es.proc_identity(p.pid)
            if ident and ident["cmdline"]:
                break
            time.sleep(0.02)
        assert ident and ident["cmdline"][0] == "sleep"
        assert es.same_process({"pid": p.pid, "proc": ident})
        assert not es.same_process({"pid": p.pid, "proc": {**ident, "start": "1"}})
        es.stop_pids({"pid": p.pid, "proc": {**ident, "start": "1"}}, wait_s=1)
        assert p.poll() is None
        es.stop_pids({"pid": p.pid, "proc": ident}, wait_s=5)
        assert p.wait(timeout=5) is not None
    finally:
        if p.poll() is None:
            p.kill()


def test_job_env_gives_each_slicer_its_own_home_and_tmp(tmp_path):
    env = es.job_env({"HOME": "/home/x", "PATH": "/bin"}, tmp_path / "s")
    assert env["HOME"] == str(tmp_path / "s" / "home") and env["TMPDIR"] == env["TMP"] == str(tmp_path / "s" / "tmp")
    assert env["PATH"] == "/bin" and all(Path(env[k]).is_dir() for k in ("HOME", "TMPDIR", "XDG_RUNTIME_DIR"))
    assert oct(os.stat(env["XDG_RUNTIME_DIR"]).st_mode & 0o777) == "0o700"


def test_exit_3_names_a_taken_port():
    log = "[slicer-mcp-bridge] FAILED TO START\nTraceback ...\nOSError: [Errno 98] Address already in use\n"
    msg = es.exit_reason(3, log, 2037)
    assert "port 2037 is already in use" in msg
    assert "already in use" in es.exit_reason(3, log)
    other = es.exit_reason(3, "[slicer-mcp-bridge] FAILED TO START\nKeyError: 'observation'\n")
    assert other.startswith("the bridge did not start") and "KeyError" in other and "in use" not in other
    assert es.exit_reason(1, log) == "Slicer exited with code 1"
