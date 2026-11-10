"""tools/xvfb_session.py without Xvfb: the command line, the path lookup, the protocol's fit on the default screen, the
environments, and the cleanup on exit and on SIGTERM (a fake Xvfb that reports a display and sleeps)."""

import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from slicer_mcp.config import REPO, load_protocol

spec = importlib.util.spec_from_file_location("xvfb_session", REPO / "tools" / "xvfb_session.py")
xs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xs)


def test_command_line_has_its_display_and_no_shared_sockets():
    cmd = xs.xvfb_command("/x/Xvfb", "1920x1200x24", 7, 12)
    assert cmd[0] == "/x/Xvfb" and cmd[1] == ":12"
    assert cmd[cmd.index("-displayfd") + 1] == "7"
    assert cmd[cmd.index("-screen") + 1:cmd.index("-screen") + 3] == ["0", "1920x1200x24"]
    nolisten = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-nolisten"]
    assert nolisten == ["tcp", "local"]


def test_free_display_skips_other_jobs_and_this_jobs_files(tmp_path):
    net = tmp_path / "unix"
    net.write_text("Num RefCount Protocol Flags Type St Inode Path\n"
                   "0: 00000002 00000000 00010000 0001 01 1 @/tmp/.X11-unix/X10\n"
                   "0: 00000002 00000000 00010000 0001 01 2 @/tmp/.X11-unix/X12\n"
                   "0: 00000002 00000000 00010000 0001 01 3 /run/user/1/bus\n")
    taken = xs.abstract_displays(net)
    assert taken == {10, 12}
    (tmp_path / ".X11").mkdir()
    (tmp_path / ".X11-unix").mkdir()
    (tmp_path / ".X11-lock").write_text("1")
    assert xs.free_display(taken, start=10, tmp=str(tmp_path)) == 13
    assert xs.abstract_displays(tmp_path / "missing") == set()


def test_xvfb_path_from_flag_env_or_runtime(monkeypatch, tmp_path):
    exe = tmp_path / "Xvfb"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.delenv("SLICER_MCP_XVFB", raising=False)
    assert xs.resolve_xvfb(None, {"xvfb_executable": str(exe)}) == str(exe)
    monkeypatch.setenv("SLICER_MCP_XVFB", str(exe))
    assert xs.resolve_xvfb(None, {"xvfb_executable": "/nope"}) == str(exe)
    with pytest.raises(SystemExit):
        xs.resolve_xvfb("/does/not/exist", {})


def test_default_screen_holds_the_observation_window():
    w, h, depth = (int(v) for v in xs.DEFAULT_SCREEN.split("x"))
    n = load_protocol()["observation"]["size_px"]
    assert w >= 2 * n and h >= 2 * n and depth == 24


def test_session_env_sets_display_and_puts_extra_libs_first():
    env = xs.session_env({"WAYLAND_DISPLAY": "wayland-0", "LD_LIBRARY_PATH": "/old", "HOME": "/h"}, ":3", ["/x/lib"])
    assert env["DISPLAY"] == ":3" and "WAYLAND_DISPLAY" not in env and env["HOME"] == "/h"
    assert env["LD_LIBRARY_PATH"] == "/x/lib:/old"
    assert "LD_LIBRARY_PATH" not in xs.session_env({}, ":3", None)


def test_xvfb_gets_the_libraries_but_no_display():
    env = xs.session_env({"DISPLAY": ":0", "PATH": "/bin"}, None, ["/x/lib"])
    assert "DISPLAY" not in env and env["LD_LIBRARY_PATH"] == "/x/lib" and env["PATH"] == "/bin"


def test_display_of_reads_a_proc_environ_blob():
    assert xs.display_of(b"HOME=/h\0DISPLAY=:42\0X=1\0") == ":42"
    assert xs.display_of(b"HOME=/h\0XDISPLAY=:1\0") is None
    assert xs.display_of(b"") is None


FAKE_XVFB = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, sys, time
    fd = int(sys.argv[sys.argv.index("-displayfd") + 1])
    os.write(fd, b"{num}\\n")
    os.close(fd)
    open(os.environ["FAKE_XVFB_PIDFILE"], "w").write(str(os.getpid()))
    time.sleep(300)
""")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def _wait_for(path, timeout=15):
    t0 = time.time()
    while not (path.exists() and path.read_text().strip()):
        assert time.time() - t0 < timeout, f"{path} never written"
        time.sleep(0.05)
    return path.read_text().split()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the display scan reads /proc")
@pytest.mark.parametrize("how", ["sigterm", "exit"])
def test_everything_on_the_display_is_stopped(tmp_path, how):
    num = 7000 + os.getpid() % 1000
    fake = tmp_path / "Xvfb"
    fake.write_text(FAKE_XVFB.format(num=num))
    fake.chmod(0o755)
    pids = tmp_path / "pids"
    script = (f"setsid sleep 300 & echo $! > {pids}.detached; echo $$ > {pids}; "
              + ("sleep 300" if how == "sigterm" else "sleep 1"))
    env = {**os.environ, "FAKE_XVFB_PIDFILE": str(tmp_path / "xvfb.pid")}
    w = subprocess.Popen([sys.executable, str(xs.__file__), "--xvfb", str(fake), "--", "sh", "-c", script], env=env)
    shell = int(_wait_for(pids)[0])
    detached = int(_wait_for(tmp_path / "pids.detached")[0])
    xvfb = int(_wait_for(tmp_path / "xvfb.pid")[0])
    assert _alive(shell) and _alive(detached) and _alive(xvfb)
    if how == "sigterm":
        w.send_signal(signal.SIGTERM)
    rc = w.wait(timeout=30)
    assert rc == (128 + signal.SIGTERM if how == "sigterm" else 0)
    time.sleep(0.2)
    assert not _alive(shell) and not _alive(detached) and not _alive(xvfb)


RUNNER = textwrap.dedent("""\
    import os, signal, sys, time
    out, slicer = sys.argv[1], int(open(sys.argv[2]).read())
    def finish(*_):
        time.sleep(1.5)  # exporting the shells, writing the run folder: Slicer must still be there
        try:
            os.kill(slicer, 0)
            alive = 1
        except OSError:
            alive = 0
        open(out, "w").write(f"slicer_alive={alive}")
        sys.exit(0)
    signal.signal(signal.SIGTERM, finish)
    open(out + ".ready", "w").write(str(os.getpid()))
    if sys.argv[3] == "exit":
        finish()
    time.sleep(300)
""")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the display scan reads /proc")
@pytest.mark.parametrize("how", ["sigterm", "exit"])
def test_the_command_group_finishes_before_slicer_is_stopped(tmp_path, how):
    """The job script's shell may die at once, but the runner in its process group gets the TERM and the grace
    to finish (Slicer, detached like the eval launch's, still alive), before everything on the display is stopped."""
    num = 7500 + os.getpid() % 400
    fake = tmp_path / "Xvfb"
    fake.write_text(FAKE_XVFB.format(num=num))
    fake.chmod(0o755)
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER)
    out, sl = tmp_path / "out", tmp_path / "slicer.pid"
    script = (f"setsid sleep 300 & echo $! > {sl}; {sys.executable} {runner} {out} {sl} {how} & "
              + ("wait" if how == "sigterm" else "exit 0"))
    env = {**os.environ, "FAKE_XVFB_PIDFILE": str(tmp_path / "xvfb.pid")}
    w = subprocess.Popen([sys.executable, str(xs.__file__), "--xvfb", str(fake), "--group", "--grace", "20", "--",
                          "bash", "-c", script], env=env, stdin=subprocess.DEVNULL)
    _wait_for(tmp_path / "out.ready")
    slicer = int(_wait_for(sl)[0])
    if how == "sigterm":
        w.send_signal(signal.SIGTERM)
    rc = w.wait(timeout=60)
    assert out.read_text() == "slicer_alive=1"
    assert rc == (128 + signal.SIGTERM if how == "sigterm" else 0)
    time.sleep(0.2)
    assert not _alive(slicer)


def test_grace_from_flag_env_or_runtime(monkeypatch):
    monkeypatch.delenv("SLICER_MCP_XVFB_GRACE_S", raising=False)
    assert xs.grace_of(None, {}) == xs.GRACE_S == 100.0
    assert xs.grace_of(None, {"xvfb_grace_s": 30}) == 30.0
    monkeypatch.setenv("SLICER_MCP_XVFB_GRACE_S", "7")
    assert xs.grace_of(None, {"xvfb_grace_s": 30}) == 7.0 and xs.grace_of(3, {}) == 3.0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the process scan reads /proc")
def test_a_group_that_ignores_sigterm_is_stopped_after_the_grace(tmp_path):
    num = 7000 + (os.getpid() + 2) % 1000
    fake = tmp_path / "Xvfb"
    fake.write_text(FAKE_XVFB.format(num=num))
    fake.chmod(0o755)
    env = {**os.environ, "FAKE_XVFB_PIDFILE": str(tmp_path / "xvfb.pid")}
    script = f"trap '' TERM; echo $$ > {tmp_path}/sh.pid; while :; do sleep 0.1; done"
    w = subprocess.Popen([sys.executable, str(xs.__file__), "--xvfb", str(fake), "--group", "--grace", "1", "--",
                          "sh", "-c", script], env=env)
    sh = int(_wait_for(tmp_path / "sh.pid")[0])
    _wait_for(tmp_path / "xvfb.pid")
    t0 = time.time()
    w.send_signal(signal.SIGTERM)
    assert w.wait(timeout=30) == 128 + signal.SIGTERM
    assert time.time() - t0 >= 1.0
    time.sleep(0.2)
    assert not _alive(sh)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the process scan reads /proc")
def test_cleanup_only_takes_this_jobs_processes():
    blob = b"A=1\0DISPLAY=:10\0SLURM_JOB_ID=42\0"
    assert xs.env_value(blob, "SLURM_JOB_ID") == "42" and xs.env_value(blob, "NOPE") is None
    env = {**os.environ, "DISPLAY": ":77"}
    a = subprocess.Popen(["sleep", "30"], env={**env, "SLURM_JOB_ID": "41"})
    b = subprocess.Popen(["sleep", "30"], env={**env, "SLURM_JOB_ID": "42"})
    try:
        time.sleep(0.2)
        assert set(xs.pids_on_display(":77", job="42")) & {a.pid, b.pid} == {b.pid}
        assert {a.pid, b.pid} <= set(xs.pids_on_display(":77", job=""))
    finally:
        a.kill(); b.kill(); a.wait(); b.wait()
