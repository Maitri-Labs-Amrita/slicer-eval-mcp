"""Run a command under its own virtual X display (Xvfb), for Linux servers without a screen.

    uv run python tools/xvfb_session.py [--screen 1920x1200x24] [--xvfb PATH] -- <command> [args...]
    uv run python tools/xvfb_session.py -- bash        # an interactive shell with DISPLAY set (e.g. inside tmux)

Every Slicer process needs an X display: the Linux build ships only the xcb Qt platform plugin (no "offscreen"), so
the eval launch, the case converters (slicer_mcp.cases) and the client must all run under the same DISPLAY. This
starts Xvfb on a free display number (-displayfd), runs the command with DISPLAY set, and stops Xvfb when the command
exits. On exit (normal, SIGTERM, SIGHUP or SIGINT) it first stops every process of this user that still runs on its
display, whatever its process group (the eval launch starts Slicer detached), then Xvfb: TERM, then KILL after 10 s. A virtual display has a fixed size, devicePixelRatio 1 and no sleep, so the frames do not depend on the
user's monitors. Rendering is Mesa software OpenGL (llvmpipe).

Set up a server first with tools/setup_headless.sh (it writes config/runtime.local.json).
The Xvfb path comes from --xvfb, else $SLICER_MCP_XVFB, else runtime.json `xvfb_executable`, else Xvfb on PATH.
runtime.json `extra_lib_dirs` (optional) go in front of LD_LIBRARY_PATH: on a server without sudo, the system libraries
Slicer needs and the server lacks (e.g. libGLU, libOpenGL, libxcb-xinerama, libpcre2-16) are extracted there from
their .deb files (`apt-get download <pkg>; dpkg -x <pkg>.deb <root>`).
Xvfb's own output goes to data/dev/xvfb_session.log. Exit code: the command's (143 / 129 when stopped by a signal).

A batch command (stdin not a terminal, e.g. a Slurm job; --group / --no-group override) runs in its own process group.
On SIGTERM (Slurm: `sbatch --signal=B:TERM@120`) the whole group gets SIGTERM first, and Slicer, the MCP
server and Xvfb are left running for up to --grace seconds (default 100: under Slurm's 120 s warning; or
$SLICER_MCP_XVFB_GRACE_S, or runtime.json `xvfb_grace_s`) while every process of the group finishes, e.g. a runner
ending its episode and writing its run folder; only then is everything on the display stopped. When the command
exits by itself, its group's leftover processes get the same grace. (A bash command whose EXIT trap stops Slicer
stops it when bash gets the TERM: let bash ignore TERM, `trap '' TERM`, or `exec` the runner.)
"""

import argparse
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from slicer_mcp.config import load_runtime  # noqa: E402

DEFAULT_SCREEN = "1920x1200x24"
LOG = REPO / "data" / "dev" / "xvfb_session.log"
STOP_WAIT_S = 10.0
SETTLE_S = 1.0
GRACE_S = 100.0


class Terminated(Exception):
    def __init__(self, signum):
        super().__init__(signum)
        self.signum = signum


def resolve_xvfb(flag, rt):
    p = flag or os.environ.get("SLICER_MCP_XVFB") or rt.get("xvfb_executable") or shutil.which("Xvfb")
    if not p:
        raise SystemExit("no Xvfb: pass --xvfb, set SLICER_MCP_XVFB or runtime.json xvfb_executable")
    p = os.path.expanduser(p)
    if not os.path.isfile(p) or not os.access(p, os.X_OK):
        raise SystemExit(f"Xvfb not found: {p}")
    return p


def xvfb_command(xvfb, screen, fd, num):
    """The Xvfb command line (pure: unit-tested). The display number is chosen here (free_display), not by Xvfb, and
    `-nolisten local` keeps the display off the node-wide abstract socket namespace: under Slurm's
    job_container/tmpfs every job has its own /tmp, so Xvfb's lock files cannot see other jobs' displays, while
    abstract sockets are shared by the whole node. Xvfb then took a number another job held and died ("Cannot write
    display number to fd 4"), or a Slicer attached to another job's display. -displayfd still reports
    when the server is ready."""
    return [str(xvfb), f":{num}", "-displayfd", str(fd), "-screen", "0", screen, "-nolisten", "tcp",
            "-nolisten", "local", "-noreset"]


def abstract_displays(proc_net_unix="/proc/net/unix"):
    """Display numbers with an abstract X socket anywhere on this node (pure on its input file: unit-tested)."""
    try:
        text = Path(proc_net_unix).read_text(errors="replace")
    except OSError:
        return set()
    return {int(m) for m in re.findall(r"@/tmp/\.X11-unix/X(\d+)", text)}


def free_display(taken, start=10, tmp="/tmp"):
    """The lowest display number from `start` with no abstract socket and no lock or socket file in this job's /tmp."""
    n = start
    while n in taken or os.path.exists(f"{tmp}/.X{n}-lock") or os.path.exists(f"{tmp}/.X11-unix/X{n}"):
        n += 1
    return n


def session_env(environ, display=None, extra_lib_dirs=()):
    """The environment (pure: unit-tested): no Wayland, extra library folders first, and DISPLAY set unless None
    (Xvfb itself gets the same library folders, without a DISPLAY)."""
    env = {k: v for k, v in environ.items() if k not in ("WAYLAND_DISPLAY", "DISPLAY")}
    if display is not None:
        env["DISPLAY"] = display
    dirs = [os.path.expanduser(d) for d in extra_lib_dirs or ()]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [p for p in [environ.get("LD_LIBRARY_PATH")] if p])
    return env


def start_xvfb(xvfb, screen, env, log, timeout_s=20.0, tries=10):
    """(process, display) for a new Xvfb on a free display number (free_display). Its output goes to `log` (a pipe
    nobody reads would block Xvfb once full). A number that fails is skipped and the next one tried."""
    skip = set()
    for _ in range(tries):
        job = os.environ.get("SLURM_JOB_ID", "")
        start = 100 + int(job) % 800 if job.isdigit() else 10
        num = free_display(abstract_displays() | skip, start=start)
        r, w = os.pipe()
        with open(log, "ab") as out:
            proc = subprocess.Popen(xvfb_command(xvfb, screen, w, num), pass_fds=(w,), env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        os.close(w)
        ready, _, _ = select.select([r], [], [], timeout_s)
        got = os.read(r, 64).decode().strip() if ready else ""
        time.sleep(SETTLE_S)
        if got.isdigit() and proc.poll() is None:
            return proc, f":{got}"
        os.close(r)
        stop_pids([proc.pid])
        print(f"Xvfb :{num} did not start (reported {got!r}); trying the next number", file=sys.stderr, flush=True)
        skip.add(num)
    tail = Path(log).read_text(errors="replace")[-2000:]
    raise SystemExit(f"Xvfb did not start on {tries} display numbers (log {log}):\n{tail}")


def display_of(environ_bytes):
    """DISPLAY from a /proc/<pid>/environ blob, or None (pure: unit-tested)."""
    for item in environ_bytes.split(b"\0"):
        if item.startswith(b"DISPLAY="):
            return item[len(b"DISPLAY="):].decode(errors="replace")
    return None


def env_value(environ_bytes, key):
    """`key` from a /proc/<pid>/environ blob, or None (pure: unit-tested)."""
    pre = key.encode() + b"="
    for item in environ_bytes.split(b"\0"):
        if item.startswith(pre):
            return item[len(pre):].decode(errors="replace")
    return None


def pids_on_display(display, exclude=(), job=None):
    """This user's processes whose environment has DISPLAY=`display` and, under Slurm, the same SLURM_JOB_ID (Linux
    /proc). Every job's display is private (-nolisten local), so two jobs can use the same number: the job filter
    keeps one job's cleanup away from another job's processes on that number."""
    job = os.environ.get("SLURM_JOB_ID") if job is None else job
    uid, out = os.getuid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in exclude:
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            blob = (d / "environ").read_bytes()
            if display_of(blob) == display and (not job or env_value(blob, "SLURM_JOB_ID") == job):
                out.append(int(d.name))
        except OSError:
            continue
    return out


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def group_pids(pgid):
    """Live (not zombie) processes in process group `pgid` (Linux /proc)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if fields[0] != "Z" and int(fields[2]) == pgid:
            out.append(int(d.name))
    return out


def wait_group(pgid, grace_s, child=None, poll_s=0.1) -> bool:
    """Wait up to grace_s until no process of group pgid is left (reaping `child`); True when the group is gone."""
    deadline = time.time() + grace_s
    while True:
        if child is not None:
            child.poll()
        if not group_pids(pgid):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(poll_s)


def grace_of(flag, rt) -> float:
    if flag is not None:
        return float(flag)
    if os.environ.get("SLICER_MCP_XVFB_GRACE_S"):
        return float(os.environ["SLICER_MCP_XVFB_GRACE_S"])
    return float(rt.get("xvfb_grace_s", GRACE_S))


def stop_pids(pids, wait_s=STOP_WAIT_S):
    """TERM, wait up to `wait_s`, then KILL whatever is left."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        deadline = time.time() + wait_s
        while time.time() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.1)
        pids = [p for p in pids if _alive(p)]
        if not pids:
            return


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except OSError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", default=DEFAULT_SCREEN, help=f"Xvfb screen WxHxD (default {DEFAULT_SCREEN})")
    ap.add_argument("--xvfb", help="Xvfb executable (default $SLICER_MCP_XVFB, else runtime.json, else PATH)")
    ap.add_argument("--runtime", help="runtime config (default $SLICER_MCP_RUNTIME, else config/runtime.json)")
    ap.add_argument("--grace", type=float, help=f"seconds the command's process group gets after SIGTERM (and after "
                                                f"the command exits) before the display is stopped (default "
                                                f"$SLICER_MCP_XVFB_GRACE_S, else runtime.json xvfb_grace_s, else {GRACE_S:g})")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--group", dest="group", action="store_true", default=None,
                   help="run the command in its own process group (default when stdin is not a terminal)")
    g.add_argument("--no-group", dest="group", action="store_false", help="keep the command in this process group "
                                                                          "(an interactive shell)")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run (after --)")
    a = ap.parse_args(argv)
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        ap.error("no command given (… -- <command>)")
    rt = load_runtime(a.runtime)
    xvfb = resolve_xvfb(a.xvfb, rt)
    grace = grace_of(a.grace, rt)
    group = a.group if a.group is not None else not sys.stdin.isatty()
    libs = rt.get("extra_lib_dirs")
    LOG.parent.mkdir(parents=True, exist_ok=True)

    def on_signal(signum, _frame):
        raise Terminated(signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    proc = child = display = None
    try:
        proc, display = start_xvfb(xvfb, a.screen, session_env(os.environ, None, libs), LOG)
        print(f"Xvfb {display} ({a.screen}, pid {proc.pid})", file=sys.stderr, flush=True)
        child = subprocess.Popen(cmd, env=session_env(os.environ, display, libs), start_new_session=group)
        while True:
            try:
                return child.wait()
            except KeyboardInterrupt:
                if group:
                    _signal_group(child.pid, signal.SIGINT)
                continue
    except Terminated as t:
        if child is not None and group:
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            _signal_group(child.pid, t.signum)
            print(f"{signal.Signals(t.signum).name}: sent to the command's process group; waiting up to {grace:g} s",
                  file=sys.stderr, flush=True)
        return 128 + t.signum
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if child is not None and group and not wait_group(child.pid, grace, child):
            print(f"the command's process group is still running after {grace:g} s: stopping it",
                  file=sys.stderr, flush=True)
        if display is not None:
            left = ([child.pid] if child is not None and child.poll() is None else [])
            left += [p for p in pids_on_display(display, exclude={os.getpid(), proc.pid}) if p not in left]
            if left:
                print(f"stopping {len(left)} process(es) still on {display}", file=sys.stderr, flush=True)
                stop_pids(left)
        if proc is not None:
            stop_pids([proc.pid])


if __name__ == "__main__":
    sys.exit(main())
