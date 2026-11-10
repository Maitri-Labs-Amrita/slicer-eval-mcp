"""Eval launch: start a Slicer that runs ONLY the Slicer MCP bridge (no stock WebServer, so no exec route).

    slicer-mcp launch [--profile P] [--no-main-window] [--slicer PATH] [--port N]   # start, wait, check
    slicer-mcp launch [--profile P] --check     # re-check the running launch (exit 1 on any problem)
    slicer-mcp launch --stop                    # stop it (and its caffeinate)

Starting refuses when anything already listens on the bridge port or on the dev WebServer port (default 2016).
Slicer is started from the unmodified install with temporary settings (--disable-settings, so the user's Slicer.ini
is neither read nor written) and without ~/.slicerrc.py; slicer_bridge/SlicerMCPBridge/start_bridge.py starts the
bridge with the profile's protocol.json. The main window is shown unless --no-main-window (the Slicer MCP
Observation window always is). On macOS, `caffeinate -d -w <pid>` keeps the display awake while this Slicer runs.
The command returns once the bridge answers and every check passes; Slicer keeps running in the background
(log: data/dev/eval_slicer.log). `slicer-mcp serve` with the same profile then runs against it.

The launch writes a state file (runtime eval_launch.state_file, mode 0600) with the process, the port, the profile,
the launch id and the bridge token. The bridge refuses every request except ping that lacks the token; the server,
the client and `--check` read it from this file (or from $SLICER_MCP_BRIDGE_TOKEN). `--check` without --profile
checks against the profile the launch recorded.

Checks: an authenticated bridge ping says launch "eval" and runs the profile's protocol config; POST /slicer/exec
and GET /slicer/system/version on the bridge port are refused (404/500); nothing listens on the dev port; the
observation window is shown at N x N.

The Slicer path comes from --slicer, else $SLICER_MCP_SLICER_EXECUTABLE, else runtime.json `slicer_executable`.

Parallel jobs on one machine: give each job its own runtime copy (--runtime) with its own eval_launch.state_file /
log, cases_dir and traces_dir (absolute paths are fine), and
  - bridge_port 0 or "auto": the bridge binds a free port chosen by the OS (a Slicer that still fails to bind is
    restarted, up to 3 times) and the launcher writes the bound port and the launch nonce back into that runtime file
    (bridge_port, launch_nonce) and into the state file, so `slicer-mcp serve` reads them from there;
  - the launcher makes a nonce (or takes $SLICER_MCP_LAUNCH_NONCE), hands it to its own Slicer and
    accepts only a ping that reports it, and only while its own Slicer process is alive;
  - Slicer runs with its own HOME, XDG_* and TMPDIR under eval_launch.scratch_dir (default: next to the state file),
    so parallel Slicers never share Slicer-tmp.ini or the temporary folder; --stop removes it;
  - eval_launch.dev_port (default 2016) is the dev WebServer port that must be closed; null skips that probe (a
    server where something else listens on 2016); --dev-port N|none overrides it;
  - --stop stops the recorded process only if it is still the one started (same start time and command line).
"""

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

from slicer_mcp.config import REPO, TOKEN_ENV, launch_state_path, load_protocol, load_runtime, read_launch_state
from slicer_mcp.jobenv import job_env

START_BRIDGE = REPO / "slicer_bridge" / "SlicerMCPBridge" / "start_bridge.py"
DEV_PORT = 2016
LAUNCH_ENV = "SLICER_MCP_LAUNCH_NONCE"
AUTO_PORTS = (0, "0", "auto")
BIND_RETRIES = 3
_UNSET = object()


def slicer_command(slicer, port, main_window=True, config=None, runtime=None, port_file=None):
    """The Slicer command line of an eval launch (pure: unit-tested)."""
    cmd = [str(slicer), "--no-splash", "--ignore-slicerrc", "--disable-settings"]
    if not main_window:
        cmd.append("--no-main-window")
    cmd += ["--python-script", str(START_BRIDGE), "--", "--port", str(port)]
    if config:
        cmd += ["--config", str(config)]
    if runtime:
        cmd += ["--runtime", str(runtime)]
    if port_file:
        cmd += ["--port-file", str(port_file)]
    return cmd


def resolve_slicer(flag, rt):
    p = flag or os.environ.get("SLICER_MCP_SLICER_EXECUTABLE") or rt.get("slicer_executable")
    if not p:
        raise SystemExit("no Slicer executable: pass --slicer, set SLICER_MCP_SLICER_EXECUTABLE or runtime.json "
                         "slicer_executable")
    if not os.path.isfile(p) or not os.access(p, os.X_OK):
        raise SystemExit(f"Slicer executable not found: {p}")
    return p


def port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _session():
    s = requests.Session()
    s.trust_env = False
    return s


def ping(url, timeout=2.0, token=None):
    """The bridge's ping result: full details with the bridge token, else only that it is alive."""
    body = {"slicer_mcp": {"token": token}} if token else {}
    r = _session().post(f"{url}/bridge/ping", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(f"ping failed: {d.get('error')}")
    return d["result"]


def dev_port_of(rt, flag=None):
    """The dev WebServer port to probe (None = skip): --dev-port, else runtime eval_launch.dev_port, else DEV_PORT."""
    if flag is not None:
        return None if str(flag).lower() in ("none", "off", "") else int(flag)
    el = rt.get("eval_launch", {})
    if "dev_port" in el:
        return None if el["dev_port"] is None else int(el["dev_port"])
    return DEV_PORT


def check_launch(host, port, protocol, expect_launch="eval", dev_port=_UNSET, launch_id=None,
                 token=None) -> tuple[list, dict]:
    """(problems, facts) for a running launch. No problems = safe to run evaluations against it. dev_port None skips
    the dev-port probe; launch_id, when given, must be the bridge's; token is the bridge token."""
    if dev_port is _UNSET:
        dev_port = DEV_PORT
    url = f"http://{host}:{port}"
    problems, facts = [], {}
    try:
        info = ping(url, token=token)
    except Exception as e:  # noqa: BLE001
        return [f"bridge not reachable at {url}: {e}"], facts
    if "launch_id" not in info:
        return [f"the bridge at {url} answered only a liveness ping: no valid bridge token (is the launch state file "
                f"missing, or ${TOKEN_ENV} wrong?)"], facts
    facts.update(launch=info.get("launch"), config_sha256=info.get("config_sha256"),
                 bridge_version=info.get("bridge_version"), main_window_visible=info.get("main_window_visible"),
                 layout_manager=info.get("layout_manager"), slice_logic_count=info.get("slice_logic_count"),
                 obs_window=info.get("obs_window"), render_sizes=info.get("render_sizes"))
    facts["launch_id"] = info.get("launch_id")
    if launch_id and info.get("launch_id") != launch_id:
        problems.append(f"bridge launch_id {str(info.get('launch_id'))[:8]} is not this launch's {launch_id[:8]} "
                        "(another job's Slicer on this port?)")
    if expect_launch and info.get("launch") != expect_launch:
        problems.append(f"bridge launch is {info.get('launch')!r}, expected {expect_launch!r} "
                        "(a dev Slicer with the stock WebServer?)")
    if info.get("config_sha256") != protocol.sha256:
        problems.append(f"bridge config sha256 {str(info.get('config_sha256'))[:12]} != {protocol.sha256[:12]}")
    s = _session()
    for method, path in (("POST", "/slicer/exec"), ("GET", "/slicer/system/version"), ("GET", "/slicer/mrml")):
        try:
            r = s.request(method, url + path, data=b"print(1)" if method == "POST" else None, timeout=5)
            facts[f"{method} {path}"] = r.status_code
            if r.status_code not in (404, 500):
                problems.append(f"{method} {path} on the bridge port returned {r.status_code} (expected 404/500)")
        except requests.RequestException as e:
            facts[f"{method} {path}"] = f"no response ({type(e).__name__})"
    facts["dev_port"] = dev_port
    facts["port_2016_listening"] = (dev_port is not None
                                    and (port_open("127.0.0.1", dev_port) or port_open("localhost", dev_port)))
    if facts["port_2016_listening"]:
        problems.append(f"something listens on :{dev_port} (the stock WebServer with exec?)")
    ow = info.get("obs_window") or {}
    if not ow.get("visible") or ow.get("minimized"):
        problems.append(f"the observation window is not shown: {ow}")
    size = protocol["observation"]["size_px"]
    bad = {k: v for k, v in (info.get("render_sizes") or {}).items() if v != [size, size]}
    if bad:
        problems.append(f"render sizes are not {size}x{size}: {bad} (screen: {info.get('screen')}; the bridge sizes the "
                        f"views for the devicePixelRatio of the screen the observation window is on)")
    return problems, facts


def _repo_path(p) -> Path:
    """A runtime path: absolute (or ~) as it is, else relative to the repo."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO / p


def _state_path(rt):
    return launch_state_path(rt)


def write_state(path, state: dict):
    """Write the launch state file atomically, readable by the owner only (it holds the bridge token)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(state, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _shown(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _alive(pid):
    """Our process `pid` is running. EPERM means the pid now belongs to another user's process (a stale state
    file): not ours, so not alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def proc_identity(pid) -> dict | None:
    """{"start", "cmdline"} of a running process (Linux /proc; elsewhere ps), or None when it is gone. The start time
    tells a reused pid apart."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        start = stat.rsplit(")", 1)[1].split()[19]
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return {"start": start, "cmdline": [c.decode(errors="replace") for c in cmd if c]}
    except (OSError, IndexError):
        pass
    if sys.platform.startswith("linux"):
        return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-o", "command=", "-p", str(int(pid))], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    parts = out.split()
    return {"start": " ".join(parts[:5]), "cmdline": parts[5:]}


def same_process(state) -> bool:
    """The state file's pid is still the process the launcher started (start time, and the command's first word)."""
    pid, want = state.get("pid"), state.get("proc")
    if not pid or not _alive(pid):
        return False
    if not want:
        return True
    now = proc_identity(pid)
    return now is not None and now["start"] == want["start"] and now["cmdline"][:1] == want["cmdline"][:1]


def write_runtime_port(runtime_path, port, nonce):
    """Write the bound port and the launch nonce into the job's runtime file (atomically), so `slicer-mcp serve` and the
    client read the port this launch got and check its launch id."""
    p = Path(runtime_path)
    rt = json.loads(p.read_text())
    rt["bridge_port"], rt["launch_nonce"] = int(port), nonce
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rt, indent=2) + "\n")
    tmp.replace(p)


def read_port_file(path, nonce):
    """The port start_bridge.py wrote after binding, if the file is there and names this launch's nonce."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return int(d["port"]) if d.get("launch_id") == nonce and d.get("port") else None


PORT_IN_USE = ("address already in use", "errno 98", "errno 48", "eaddrinuse")


def exit_reason(code, log_text, port=None) -> str:
    """Why Slicer exited before its bridge came up (pure: unit-tested). Exit 3 is start_bridge.py's 'the bridge
    did not start'; its traceback in the log says whether the port was taken (another job bound it between our
    free-port probe and Slicer's bind)."""
    text = (log_text or "").lower()
    if code == 3 and any(s in text for s in PORT_IN_USE):
        where = f"port {port}" if port else "the bridge port"
        return f"{where} is already in use (another process bound it first): the bridge could not listen"
    if code == 3:
        last = [ln.strip() for ln in (log_text or "").splitlines() if ln.strip()][-1:]
        return "the bridge did not start" + (f" ({last[0][:200]})" if last else "")
    return f"Slicer exited with code {code}"


def wait_for_bridge(proc, host, port, nonce, timeout, port_file=None, poll_s=1.0, ping_fn=None, token=None):
    """Wait until this launch's own bridge answers: (port, info), or ("exited", returncode), or ("timeout", None).
    A ping counts only when it reports this launch's nonce and our Slicer process is still alive: an answer
    from another job's Slicer that took the port first is ignored."""
    ping_fn = ping_fn or (lambda url, timeout: ping(url, timeout=timeout, token=token))
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            return "exited", proc.returncode
        p = port
        if port_file is not None:
            p = read_port_file(port_file, nonce)
        if p:
            try:
                info = ping_fn(f"http://{host}:{p}", timeout=1.0)
                if info.get("launch_id") == nonce and proc.poll() is None:
                    return p, info
            except Exception:  # noqa: BLE001
                pass
        if time.time() - t0 > timeout:
            return "timeout", None
        time.sleep(poll_s)


def start(a, rt, protocol):
    host = rt["bridge_host"]
    want = a.port if a.port is not None else rt["bridge_port"]
    auto = want in AUTO_PORTS
    dev_port = dev_port_of(rt, a.dev_port)
    busy = [p for p in ((None if auto else int(want)), dev_port) if p is not None and port_open(host, p)]
    if busy:
        raise SystemExit(f"refusing to start: ports {busy} are in use (close the dev Slicer, or stop the running eval "
                         "launch with --stop)")
    slicer = resolve_slicer(a.slicer, rt)
    el = rt.get("eval_launch", {})
    log_path = _repo_path(el.get("log", "data/dev/eval_slicer.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sp = _state_path(rt)
    sp.parent.mkdir(parents=True, exist_ok=True)
    nonce = os.environ.get(LAUNCH_ENV) or secrets.token_hex(16)
    token = os.environ.get(TOKEN_ENV) or secrets.token_hex(32)
    scratch = _repo_path(el["scratch_dir"]) if el.get("scratch_dir") else sp.parent / f"slicer_home_{nonce[:12]}"
    env = job_env(os.environ, scratch)
    env[LAUNCH_ENV] = nonce
    env[TOKEN_ENV] = token
    timeout = a.timeout or el.get("startup_timeout_s", 180)
    runtime_path = a.runtime or os.environ.get("SLICER_MCP_RUNTIME")
    for attempt in range(BIND_RETRIES + 1 if auto else 1):
        port_file = sp.parent / f".{sp.name}.port" if auto else None
        if port_file is not None:
            port_file.unlink(missing_ok=True)
        cmd = slicer_command(slicer, 0 if auto else int(want), main_window=not a.no_main_window, config=protocol.path,
                             runtime=a.runtime, port_file=port_file)
        log = open(log_path, "w")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
        caff = None
        if el.get("caffeinate", True) and not a.no_caffeinate and sys.platform == "darwin":
            caff = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(proc.pid)], start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        state = {"pid": proc.pid, "proc": proc_identity(proc.pid), "caffeinate_pid": caff.pid if caff else None,
                 "port": None if auto else int(want), "host": host, "main_window": not a.no_main_window,
                 "command": cmd, "log": str(log_path), "started_unix": round(t0, 1), "launch_id": nonce,
                 "scratch_dir": str(scratch), "runtime": str(Path(runtime_path).resolve()) if runtime_path else None,
                 "profile": str(protocol.path.parent), "token": token}
        write_state(sp, state)
        print(f"starting Slicer (pid {proc.pid}, {'main window shown' if not a.no_main_window else '--no-main-window'}"
              f", port {'auto' if auto else want}); log {_shown(log_path)}", flush=True)
        got, info = wait_for_bridge(proc, host, None if auto else int(want), nonce, timeout, port_file, token=token)
        if got == "exited":
            if auto and info == 3 and attempt < BIND_RETRIES:
                print(f"the bridge did not start (exit 3); retrying ({attempt + 1}/{BIND_RETRIES})", flush=True)
                continue
            log.flush()
            reason = exit_reason(info, log_path.read_text(errors="replace")[-4000:], None if auto else want)
            raise SystemExit(f"{reason} (Slicer exit {info}, before the bridge came up); see {log_path}")
        if got == "timeout":
            stop_pids(state)
            raise SystemExit(f"bridge did not answer within {timeout} s; Slicer stopped; see {log_path}")
        break
    port = got
    state["port"] = port
    write_state(sp, state)
    up = time.time() - t0
    problems, facts = check_launch(host, port, protocol, dev_port=dev_port, launch_id=nonce, token=token)
    print(json.dumps({"bridge_up_after_s": round(up, 1), "port": port, **facts}, indent=2))
    if problems:
        print("eval launch FAILED its checks; stopping Slicer:\n  " + "\n  ".join(problems), flush=True)
        stop_pids(state)
        raise SystemExit("eval launch FAILED its checks (Slicer stopped):\n  " + "\n  ".join(problems))
    if runtime_path:
        write_runtime_port(runtime_path, port, nonce)
    print(f"eval Slicer ready on {host}:{port} (launch {nonce[:8]}, no exec route"
          + (f", :{dev_port} closed" if dev_port is not None else "") + "). Stop it with: "
          "slicer-mcp launch --stop" + (f" --runtime {runtime_path}" if runtime_path else ""))


def stop_pids(state, wait_s=30):
    pid = state.get("pid")
    if pid and same_process(state):
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        for _ in range(wait_s * 10):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            if not _alive(pid):
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    cp = state.get("caffeinate_pid")
    if cp and _alive(cp):
        try:
            os.kill(cp, signal.SIGTERM)
        except OSError:
            pass
    try:
        while os.waitpid(-1, os.WNOHANG)[0]:
            pass
    except ChildProcessError:
        pass


def stop(rt):
    sp = _state_path(rt)
    if not sp.exists():
        raise SystemExit(f"no eval launch recorded ({_shown(sp)} missing)")
    state = json.loads(sp.read_text())
    pid = state.get("pid")
    if pid and _alive(pid) and not same_process(state):
        print(f"pid {pid} is no longer the recorded Slicer (start time or command differ): not stopped")
    else:
        stop_pids(state)
        print(f"stopped eval Slicer (pid {pid})")
    scratch = state.get("scratch_dir")
    if scratch and Path(scratch).name.startswith("slicer_home_"):
        shutil.rmtree(scratch, ignore_errors=True)
    sp.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slicer-mcp launch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-main-window", action="store_true", help="hide Slicer's main window (the observation "
                                                                  "window still shows)")
    ap.add_argument("--slicer", help="Slicer executable (default $SLICER_MCP_SLICER_EXECUTABLE, else runtime.json)")
    ap.add_argument("--port", help="bridge port, or 0 / auto for a free one (default runtime.json bridge_port)")
    ap.add_argument("--dev-port", help="the dev WebServer port that must be closed, or none to skip the probe "
                                       "(default runtime.json eval_launch.dev_port, else 2016)")
    ap.add_argument("--profile", help="the task profile whose protocol.json the bridge loads: a folder or a name in "
                                      "profiles/ (default: $SLICER_MCP_PROFILE, else ct_basic; --check: the profile "
                                      "the running launch recorded)")
    ap.add_argument("--runtime", help="runtime config (default config/runtime.json)")
    ap.add_argument("--timeout", type=float, help="startup timeout in s (default runtime.json eval_launch)")
    ap.add_argument("--no-caffeinate", action="store_true", help="do not keep the display awake")
    ap.add_argument("--check", action="store_true", help="check the running launch and exit (1 on any problem)")
    ap.add_argument("--stop", action="store_true", help="stop the running eval launch")
    a = ap.parse_args(argv)
    rt = load_runtime(a.runtime)
    if a.stop:
        return stop(rt)
    if a.port is not None and a.port not in AUTO_PORTS:
        a.port = int(a.port)
    if a.check:
        state = read_launch_state(rt) or {}
        protocol = load_protocol(a.profile or state.get("profile"))
        problems, facts = check_launch(rt["bridge_host"], int(a.port or rt["bridge_port"]), protocol,
                                       dev_port=dev_port_of(rt, a.dev_port),
                                       launch_id=rt.get("launch_nonce") or state.get("launch_id"),
                                       token=os.environ.get(TOKEN_ENV) or state.get("token"))
        print(json.dumps(facts, indent=2))
        if problems:
            print("FAILED:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print("OK: eval launch, no exec route, dev port closed")
        return
    start(a, rt, load_protocol(a.profile))


if __name__ == "__main__":
    main()
