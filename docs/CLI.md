# slicer-mcp command reference

The `--help` text of every `slicer-mcp` command, as of version 0.1.0 (run them with `uv run`). Each command also
reads `$SLICER_MCP_RUNTIME` (the runtime config) and `$SLICER_MCP_PROFILE` (the default profile) where it takes one.

```text
The slicer-mcp command.

    slicer-mcp serve   [--profile P] [--runtime R] [--slicer-url U]   the MCP server (stdio; an MCP client starts it)
    slicer-mcp launch  [--profile P] [--runtime R] [--no-main-window] [--check | --stop]   the Slicer that serves it
    slicer-mcp case    prepare ... | remove <case_id>                  temporary anonymised cases (cases.py)
    slicer-mcp tools-md [--check]                                      the tool reference (docs/tools/<profile>.md)
    slicer-mcp replay  <trace dir>                                     replay a trace against the live bridge
    slicer-mcp setup-headless [options]                                Linux without a screen: Xvfb, Slicer, libraries
    slicer-mcp xvfb-session [options] -- <command>                     run a command on its own virtual display
```

## `slicer-mcp launch`

```text
usage: slicer-mcp launch [-h] [--no-main-window] [--slicer SLICER] [--port PORT]
                         [--dev-port DEV_PORT] [--profile PROFILE] [--runtime RUNTIME]
                         [--timeout TIMEOUT] [--no-caffeinate] [--check] [--stop]

Eval launch: start a Slicer that runs ONLY the Slicer MCP bridge (no stock WebServer, so no exec route).

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

options:
  -h, --help           show this help message and exit
  --no-main-window     hide Slicer's main window (the observation window still shows)
  --slicer SLICER      Slicer executable (default $SLICER_MCP_SLICER_EXECUTABLE, else
                       runtime.json)
  --port PORT          bridge port, or 0 / auto for a free one (default runtime.json bridge_port)
  --dev-port DEV_PORT  the dev WebServer port that must be closed, or none to skip the probe
                       (default runtime.json eval_launch.dev_port, else 2016)
  --profile PROFILE    the task profile whose protocol.json the bridge loads: a folder or a name
                       in profiles/ (default: $SLICER_MCP_PROFILE, else ct_basic; --check: the
                       profile the running launch recorded)
  --runtime RUNTIME    runtime config (default config/runtime.json)
  --timeout TIMEOUT    startup timeout in s (default runtime.json eval_launch)
  --no-caffeinate      do not keep the display awake
  --check              check the running launch and exit (1 on any problem)
  --stop               stop the running eval launch
```

## `slicer-mcp case prepare`

```text
usage: slicer-mcp case prepare [-h] (--dicom DICOM | --volume VOLUME) [--series SERIES]
                               [--spacing X Y Z] [--lps-identity]
                               [--intensity-offset INTENSITY_OFFSET] [--case-key CASE_KEY]
                               [--labels LABELS] [--label-map LABEL_MAP] [--profile PROFILE]
                               [--runtime RUNTIME]

options:
  -h, --help            show this help message and exit
  --dicom DICOM         a DICOM folder (only read)
  --volume VOLUME       a volume file: NIfTI, NRRD, MHA, ... (only read)
  --series SERIES       with --dicom: the series UID (default: the CT series with the most files)
  --spacing X Y Z       with --volume: the voxel spacing in mm, instead of the header's
  --lps-identity        with --volume: direction and origin = an LPS identity at 0, instead of the
                        header's
  --intensity-offset INTENSITY_OFFSET
                        with --volume: added to every value
  --case-key CASE_KEY   with --volume: the case id is derived from this key (default: the file's
                        sha256)
  --labels LABELS       a label map on the case's voxel grid: also build the segmentation shells
  --label-map LABEL_MAP
                        with --labels: its codes -> the profile's class names, e.g. 1=LV,2=RV
  --profile PROFILE     the profile whose segmentation classes the shells use (default:
                        $SLICER_MCP_PROFILE, else ct_basic)
  --runtime RUNTIME
```

## `slicer-mcp case remove`

```text
usage: slicer-mcp case remove [-h] [--lease LEASE] [--runtime RUNTIME] case_id

positional arguments:
  case_id

options:
  -h, --help         show this help message and exit
  --lease LEASE      the lease token prepare printed (default: the case's oldest lease)
  --runtime RUNTIME
```

## `slicer-mcp serve`

```text
usage: slicer-mcp serve [-h] [--profile PROFILE] [--slicer-url SLICER_URL] [--runtime RUNTIME]
                        [--report REPORT]

Slicer MCP server (stdio)

options:
  -h, --help            show this help message and exit
  --profile PROFILE     the task profile: a folder with protocol.json, or a name in profiles/
                        (default: $SLICER_MCP_PROFILE, else ct_basic)
  --slicer-url SLICER_URL
                        bridge URL (default: SLICER_URL, else config/runtime.json)
  --runtime RUNTIME     runtime config (default: $SLICER_MCP_RUNTIME, else config/runtime.json)
  --report REPORT       the task's report plugin, a Python file (default: the profile's
                        mcp.report, else the built-in JSON report; docs:
                        src/slicer_mcp/reports.py)
```

## `slicer-mcp replay`

```text
usage: slicer-mcp replay [-h] [--ssim-min SSIM_MIN] [--force] trace_dir

Replay a trace against the live bridge and diff the regenerated frames against the logged ones.

    slicer-mcp replay traces/<episode_id> [--ssim-min 0.99] [--force]

Resets the same case, then re-issues every successful bridge step with its logged args_after_validation
(bookmark_view re-captures all four views; submit_report and failed steps have no live frames
and are skipped). Each frame passes when it is pixel-identical or SSIM >= ssim-min. The logged state_after
must match too. Refuses to run when the bridge's protocol config differs from the trace's (unless --force).
Writes <trace>/replay.json and exits non-zero on any failure.

positional arguments:
  trace_dir

options:
  -h, --help           show this help message and exit
  --ssim-min SSIM_MIN
  --force
```

## `slicer-mcp tools-md`

```text
usage: slicer-mcp tools-md [-h] [--check]

Generate docs/tools/<profile>.md, the model-facing tool reference of each shipped profile, from
code (never edited

options:
  -h, --help  show this help message and exit
  --check     exit 1 if a docs/tools/<profile>.md differs from the generated text
```

## `slicer-mcp setup-headless`

```text
Set up a headless Linux server (Ubuntu/Debian, no sudo) to run slicer-mcp.

  tools/setup_headless.sh [--shared DIR] [--prefix DIR] [--slicer-dir DIR] [--slicer-tarball FILE] [--runtime-out FILE]
  slicer-mcp setup-headless [the same options]       (the copy shipped in the installed package)

1. Xvfb: the system one if on PATH, else extracted from its .deb into <prefix>/root (apt-get download; dpkg -x).
2. Slicer 5.10.0: reused if <slicer-dir>/Slicer-5.10.0-linux-amd64 exists, else from --slicer-tarball (no network),
   else downloaded from download.slicer.org; the tarball is checked against a pinned sha512.
3. The system libraries Slicer needs and the server lacks (found with ldd), extracted the same way.
4. config/runtime.local.json (gitignored; --runtime-out FILE writes another file): runtime.json with this user's
   paths. A rerun keeps every other setting already in the file (port, state file, ...).
5. A check: Slicer starts under Xvfb and gets an OpenGL context.

--shared DIR sets --prefix and --slicer-dir to one group folder, so a team on one cluster installs once and every
member only writes their own config.

Run it again at any time: every step skips what is already there. Then run everything under the virtual display:
  SLICER_MCP_RUNTIME=config/runtime.local.json uv run python tools/xvfb_session.py -- <command>
```

## `slicer-mcp xvfb-session`

```text
usage: xvfb_session.py [-h] [--screen SCREEN] [--xvfb XVFB] [--runtime RUNTIME] [--grace GRACE]
                       [--group | --no-group]
                       ...

Run a command under its own virtual X display (Xvfb), for Linux servers without a screen.

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

positional arguments:
  command            the command to run (after --)

options:
  -h, --help         show this help message and exit
  --screen SCREEN    Xvfb screen WxHxD (default 1920x1200x24)
  --xvfb XVFB        Xvfb executable (default $SLICER_MCP_XVFB, else runtime.json, else PATH)
  --runtime RUNTIME  runtime config (default $SLICER_MCP_RUNTIME, else config/runtime.json)
  --grace GRACE      seconds the command's process group gets after SIGTERM (and after the command
                     exits) before the display is stopped (default $SLICER_MCP_XVFB_GRACE_S, else
                     runtime.json xvfb_grace_s, else 100)
  --group            run the command in its own process group (default when stdin is not a
                     terminal)
  --no-group         keep the command in this process group (an interactive shell)
```

`examples/scripted_episode.py` is a scripted MCP client that calls every tool.
