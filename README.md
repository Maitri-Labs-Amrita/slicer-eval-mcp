# slicer-mcp

A secure [3D Slicer](https://www.slicer.org) MCP server for vision-language-model agents. The model examines a CT
the way a clinician does (rotates the 3D view, changes presets, clips, scrolls slices, shows segmented structures)
through **named, bounded tools only**: there is no Python or code execution. Every call is logged with its
arguments, the resulting state and the images, so a run can be audited, replayed, scored and used as training data.

Status: version 0.1.0, an early research release. The tool surface of 0.1.0 is frozen (see
[ROADMAP.md](ROADMAP.md)), so results obtained with it stay reproducible.

## Features

| # | Feature | 0.1.0 |
|---|---|---|
| 1 | **Security:** only named, typed tools; no code execution; the bridge listens on 127.0.0.1 only and needs a per-launch token | yes |
| 2 | **Bounded arguments:** every argument is range-checked; out-of-range values are clamped with a warning | yes |
| 3 | **Real clinical CT:** DICOM folders, NIfTI, NRRD and other volume files | yes (`slicer-mcp case prepare`) |
| 4 | **Full 3D view control:** orbit, six view directions, zoom, pan, centre on structures | yes |
| 5 | **Volume-rendering presets:** named global presets, never tuned per case | yes (Slicer's stock presets plus a profile's own) |
| 6 | **Clipping:** a box or a plane | yes |
| 7 | **Slice tools:** set, scroll and jump slices, window presets, thick slabs, HU at a pixel, distance measurement | yes |
| 8 | **Segmentation-aware views:** show, hide, fade, outline or isolate structures, or the CT inside a segmentation | yes (profiles with a segmentation) |
| 9 | **Configurable settings:** image size, presets, bounds and defaults per profile | yes (`protocol.json`) |
| 10 | **Deterministic reset:** one call returns the scene to a known start | yes (`reset_episode`, `reset_view`) |
| 11 | **Auditable traces:** arguments, warnings, state and images per call, stamped with the profile's hashes | yes |
| 12 | **Per-task tool allowlist:** the server lists only the profile's tools and refuses and logs any other call | yes |
| 13 | **Leakage controls:** anonymous node names, no text in the images, geometry-only state | yes |
| 14 | **Verifiable answers:** the final report is JSON, validated by the task's report plugin | yes; scoring is the task harness's job |
| 15 | **Headless and parallel:** one Slicer per port; runs on a GPU-less Linux server under Xvfb | yes |
| 16 | **Model-agnostic:** any MCP client | yes (stdio) |

Planned work (cardiac view planes, more slice tools, a compact state, named toolsets) is listed in
[ROADMAP.md](ROADMAP.md).

## How it works

```
MCP client (your runner, Claude Code, ...)
   │ stdio
slicer-mcp serve --profile P          validates and clamps arguments, enforces the allowlist, writes the trace
   │ HTTP POST 127.0.0.1:<port>/bridge/<name>   (no exec route)
Slicer + SlicerMCPBridge               renders the observation window: 3D, axial, sagittal, coronal (N x N each)
```

A **task profile** is a folder that holds everything task-specific:
- `protocol.json`: the protocol values (frame size, bounds, presets, defaults, segmentation classes) and an `mcp`
  section with the tool allowlist and the first line of the server instructions. Its sha256 goes into every trace,
  and the server refuses to run against a bridge that loaded another protocol.
- `presets.xml` (optional): extra volume-rendering presets.
- `report.py` (optional): a report plugin (below).

Shipped profiles (examples; a task brings its own, passed as `--profile <folder>`):
- `ct_basic` (the default): a general CT viewer with 17 tools and no segmentation.
- `ct_segments`: all 22 tools: the same viewer plus a segmentation given at case preparation (example classes: liver,
  spleen, kidneys, aorta, pancreas) shown as hollow shells, and the five segment tools.

The model-facing reference of each profile (tools, descriptions, schemas, instructions, errors) is generated from
the code: [docs/tools/ct_basic.md](docs/tools/ct_basic.md), [docs/tools/ct_segments.md](docs/tools/ct_segments.md).

### Reports

The model ends an episode with `submit_report`. The report's format belongs to the task, so the server only
validates it, writes it into the trace and ends the episode:
- **Built in:** `submit_report` takes any JSON object (at most 64 KB) and writes it to the trace as `report.json`
  (schema `json-object-v1`).
- **A harness's own report:** `slicer-mcp serve --report <plugin.py>` (or `report.py` in the profile) defines what
  `submit_report` accepts: a pydantic model, its description and what goes into the trace
  (`src/slicer_mcp/reports.py`; example: [examples/report_plugin.py](examples/report_plugin.py)). An invalid report
  is rejected with every reason at once and the episode continues.

Rendering a report (a PDF, a page) and scoring it are the harness's job.

**A report plugin is trusted code.** It is a Python file that runs inside the server process, with the same rights
as the server itself. Load only plugins you trust. Neither the MCP client nor the model can choose the plugin: only
the operator can, with `serve --report <path>` (any path) or with `mcp.report` in the profile. A profile's
`mcp.report` must name a file inside the profile folder; a path that leaves the folder (`..`, an absolute path, a
symbolic link to a file outside) is refused.

## Security model

- The model gets named, typed tools only. Arguments are validated and clamped; there is no code execution route in
  the server or in the bridge.
- The bridge listens on 127.0.0.1 only. `slicer-mcp launch` makes a random **bridge token** for every launch and
  writes it to the launch state file (`eval_launch.state_file` in the runtime config, file mode 0600). The bridge
  refuses every request that does not carry the token, before any handler runs. Only `ping` works without it, and
  then it reports only that the bridge is alive (no launch id, no versions). This protects the bridge against other
  local users and against web pages that reach it through DNS rebinding.
- `slicer-mcp serve`, `slicer-mcp launch --check`, `slicer-mcp replay` and the live tests read the token from the
  state file of the runtime config they use, or from `$SLICER_MCP_BRIDGE_TOKEN`.
- Report plugins and profiles are trusted operator input (see Reports).

## Installation

slicer-mcp is installed from a source checkout; there is no PyPI package. The profiles, the Slicer bridge,
`config/` and `tools/` are used from the checkout.

```bash
git clone <this repository> slicer-mcp
cd slicer-mcp
uv sync
uv run pytest            # unit tests, no Slicer needed
```

## Requirements

- 3D Slicer 5.10.0. On macOS the x86_64 build (it runs under Rosetta on Apple silicon); on Linux the official
  build under Xvfb (below). The Slicer install is never modified and nothing is installed into its Python.
- Python 3.10+ and [uv](https://docs.astral.sh/uv/).
- `config/runtime.json` sets the Slicer path, the bridge port and the data folders. Copy it to
  `config/runtime.local.json` (gitignored) for local changes and pass `--runtime` or set `SLICER_MCP_RUNTIME`.

## Quickstart

```bash
# 1. Start a Slicer that runs only the bridge (no stock WebServer, temporary settings). It returns once the bridge
#    answers; Slicer keeps running. On macOS keep the display awake: frames freeze when it sleeps.
uv run slicer-mcp launch --profile ct_basic          # add --no-main-window to hide Slicer's main window

# 2. Prepare a temporary, anonymised case. It prints JSON with the case_id and a lease.
uv run slicer-mcp case prepare --dicom /path/to/dicom/folder
uv run slicer-mcp case prepare --volume scan.nii.gz [--spacing X Y Z] [--intensity-offset N]

# 3. Point an MCP client at the server (stdio). The runner calls the runner-only reset_episode with the case_id,
#    then the model calls the tools.
uv run slicer-mcp serve --profile ct_basic

# 4. Check the running launch at any time (it uses the profile the launch recorded)
uv run slicer-mcp launch --check

# 5. Clean up
uv run slicer-mcp case remove <case_id> --lease <lease>
uv run slicer-mcp launch --stop
```

A case id depends only on the case key (or the file's sha256). Preparing the same case again with other options
(spacing, intensity offset) while another run holds a lease on it is refused, so a running episode never sees its
case change.

A scripted episode with no model and no API cost (it calls every tool, including clamped and invalid arguments):

```bash
uv run python examples/scripted_episode.py <out dir> <case_id> slicer_mcp/ -- uv run slicer-mcp serve --profile ct_basic
```

### A profile with a segmentation (`ct_segments`)

```bash
uv run slicer-mcp launch --profile ct_segments
# a volume with a label map: the segment tools show its structures as hollow shells
uv run slicer-mcp case prepare --profile ct_segments --volume ct.nii.gz --labels labels.nii.gz \
    --label-map 1=liver,2=spleen,3=kidney_left,4=kidney_right,5=aorta,6=pancreas
uv run slicer-mcp serve --profile ct_segments --report examples/report_plugin.py
```

A task's own setup works the same way from outside the repo: `--profile /path/to/its/profile` for `launch`, `case
prepare` and `serve`, and `--report /path/to/its/report.py` for `serve`.

When a dataset's headers are wrong or its values are stored with an offset, pass the corrections:
`--spacing X Y Z`, `--lps-identity`, `--intensity-offset -1024` (for values stored as HU + 1024), and `--case-key
<dataset>:<id>` so the same case always gets the same id.

### Using it from an MCP client

- `reset_episode`, `end_episode` and `observe` carry the tool `_meta` `{"slicer_mcp/runner_only": true}`: the
  client must hide them from the model.
- Every result is one JSON text block `{step_index, state, result, warnings, error, frames}` followed by one PNG per
  view named in `frames`. Errors the model caused (`invalid_argument`, `not_found`) are tool errors and the episode
  continues; `environment` means the runner should abort.
- The client may send `slicer_mcp/model_reasoning` and `slicer_mcp/tokens` in the request `_meta`; both are logged.
- Traces go to `traces/<episode_id>/`: `meta.json`, `steps.jsonl`, `frames/`, and the report files. Replay one
  against the live bridge with `uv run slicer-mcp replay traces/<episode_id>`.

## Headless Linux server (no screen, no sudo)

Every Slicer process needs an X display, so on a server everything runs under a virtual one (Xvfb, Mesa software
OpenGL). `tools/setup_headless.sh` installs Xvfb, Slicer 5.10.0 and the missing system libraries without sudo (from
.deb files and download.slicer.org, sha512-checked), writes `config/runtime.local.json` and checks that Slicer gets
an OpenGL context. It needs Ubuntu/Debian x86_64 and `uv` on PATH.

```bash
tools/setup_headless.sh                    # or --shared <group folder>: a team installs once
SLICER_MCP_RUNTIME=config/runtime.local.json uv run python tools/xvfb_session.py -- bash   # inside tmux
```

For parallel runs give each job its own runtime copy (`bridge_port: "auto"`, its own state file, log, cases and
traces folders) and run it inside `tools/xvfb_session.py`. Frames rendered with Mesa differ slightly from Mac
frames, so every run of one comparison must run on the same machine.

## Tests

```bash
uv run pytest                                  # unit tests (no Slicer)
uv run slicer-mcp tools-md --check             # the generated tool references are up to date
uv run pytest -m slicer tests/slicer           # live tests: need `slicer-mcp launch` running and a test CT
```

The live tests need a CT of your own: `SLICER_MCP_TEST_VOLUME=<file>` (with `SLICER_MCP_TEST_PREPARE` for geometry
overrides, and `SLICER_MCP_TEST_LABELS` / `SLICER_MCP_TEST_LABEL_MAP` for the segment tests), or an already prepared
case with `SLICER_MCP_TEST_CASE=<case_id>`. No patient data ships with this repository.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): setup, tests, live tests, the generated tool references, the schema-version rule
- [ROADMAP.md](ROADMAP.md): planned work
- `docs/tools/<profile>.md`: the generated model-facing tool references

## License

Apache License 2.0; see [LICENSE](LICENSE).

## Acknowledgement

slicer-mcp is developed by its authors at MAITRI Labs, Amrita Vishwa Vidyapeetham. It was used to evaluate
vision-language models in an accompanying preprint.
