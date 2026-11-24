# slicer-mcp

slicer-mcp is an MCP server through which vision-language model agents operate [3D Slicer](https://www.slicer.org).
The model examines a volumetric scan using tools for controlling the 3D view, selecting rendering presets,
clipping, viewing slices, taking measurements and displaying segmentations. The tools do not execute model-generated
code. Every tool call is logged with its arguments, the resulting state and the images, so a session can be
audited, replayed and scored.

![3D Slicer and the MCP tool groups](docs/images/overview.png)

The images below show example views returned to the model: segmentation shells in 3D, an isolated structure,
and sagittal and coronal slices.

![Example views](docs/images/agent_views.png)

```
MCP client ──stdio──► slicer-mcp serve --profile P ──HTTP 127.0.0.1 + token──► 3D Slicer + SlicerMCPBridge
                      (validates and clamps arguments,                          (renders 3D, axial, sagittal,
                       enforces the tool allowlist, writes the trace)            coronal views)
```

The current version is 0.1.0. The tool definitions are fixed for this version to support reproducible evaluations.

## Features

- **Defined tools:** Clients control Slicer through the available tools. Arbitrary code execution is not supported.
- **Scan formats:** Load scans from DICOM folders, NIfTI files or NRRD files.
- **Custom task profiles:** Choose the tools, image size, presets, argument bounds and optional report format for
  your task. Profiles make it easy to compare different evaluation settings without modifying the server.
- **Reproducible sessions:** Traces record each tool call, its results and the profile's hashes. Use `replay` to
  render the recorded views again, compare frames and inspect what happened during a session.
- **Headless and parallel operation:** Run evaluations on a Linux server without a screen or GPU. Run multiple
  Slicer instances in parallel to test different cases or models at the same time.

## Requirements

- Linux or macOS
- [uv](https://docs.astral.sh/uv/)
- [3D Slicer 5.10.0](https://download.slicer.org/) (macOS: the x86_64 build). On a Linux server, `setup-headless`
  installs it for you ([Headless Linux](#headless-linux)).

## Install

```bash
git clone https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp.git
cd slicer-eval-mcp
uv sync --inexact
```

Create a local runtime configuration and select it for the current shell:

```bash
cp config/runtime.json config/runtime.local.json      # your copy (gitignored)
export SLICER_MCP_RUNTIME=config/runtime.local.json   # every slicer-mcp command now reads it
```

In `config/runtime.local.json`, set `slicer_executable` to the path of your Slicer executable, for example
`"/Applications/Slicer.app/Contents/MacOS/Slicer"` (macOS, the default) or
`"/opt/Slicer-5.10.0-linux-amd64/Slicer"` (Linux). The file also sets the local port Slicer listens on
(`bridge_port`, 2017) and the folders for prepared cases (`cases_dir`) and run traces (`traces_dir`).

### Use in a benchmark

The package includes the Slicer bridge and its helper scripts, so you can use it as a dependency in your own
benchmark. Your runner manages cases and model calls, while slicer-mcp provides the Slicer tools and session
traces. [CHD-Bench](https://github.com/Maitri-Labs-Amrita/CHD-Bench) uses this structure to evaluate agents on
cardiac CT and MRI scans.

For example, run the following command from your benchmark project to add slicer-mcp as a dependency. Replace
`<commit>` with the commit you want to use so your benchmark uses a fixed version:

```bash
uv add "slicer-mcp @ git+https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp@<commit>"
```

## Quick start

The steps below use a Linux desktop with Slicer installed and configured as described above. For a server without
a display, first follow [Headless Linux](#headless-linux).

1. Start Slicer with the bridge and prepare a scan:

   ```bash
   uv run slicer-mcp launch --profile ct_basic
   uv run slicer-mcp case prepare --volume /path/to/scan.nii.gz
   ```

   For a DICOM folder, replace `--volume /path/to/scan.nii.gz` with `--dicom /path/to/dicom_folder`.
   The preparation command prints a `case_id` and a `lease`. Keep both values for the following steps.

2. Run the scripted client to test the tools without a model. Replace `<case_id>` with the value printed above:

   ```bash
   uv run python examples/scripted_episode.py out/ <case_id> slicer_mcp/ -- \
       uv run slicer-mcp serve --profile ct_basic --runtime config/runtime.local.json
   ```

   The client starts the MCP server and calls every tool. The `ct_basic` profile excludes the five segmentation
   tools, so calls to those tools are rejected as expected. To connect a model, use your own runner or an MCP host
   as described in the following sections.

   Replies are saved in `out/`. Session traces are saved in `traces/<episode_id>/`, including `steps.jsonl`,
   a `frames/` directory and the report. The `slicer_mcp/` argument sets the prefix for extra fields logged with
   each call.

3. When you have finished, remove the prepared case and stop Slicer. Replace `<case_id>` and `<lease>` with the
   values from step 1:

   ```bash
   uv run slicer-mcp case remove <case_id> --lease <lease>
   uv run slicer-mcp launch --stop
   ```

## Use from your own runner

For evaluations, use a runner that starts the MCP server, loads a case and forwards the model's tool calls to the
server. After running `launch` and `case prepare`, use the following example to connect the runner:

```python
import asyncio, json
from mcp import Client, StdioServerParameters

async def main():
    server = StdioServerParameters(command="uv", args=["run", "slicer-mcp", "serve", "--profile", "ct_basic",
                                                       "--runtime", "config/runtime.local.json"])
    async with Client(server) as c:
        tools = [t for t in (await c.list_tools()).tools
                 if not (t.meta or {}).get("slicer_mcp/runner_only")]   # give only these to the model
        await c.call_tool("reset_episode", {"case_id": "<case_id>"})    # load the case (runner only)
        res = await c.call_tool("orbit", {"azimuth_deg": 30})           # a model's tool call
        print(json.loads(res.content[0].text)["state"])
        await c.call_tool("end_episode", {})                            # close the episode (runner only)

asyncio.run(main())
```

Each tool result contains one JSON text block with the fields `step_index`, `state`, `result`, `warnings`, `error`
and `frames`, followed by one PNG image for each view listed in `frames`.

The model submits its final report with `submit_report`, which accepts any JSON object by default. To define and
validate a custom report format, provide a report plugin with `serve --report plugin.py`. See
[examples/report_plugin.py](examples/report_plugin.py) for an example.

## Use from an MCP host

To use the server interactively in an MCP host such as Claude Desktop, VS Code or Cursor, run `launch` and
`case prepare` first. Then add the server to the host's MCP configuration. For example, add the following to
Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "slicer": {
      "command": "uv",
      "args": ["--directory", "/path/to/slicer-eval-mcp", "run", "slicer-mcp", "serve",
               "--profile", "ct_basic", "--runtime", "config/runtime.local.json"]
    }
  }
}
```

In this configuration, the host exposes every tool to the model, including tools intended for the runner. The model
must call `reset_episode` with the `case_id` to load the case. For evaluations, use your own runner to handle these
calls and hide the runner-only tools from the model.

## Tools

The `ct_segments` profile exposes all 22 model tools. The `ct_basic` profile exposes 17 tools and excludes the
segmentation tools. The full descriptions and argument schemas provided to the model are documented in
[docs/tools/ct_basic.md](docs/tools/ct_basic.md) and [docs/tools/ct_segments.md](docs/tools/ct_segments.md).

| Group | Tool | What it does |
|---|---|---|
| Slices | `set_slice` | Move one slice view to a position in mm along its axis |
| | `scroll_slice` | Scroll one slice view by a number of slices |
| | `set_window_preset` | Set the grey-scale window/level of the slice views to a preset |
| | `jump_to_point` | Move all three slice views through a point [R, A, S] |
| | `point_at` | Return the position and intensity of a pixel in a slice image |
| | `measure` | Measure the distance in mm between two pixels of a slice image |
| 3D view | `orbit` | Rotate the 3D camera about its focal point |
| | `set_view_direction` | Look at the patient from one side (anterior, left, superior, ...) |
| | `zoom` | Zoom the 3D view, or the slice views with `view: "slices"`, by a factor of 0.25 to 4 per call |
| | `pan_to` | Move the 3D camera's focal point to a point |
| | `set_render_preset` | Change the volume-rendering preset |
| | `set_clip_box` | Crop the 3D rendering to a box |
| | `set_clip_plane` | Cut away one side of the 3D rendering with a plane |
| | `clear_clip` | Remove the clip box or plane |
| Segmentation (`ct_segments`) | `set_segment_visibility` | Show or hide individual structure shells |
| | `set_segment_opacity` | Set the 3D opacity of one or all structure shells |
| | `set_3d_content` | Show the shells alone, inside the CT, or the CT alone |
| | `set_slice_outlines` | Turn the structure outlines on the slices on or off |
| | `center_on_segments` | Centre the 3D view on chosen structures |
| Episode | `bookmark_view` | Save the current view as evidence (`v001`, `v002`, ...) |
| | `reset_view` | Restore the initial view |
| | `submit_report` | Submit the final report (JSON) |
| Runner only | `reset_episode` | Load a case and start an episode |
| | `end_episode` | Close the open episode |
| | `observe` | Capture the current views without changing them |

## Profiles

Use a profile to configure Slicer for your evaluation. You can choose which tools the model uses, set image sizes
and argument bounds, and supply presets for slice display and volume rendering. You can also choose which
segmentation structures to display and define a report format for the model's answer.

To compare evaluation settings, create a profile for each setting and select it when launching Slicer and starting
the MCP server. No server changes are needed. Each trace records the profile's hashes so you can identify the
configuration used in a run.

For example, [CHD-Bench](https://github.com/Maitri-Labs-Amrita/CHD-Bench) uses separate profiles for its full
configuration, its evaluation without a 3D view and its evaluation without segmentation.

Each profile is a folder containing the following files:

| File | Purpose |
|---|---|
| `protocol.json` | the tool allowlist, argument bounds, presets, image size and segmentation structures |
| `presets.xml` (optional) | the profile's own volume-rendering presets, added to Slicer's stock ones |
| `report.py` (optional) | the report format `submit_report` accepts, and its validation |

This repository includes two example profiles. The `ct_basic` profile provides 17 tools without segmentation.
The `ct_segments` profile provides 22 tools and displays a label map as hollow shells. For a profile that uses
segmentation, provide the label map when preparing the case:

```bash
uv run slicer-mcp case prepare --profile ct_segments --volume ct.nii.gz --labels labels.nii.gz \
    --label-map 1=liver,2=spleen,3=kidney_left,4=kidney_right,5=aorta,6=pancreas
```

To use your own profile, pass `--profile /path/to/profile`. See [docs/CLI.md](docs/CLI.md) for all commands and options.

## Headless Linux

Install Slicer, Xvfb and the required libraries without sudo, then open a shell on a virtual display:

```bash
uv run slicer-mcp setup-headless    # Xvfb, Slicer 5.10.0 and missing libraries, no sudo; writes config/runtime.local.json
uv run slicer-mcp xvfb-session --runtime config/runtime.local.json -- bash   # a shell on a virtual display
```

For parallel runs, give each job its own runtime copy with `"bridge_port": "auto"` and its own state file, cases and
traces folders. Frames rendered by Mesa on the CPU differ slightly from frames rendered on a GPU. Use the same
rendering hardware for runs whose frames you intend to compare.

## Security

- The MCP client can call the tools enabled by the profile and the runner-only tools for managing episodes.
  The evaluation runner is responsible for hiding runner-only tools from the model. Arbitrary code execution
  is not supported.
- Every argument is validated against its schema and the profile's bounds; out-of-range values are clamped, with a
  warning in the result.
- The bridge listens only on 127.0.0.1 and requires the token generated for the current launch.
- You are responsible for de-identifying scans before use, including any identifying text burned into the images.
- Report plugins (`serve --report`) execute Python code in the server. Use only plugins you trust.

## Tests

Run the unit tests without Slicer. The live tests require a running Slicer instance started with `launch` and the
`SLICER_MCP_TEST_VOLUME` environment variable set to the path of a CT file, such as a NIfTI file.

```bash
uv run pytest -q                        # unit tests, no Slicer
uv run pytest -m slicer tests/slicer    # live tests: a running launch and SLICER_MCP_TEST_VOLUME=<a CT file, e.g. NIfTI>
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This repository is licensed under Apache-2.0. See [LICENSE](LICENSE).
