# Contributing

## Setup

slicer-mcp runs from a source checkout. You need Python 3.10+, [uv](https://docs.astral.sh/uv/) and, for the live
tests, 3D Slicer 5.10.0.

```bash
uv sync
```

## Tests

```bash
uv run pytest                          # unit tests: no Slicer needed
uv run slicer-mcp tools-md --check     # the generated tool references are up to date
```

Every change needs a unit test where one is possible. `tests/unit/fakes.py` has a fake bridge client, and
`tests/unit/test_run_isolation.py` runs the real bridge dispatch with Slicer stubbed.

## Live tests

The live tests (marker `slicer`, in `tests/slicer/`) need a running bridge and a CT of your own. No patient data
ships with this repository.

```bash
uv run slicer-mcp launch --profile ct_segments
SLICER_MCP_TEST_VOLUME=<ct file> uv run pytest -m slicer tests/slicer
uv run slicer-mcp launch --stop
```

`tests/slicer/conftest.py` lists the other variables: `SLICER_MCP_TEST_PREPARE` (geometry overrides),
`SLICER_MCP_TEST_LABELS` and `SLICER_MCP_TEST_LABEL_MAP` (segment tests), or `SLICER_MCP_TEST_CASE` (an already
prepared case). On a Linux server, run them inside `tools/xvfb_session.py` (see the README).

After a change to the bridge code, stop the launch and start it again: the bridge has no hot reload.

## Generated tool references

`docs/tools/<profile>.md` is generated from the code and the profiles. Do not edit it by hand.

```bash
uv run slicer-mcp tools-md
```

## Rules

- **The tool surface is versioned.** Any change to a tool name, a tool description, an argument schema or the
  server instructions bumps `TOOL_SCHEMA_VERSION` (`src/slicer_mcp/session.py`), regenerates `docs/tools/`, and
  ships as a new profile. Existing profiles stay unchanged, so earlier results stay reproducible.
- **No code execution.** The server and the bridge never use Slicer's stock WebServer or any exec route. A unit
  test enforces this.
- **Protocol values go in the profile's `protocol.json`** (hashed into every trace). Deployment values go in
  `config/runtime.json` (not hashed). Do not hard-code bounds.
- **No leakage.** Node names are anonymous, frames carry no text, and the state holds geometry and settings only.
- Never commit `data/`, `traces/` or any patient images.
