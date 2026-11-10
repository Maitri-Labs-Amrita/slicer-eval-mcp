"""Live tests on the eval launch (slicer-mcp launch): no exec route and :2016 closed, checked during a real
stdio episode; the slice-logic count for each window mode; the reset's window check in the trace.

Skipped unless the bridge reports launch "eval". Run once per window mode:
    uv run slicer-mcp launch --profile ct_segments [--no-main-window] && uv run pytest -m slicer tests/slicer
"""

import json
import os
import sys

import anyio
import pytest
import requests
from mcp import Client, StdioServerParameters

from slicer_mcp import launch
from slicer_mcp.bridge_client import VIEWS
from slicer_mcp.config import REPO, launch_token
from slicer_mcp.tracing.logger import read_trace

pytestmark = pytest.mark.slicer


@pytest.fixture(scope="module")
def eval_info(bridge):
    info = bridge.ping()
    if info.get("launch") != "eval":
        pytest.skip(f"bridge launch is {info.get('launch')!r}, not the eval launch")
    return info


@pytest.fixture(scope="module")
def es():
    return launch


@pytest.fixture()
def runtime_file(tmp_path, runtime):
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    rt.update(traces_dir=str(tmp_path / "traces"))
    p = tmp_path / "runtime.json"
    p.write_text(json.dumps(rt))
    return p


def _no_exec_anywhere(es, runtime, protocol):
    problems, facts = es.check_launch(runtime["bridge_host"], runtime["bridge_port"], protocol,
                                      token=launch_token(runtime))
    assert problems == [], (problems, facts)
    assert facts["POST /slicer/exec"] in (404, 500) and facts["port_2016_listening"] is False
    return facts


def test_launch_checks_pass(eval_info, es, runtime, protocol):
    facts = _no_exec_anywhere(es, runtime, protocol)
    r = requests.post(f"http://{runtime['bridge_host']}:{runtime['bridge_port']}/slicer/exec", data=b"print(1)",
                      proxies={"http": None, "https": None}, timeout=5)
    assert r.status_code in (404, 500)
    assert facts["launch"] == "eval"


def test_slice_logic_count_by_window_mode(eval_info, bridge, case_id):
    """With the main window there is a layout manager and its hidden twins: 9 slice logics after a reset.
    With --no-main-window there is none, so no main slice views and no twins: only our 3 (measured; 6 had
    been expected)."""
    bridge.call("reset_episode", {"case_id": case_id}, capture=VIEWS)
    info = bridge.ping()
    assert info["main_window_visible"] == info["layout_manager"]
    if info["layout_manager"]:
        assert info["slice_logic_count"] == 9, info
        assert len(info["views_adopted_by_main_layout"]) == 4
    else:
        assert info["slice_logic_count"] == 3, info
        assert info["views_adopted_by_main_layout"] == []
    assert info["obs_window"] == {"visible": True, "minimized": False}


def test_stdio_episode_without_exec(eval_info, es, runtime, profile, protocol, runtime_file, case_id, make_report):
    """A full episode through the real slicer-mcp serve subprocess; between steps, the exec route stays absent and
    nothing listens on :2016. The trace records the eval launch and the reset's window check."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "slicer_mcp.cli", "serve", "--profile",
                                                                 profile.name, "--runtime", str(runtime_file)],
                                   env=dict(os.environ), cwd=str(REPO))
    calls = [("reset_episode", {"case_id": case_id}), ("set_slice", {"view": "axial", "position_mm": -400.0}),
             ("orbit", {"azimuth_deg": 30}), ("point_at", {"view": "axial", "x": 256, "y": 256}),
             ("set_clip_box", {"size_mm": [120, 120, 120]}), ("bookmark_view", {"label": "box"}),
             ("submit_report", {"report": make_report()})]

    async def go():
        out = []
        async with Client(params) as c:
            for tool, args in calls:
                out.append(await c.call_tool(tool, args))
                await anyio.to_thread.run_sync(_no_exec_anywhere, es, runtime, protocol)
        return out
    res = anyio.run(go)
    assert [r.is_error for r in res] == [False] * len(calls), [r.content[0].text for r in res if r.is_error]
    meta, steps = read_trace(next((runtime_file.parent / "traces").iterdir()))
    assert meta["status"] == "submitted" and len(steps) == len(calls)
    assert meta["launch"] == "eval" and meta["bridge"]["version"] == eval_info["bridge_version"]
    assert meta["obs_window"] == {"repairs": []}
    assert meta["host"]["display"]["asleep"] in (False, None)


def test_views_are_sized_for_the_windows_screen(bridge):
    """The views are N / devicePixelRatio logical pixels for the screen the window is on, so the render windows
    are N x N device pixels on a Retina screen (ratio 2) and on an ordinary monitor (ratio 1) alike."""
    info = bridge.ping()
    n, scr = info["obs_size"], info["screen"]
    assert scr["fit_error"] is None and scr["device_pixel_ratio"] == info["device_pixel_ratio"]
    assert scr["logical_px"] * scr["device_pixel_ratio"] == n
    assert info["render_sizes"] == {v: [n, n] for v in ("3d", "axial", "sagittal", "coronal")}
