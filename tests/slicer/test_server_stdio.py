"""Live MCP server tests: the MCP server over the live bridge. One test drives the real CLI over stdio (a subprocess);
the others use the in-process client. Each checks the resulting trace, and the traced episodes replay."""

import base64
import json
import os
import re
import shutil
import sys
import time

import anyio
import pytest
from mcp import Client, StdioServerParameters

from slicer_mcp import server as server_mod
from slicer_mcp.bridge_client import png_size
from slicer_mcp.config import REPO
from slicer_mcp.session import RUNNER_TOOLS
from slicer_mcp.tracing.logger import read_trace
from slicer_mcp.tracing.replay import replay

pytestmark = pytest.mark.slicer


@pytest.fixture()
def runtime_file(tmp_path, runtime):
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    rt.update(traces_dir=str(tmp_path / "traces"))
    p = tmp_path / "runtime.json"
    p.write_text(json.dumps(rt))
    return p


def body(res):
    payload = json.loads(res.content[0].text)
    imgs = [base64.b64decode(c.data) for c in res.content[1:]]
    assert len(imgs) == len(payload["frames"])
    assert all(png_size(i) == (512, 512) for i in imgs)
    return payload, imgs


async def _session(client, calls):
    out, walls = [], []
    for tool, args, meta in calls:
        t0 = time.perf_counter()
        out.append(await client.call_tool(tool, args, meta=meta))
        walls.append((time.perf_counter() - t0) * 1000)
    return out, walls


def run_inproc(srv, calls):
    async def go():
        async with Client(srv.server) as c:
            tools = (await c.list_tools()).tools
            return tools, *(await _session(c, calls))
    return anyio.run(go)


def run_stdio(profile, runtime_file, calls):
    params = StdioServerParameters(command=sys.executable, args=["-m", "slicer_mcp.cli", "serve", "--profile", profile,
                                                                 "--runtime", str(runtime_file)],
                                   env=dict(os.environ), cwd=str(REPO))

    async def go():
        async with Client(params) as c:
            tools = (await c.list_tools()).tools
            return tools, *(await _session(c, calls))
    return anyio.run(go)


def test_stdio_episode_end_to_end(bridge, profile, protocol, case_id, runtime_file, record, make_report):
    """The real `slicer-mcp serve --profile <the bridge's>` over stdio: tool list, a short episode, the trace, then
    replay."""
    center = bridge.call("get_case_info").result["center_ras_mm"]
    r = {"slicer_mcp/model_reasoning": "look at the great arteries from the left", "slicer_mcp/tokens": {"in": 1500, "out": 42}}
    calls = [
        ("reset_episode", {"case_id": case_id, "client_meta": {"model_id": "scripted", "seed": 0}}, None),
        ("orbit", {"azimuth_deg": 270, "elevation_deg": 20}, r),
        ("set_render_preset", {"name": protocol["render"]["presets"][-1]}, None),
        ("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 15}, None),
        ("set_slice", {"view": "axial", "position_mm": "mid"}, None),
        ("jump_to_point", {"ras": [center[0], center[1], center[2] + 1000], "composite": True}, None),
        ("scroll_slice", {"view": "axial", "delta_slices": 999}, None),
        ("bookmark_view", {"label": "anterior cut"}, None),
        ("point_at", {"view": "axial", "x": 400, "y": 320}, None),
        ("reset_view", {}, None),
        ("submit_report", {"report": make_report("v001")}, None),
        ("orbit", {"azimuth_deg": 5}, None),
    ]
    tools, res, walls = run_stdio(profile.name, runtime_file, calls)
    assert [t.name for t in tools] == list(RUNNER_TOOLS + profile.tools)
    assert {t.name for t in tools if t.meta.get("slicer_mcp/runner_only")} == set(RUNNER_TOOLS)
    reset, orbit, preset, clip, bad, jump, scroll, book, pt, rv, rep, after = res
    episode_id = reset.meta["slicer_mcp/episode_id"]

    assert body(reset)[0]["frames"] == ["3d", "axial", "sagittal", "coronal"]
    assert set(body(reset)[0]["result"]["case_info"]) >= {"ras_bounds_mm", "center_ras_mm", "slice_position_range_mm"}
    p, _ = body(orbit)
    assert not orbit.is_error and p["frames"] == ["3d"]
    assert p["warnings"] == ["azimuth_deg=270 clamped to 180 (allowed [-180, 180])"]
    assert body(preset)[0]["state"]["render"]["preset"] == protocol["render"]["presets"][-1]
    assert body(clip)[0]["state"]["clip"]["type"] == "plane"
    p, imgs = body(bad)
    assert bad.is_error and p["error"]["type"] == "invalid_argument" and imgs == []
    p, _ = body(jump)
    assert p["frames"] == ["3d", "axial", "sagittal", "coronal"]
    assert any(w.startswith("ras[2]=") and "clamped" in w for w in p["warnings"])
    p, _ = body(scroll)
    assert p["frames"] == ["axial"] and any("clamped to 200" in w for w in p["warnings"])
    assert any("volume edge" in w for w in p["warnings"])
    assert body(book)[0]["result"] == {"view_ref": "v001"}
    p, _ = body(pt)
    assert p["frames"] == ["axial", "sagittal", "coronal"] and len(p["result"]["ras"]) == 3
    assert all(abs(a - b) < 0.01 for a, b in zip(p["state"]["focus_ras"], p["result"]["ras"]))
    assert "camera" in p["state"] and "builds" not in p["state"]
    assert body(rv)[0]["frames"] == ["3d", "axial", "sagittal", "coronal"]
    assert not rep.is_error and body(rep)[0]["result"] == {"accepted": True}
    assert after.is_error and body(after)[0]["error"]["type"] == "episode_closed"

    d = runtime_file.parent / "traces" / episode_id
    meta, steps = read_trace(d)
    assert meta["profile"] == profile.name and meta["status"] == "submitted" and meta["n_steps"] == 10
    assert meta["protocol"]["config_sha256"] == protocol.sha256 == bridge.ping()["config_sha256"]
    assert [s["step"] for s in steps] == list(range(11))
    assert steps[1]["args"] == {"azimuth_deg": 270, "elevation_deg": 20}
    assert steps[1]["args_after_validation"] == {"azimuth_deg": 180.0, "elevation_deg": 20.0, "roll_deg": 0.0}
    assert steps[1]["model_reasoning"] == r["slicer_mcp/model_reasoning"] and steps[1]["tokens"] == {"in": 1500, "out": 42}
    assert steps[2]["model_reasoning"] is None and steps[2]["tokens"] is None
    assert steps[4]["error"]["type"] == "invalid_argument" and steps[4]["args"] == {"view": "axial", "position_mm": "mid"}
    assert steps[6]["args"]["delta_slices"] == 999 and steps[6]["args_after_validation"]["delta_slices"] == 200
    assert len(steps[5]["frames"]) == 4 and steps[5]["args_after_validation"]["composite"] is True
    assert (d / "report.json").exists()
    for s in steps:
        for f in s["frames"]:
            assert png_size((d / f).read_bytes()) == (512, 512)

    over = [w - s["latency_ms"]["server_ms"] for w, s in zip(walls[1:], steps[1:]) if s["frames"]]
    record("m6_stdio_overhead_ms", {"median": sorted(over)[len(over) // 2], "max": max(over), "n": len(over)})

    rep_ = replay(d, client=bridge)
    record("m6_replay_stdio", {k: rep_[k] for k in ("pass", "n_steps", "n_frames", "n_identical", "min_ssim")})
    assert rep_["pass"], [x for x in rep_["steps"] if not x.get("skipped") and not x["pass"]]


def test_nothing_model_facing_names_the_case(bridge, alias_cases, runtime_file):
    """Load the case under the name LEAK_TEST: no tool listing, instruction, result or state may contain it."""
    srv = server_mod.build(bridge.profile.dir, runtime=str(runtime_file))
    tools, res, _ = run_inproc(srv, [
        ("reset_episode", {"case_id": "LEAK_TEST"}, None),
        ("reset_view", {}, None), ("point_at", {"view": "axial", "x": 1, "y": 1}, None),
        ("orbit", {"azimuth_deg": 999}, None),
        ("set_slice", {"view": "axial", "position_mm": 1e6}, None), ("bookmark_view", {"label": "x"}, None),
        ("set_slice", {"view": "nope", "position_mm": 0}, None),
        ("submit_report", {"report": "no finding"}, None),
    ])
    blob = json.dumps([t.model_dump(mode="json") for t in tools]) + (srv.server.instructions or "")
    blob += "".join(r.content[0].text for r in res)
    assert not re.search("LEAK|case_0", blob)
    srv.session.close("abandoned")


def test_server_refuses_a_bridge_on_another_config(bridge, tmp_path, runtime_file):
    prof = tmp_path / "other_profile"
    shutil.copytree(bridge.profile.dir, prof)
    cfg = json.loads((prof / "protocol.json").read_text())
    cfg["camera"]["pan_margin_mm"] = 51.0
    (prof / "protocol.json").write_text(json.dumps(cfg))
    with pytest.raises(server_mod.StartupError, match="sha256"):
        server_mod.build(prof, runtime=str(runtime_file))
