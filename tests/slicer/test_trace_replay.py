"""Live trace test: a scripted 10-step episode is traced on the live bridge and replays within tolerance."""

import json

import pytest

from slicer_mcp.session import EnvSession
from slicer_mcp.tracing.logger import read_trace
from slicer_mcp.tracing.replay import replay

pytestmark = pytest.mark.slicer


def scripted_episode(sess, center, render_preset):
    """10 steps across the 2D/3D/episode tools (with one clamp and one invalid call)."""
    sess.call("orbit", {"azimuth_deg": 30, "elevation_deg": 15}, model_reasoning="rotate to see the outflow tracts")
    sess.call("zoom", {"factor": 1.6})
    sess.call("set_render_preset", {"name": render_preset})
    sess.call("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 10})
    b = sess.call("bookmark_view", {"label": "anterior cut"})
    sess.call("jump_to_point", {"ras": center})
    sess.call("scroll_slice", {"view": "axial", "delta_slices": 999})
    sess.call("set_slice", {"view": "oblique", "position_mm": 0})
    sess.call("point_at", {"view": "axial", "x": 400, "y": 320})
    sess.call("set_window_preset", {"name": "CT-Angio", "composite": True})
    return b


def test_scripted_episode_traces_and_replays(bridge, profile, protocol, case_id, tmp_path, record, make_report):
    sess = EnvSession(bridge, profile, tmp_path)
    sess.reset(case_id, {"model_id": "scripted", "seed": 0})
    center = bridge.call("get_case_info").result["center_ras_mm"]
    b = scripted_episode(sess, center, protocol["render"]["presets"][-1])
    done = sess.call("submit_report", {"report": make_report(b.view_ref)})
    assert done.ok and done.done
    d = sess.trace_dir
    meta, steps = read_trace(d)

    info = bridge.ping()
    assert meta["protocol"] == {"config_version": protocol.version, "config_sha256": protocol.sha256}
    assert meta["presets_sha256"] == info["presets_sha256"] and meta["profile"] == profile.name
    assert meta["bridge"]["version"] == info["bridge_version"]
    assert meta["slicer"] == {"version": info["slicer_version"], "revision": info["slicer_revision"]}
    assert "commit" in meta["server_git"]
    assert meta["image"]["size_px"] == 512 and meta["status"] == "submitted" and meta["n_steps"] == 11
    assert meta["reset_latency_ms"]["bridge"]["total"] > 0

    assert [s["step"] for s in steps] == list(range(12)) and len(steps) == 12
    assert steps[1]["model_reasoning"] == "rotate to see the outflow tracts"
    scroll = steps[7]
    assert scroll["args"]["delta_slices"] == 999 and scroll["args_after_validation"]["delta_slices"] == 200
    assert any("clamped" in w for w in scroll["warnings"])
    assert steps[8]["error"]["type"] == "invalid_argument" and steps[8]["frames"] == []
    assert steps[5]["view_ref"] == "v001" and len(steps[5]["frames"]) == 4
    assert len(steps[10]["frames"]) == 4
    assert (d / "report.json").exists()
    for s in steps:
        for f in s["frames"]:
            png = (d / f).read_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n"
    blob = json.dumps([s["state_after"] for s in steps])
    assert case_id not in blob

    rep = replay(d, client=bridge)
    record("m5_replay", {k: rep[k] for k in ("pass", "n_steps", "n_frames", "n_identical", "min_ssim")})
    assert rep["pass"], [r for r in rep["steps"] if not r.get("skipped") and not r["pass"]]
    assert rep["n_frames"] == 4 + 1 + 1 + 1 + 1 + 4 + 3 + 1 + 3 + 4


def test_replay_of_tampered_trace_fails(bridge, profile, case_id, tmp_path):
    sess = EnvSession(bridge, profile, tmp_path)
    sess.reset(case_id)
    pos = bridge.call("get_case_info").result["center_ras_mm"][2]
    sess.call("set_slice", {"view": "axial", "position_mm": pos})
    d = sess.trace_dir
    sess.close("abandoned")
    lines = (d / "steps.jsonl").read_text().splitlines()
    s = json.loads(lines[1])
    s["args_after_validation"]["position_mm"] = pos + 20
    lines[1] = json.dumps(s)
    (d / "steps.jsonl").write_text("\n".join(lines) + "\n")
    rep = replay(d, client=bridge)
    assert not rep["pass"]
