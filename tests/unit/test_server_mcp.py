"""The MCP layer end to end over a fake bridge: content shape, validation -> trace, allowlist enforcement, errors."""

import base64
import json

import anyio
import pytest
from mcp import Client

from slicer_mcp.bridge_client import png_size
from slicer_mcp.tracing.logger import read_trace

from tests.unit.conftest import BASIC, SEG
from tests.unit.fakes import answer_report

ANSWER = answer_report()


def run(srv, calls):
    """calls: [(tool, args, meta)] -> [CallToolResult], in one MCP session (in-process client)."""
    async def go():
        out = []
        async with Client(srv.server) as c:
            for tool, args, meta in calls:
                out.append(await c.call_tool(tool, args, meta=meta))
        return out
    return anyio.run(go)


def body(res):
    """(json payload, [png bytes]) from a CallToolResult: one JSON text block first, then the images."""
    assert res.content[0].type == "text"
    payload = json.loads(res.content[0].text)
    imgs = [base64.b64decode(c.data) for c in res.content[1:]]
    assert all(c.type == "image" and c.mime_type == "image/png" for c in res.content[1:])
    assert len(imgs) == len(payload["frames"])
    return payload, imgs


def test_seg_episode_through_mcp(make_server, fake_bridge):
    srv = make_server(SEG)
    meta = {"slicer_mcp/model_reasoning": "turn left to see the aorta", "slicer_mcp/tokens": {"in": 120, "out": 30}}
    reset, orbit, bad, zoom, book, pt, hide, rep = run(srv, [
        ("reset_episode", {"case_id": "case_test", "client_meta": {"model_id": "fake-model", "seed": 3}}, None),
        ("orbit", {"azimuth_deg": 250, "elevation_deg": 10}, meta),
        ("orbit", {"azimuth_deg": "left"}, {"slicer_mcp/model_reasoning": "oops"}),
        ("zoom", {"factor": 1.5, "composite": True}, None),
        ("bookmark_view", {"label": "outflow tracts"}, None),
        ("point_at", {"view": "axial", "x": 900, "y": 300}, None),
        ("set_segment_visibility", {"hide": ["liver", "spleen"]}, None),
        ("submit_report", {"report": answer_report(["v001"])}, None),
    ])
    assert fake_bridge.calls[0][:2] == ("reset_episode", {"case_id": "case_test", "with_segments": True})
    p, imgs = body(reset)
    assert not reset.is_error and p["step_index"] == 0 and p["frames"] == ["3d", "axial", "sagittal", "coronal"]
    ep = reset.meta["slicer_mcp/episode_id"]
    assert ep.endswith(tuple("0123456789abcdef")) and f"_{SEG}_" in ep
    assert reset.meta["slicer_mcp/case_info"]["volume_sha256"] == "v" * 64
    assert "volume_sha256" not in reset.content[0].text
    assert all(png_size(i) == (512, 512) for i in imgs)
    assert p["result"]["case_info"]["slice_position_range_mm"]["axial"] == [-100.0, 100.0]
    assert p["state"]["segments"]["mode"] == "shell_only"

    p, imgs = body(orbit)
    assert not orbit.is_error and p["frames"] == ["3d"] and p["step_index"] == 1
    assert p["warnings"] == ["azimuth_deg=250 clamped to 180 (allowed [-180, 180])"]
    assert fake_bridge.calls[1] == ("orbit", {"azimuth_deg": 180.0, "elevation_deg": 10.0, "roll_deg": 0.0}, ("3d",))

    p, imgs = body(bad)
    assert bad.is_error and p["error"]["type"] == "invalid_argument" and "azimuth_deg" in p["error"]["message"]
    assert imgs == [] and [c[0] for c in fake_bridge.calls].count("orbit") == 1

    assert body(zoom)[0]["frames"] == ["3d", "axial", "sagittal", "coronal"]
    p, _ = body(book)
    assert p["result"] == {"view_ref": "v001"} and len(p["frames"]) == 4
    p, _ = body(pt)
    assert p["frames"] == ["axial", "sagittal", "coronal"] and p["warnings"] == ["x=900 clamped to 511 (allowed [0, 511])"]
    assert not hide.is_error and body(hide)[0]["frames"] == ["3d", "axial", "sagittal", "coronal"]
    p, _ = body(rep)
    assert not rep.is_error and p["result"] == {"accepted": True}

    meta_json, steps = read_trace(srv.session.trace_dir)
    assert meta_json["profile"] == SEG and meta_json["status"] == "submitted" and meta_json["model"]["id"] == "fake-model"
    assert meta_json["image"]["views"] == ["3d", "axial", "sagittal", "coronal"]
    assert meta_json["segments_manifest_sha256"] == "s" * 64 and meta_json["tools"] == list(srv.profile.tools)
    assert [s["tool"] for s in steps] == ["reset_episode", "orbit", "orbit", "zoom", "bookmark_view",
                                          "point_at", "set_segment_visibility", "submit_report"]
    s1 = steps[1]
    assert s1["args"] == {"azimuth_deg": 250, "elevation_deg": 10}
    assert s1["args_after_validation"] == {"azimuth_deg": 180.0, "elevation_deg": 10.0, "roll_deg": 0.0}
    assert s1["model_reasoning"] == "turn left to see the aorta" and s1["tokens"] == {"in": 120, "out": 30}
    assert s1["warnings"] == ["azimuth_deg=250 clamped to 180 (allowed [-180, 180])"]
    s2 = steps[2]
    assert s2["args"] == {"azimuth_deg": "left"} and s2["args_after_validation"] is None
    assert s2["error"]["type"] == "invalid_argument" and s2["model_reasoning"] == "oops" and s2["frames"] == []
    assert steps[3]["model_reasoning"] is None and steps[3]["tokens"] is None
    assert steps[3]["args_after_validation"] == {"factor": 1.5, "composite": True}


def test_basic_profile_loads_no_segments(make_server, fake_bridge):
    srv = make_server(BASIC)
    reset, seg, rep = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("set_3d_content", {"mode": "ct_only"}, None),
        ("submit_report", {"report": {"answer": "no finding"}}, None),
    ])
    assert fake_bridge.calls[0][:2] == ("reset_episode", {"case_id": "case_test"})
    assert "segments" not in body(reset)[0]["state"]
    assert seg.is_error and "not available" in body(seg)[0]["error"]["message"]
    assert not rep.is_error and body(rep)[0]["result"] == {"accepted": True}
    meta = json.loads((srv.session.trace_dir / "meta.json").read_text())
    assert "segments_manifest_sha256" not in meta and meta["report_schema_version"] == "example-answer-v1"
    assert json.loads((srv.session.trace_dir / "report.json").read_text())["answer"] == "no finding"


def test_a_tool_outside_the_profile_is_refused_and_logged(make_server, fake_bridge):
    srv = make_server(SEG)
    reset, other = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("make_blood_pool_shell", {}, None),
    ])
    p, _ = body(other)
    assert other.is_error and p["error"]["type"] == "invalid_argument" and "not available" in p["error"]["message"]
    assert not any(c[0] == "make_blood_pool_shell" for c in fake_bridge.calls)
    _, steps = read_trace(srv.session.trace_dir)
    assert steps[1]["tool"] == "make_blood_pool_shell" and steps[1]["error"]["type"] == "invalid_argument"


def test_calls_outside_an_episode(make_server):
    srv = make_server(SEG)
    before, reset, done, after = run(srv, [
        ("reset_view", {}, None),
        ("reset_episode", {"case_id": "case_test"}, None),
        ("submit_report", {"report": ANSWER}, None),
        ("orbit", {"azimuth_deg": 5}, None),
    ])
    assert before.is_error and body(before)[0]["error"]["type"] == "no_episode"
    assert not done.is_error
    assert after.is_error and body(after)[0]["error"]["type"] == "episode_closed"
    _, steps = read_trace(srv.session.trace_dir)
    assert len(steps) == 2


def test_invalid_report_keeps_episode_open(make_server):
    srv = make_server(SEG)
    _, bad, good = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("submit_report", {"report": {"answer": 42}}, None),
        ("submit_report", {"report": ANSWER}, None),
    ])
    assert bad.is_error and "answer: Input should be a valid string" in body(bad)[0]["error"]["message"]
    assert not good.is_error


def test_invalid_report_lists_every_problem_at_once(make_server):
    srv = make_server(SEG)
    _, bad = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("submit_report", {"report": {"answer": "", "evidence": ["v042"], "extra": 1}}, None),
    ])
    msg = body(bad)[0]["error"]["message"]
    for part in ("answer: String should have at least 1 character", "evidence: cites unknown view_refs ['v042']",
                 "extra: unknown field"):
        assert part in msg, part
    assert "pydantic" not in msg and "http" not in msg and "input_value" not in msg


def test_the_model_sees_the_bridge_state(make_server):
    keys = {"case_loaded", "camera", "render", "clip", "slices", "focus_ras", "wl", "segments"}
    srv = make_server(SEG)
    results = run(srv, [("reset_episode", {"case_id": "case_test"}, None), ("reset_view", {}, None),
                        ("submit_report", {"report": {"answer": ""}}, None),
                        ("submit_report", {"report": ANSWER}, None), ("reset_view", {}, None)])
    for r in results:
        assert set(body(r)[0]["state"]) == keys
    _, steps = read_trace(srv.session.trace_dir)
    assert body(results[1])[0]["state"] == steps[1]["state_after"]
    assert json.loads((srv.session.trace_dir / "meta.json").read_text())["state_format"] == "full"


def test_bridge_internal_error_is_a_tool_error_not_a_crash(make_server):
    srv = make_server(SEG)
    _, boom, again = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("orbit", {"azimuth_deg": 123}, None),
        ("orbit", {"azimuth_deg": 5}, None),
    ])
    p, _ = body(boom)
    assert boom.is_error and p["error"]["type"] == "environment" and "traceback" not in json.dumps(p)
    assert not again.is_error
    _, steps = read_trace(srv.session.trace_dir)
    assert steps[1]["error"]["type"] == "internal"


def test_reset_rejects_bad_case_id(make_server):
    srv = make_server(SEG)
    (r,) = run(srv, [("reset_episode", {"case_id": "../../x"}, None)])
    assert r.is_error and body(r)[0]["error"]["type"] == "invalid_argument"


def test_model_facing_content_never_names_the_case(make_server):
    srv = make_server(SEG)
    res = run(srv, [("reset_episode", {"case_id": "case_test"}, None), ("reset_view", {}, None),
                    ("point_at", {"view": "axial", "x": 1, "y": 1}, None), ("orbit", {"azimuth_deg": 999}, None)])
    for r in res:
        assert "case_test" not in r.content[0].text


def test_end_episode_closes_the_episode(make_server, fake_bridge):
    """The runner ends the episode before disconnecting; a second end_episode changes nothing."""
    srv = make_server(SEG)
    reset, end, again, late = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("end_episode", {}, None),
        ("end_episode", {}, None),
        ("orbit", {"azimuth_deg": 10}, None),
    ])
    p, _ = body(end)
    assert not end.is_error and p["result"]["status"] == "abandoned"
    assert p["result"]["episode_id"] == reset.meta["slicer_mcp/episode_id"]
    assert body(again)[0]["result"] == p["result"]
    assert late.is_error and body(late)[0]["error"]["type"] == "no_episode"


def test_observe_is_a_logged_step_that_changes_nothing(make_server, fake_bridge):
    srv = make_server(SEG)
    _, obs = run(srv, [("reset_episode", {"case_id": "case_test"}, None), ("observe", {}, None)])
    p, imgs = body(obs)
    assert not obs.is_error and p["frames"] == ["3d", "axial", "sagittal", "coronal"] and len(imgs) == 4
    assert fake_bridge.calls[-1][0] == "capture"
    _, steps = read_trace(srv.session.trace_dir)
    assert steps[1]["tool"] == "observe"
