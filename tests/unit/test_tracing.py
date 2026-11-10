"""Trace writer, EnvSession logging and replay, against a fake bridge client (no Slicer)."""

import json
from pathlib import Path

import pytest

from slicer_mcp.config import load_profile
from slicer_mcp.session import TOOL_SCHEMA_VERSION, EnvSession
from slicer_mcp.tracing.logger import STEP_EXTRA_FIELDS, STEP_FIELDS, read_trace
from slicer_mcp.tracing.replay import replay
from tests.unit.fakes import FakeClient
from tests.unit.fakes import answer_report
from tests.unit.fakes import png as _png

PROFILE = load_profile("ct_segments", report=str(Path(__file__).resolve().parents[2] / "examples" / "report_plugin.py"))
PROTOCOL = PROFILE.protocol


@pytest.fixture()
def session(tmp_path):
    return EnvSession(FakeClient(PROTOCOL), PROFILE, tmp_path)


def test_episode_trace_layout(session):
    session.reset("case_test", {"model_id": "m", "system_prompt_sha256": "abc", "seed": 7, "extra": 1})
    session.call("orbit", {"azimuth_deg": 10}, model_reasoning="look left", tokens={"in": 5, "out": 2})
    session.call("zoom", {"factor": 9})
    session.call("set_slice", {"view": "oblique", "position_mm": 1})
    b = session.call("bookmark_view", {"label": "overview"})
    session.call("point_at", {"view": "axial", "x": 400, "y": 300})
    bad = session.call("submit_report", {"report": answer_report(["v009"])})
    assert not bad.ok and bad.error["type"] == "invalid_argument"
    done = session.call("submit_report", {"report": answer_report([b.view_ref])})
    assert done.ok and done.done
    d = session.trace_dir
    meta, steps = read_trace(d)
    assert meta["case_id"] == "case_test" and meta["profile"] == "ct_segments" and meta["status"] == "submitted"
    assert meta["model"] == {"id": "m", "version": None} and meta["seed"] == 7 and meta["client_meta"] == {"extra": 1}
    assert meta["tool_schema_version"] == TOOL_SCHEMA_VERSION
    assert meta["protocol"]["config_sha256"] == PROTOCOL.sha256 and meta["presets_sha256"] == "p"
    assert meta["slicer"] == {"version": "5.10.0", "revision": "34045"} and meta["end_time"]
    assert "commit" in meta["server_git"] and meta["reset_latency_ms"]["bridge"]["total"] == 2.0
    assert [s["step"] for s in steps] == list(range(8))
    assert all(tuple(s)[:len(STEP_FIELDS)] == STEP_FIELDS for s in steps)
    assert all(set(tuple(s)[len(STEP_FIELDS):]) <= set(STEP_EXTRA_FIELDS) for s in steps)
    assert steps[0]["identity"]["case_id"] and steps[0]["identity"]["volume_sha256"]
    assert steps[0]["tool"] == "reset_episode" and len(steps[0]["frames"]) == 4
    assert steps[0]["result"]["case_info"]["center_ras_mm"] == [0.0, 100.0, 0.0]
    assert steps[1]["model_reasoning"] == "look left" and steps[1]["tokens"] == {"in": 5, "out": 2}
    assert steps[1]["frames"] == ["frames/001_3d.png"] and (d / "frames/001_3d.png").exists()
    assert steps[2]["args"] == {"factor": 9} and steps[2]["args_after_validation"] == {"factor": 4.0}
    assert steps[2]["warnings"] == ["factor clamped"]
    assert steps[3]["error"]["type"] == "invalid_argument" and steps[3]["frames"] == []
    assert steps[3]["state_after"] == steps[2]["state_after"]
    assert steps[4]["view_ref"] == "v001" and len(steps[4]["frames"]) == 4
    assert steps[5]["tool"] == "point_at" and steps[5]["frames"] == [f"frames/005_{v}.png" for v in
                                                                     ("axial", "sagittal", "coronal")]
    rep = json.loads((d / "report.json").read_text())
    assert rep["answer"] == "no finding" and rep["evidence"] == ["v001"]
    assert meta["report_schema_version"] == "example-answer-v1"
    for s in steps:
        assert set(s["latency_ms"]) >= {"server_ms"}


def test_no_calls_after_submit(session):
    session.reset("case_test")
    session.call("bookmark_view", {"label": "a"})
    session.call("submit_report", {"report": answer_report()})
    with pytest.raises(RuntimeError):
        session.call("orbit", {"azimuth_deg": 1})


def test_new_reset_abandons_open_episode(session):
    session.reset("case_test")
    first = session.trace_dir
    session.reset("case_test")
    assert json.loads((first / "meta.json").read_text())["status"] == "abandoned"
    assert session.trace_dir != first


def test_replay_reissues_validated_args(session):
    session.reset("case_test")
    session.call("orbit", {"azimuth_deg": 10})
    session.call("zoom", {"factor": 9})
    session.call("set_slice", {"view": "axial", "position_mm": 5})
    session.call("bookmark_view", {"label": "a"})
    session.call("pan_to", {})
    d = session.trace_dir
    fresh = FakeClient(PROTOCOL)
    rep = replay(d, client=fresh)
    assert rep["pass"] and rep["n_frames"] == 4 + 1 + 1 + 1 + 4 + 1
    assert ("zoom", {"factor": 4.0}, ("3d",)) in fresh.calls
    assert ("capture", {}, ("3d", "axial", "sagittal", "coronal")) in fresh.calls
    assert (d / "replay.json").exists()


def test_replay_detects_changed_frames(session):
    session.reset("case_test")
    session.call("orbit", {"azimuth_deg": 10})
    d = session.trace_dir
    (d / "frames/001_3d.png").write_bytes(_png(200))
    assert not replay(d, client=FakeClient(PROTOCOL))["pass"]


def test_replay_refuses_other_protocol(session):
    session.reset("case_test")
    d = session.trace_dir
    m = json.loads((d / "meta.json").read_text())
    m["protocol"]["config_sha256"] = "0" * 64
    (d / "meta.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        replay(d, client=FakeClient(PROTOCOL))
