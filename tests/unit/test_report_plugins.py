"""Reports (reports.py): the interface check, the trace files, the built-in JSON report, and a
harness's plugin (examples/report_plugin.py: a free-text answer with evidence) through `--report`."""

import json
import types

import pytest
from pydantic import BaseModel

from slicer_mcp import reports
from slicer_mcp.config import load_profile
from slicer_mcp.episode import EpisodeError, EpisodeSession
from slicer_mcp.tools.registry import tool_specs

from tests.unit.conftest import BASIC, EXAMPLE_REPORT, SEG
from tests.unit.fakes import answer_report


def submit(profile, rep, report=None):
    p = load_profile(profile, report=report)
    return EpisodeSession("c", report_model=p.report.Model).submit_report(rep)


def test_check_plugin_names_what_is_missing():
    with pytest.raises(ValueError, match="lacks"):
        reports.check_plugin(types.SimpleNamespace(SCHEMA_VERSION="x", Model=BaseModel))
    mod = types.SimpleNamespace(SCHEMA_VERSION="x", DESCRIPTION="d", ARGUMENT_DESCRIPTION="a", Model=dict,
                                accept=lambda r: ({}, {}))
    with pytest.raises(ValueError, match="pydantic"):
        reports.check_plugin(mod)


def test_check_files_refuses_reserved_and_odd_names():
    assert reports.check_files({"report.json": {}, "extra.json": {}})
    for bad in ("meta.json", "steps.jsonl", "../x.json", "Report.json", "report.json\n", "meta.json\n"):
        with pytest.raises(ValueError):
            reports.check_files({bad: {}})


def test_shipped_profiles_take_the_builtin_json_report():
    for name in (BASIC, SEG):
        p = load_profile(name)
        assert p.report is reports.BUILTIN and p.report_sha256 is None


def test_builtin_report_takes_any_json_object_up_to_the_limit():
    r = submit(BASIC, {"finding": "a 3 cm mass", "nested": {"size_mm": [30, 28]}, "list": [1, "a", None]})
    result, files = reports.BUILTIN.accept(r)
    assert result == {"accepted": True}
    assert files == {"report.json": {"report_schema_version": "json-object-v1",
                                     "report": {"finding": "a 3 cm mass", "nested": {"size_mm": [30, 28]},
                                                "list": [1, "a", None]}}}
    assert reports.BUILTIN.accept(submit(BASIC, {}))[1]["report.json"]["report"] == {}
    with pytest.raises(EpisodeError, match="larger than"):
        submit(BASIC, {"text": "x" * reports.JSON_REPORT_MAX_BYTES})


def test_builtin_report_limit_counts_utf8_bytes_not_escapes():
    """The 64 KB limit is compact UTF-8; json.dumps' \\u escapes would count 6 bytes per character."""
    n = (reports.JSON_REPORT_MAX_BYTES - 20) // 2
    assert submit(BASIC, {"text": "é" * n})
    with pytest.raises(EpisodeError, match="larger than"):
        submit(BASIC, {"text": "é" * (n + 20)})


def test_a_harness_plugin_replaces_the_builtin_report():
    p = load_profile(BASIC, report=EXAMPLE_REPORT)
    assert p.report.SCHEMA_VERSION == "example-answer-v1" and len(p.report_sha256) == 64
    with pytest.raises(EpisodeError, match="answer"):
        submit(BASIC, {"answer": ""}, EXAMPLE_REPORT)
    with pytest.raises(EpisodeError, match="unknown view_refs"):
        submit(BASIC, answer_report(["v001"]), EXAMPLE_REPORT)
    assert p.report.accept(submit(BASIC, answer_report(), EXAMPLE_REPORT))[0] == {"accepted": True}


def test_a_plugin_needs_submit_report_in_the_profile(tmp_path):
    cfg = json.loads(load_profile(BASIC).protocol.path.read_text())
    cfg["mcp"]["tools"] = [t for t in cfg["mcp"]["tools"] if t != "submit_report"]
    (tmp_path / "protocol.json").write_text(json.dumps(cfg))
    assert load_profile(tmp_path).report is None
    with pytest.raises(ValueError, match="needs submit_report"):
        load_profile(tmp_path, report=EXAMPLE_REPORT)


def test_submit_report_text_and_schema_come_from_the_report():
    builtin = {s.name: s for s in tool_specs(load_profile(BASIC))}["submit_report"]
    plugin = {s.name: s for s in tool_specs(load_profile(BASIC, report=EXAMPLE_REPORT))}["submit_report"]
    assert "one JSON object" in builtin.description and "answer" not in json.dumps(builtin.input_schema)
    assert "answer is your answer" in plugin.description and '"answer"' in json.dumps(plugin.input_schema)


def test_session_writes_the_reports_files(make_server, fake_bridge):
    sess = make_server(SEG).session
    sess.reset("case_test")
    out = sess.call("submit_report", {"report": answer_report()})
    assert out.ok and out.result == {"accepted": True}
    d = sess.last_trace_dir
    assert json.loads((d / "report.json").read_text())["answer"] == "no finding"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["report_schema_version"] == "example-answer-v1" and meta["status"] == "submitted"
    assert meta["report_plugin_sha256"] == load_profile(SEG, report=EXAMPLE_REPORT).report_sha256


def test_session_with_the_builtin_report(make_server, fake_bridge):
    sess = make_server(SEG, report=None).session
    sess.reset("case_test")
    assert not sess.call("submit_report", {"report": "not an object"}).ok
    out = sess.call("submit_report", {"report": {"structures_seen": ["liver"]}})
    assert out.ok and out.result == {"accepted": True}
    rep = json.loads((sess.last_trace_dir / "report.json").read_text())
    assert rep == {"report_schema_version": "json-object-v1", "report": {"structures_seen": ["liver"]}}
    meta = json.loads((sess.last_trace_dir / "meta.json").read_text())
    assert meta["report_schema_version"] == "json-object-v1" and meta["report_plugin_sha256"] is None


def test_a_non_finite_report_is_refused_and_the_episode_continues(make_server, fake_bridge):
    """Python's JSON decoder reads 1e400 as inf. Such a report is refused, the step is logged (the trace stays
    strict JSON), and a valid report is still accepted."""
    srv = make_server(SEG, report=None)
    assert not srv.handle("reset_episode", {"case_id": "case_test"}).is_error
    res = srv.handle("submit_report", {"report": {"n": float("inf"), "m": [1, float("nan")]}})
    body = json.loads(res.content[0].text)
    assert res.is_error and body["error"]["type"] == "invalid_argument" and "report.n" in body["error"]["message"]
    assert not srv.session.episode.done
    out = srv.session.call("submit_report", {"report": {"n": 1}})
    assert out.ok and out.done
    steps = [json.loads(line) for line in (srv.session.last_trace_dir / "steps.jsonl").read_text().splitlines()]
    assert steps[1]["args"] == {"report": {"n": "inf", "m": [1, "nan"]}} and steps[1]["error"]
    assert steps[2]["error"] is None


def test_a_non_finite_tool_argument_is_refused_and_logged(make_server, fake_bridge):
    srv = make_server(SEG)
    srv.handle("reset_episode", {"case_id": "case_test"})
    res = srv.handle("orbit", {"azimuth_deg": float("inf"), "elevation_deg": 0})
    assert res.is_error and json.loads(res.content[0].text)["error"]["type"] == "invalid_argument"
    steps = (srv.session.trace_dir / "steps.jsonl").read_text().splitlines()
    assert json.loads(steps[-1])["args"]["azimuth_deg"] == "inf"


def test_the_episode_ends_only_after_the_plugin_accepted(make_server, fake_bridge, monkeypatch):
    """A plugin whose accept() fails, or whose files cannot be written, leaves the episode open."""
    sess = make_server(SEG, report=None).session
    sess.reset("case_test")
    for bad in (lambda r: 1 / 0, lambda r: ({"accepted": True}, {"meta.json": {}}),
                lambda r: ({"accepted": True}, {"report.json": {"x": float("inf")}})):
        monkeypatch.setattr(sess.profile.report, "accept", bad)
        with pytest.raises((ZeroDivisionError, ValueError)):
            sess.call("submit_report", {"report": {"ok": True}})
        assert not sess.episode.done and not (sess.trace_dir / "report.json").exists()
    monkeypatch.setattr(sess.profile.report, "accept", reports._accept_json)
    assert sess.call("submit_report", {"report": {"ok": True}}).done


def _steps(sess):
    return [json.loads(x) for x in (sess.trace_dir / "steps.jsonl").read_text().splitlines()]


def test_a_failing_plugin_or_a_lost_connection_logs_an_environment_step(make_server, fake_bridge, monkeypatch):
    """Both failures are logged as an `environment` error step before the exception propagates: the trace has no
    step number without a logged step."""
    import requests
    sess = make_server(SEG, report=None).session
    sess.reset("case_test")
    monkeypatch.setattr(sess.profile.report, "accept", lambda r: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        sess.call("submit_report", {"report": {"ok": True}})
    last = _steps(sess)[-1]
    assert last["step"] == 1 and last["tool"] == "submit_report"
    assert last["error"]["type"] == "environment" and "ZeroDivisionError" in last["error"]["message"]
    real = fake_bridge.call

    def lost(name, *a, **k):
        if name == "orbit":
            raise requests.ConnectionError("connection reset")
        return real(name, *a, **k)
    monkeypatch.setattr(fake_bridge, "call", lost)
    with pytest.raises(requests.ConnectionError):
        sess.call("orbit", {"azimuth_deg": 10})
    last = _steps(sess)[-1]
    assert last["step"] == 2 and last["tool"] == "orbit"
    assert last["error"] == {"type": "environment", "message": "ConnectionError: connection reset"}
    assert not sess.episode.done


def test_report_files_are_written_all_or_none(tmp_path, monkeypatch):
    """When publishing the second file fails, the first is taken back and no temporary file is left."""
    from pathlib import Path

    from slicer_mcp.tracing.logger import TraceLogger
    log = TraceLogger(tmp_path, "ep_test")
    real, n = Path.replace, []

    def replace(self, target):
        n.append(target)
        if len(n) == 2:
            raise OSError("disk full")
        return real(self, target)
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="disk full"):
        log.write_report_files({"report.json": {"a": 1}, "extra.json": {"b": 2}})
    assert not [f for f in log.dir.iterdir() if f.name.startswith((".report", ".extra", "report", "extra"))]
    real_unlink, n[:] = Path.unlink, []

    def unlink(self, missing_ok=False):
        if self.name == ".report.json.tmp":
            raise PermissionError("locked")
        return real_unlink(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="disk full"):
        log.write_report_files({"report.json": {"a": 1}, "extra.json": {"b": 2}})
    assert not (log.dir / "report.json").exists() and not (log.dir / ".extra.json.tmp").exists()


def test_non_finite_client_meta_is_refused_before_the_reset(make_server, fake_bridge):
    srv = make_server(SEG)
    res = srv.handle("reset_episode", {"case_id": "case_test", "client_meta": {"seed": float("inf")}})
    assert res.is_error and json.loads(res.content[0].text)["error"]["type"] == "invalid_argument"
    assert fake_bridge.calls == [] and srv.session.logger is None
    with pytest.raises(ValueError, match="client_meta"):
        srv.session.reset("case_test", {"x": [float("nan")]})
    assert fake_bridge.calls == []
