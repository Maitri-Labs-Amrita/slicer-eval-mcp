"""Display power check: parsing, evaluation, and the server's refusal at start and at reset."""

import json
import sys

import pytest

from slicer_mcp import display
from slicer_mcp import server as server_mod
from tests.unit.fakes import FakeClient

ASSERTIONS_ON = """Assertion status system-wide:
   BackgroundTask                 0
   PreventUserIdleDisplaySleep    1
   PreventSystemSleep             0
"""
ASSERTIONS_OFF = ASSERTIONS_ON.replace("PreventUserIdleDisplaySleep    1", "PreventUserIdleDisplaySleep    0")
PMSET = """ sleep                1 (sleep prevented by powerd)
 displaysleep         5 (display sleep prevented by caffeinate)
 disksleep            10
"""


def test_parsers():
    assert display.parse_assertions(ASSERTIONS_ON) is True
    assert display.parse_assertions(ASSERTIONS_OFF) is False
    assert display.parse_assertions("nothing") is None and display.parse_assertions(None) is None
    assert display.parse_displaysleep(PMSET) == 5
    assert display.parse_displaysleep(" displaysleep 0\n") == 0


def st(asleep=False, prevent=True, timer=5):
    return {"platform": "darwin", "asleep": asleep, "prevent_display_sleep": prevent, "displaysleep_min": timer}


def test_evaluate():
    assert display.evaluate(st()) == ([], [])
    errors, _ = display.evaluate(st(asleep=True))
    assert errors and "asleep" in errors[0]
    assert display.evaluate(st(asleep=True), "warn")[0] == []
    assert display.evaluate(st(asleep=True), "off") == ([], [])
    _, warnings = display.evaluate(st(prevent=False))
    assert any("caffeinate" in w for w in warnings)
    assert display.evaluate(st(prevent=False, timer=0)) == ([], [])
    _, warnings = display.evaluate({"platform": "linux", "asleep": None})
    assert warnings and "unknown" in warnings[0]
    with pytest.raises(ValueError):
        display.evaluate(st(), "maybe")


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreGraphics / pmset are macOS only")
def test_real_status_on_this_mac():
    s = display.display_status()
    assert s["asleep"] in (True, False) and s["prevent_display_sleep"] in (True, False, None)


@pytest.fixture()
def require_runtime(runtime_file):
    rt = json.loads(runtime_file.read_text())
    rt["display_check"] = "require"
    runtime_file.write_text(json.dumps(rt))
    return runtime_file


def test_server_refuses_to_start_on_a_sleeping_display(protocol, monkeypatch, require_runtime):
    monkeypatch.setattr(server_mod, "BridgeClient", lambda *a, **k: FakeClient(protocol))
    monkeypatch.setattr(display, "display_status", lambda asleep_only=False: st(asleep=True))
    with pytest.raises(server_mod.StartupError, match="display"):
        server_mod.build("ct_segments", runtime=str(require_runtime))
    with pytest.raises(SystemExit) as ei:
        server_mod.main(["--profile", "ct_segments", "--runtime", str(require_runtime)])
    assert ei.value.code == 2


def test_server_starts_awake_and_records_the_display(protocol, monkeypatch, require_runtime):
    monkeypatch.setattr(server_mod, "BridgeClient", lambda *a, **k: FakeClient(protocol))
    monkeypatch.setattr(display, "display_status", lambda asleep_only=False: st())
    srv = server_mod.build("ct_segments", runtime=str(require_runtime))
    assert not srv.handle("reset_episode", {"case_id": "case_test"}).is_error
    meta = json.loads((srv.session.trace_dir / "meta.json").read_text())
    assert meta["host"]["display"]["asleep"] is False
    monkeypatch.setattr(display, "display_status", lambda asleep_only=False: st(asleep=True))
    res = srv.handle("reset_episode", {"case_id": "case_test"})
    body = json.loads(res.content[0].text)
    assert res.is_error and body["error"]["type"] == "environment" and "asleep" in body["error"]["message"]


def test_injected_client_skips_the_check(protocol, monkeypatch, require_runtime):
    """build(client=...) is the test path (the client repo's fakes): it never depends on the display."""
    monkeypatch.setattr(display, "display_status", lambda asleep_only=False: st(asleep=True))
    srv = server_mod.build("ct_segments", runtime=str(require_runtime), client=FakeClient(protocol))
    assert srv.display_check == "off"
