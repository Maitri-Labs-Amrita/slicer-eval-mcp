"""Unit fixtures: a temp runtime config and MCP servers built through the real CLI path over a fake bridge."""

import json

import pytest

from slicer_mcp import server as server_mod
from slicer_mcp.config import REPO, load_profile
from tests.unit.fakes import FakeClient

SEG = "ct_segments"
BASIC = "ct_basic"
EXAMPLE_REPORT = str(REPO / "examples" / "report_plugin.py")


@pytest.fixture()
def seg_profile():
    return load_profile(SEG, report=EXAMPLE_REPORT)


@pytest.fixture()
def protocol(seg_profile):
    return seg_profile.protocol


@pytest.fixture()
def runtime_file(tmp_path):
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    rt.update(traces_dir=str(tmp_path / "traces"), display_check="off")
    p = tmp_path / "runtime.json"
    p.write_text(json.dumps(rt))
    return p


@pytest.fixture()
def fake_bridge(monkeypatch, protocol):
    """Every BridgeClient the server creates is this one fake (for the ct_segments protocol)."""
    fake = FakeClient(protocol)
    monkeypatch.setattr(server_mod, "BridgeClient", lambda *a, **k: fake)
    return fake


@pytest.fixture()
def make_server(monkeypatch, fake_bridge, runtime_file):
    """make_server(profile name, report plugin) -> an EnvServer over the fake bridge, built through the real build()
    path. The default report is the example plugin; report=None gives the built-in JSON report."""
    def _make(profile=SEG, report=EXAMPLE_REPORT):
        fake_bridge.protocol = load_profile(profile).protocol
        return server_mod.build(profile, runtime=str(runtime_file), report=report)
    return _make
