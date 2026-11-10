"""Each profile exposes exactly its allowlist (listed through the MCP layer); descriptions and schemas
are clean and follow the profile."""

import json
import re
from pathlib import Path

import anyio
import pytest
from mcp import Client

from slicer_mcp import server as server_mod
from slicer_mcp.config import load_profile
from slicer_mcp.tools.registry import MODEL_TOOLS, RUNNER_ONLY_META, describe, instructions, tool_specs

from tests.unit.conftest import BASIC, EXAMPLE_REPORT, SEG

RUNNER = {"reset_episode", "end_episode", "observe"}
EPISODE = {"reset_view", "bookmark_view", "submit_report"}
TWO_D = {"set_slice", "scroll_slice", "set_window_preset", "jump_to_point", "point_at"}
THREE_D = {"orbit", "set_view_direction", "zoom", "pan_to", "set_render_preset", "set_clip_box", "set_clip_plane",
           "clear_clip"}
SEGMENT = {"set_segment_visibility", "set_segment_opacity", "set_3d_content", "set_slice_outlines", "center_on_segments"}
EXPECTED = {
    SEG: RUNNER | EPISODE | TWO_D | THREE_D | SEGMENT | {"measure"},
    BASIC: RUNNER | EPISODE | TWO_D | THREE_D | {"measure"},
}


def list_tools(srv):
    async def go():
        async with Client(srv.server) as c:
            return (await c.list_tools()).tools
    return anyio.run(go)


@pytest.mark.parametrize("profile", [SEG, BASIC])
def test_profile_exposes_exactly_its_tools(make_server, profile):
    tools = list_tools(make_server(profile))
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED[profile]
    runner = {t.name for t in tools if (t.meta or {}).get(RUNNER_ONLY_META)}
    assert runner == RUNNER
    for t in tools:
        assert t.input_schema["type"] == "object" and t.input_schema["additionalProperties"] is False
        assert t.description and t.annotations.open_world_hint is False


def test_the_listing_follows_the_profiles_order(make_server):
    names = [t.name for t in list_tools(make_server(SEG))]
    assert names[:3] == ["reset_episode", "end_episode", "observe"]
    assert names[3:] == list(load_profile(SEG).tools)


def test_bad_profiles_are_refused(tmp_path):
    base = json.loads((load_profile(BASIC).protocol.path).read_text())

    def write(**mcp):
        d = tmp_path / f"p{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        (d / "protocol.json").write_text(json.dumps({**base, "mcp": {**base["mcp"], **mcp}}))
        (d / "report.py").write_text(Path(EXAMPLE_REPORT).read_text())
        return d
    with pytest.raises(ValueError, match="unknown"):
        load_profile(write(tools=["orbit", "make_blood_pool_shell"]))
    with pytest.raises(ValueError, match="distinct"):
        load_profile(write(tools=["orbit", "orbit"]))
    with pytest.raises(ValueError, match="need a segmentation"):
        load_profile(write(tools=["orbit", "set_3d_content"]))
    with pytest.raises(ValueError, match="needs submit_report"):
        load_profile(write(tools=["orbit"], report="report.py"))
    with pytest.raises(ValueError, match="needs submit_report"):
        load_profile(write(tools=["orbit"]), report=EXAMPLE_REPORT)
    with pytest.raises(OSError):
        load_profile(write(report="missing.py"))
    p = load_profile(write(tools=["orbit"]))
    assert p.tools == ("orbit",) and p.report is None


def test_unknown_profile_rejected(fake_bridge, runtime_file):
    with pytest.raises(server_mod.StartupError, match="cannot load the profile"):
        server_mod.build("no_such_profile", runtime=str(runtime_file))


def test_refuses_bridge_with_other_config(fake_bridge, runtime_file, monkeypatch):
    real_ping = fake_bridge.ping
    monkeypatch.setattr(fake_bridge, "ping", lambda: {**real_ping(), "config_sha256": "0" * 64})
    with pytest.raises(server_mod.StartupError, match="sha256"):
        server_mod.build(SEG, runtime=str(runtime_file))
    monkeypatch.setattr(fake_bridge, "ping", lambda: {**real_ping(), "obs_size": 768})
    with pytest.raises(server_mod.StartupError, match="frame size"):
        server_mod.build(SEG, runtime=str(runtime_file))
    monkeypatch.setattr(fake_bridge, "ping", lambda: {**real_ping(), "handlers": ["ping", "reset_episode", "capture"]})
    with pytest.raises(server_mod.StartupError, match="lacks handlers"):
        server_mod.build(SEG, runtime=str(runtime_file))


def test_main_exits_nonzero_when_refusing(fake_bridge, runtime_file, monkeypatch, capsys):
    monkeypatch.setattr(fake_bridge, "ping", lambda: {"config_sha256": "0" * 64})
    with pytest.raises(SystemExit) as e:
        server_mod.main(["--profile", SEG, "--runtime", str(runtime_file)])
    assert e.value.code == 2 and "refusing to start" in capsys.readouterr().err


@pytest.mark.parametrize("profile", [SEG, BASIC])
def test_descriptions_never_name_a_case_or_a_path(profile):
    p = load_profile(profile)
    blob = json.dumps([[s.name, s.title, s.description, s.input_schema] for s in tool_specs(p)])
    blob += json.dumps(instructions(p))
    assert not re.search(r"case_\d|LEAK|/Users/|\.nrrd|dicom", blob, re.I)


@pytest.mark.parametrize("profile", [SEG, BASIC])
def test_descriptions_follow_the_config(profile):
    p = load_profile(profile)
    protocol, d = p.protocol, describe(p)
    for name in protocol["render"]["presets"]:
        assert name in d["set_render_preset"]
    for name in protocol["window_level"]["presets"]:
        assert name in d["set_window_preset"]
    for name in protocol["camera"]["view_directions"]:
        assert name in d["set_view_direction"]
    assert "REMOVED" in d["set_clip_plane"] and "azimuth 0 = anterior" in d["set_clip_plane"]
    assert "+90 moves the camera to the patient's left" in d["orbit"]
    assert "axial -> S" in d["set_slice"] and "slab_mm" in d["set_slice"]
    assert "0 to 511" in d["point_at"] and "pan_to and set_clip_box use the focus" in d["point_at"]
    assert "result.hu" in d["point_at"] and "result.hu" in d["jump_to_point"]
    assert "the focus" in d["pan_to"] and "Omit center_ras" in d["set_clip_box"]
    assert "case_info" in d["reset_episode"]
    assert f"A {protocol['slice_tools']['measure_color_name']} line" in d["measure"]
    ins = instructions(p)
    assert ins.startswith(protocol["mcp"]["instructions_title"])
    assert "How the tools work" in ins and "clamped and reported in warnings" in ins and len(ins) < 800
    assert d["submit_report"] == p.report.DESCRIPTION


def test_segment_wording_only_with_a_segmentation():
    seg_d, basic = describe(load_profile(SEG)), describe(load_profile(BASIC))
    seg = load_profile(SEG).protocol["segmentation"]
    legend = ", ".join(f"{c['name']} {c['color_name']}" for c in seg["classes"])
    assert legend in seg_d["set_segment_visibility"]
    assert "cuts the structure shells too" in seg_d["set_clip_box"] and "inside_segment" in seg_d["point_at"]
    assert "the structures (all shown, opaque" in seg_d["reset_view"]
    assert f"shell_only (the structure shells alone: {seg['shell_only_note']})" in seg_d["set_3d_content"]
    assert not set(basic) & SEGMENT
    text = json.dumps(basic)
    for w in ("shell", "structure", "inside_segment"):
        assert w not in text, w


def test_every_registry_tool_has_a_description_with_a_segmentation_and_a_report():
    assert set(describe(load_profile(SEG))) >= set(MODEL_TOOLS) | RUNNER
