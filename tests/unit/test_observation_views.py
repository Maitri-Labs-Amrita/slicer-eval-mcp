"""observation.views: a profile chooses the live views the model is shown. Without
"3d", no frame of the 3D view is ever captured, no 3D-only tool may be offered, and no description names a 3D view;
profiles with the 3D view are unchanged, byte for byte."""

import hashlib
import json

import pytest

from slicer_mcp.config import REPO, VIEWS, load_profile
from slicer_mcp.schemas.tools import ArgsRejected, validate_args
from slicer_mcp.tools.registry import THREED_TOOLS, instructions, tool_specs
from slicer_mcp.tools_md import render
from slicer_mcp.tracing.logger import read_trace

from tests.unit.conftest import BASIC, SEG
from tests.unit.test_registry import list_tools
from tests.unit.test_server_mcp import body, run

SLICES = ["axial", "sagittal", "coronal"]
# The shipped profiles' tool listings and instructions at slicer-eval-mcp ddfeb58 (v0.1.0, before observation.views):
# (model tools as a harness hashes them, every ToolSpec, the server instructions), each a sha256.
GOLDEN = {
    BASIC: ("45a0886b10162ef19436d9d73e15c78f04b1e1ec1c9bea53e72e0e6bdf0cee43",
            "53c30dbbbf3700af89c125339b8dcdb5912c78f57bdf976ba1f0e57c64d4797a",
            "77104607b9ee1dcd27542b0ae2cfc6d6350bd417c6c2338be7d89cb2bf68f90d"),
    SEG: ("5da09d9ef6cfb2e81c1bf05e1911e1cb5bdbdc6aa4648f0d18ba3158b3793b28",
          "4255fb062fbcf75dfb8f50d8304b1504f76f5c74a922695f02af7c44bb0b3e74",
          "52d73f0c8cbdcc114200091ba349f6ad81dd41acbb68819d556c4d170bf0e955"),
}


def _sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def listing_hashes(profile) -> tuple:
    specs = tool_specs(profile)
    model = [{"name": s.name, "description": s.description, "input_schema": s.input_schema}
             for s in specs if not s.runner_only]
    full = [{"name": s.name, "title": s.title, "description": s.description, "input_schema": s.input_schema,
             "read_only": s.read_only, "runner_only": s.runner_only} for s in specs]
    return _sha(model), _sha(full), hashlib.sha256(instructions(profile).encode()).hexdigest()


@pytest.mark.parametrize("name", [BASIC, SEG])
def test_profiles_with_the_3d_view_are_unchanged(name):
    p = load_profile(name)
    assert p.views == VIEWS and p.has_3d
    assert listing_hashes(p) == GOLDEN[name]


def _write(tmp_path, views=SLICES, drop=THREED_TOOLS, name="slices_seg"):
    """ct_segments with observation.views and without the tools in `drop`."""
    base = json.loads((REPO / "profiles" / SEG / "protocol.json").read_text())
    base["config_version"] = f"{name}-v1"
    base["observation"] = {**base["observation"], **({"views": views} if views is not None else {})}
    base["mcp"] = {**base["mcp"], "tools": [t for t in base["mcp"]["tools"] if t not in drop]}
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "protocol.json").write_text(json.dumps(base, indent=2))
    return d


@pytest.fixture()
def slices_profile(tmp_path):
    return _write(tmp_path)


def test_a_slices_only_profile_loads(slices_profile):
    p = load_profile(slices_profile)
    assert p.views == tuple(SLICES) and not p.has_3d
    assert not set(p.tools) & set(THREED_TOOLS)
    assert {"zoom", "set_segment_visibility", "set_slice_outlines"} <= set(p.tools)


def test_views_come_in_capture_order(tmp_path):
    p = load_profile(_write(tmp_path, views=["coronal", "3d", "axial", "sagittal"], drop=()))
    assert p.views == VIEWS


@pytest.mark.parametrize("views, why", [
    (["axial", "sagittal"], "including"),
    (["3d", "axial", "sagittal"], "including"),
    (["axial", "sagittal", "coronal", "oblique"], "among"),
    (["axial", "sagittal", "coronal", "axial"], "distinct"),
    ("axial,sagittal,coronal", "distinct"),
])
def test_bad_views_are_refused(tmp_path, views, why):
    with pytest.raises(ValueError, match=why):
        load_profile(_write(tmp_path, views=views, drop=()))


@pytest.mark.parametrize("tool", THREED_TOOLS)
def test_a_3d_tool_needs_the_3d_view(tmp_path, tool):
    with pytest.raises(ValueError, match="need the '3d' view"):
        load_profile(_write(tmp_path, drop=tuple(t for t in THREED_TOOLS if t != tool)))


def test_no_description_or_schema_names_the_3d_view(slices_profile):
    p = load_profile(slices_profile)
    specs = [s for s in tool_specs(p)]
    for s in specs:
        text = s.description + json.dumps(s.input_schema)
        assert "3D" not in text and "3d" not in text, s.name
        for t in THREED_TOOLS:  # no tool the profile does not offer is named
            assert t not in text, (s.name, t)
    assert "3D" not in instructions(p)
    zoom = next(s for s in specs if s.name == "zoom")
    assert zoom.input_schema["properties"]["view"]["enum"] == ["slices"]
    assert zoom.input_schema["properties"]["view"]["default"] == "slices"


def test_slices_only_descriptions(slices_profile):
    d = {s.name: s.description for s in tool_specs(load_profile(slices_profile))}
    assert d["reset_view"].startswith("Restore the initial view: slice positions and window/level")
    assert d["reset_view"].endswith("Returns the axial, sagittal and coronal slice views.")
    assert "(all shown, slice outlines at their start values)" in d["reset_view"]
    assert "Slice views only. Returns the three slice views." in d["set_window_preset"]
    assert d["zoom"].startswith("Zoom the three slice views by a factor")
    assert "Hidden structures disappear from the slice outlines." in d["set_segment_visibility"]
    assert d["set_segment_visibility"].endswith("Returns the three slice views.")
    assert d["point_at"].endswith("so point at the new frames. Returns the three slice views.")  # no pan_to note


def test_zoom_takes_the_slices_only(slices_profile):
    p = load_profile(slices_profile)
    zoom = next(s for s in tool_specs(p) if s.name == "zoom").model
    assert validate_args(zoom, {"factor": 2.0}) == ({"factor": 2.0, "view": "slices"}, [])
    assert validate_args(zoom, {"factor": 2.0, "view": "slices"})[0]["view"] == "slices"
    with pytest.raises(ArgsRejected, match="view"):
        validate_args(zoom, {"factor": 2.0, "view": "3d"})


def test_no_3d_frame_is_ever_captured(make_server, fake_bridge, slices_profile):
    srv = make_server(str(slices_profile), report=None)
    names = [t.name for t in list_tools(srv)]
    assert not set(names) & set(THREED_TOOLS)
    res = run(srv, [
        ("reset_episode", {"case_id": "case_test"}, None),
        ("zoom", {"factor": 1.5}, None),
        ("zoom", {"factor": 1.5, "view": "3d"}, None),
        ("point_at", {"view": "axial", "x": 10, "y": 10, "composite": True}, None),
        ("set_segment_visibility", {"hide": ["liver"]}, None),
        ("set_slice_outlines", {"visible": False}, None),
        ("bookmark_view", {"label": "all views"}, None),
        ("reset_view", {}, None),
        ("orbit", {"azimuth_deg": 30}, None),  # not offered: refused before the bridge
    ])
    reset, zoom, zoom3d, point, vis, outl, book, rv, orbit = res
    for r in (reset, point, vis, book, rv):
        p, imgs = body(r)
        assert not r.is_error and p["frames"] == SLICES and len(imgs) == 3
    assert body(zoom)[0]["frames"] == SLICES and not zoom.is_error
    assert zoom3d.is_error and orbit.is_error
    assert ("zoom", {"factor": 1.5, "view": "slices"}, tuple(SLICES)) in fake_bridge.calls
    assert all("3d" not in capture for _, _, capture in fake_bridge.calls)
    assert not any(name == "orbit" for name, _, _ in fake_bridge.calls)
    meta, _ = read_trace(srv.session.trace_dir)
    assert meta["image"]["views"] == SLICES


def test_tools_md_renders_a_slices_only_profile(slices_profile):
    md = render(str(slices_profile))
    assert "| axial, sagittal, coronal |" in md
    assert "returns frames: every view (axial, sagittal, coronal)" in md
    assert "3d" not in md.split("## Tools", 1)[1].replace("observation.views", "")
