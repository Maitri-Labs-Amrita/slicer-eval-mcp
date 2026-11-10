"""Pydantic argument models: bounds come from the config; out-of-range numbers clamp with a warning; wrong
names / types / choices are rejected."""

import copy
import hashlib
import json

import pytest

import dataclasses

from slicer_mcp.config import REPO, ProtocolConfig, load_profile
from slicer_mcp.schemas.tools import ArgsRejected, build_arg_models, input_schema, validate_args
from slicer_mcp.tools.registry import MODEL_TOOLS

PROFILE = load_profile("ct_segments")
P = PROFILE.protocol
M = build_arg_models(PROFILE)
MB = build_arg_models(load_profile("ct_basic"))


def ok(tool, args):
    return validate_args(M[tool], args)


def rejected(tool, args):
    with pytest.raises(ArgsRejected) as e:
        validate_args(M[tool], args)
    return str(e.value)


def test_every_tool_has_a_model():
    assert set(MODEL_TOOLS) | {"reset_episode", "end_episode", "observe"} == set(M)
    assert not {"set_segment_visibility", "set_3d_content"} & set(MB)


def test_clamp_with_warning_in_bridge_words():
    args, w = ok("orbit", {"azimuth_deg": 200, "elevation_deg": -95})
    assert args == {"azimuth_deg": 180.0, "elevation_deg": -90.0, "roll_deg": 0.0}
    assert w == ["azimuth_deg=200 clamped to 180 (allowed [-180, 180])",
                 "elevation_deg=-95 clamped to -90 (allowed [-90, 90])"]
    args, w = ok("point_at", {"view": "axial", "x": -3, "y": 900})
    assert args == {"view": "axial", "x": 0.0, "y": 511.0} and len(w) == 2
    args, w = ok("zoom", {"factor": 9})
    assert args == {"factor": 4.0} and "factor=9 clamped to 4" in w[0]
    args, w = ok("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": -300})
    assert args["offset_mm"] == -200.0 and len(w) == 1
    args, w = ok("set_clip_box", {"center_ras": [0, 0, 0], "size_mm": [1, 50, 2]})
    assert args["size_mm"] == [5.0, 50.0, 5.0] and len(w) == 2
    args, w = ok("scroll_slice", {"view": "axial", "delta_slices": -999})
    assert args == {"view": "axial", "delta_slices": -200} and len(w) == 1


def test_in_range_values_pass_unchanged():
    assert ok("orbit", {"azimuth_deg": 30, "elevation_deg": 15, "roll_deg": -5}) == (
        {"azimuth_deg": 30.0, "elevation_deg": 15.0, "roll_deg": -5.0}, [])
    assert ok("scroll_slice", {"view": "sagittal", "delta_slices": 3.0}) == ({"view": "sagittal", "delta_slices": 3}, [])
    assert ok("point_at", {"view": "coronal", "x": 400, "y": 300}) == ({"view": "coronal", "x": 400.0, "y": 300.0}, [])
    assert ok("set_slice", {"view": "axial", "position_mm": -9999}) == (
        {"view": "axial", "position_mm": -9999.0}, [])


def test_omitted_point_means_the_focus():
    assert ok("pan_to", {}) == ({}, [])
    assert ok("set_clip_box", {"size_mm": [90, 80, 90]}) == ({"size_mm": [90.0, 80.0, 90.0]}, [])
    assert ok("pan_to", {"ras": [1, 2, 3]})[0] == {"ras": [1.0, 2.0, 3.0]}
    assert "size_mm" in rejected("set_clip_box", {})


def test_composite_kept_only_when_requested():
    assert ok("zoom", {"factor": 2, "composite": True})[0] == {"factor": 2.0, "composite": True}
    assert ok("zoom", {"factor": 2, "composite": False})[0] == {"factor": 2.0}


@pytest.mark.parametrize("tool,args,needle", [
    ("orbit", {"azimuth_deg": True}, "valid number"),
    ("orbit", {"azimuth_deg": "30"}, "valid number"),
    ("orbit", {"azimuth_deg": float("nan")}, "finite"),
    ("orbit", {"azimuth": 30}, "unknown argument"),
    ("set_slice", {"view": "oblique", "position_mm": 0}, "view"),
    ("set_slice", {"view": "axial"}, "position_mm: Field required"),
    ("scroll_slice", {"view": "axial", "delta_slices": 2.5}, "whole number"),
    ("set_render_preset", {"name": "MyTunedPreset"}, "name"),
    ("set_window_preset", {"name": "ct-angio"}, "name"),
    ("set_view_direction", {"name": "apical four-chamber"}, "name"),
    ("jump_to_point", {"ras": [1, 2]}, "ras"),
    ("jump_to_point", {"ras": [1, 2, "x"]}, "ras.2"),
    ("point_at", {"view": "3d", "x": 1, "y": 1}, "view"),
    ("point_at", {"view": "axial", "x": 1}, "y: Field required"),
    ("point_at", {"view": "axial", "x": "10", "y": 1}, "valid number"),
    ("pan_to", {"ras": [1, 2]}, "ras"),
    ("bookmark_view", {"label": ""}, "label"),
    ("submit_report", {"report": "{S,D,S}"}, "report"),
    ("reset_view", {"verbose": True}, "unknown argument"),
    ("reset_episode", {"case_id": "../../etc"}, "case_id"),
])
def test_rejections(tool, args, needle):
    assert needle in rejected(tool, args)


def test_non_object_arguments_rejected():
    with pytest.raises(ArgsRejected):
        validate_args(M["orbit"], [1, 2, 3])


def test_bounds_follow_the_config():
    data = copy.deepcopy(P.data)
    data["camera"]["orbit_azimuth_range"] = [-45.0, 45.0]
    data["render"]["presets"] = ["CT-Bone", "CT-MIP"]
    raw = json.dumps(data).encode()
    alt = ProtocolConfig(data, P.path, hashlib.sha256(raw).hexdigest())
    m = build_arg_models(dataclasses.replace(PROFILE, protocol=alt))
    args, w = validate_args(m["orbit"], {"azimuth_deg": 90})
    assert args["azimuth_deg"] == 45.0 and w
    s = input_schema(m["orbit"])["properties"]["azimuth_deg"]
    assert (s["minimum"], s["maximum"]) == (-45.0, 45.0)
    assert input_schema(m["set_render_preset"])["properties"]["name"]["enum"] == ["CT-Bone", "CT-MIP"]
    with pytest.raises(ArgsRejected):
        validate_args(m["set_render_preset"], {"name": "CT-Cardiac3"})


def test_report_schema_is_self_contained():
    s = json.dumps(input_schema(M["submit_report"]))
    assert "$ref" not in s and "$defs" not in s
    report = input_schema(M["submit_report"])["properties"]["report"]
    assert report["type"] == "object" and report["additionalProperties"] is True and report["properties"] == {}
    plugin = build_arg_models(load_profile("ct_segments", report=str(REPO / "examples" / "report_plugin.py")))
    s = json.dumps(input_schema(plugin["submit_report"]))
    assert "$ref" not in s and "$defs" not in s
    report = input_schema(plugin["submit_report"])["properties"]["report"]
    assert report["required"] == ["answer"] and set(report["properties"]) == {"answer", "evidence"}
    assert report["additionalProperties"] is False and report["properties"]["answer"]["maxLength"] == 4000


def test_slab_is_an_option_of_set_slice_clamped_and_dropped_when_omitted():
    args, w = validate_args(M["set_slice"], {"view": "coronal", "position_mm": -60, "slab_mm": 35})
    assert args == {"view": "coronal", "position_mm": -60.0, "slab_mm": 20.0} and "slab_mm=35 clamped to 20" in w[0]
    assert validate_args(M["set_slice"], {"view": "axial", "position_mm": 10})[0] == {"view": "axial",
                                                                                    "position_mm": 10.0}
    for extra in ("slab_mode", "radius_mm"):
        with pytest.raises(ArgsRejected, match="unknown argument"):
            validate_args(M["set_slice"], {"view": "axial", "position_mm": 1, extra: 1})


def test_zoom_view_slices():
    assert validate_args(M["zoom"], {"factor": 2, "view": "slices"})[0] == {"factor": 2.0, "view": "slices"}
    assert validate_args(M["zoom"], {"factor": 2})[0] == {"factor": 2.0}
    with pytest.raises(ArgsRejected, match="view"):
        validate_args(M["zoom"], {"factor": 2, "view": "axial"})


def test_every_window_preset_is_offered():
    for name in P["window_level"]["presets"]:
        assert validate_args(M["set_window_preset"], {"name": name})[0] == {"name": name}


def test_segment_tools_take_the_six_classes():
    def okS(tool, args):
        return validate_args(M[tool], args)

    assert okS("set_segment_visibility", {"show": ["all"]})[0] == {"show": ["all"], "hide": []}
    assert okS("center_on_segments", {}) == ({}, [])
    for bad in ({"names": []}, {"names": ["all"]}, {"names": ["kidney_left", "XX"]}):
        with pytest.raises(ArgsRejected):
            okS("center_on_segments", bad)
    assert okS("set_slice_outlines", {"visible": False}) == ({"visible": False}, [])
    with pytest.raises(ArgsRejected):
        okS("set_slice_outlines", {"visible": "no"})

    assert okS("set_segment_visibility", {"hide": ["liver", "spleen"]}) == ({"show": [], "hide": ["liver", "spleen"]}, [])
    for bad in ({"show": ["XX"]}, {"show": "kidney_left"}, {"hide": ["lv"]}, {"show": ["kidney_left"], "colour": 1}, {"hide": ["Myo"]},
                {"show": ["Pool"]}):
        with pytest.raises(ArgsRejected):
            okS("set_segment_visibility", bad)
    args, w = okS("set_segment_opacity", {"name": "kidney_right", "opacity": 1.5})
    assert args == {"name": "kidney_right", "opacity": 1.0} and w == ["opacity=1.5 clamped to 1 (allowed [0, 1])"]
    with pytest.raises(ArgsRejected):
        okS("set_segment_opacity", {"name": "PV", "opacity": 0.5})
    assert okS("center_on_segments", {"names": ["kidney_left", "aorta"]}) == ({"names": ["kidney_left", "aorta"]}, [])
    with pytest.raises(ArgsRejected):
        okS("set_3d_content", {"mode": "ct_in_shell"})
    assert input_schema(M["set_3d_content"])["properties"]["mode"]["enum"] == P["segmentation"]["modes"]


def test_measure_args():
    args, w = validate_args(M["measure"], {"view": "axial", "x1": -4, "y1": 10, "x2": 600, "y2": 10.5})
    assert args == {"view": "axial", "x1": 0.0, "y1": 10.0, "x2": 511.0, "y2": 10.5} and len(w) == 2
    with pytest.raises(ArgsRejected, match="y2"):
        validate_args(M["measure"], {"view": "axial", "x1": 1, "y1": 1, "x2": 2})
