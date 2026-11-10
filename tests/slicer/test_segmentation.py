"""Segment tools: segmentation shells in the live bridge (a profile with a segmentation, e.g. ct_segments).
Each tool: state changed, frame changed, out-of-range values clamped with a warning, bad names refused with
nothing changed. Needs <case>.shell.nrrd / .labels.nrrd / .shell.json (tools/prepare_shell.py)."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from slicer_mcp.bridge_client import VIEWS, BridgeError
from slicer_mcp.imaging import changed, compare, load_rgb
from tests.slicer.test_episode import diff_pixels, marker_flicker_only

pytestmark = pytest.mark.slicer
SLICES = ("axial", "sagittal", "coronal")
SHELL_SUFFIXES = (".shell.nrrd", ".labels.nrrd", ".shell.json")


@pytest.fixture(scope="module")
def seg_case(runtime, profile, case_id):
    if not profile.segmentation:
        pytest.skip(f"profile {profile.name} has no segmentation")
    d = Path(runtime["cases_dir"])
    if not all((d / f"{case_id}{s}").exists() for s in SHELL_SUFFIXES):
        pytest.skip(f"{case_id} has no shells (slicer-mcp case prepare --labels ...)")
    return case_id


def _reset(bridge, case, seg=True, capture=VIEWS):
    return bridge.call("reset_episode", {"case_id": case, "with_segments": seg}, capture=capture)


@pytest.fixture()
def fresh(bridge, seg_case):
    return _reset(bridge, seg_case)


def _rejected(bridge, name, args, match, before):
    with pytest.raises(BridgeError) as e:
        bridge.call(name, args)
    assert e.value.type == "invalid_argument" and match in e.value.message, e.value.message
    after = bridge.call("capture", {}, capture=("3d",))
    assert after.state == before.state
    return after


def test_reset_loads_shells_with_defaults(bridge, fresh, protocol):
    seg = protocol["segmentation"]
    st = fresh.state["segments"]
    names = [c["name"] for c in sorted(seg["classes"], key=lambda c: c["label"])]
    assert st == {"mode": seg["default_mode"], "slice_outlines": seg["default_slice_outlines"],
                  "visible": dict.fromkeys(names, True), "opacity": dict.fromkeys(names, seg["default_opacity"])}
    assert fresh.state["render"]["visible"] is False
    info = bridge.call("get_case_info").result["segments"]
    assert info["classes"] == names and info["colors"]["kidney_left"] == "pink"
    inv = fresh.result["node_inventory"]
    assert inv["vtkMRMLSegmentationNode"] == 2 and "vtkMRMLLabelMapVolumeNode" not in inv
    assert len(fresh.result["segments_manifest_sha256"]) == 64
    rgb = load_rgb(fresh.images["3d"])
    assert (rgb.max(axis=2) > 40).sum() > 5000


def test_no_segments_without_with_segments(bridge, seg_case):
    r = _reset(bridge, seg_case, seg=False, capture=("3d",))
    assert "segments" not in r.state and r.state["render"]["visible"] is True
    assert "vtkMRMLSegmentationNode" not in r.result["node_inventory"]
    for name, args in (("set_3d_content", {"mode": "shell_only"}), ("set_segment_visibility", {"hide": ["kidney_left"]})):
        with pytest.raises(BridgeError, match="no segmentation is loaded"):
            bridge.call(name, args)


def test_plain_reset_frames_unchanged_by_shells_on_disk(bridge, seg_case):
    """A reset without segments does not change because the case has shells on disk (with_segments off)."""
    a = _reset(bridge, seg_case, seg=False)
    s = _reset(bridge, seg_case)
    b = _reset(bridge, seg_case, seg=False)
    for v in VIEWS:
        assert marker_flicker_only(v, a.images[v], b.images[v]), (v, diff_pixels(a.images[v], b.images[v])[:10])
    assert a.state == b.state and a.result["node_inventory"] == b.result["node_inventory"]
    assert changed(a.images["3d"], s.images["3d"])


def test_no_leftover_nodes_across_resets(bridge, seg_case):
    inv = [_reset(bridge, seg_case, capture=()).result["node_inventory"] for _ in range(3)]
    assert inv[0] == inv[1] == inv[2]


def test_set_3d_content(bridge, fresh):
    shell = fresh.images["3d"]
    r = bridge.call("set_3d_content", {"mode": "ct_and_shell"}, capture=("3d",))
    assert r.state["segments"]["mode"] == "ct_and_shell" and r.state["render"]["visible"] is True
    assert changed(shell, r.images["3d"])
    r = bridge.call("set_3d_content", {"mode": "ct_only"}, capture=("3d",))
    ct = r.images["3d"]
    assert r.state["render"]["visible"] is True and changed(shell, ct)
    r = bridge.call("set_3d_content", {"mode": "ct_only"}, capture=("3d",))
    assert any("already shows ct_only" in w for w in r.warnings)
    with pytest.raises(BridgeError):
        bridge.call("set_3d_content", {"mode": "solid"})
    with pytest.raises(BridgeError, match="mode must be one of"):
        bridge.call("set_3d_content", {"mode": "ct_in_shell"})


def test_ct_only_matches_a_reset_without_segments(bridge, seg_case):
    plain = _reset(bridge, seg_case, seg=False, capture=("3d",)).images["3d"]
    _reset(bridge, seg_case, capture=())
    ct = bridge.call("set_3d_content", {"mode": "ct_only"}, capture=("3d",)).images["3d"]
    assert marker_flicker_only("3d", plain, ct), compare(plain, ct)


def test_set_segment_visibility(bridge, fresh):
    r = bridge.call("set_segment_visibility", {"hide": ["liver", "spleen", "kidney_left"]}, capture=VIEWS)
    v = r.state["segments"]["visible"]
    assert [n for n in v if v[n]] == ["kidney_right", "aorta", "pancreas"] and r.result["visible"] == ["kidney_right", "aorta", "pancreas"]
    assert changed(fresh.images["3d"], r.images["3d"])
    assert any(changed(fresh.images[s], r.images[s]) for s in SLICES)
    r = bridge.call("set_segment_visibility", {"show": ["all"], "hide": ["spleen"]})
    assert [n for n, on in r.state["segments"]["visible"].items() if not on] == ["spleen"]
    r = bridge.call("set_segment_visibility", {"hide": ["spleen"]})
    assert any("spleen was already hidden" in w for w in r.warnings)


def test_hiding_every_shell_in_shell_only_leaves_an_empty_frame(bridge, fresh):
    r = bridge.call("set_segment_visibility", {"hide": ["all"]}, capture=("3d",))
    rgb = load_rgb(r.images["3d"])
    marker = np.zeros(rgb.shape[:2], bool)
    marker[-120:, -120:] = True
    assert rgb.max(axis=2)[~marker].max() == 0


def test_bad_names_are_refused_and_change_nothing(bridge, fresh):
    before = bridge.call("capture", {}, capture=("3d",))
    _rejected(bridge, "set_segment_visibility", {"show": ["XX"]}, "unknown structure 'XX'", before)
    _rejected(bridge, "set_segment_visibility", {"show": ["kidney_left"], "hide": ["kidney_left"]}, "both shown and hidden", before)
    _rejected(bridge, "set_segment_visibility", {}, "at least one structure", before)
    _rejected(bridge, "set_segment_opacity", {"name": "bladder", "opacity": 0.5}, "unknown structure", before)
    _rejected(bridge, "center_on_segments", {"names": ["kidney_left", "XX"]}, "unknown structure", before)


def test_structure_absent_from_the_case_is_refused(bridge, runtime, seg_case):
    """A case whose segmentation has no pancreas: naming it is an error, and nothing changes."""
    d = Path(runtime["cases_dir"])
    alias = "case_9901"
    made = []
    try:
        (d / f"{alias}.nrrd").symlink_to(f"{seg_case}.nrrd")
        made.append(d / f"{alias}.nrrd")
        for suffix in (".shell.nrrd", ".labels.nrrd"):
            _write_without_label(d / f"{seg_case}{suffix}", d / f"{alias}{suffix}", 6)
            made.append(d / f"{alias}{suffix}")
        man = json.loads((d / f"{seg_case}.shell.json").read_text())
        man["classes"].remove("pancreas")
        for k in ("labels", "voxels", "shell_voxels", "bbox_ras", "registration"):
            man[k].pop("pancreas", None)
        (d / f"{alias}.shell.json").write_text(json.dumps(man))
        made.append(d / f"{alias}.shell.json")
        r = _reset(bridge, alias, capture=("3d",))
        assert list(r.state["segments"]["visible"]) == ["liver", "spleen", "kidney_left", "kidney_right", "aorta"]
        _rejected(bridge, "set_segment_visibility", {"show": ["pancreas"]}, "pancreas is not segmented in this case", r)
        _rejected(bridge, "set_segment_opacity", {"name": "pancreas", "opacity": 0.2}, "pancreas is not segmented", r)
        _rejected(bridge, "center_on_segments", {"names": ["pancreas"]}, "pancreas is not segmented", r)
    finally:
        for p in made:
            p.unlink()


def _write_without_label(src, dst, label):
    """Copy a gzip NRRD label map with one label value set to 0 (header unchanged)."""
    raw = src.read_bytes()
    i = raw.index(b"\n\n") + 2
    header, data = raw[:i], np.frombuffer(gzip.decompress(raw[i:]), dtype=np.uint8).copy()
    data[data == label] = 0
    dst.write_bytes(header + gzip.compress(data.tobytes()))


def test_set_segment_opacity(bridge, fresh):
    r = bridge.call("set_segment_opacity", {"name": "kidney_right", "opacity": 0.3}, capture=("3d",))
    assert r.state["segments"]["opacity"]["kidney_right"] == 0.3 and r.state["segments"]["opacity"]["kidney_left"] == 1.0
    assert changed(fresh.images["3d"], r.images["3d"])
    r = bridge.call("set_segment_opacity", {"name": "all", "opacity": 7})
    assert set(r.state["segments"]["opacity"].values()) == {1.0}
    assert any(w.startswith("opacity=7 clamped to 1") for w in r.warnings)


def test_set_slice_outlines(bridge, fresh):
    r = bridge.call("set_slice_outlines", {"visible": False}, capture=SLICES)
    assert r.state["segments"]["slice_outlines"] is False
    for s in SLICES:
        assert changed(fresh.images[s], r.images[s]), s
    r2 = bridge.call("set_segment_visibility", {"hide": ["kidney_left"]}, capture=SLICES)
    for s in SLICES:
        assert marker_flicker_only(s, r.images[s], r2.images[s]), s
    r3 = bridge.call("set_slice_outlines", {"visible": True}, capture=SLICES)
    assert any(changed(r2.images[s], r3.images[s]) for s in SLICES)


def test_slice_outlines_stay_off_through_opacity(bridge, fresh):
    """set_segment_opacity leaves the slice outlines off (only a reset restores the default)."""
    off = bridge.call("set_slice_outlines", {"visible": False}, capture=SLICES)
    r = bridge.call("set_segment_opacity", {"name": "all", "opacity": 0.5}, capture=SLICES)
    assert r.state["segments"]["slice_outlines"] is False
    for s in SLICES:
        assert marker_flicker_only(s, off.images[s], r.images[s]), s
    assert bridge.call("reset_view", {}).state["segments"]["slice_outlines"] is True


def test_center_on_segments(bridge, fresh, runtime, seg_case, protocol):
    man = json.loads((Path(runtime["cases_dir"]) / f"{seg_case}.shell.json").read_text())
    r = bridge.call("center_on_segments", {"names": ["kidney_left"]}, capture=("3d",))
    box = man["bbox_ras"]["kidney_left"]
    assert r.state["camera"]["focal_point"] == pytest.approx(box["center"], abs=0.01)
    assert r.state["camera"]["zoom"] > 1 and r.result["center_ras"] == pytest.approx(box["center"], abs=0.01)
    assert changed(fresh.images["3d"], r.images["3d"])
    d0 = fresh.state["camera"]
    v0 = np.subtract(d0["position"], d0["focal_point"])
    v1 = np.subtract(r.state["camera"]["position"], r.state["camera"]["focal_point"])
    assert np.allclose(v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1), atol=1e-3)
    r = bridge.call("center_on_segments", {})
    assert r.result["args_after_validation"]["names"] == ["liver", "spleen", "kidney_left", "kidney_right", "aorta", "pancreas"]
    bridge.call("set_segment_visibility", {"hide": ["all"]})
    with pytest.raises(BridgeError, match="no structure is shown"):
        bridge.call("center_on_segments", {})


def test_clip_plane_and_box_cut_the_shells(bridge, fresh):
    r0 = bridge.call("center_on_segments", {}, capture=("3d",))
    r = bridge.call("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 0}, capture=("3d",))
    assert changed(r0.images["3d"], r.images["3d"])
    r = bridge.call("set_clip_box", {"size_mm": [30, 30, 30], "center_ras": r0.result["center_ras"]},
                    capture=("3d",))
    assert changed(r0.images["3d"], r.images["3d"])
    r = bridge.call("clear_clip", {}, capture=("3d",))
    assert marker_flicker_only("3d", r0.images["3d"], r.images["3d"])


def test_inside_segment(bridge, fresh, runtime, seg_case):
    man = json.loads((Path(runtime["cases_dir"]) / f"{seg_case}.shell.json").read_text())
    box = man["bbox_ras"]["kidney_left"]
    hits = []
    for f in (0.5, 0.4, 0.6, 0.3, 0.7):
        p = [box["min"][i] + f * (box["max"][i] - box["min"][i]) for i in range(3)]
        hits.append(bridge.call("jump_to_point", {"ras": p}).result["inside_segment"])
    assert "kidney_left" in hits
    far = bridge.call("jump_to_point", {"ras": [1e4, 1e4, 1e4]}).result
    assert far["inside_segment"] is None
    r = bridge.call("point_at", {"view": "axial", "x": 2, "y": 2}).result
    assert "inside_segment" in r and r["inside_segment"] is None


def test_reset_view_restores_the_shell_defaults(bridge, fresh):
    bridge.call("set_3d_content", {"mode": "ct_and_shell"})
    bridge.call("set_segment_visibility", {"hide": ["kidney_right"]})
    bridge.call("set_segment_opacity", {"name": "kidney_left", "opacity": 0.1})
    bridge.call("set_slice_outlines", {"visible": False})
    bridge.call("set_clip_plane", {"azimuth_deg": 90, "elevation_deg": 0, "offset_mm": 0})
    r = bridge.call("reset_view", {}, capture=VIEWS)
    assert r.state == fresh.state
    for v in VIEWS:
        assert marker_flicker_only(v, fresh.images[v], r.images[v]), v


def test_leak_test_name_never_visible_with_shells(bridge, runtime, seg_case):
    d = Path(runtime["cases_dir"])
    made = []
    try:
        for s in (".nrrd",) + SHELL_SUFFIXES:
            p = d / f"LEAK_TEST{s}"
            if not p.exists():
                p.symlink_to(f"{seg_case}{s}")
                made.append(p)
        a = _reset(bridge, seg_case)
        b = _reset(bridge, "LEAK_TEST")
        for v in VIEWS:
            assert marker_flicker_only(v, a.images[v], b.images[v]), (v, diff_pixels(a.images[v], b.images[v])[:10])
        blob = json.dumps([b.state, {k: x for k, x in b.result.items() if k != "segments_manifest_sha256"},
                           bridge.call("get_case_info").result])
        assert "LEAK" not in blob and "leak" not in blob.lower() and "/" not in json.dumps(b.state)
        assert a.state == b.state
    finally:
        for p in made:
            p.unlink()


def test_slice_options_and_measure_with_segments(bridge, fresh):
    """A with_segments episode has the slice zoom, the slab, the HU and measure (no cardiac planes), and
    reset_view clears them."""
    assert fresh.state["slice_tools"]["zoom"] == 1.0 and fresh.state["slice_tools"]["measure"] is None
    pos = fresh.state["slices"]["axial"]["position_mm"]
    r = bridge.call("set_slice", {"view": "axial", "position_mm": pos, "slab_mm": 10}, capture=("axial",))
    assert r.state["slice_tools"]["slab_mm"]["axial"] == 10.0
    z = bridge.call("zoom", {"factor": 2, "view": "slices"}, capture=SLICES)
    assert z.state["slice_tools"]["zoom"] == 2.0
    p = bridge.call("point_at", {"view": "axial", "x": 256, "y": 256}, capture=SLICES)
    assert isinstance(p.result["hu"], int) and "inside_segment" in p.result
    m = bridge.call("measure", {"view": "axial", "x1": 200, "y1": 256, "x2": 300, "y2": 256}, capture=("axial",))
    assert m.result["distance_mm"] > 0 and m.state["slice_tools"]["measure"]["view"] == "axial"
    with pytest.raises(BridgeError):
        bridge.call("set_slice_plane", {"name": "four_chamber"})
    rv = bridge.call("reset_view", {}, capture=VIEWS)
    t = rv.state["slice_tools"]
    assert t["zoom"] == 1.0 and t["measure"] is None and set(t["slab_mm"].values()) == {0.0}
