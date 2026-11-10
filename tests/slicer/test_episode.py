"""Live episode tests: deterministic reset, name-leak guard, episode read-outs."""

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slicer_mcp.bridge_client import VIEWS, BridgeError
from slicer_mcp.imaging import changed, compare
from slicer_mcp.session import CASE_INFO_RUNNER_ONLY

pytestmark = pytest.mark.slicer

SSIM_MIN = 0.99


def _reset(bridge, case):
    return bridge.call("reset_episode", {"case_id": case}, capture=VIEWS)


def test_ping_reports_fixed_size_and_hashes(bridge, protocol):
    info = bridge.ping()
    n = protocol["observation"]["size_px"]
    assert all(s == [n, n] for s in info["render_sizes"].values()), info["render_sizes"]
    assert re.fullmatch(r"[0-9a-f]{64}", info["presets_sha256"])
    assert isinstance(info["views_adopted_by_main_layout"], list)


def test_slice_views_fill_the_frame(bridge, case_id):
    """Slice nodes must have N x N dimensions: a square field of view that fits the volume."""
    r = _reset(bridge, case_id)
    ci = bridge.call("get_case_info").result
    for v, s in r.state["slices"].items():
        assert s["fov_mm"][0] == s["fov_mm"][1], (v, s)
        lo, hi = ci["slice_position_range_mm"][v]
        assert lo <= s["position_mm"] <= hi


def test_reset_is_deterministic(bridge, case_id, record):
    a = _reset(bridge, case_id)
    b = _reset(bridge, case_id)
    assert a.state == b.state
    assert a.result["node_inventory"] == b.result["node_inventory"], "leftover nodes across resets"
    cmp = {v: compare(a.images[v], b.images[v]) for v in VIEWS}
    record("m2_reset_determinism", {"views": cmp, "reset_latency_ms": [a.latency_ms, b.latency_ms],
                                    "wall_ms": [a.wall_ms, b.wall_ms]})
    for v, c in cmp.items():
        assert c["identical"] or c["ssim"] >= SSIM_MIN, (v, c)


def test_reset_after_changes_restores_state(bridge, case_id):
    """Reset must not trust leftovers: change slices, W/L (and later camera/render/clip), reset, compare with a
    fresh reset (state, frames, node inventory). reset_view must restore the same state without reloading."""
    a = _reset(bridge, case_id)
    for view in ("reset_view", "reset_episode"):
        bridge.call("set_window_preset", {"name": "CT-Angio"})
        bridge.call("point_at", {"view": "axial", "x": 300, "y": 420})
        bridge.call("jump_to_point", {"ras": a.state["camera"]["focal_point"]})
        bridge.call("scroll_slice", {"view": "axial", "delta_slices": 40})
        bridge.call("set_slice", {"view": "coronal", "position_mm": a.state["slices"]["coronal"]["position_mm"] + 30})
        bridge.call("set_window_preset", {"name": "CT-Lung"})
        for tool, args in CHANGES_3D:
            bridge.call(tool, args)
        mid = bridge.call("get_state")
        assert mid.state != a.state
        b = bridge.call(view, {"case_id": case_id} if view == "reset_episode" else {}, capture=VIEWS)
        assert b.state == a.state, view
        if view == "reset_episode":
            assert b.result["node_inventory"] == a.result["node_inventory"]
        for v in VIEWS:
            c = compare(a.images[v], b.images[v])
            assert c["identical"] or c["ssim"] >= SSIM_MIN, (view, v, c)


CHANGES_3D = [
    ("orbit", {"azimuth_deg": 35, "elevation_deg": -20, "roll_deg": 10}),
    ("zoom", {"factor": 1.8}),
    ("pan_to", {"ras": [0.0, -170.0, -400.0]}),
    ("set_render_preset", {"name": "CT-Bone"}),
    ("set_clip_box", {"center_ras": [0.0, -170.0, -400.0], "size_mm": [90, 90, 90]}),
    ("set_clip_plane", {"azimuth_deg": 30, "elevation_deg": 10, "offset_mm": -15}),
]


MARKER_BOX = (447, 460, 500, 490)
FLICKER_MAX_PIXELS = 4
FLICKER_MAX_DIFF = 8


def _rgb(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)


def diff_pixels(png_a, png_b):
    """[(x, y, max channel difference)] of the pixels that differ."""
    d = np.abs(_rgb(png_a) - _rgb(png_b)).max(axis=2)
    ys, xs = np.nonzero(d)
    return [(int(x), int(y), int(d[y, x])) for y, x in zip(ys, xs)]


def marker_flicker_only(view, png_a, png_b):
    """True when the two frames are identical, or differ only by the orientation-marker flicker."""
    d = np.abs(_rgb(png_a) - _rgb(png_b)).max(axis=2)
    ys, xs = np.nonzero(d)
    if len(xs) == 0:
        return True
    x0, y0, x1, y1 = MARKER_BOX
    return (view != "3d" and len(xs) <= FLICKER_MAX_PIXELS and int(d.max()) <= FLICKER_MAX_DIFF
            and bool(((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).all()))


def test_marker_flicker_tolerance_is_narrow():
    """The tolerance accepts the measured flicker and nothing like a rendered label."""
    base = np.zeros((512, 512, 3), dtype=np.uint8)

    def png(mod):
        a = base.copy()
        mod(a)
        b = io.BytesIO()
        Image.fromarray(a).save(b, "PNG")
        return b.getvalue()
    ref = png(lambda a: None)
    assert marker_flicker_only("axial", ref, ref)
    assert marker_flicker_only("axial", ref, png(lambda a: a.__setitem__((slice(471, 473), 465), 2)))
    assert marker_flicker_only("sagittal", ref, png(lambda a: a.__setitem__((477, 477), 2)))
    assert not marker_flicker_only("3d", ref, png(lambda a: a.__setitem__((471, 465), 2)))
    assert marker_flicker_only("sagittal", ref, png(lambda a: a.__setitem__((slice(477, 479), 477), 6)))
    assert not marker_flicker_only("axial", ref, png(lambda a: a.__setitem__((471, 465), 40)))
    assert not marker_flicker_only("axial", ref, png(lambda a: a.__setitem__((slice(467, 470), slice(460, 462)), 2)))
    assert not marker_flicker_only("axial", ref, png(lambda a: a.__setitem__((10, 10), 2)))
    assert not marker_flicker_only("axial", ref, png(lambda a: a.__setitem__((slice(5, 20), slice(5, 120)), 255)))


def test_leak_test_name_never_visible(bridge, alias_cases, record):
    clean, leaky = alias_cases
    a = _reset(bridge, clean)
    b = _reset(bridge, leaky)
    info = bridge.ping()
    cmp = {v: compare(a.images[v], b.images[v]) for v in VIEWS}
    record("m2_leak_test", {"views": cmp, "window_title": info["window_title"],
                            "main_window_title": info["main_window_title"]})
    for v, c in cmp.items():
        assert marker_flicker_only(v, a.images[v], b.images[v]), (v, c, diff_pixels(a.images[v], b.images[v])[:20])
    blob = json.dumps([b.state, b.result, bridge.call("get_case_info").result, info])
    assert "LEAK_TEST" not in blob and "leak" not in blob.lower()
    assert a.state == b.state


def test_state_has_geometry_and_settings_only(bridge, case_id, protocol):
    r = _reset(bridge, case_id)
    assert set(r.state) == {"case_loaded", "camera", "render", "clip", "slices", "focus_ras", "wl", "slice_tools"}
    strings = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            strings.append(x)
    walk(r.state)
    assert set(strings) <= {protocol["render"]["default_preset"], protocol["window_level"]["default_preset"]}, strings


def test_case_info_is_geometry(bridge, case_id, runtime):
    _reset(bridge, case_id)
    ci = bridge.call("get_case_info").result
    head = (Path(runtime["cases_dir"]) / f"{case_id}.nrrd").read_bytes()[:4096].split(b"\n\n")[0].decode()
    sizes = next(line for line in head.splitlines() if line.startswith("sizes:")).split()[1:]
    assert ci["dims"] == [int(s) for s in sizes]
    assert set(ci) - set(CASE_INFO_RUNNER_ONLY) == {"dims", "spacing_mm", "ras_bounds_mm", "center_ras_mm",
                                                    "slice_position_range_mm", "slice_axes"}


def test_reset_view_restores_camera(bridge, case_id):
    r0 = _reset(bridge, case_id)
    r1 = bridge.call("reset_view", capture=("3d",))
    assert r1.state == r0.state
    assert not changed(r0.images["3d"], r1.images["3d"])


def test_capture_is_pure(bridge, case_id):
    r0 = _reset(bridge, case_id)
    r1 = bridge.call("capture", capture=VIEWS)
    assert r1.state == r0.state
    for v in VIEWS:
        assert not changed(r0.images[v], r1.images[v]), v


def test_errors(bridge):
    with pytest.raises(BridgeError) as e:
        bridge.call("reset_episode", {"case_id": "../etc/passwd"})
    assert e.value.type == "invalid_argument"
    with pytest.raises(BridgeError) as e:
        bridge.call("reset_episode", {"case_id": "case_9999"})
    assert e.value.type == "invalid_argument"
    with pytest.raises(BridgeError) as e:
        bridge.call("exec")
    assert e.value.type == "not_found"
