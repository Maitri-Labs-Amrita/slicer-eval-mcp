"""Live 2D slice tool tests. Each tool: state changed as expected, frame changed, clamping warns."""

import json

import pytest

from slicer_mcp.bridge_client import VIEWS, BridgeError
from slicer_mcp.imaging import changed

pytestmark = pytest.mark.slicer

SLICES = ("axial", "sagittal", "coronal")


@pytest.fixture()
def fresh(bridge, case_id):
    """A fresh reset with all four frames, plus the case geometry."""
    r = bridge.call("reset_episode", {"case_id": case_id}, capture=VIEWS)
    return r, bridge.call("get_case_info").result


def _clamp_warned(warnings, name):
    return any(w.startswith(f"{name}=") and "clamped" in w for w in warnings)


@pytest.mark.parametrize("view", SLICES)
def test_set_slice(bridge, fresh, view):
    r0, ci = fresh
    lo, hi = ci["slice_position_range_mm"][view]
    target = round(lo + 0.3 * (hi - lo), 2)
    r = bridge.call("set_slice", {"view": view, "position_mm": target}, capture=(view,))
    assert r.state["slices"][view]["position_mm"] == pytest.approx(target, abs=1e-3)
    assert r.result["args_after_validation"] == {"view": view, "position_mm": target}
    assert r.warnings == []
    assert changed(r0.images[view], r.images[view])
    others = [v for v in SLICES if v != view]
    assert all(r.state["slices"][v] == r0.state["slices"][v] for v in others)
    assert r.state["camera"] == r0.state["camera"]

    r = bridge.call("set_slice", {"view": view, "position_mm": hi + 500}, capture=(view,))
    assert r.state["slices"][view]["position_mm"] == pytest.approx(hi, abs=1e-3)
    assert _clamp_warned(r.warnings, "position_mm"), r.warnings


def test_scroll_slice(bridge, fresh):
    r0, ci = fresh
    sp = r0.state["slices"]["axial"]["spacing_mm"]
    p0 = r0.state["slices"]["axial"]["position_mm"]
    r = bridge.call("scroll_slice", {"view": "axial", "delta_slices": 25}, capture=("axial",))
    assert r.state["slices"]["axial"]["position_mm"] == pytest.approx(p0 + 25 * sp, abs=1e-3)
    assert r.result["moved_slices"] == pytest.approx(25)
    assert r.warnings == []
    assert changed(r0.images["axial"], r.images["axial"])
    r = bridge.call("scroll_slice", {"view": "axial", "delta_slices": -25}, capture=("axial",))
    assert r.state["slices"]["axial"]["position_mm"] == pytest.approx(p0, abs=1e-3)
    assert not changed(r0.images["axial"], r.images["axial"])

    r = bridge.call("scroll_slice", {"view": "sagittal", "delta_slices": 100000}, capture=("sagittal",))
    assert r.result["args_after_validation"]["delta_slices"] == 200
    assert _clamp_warned(r.warnings, "delta_slices"), r.warnings
    for _ in range(3):
        r = bridge.call("scroll_slice", {"view": "sagittal", "delta_slices": 200})
    assert r.state["slices"]["sagittal"]["position_mm"] == pytest.approx(ci["slice_position_range_mm"]["sagittal"][1], abs=1e-3)
    assert any("volume edge" in w for w in r.warnings), r.warnings


def test_scroll_slice_rejects_non_integer(bridge, fresh):
    with pytest.raises(BridgeError) as e:
        bridge.call("scroll_slice", {"view": "axial", "delta_slices": 2.5})
    assert e.value.type == "invalid_argument"
    r = bridge.call("scroll_slice", {"view": "axial", "delta_slices": 3.0})
    assert r.result["args_after_validation"]["delta_slices"] == 3


def test_set_window_preset(bridge, fresh, protocol):
    r0, _ = fresh
    presets = protocol["window_level"]["presets"]
    r = bridge.call("set_window_preset", {"name": "CT-Lung"}, capture=SLICES)
    assert r.state["wl"] == {"window": presets["CT-Lung"][0], "level": presets["CT-Lung"][1], "preset": "CT-Lung"}
    for v in SLICES:
        assert changed(r0.images[v], r.images[v]), v
    with pytest.raises(BridgeError) as e:
        bridge.call("set_window_preset", {"name": "CT-Not-A-Preset"})
    assert e.value.type == "invalid_argument"


def test_set_window_level_is_gone(bridge, fresh):
    with pytest.raises(BridgeError) as e:
        bridge.call("set_window_level", {"window": 600, "level": 150})
    assert e.value.type == "not_found"


def test_window_changes_do_not_touch_3d(bridge, fresh):
    r0, _ = fresh
    r = bridge.call("set_window_preset", {"name": "CT-Lung"}, capture=("3d",))
    assert not changed(r0.images["3d"], r.images["3d"])


def test_jump_to_point(bridge, fresh):
    r0, ci = fresh
    b = ci["ras_bounds_mm"]
    p = [b["R"][0] + 0.6 * (b["R"][1] - b["R"][0]),
         b["A"][0] + 0.4 * (b["A"][1] - b["A"][0]),
         b["S"][0] + 0.7 * (b["S"][1] - b["S"][0])]
    p = [round(x, 2) for x in p]
    r = bridge.call("jump_to_point", {"ras": p}, capture=SLICES)
    s = r.state["slices"]
    assert (s["sagittal"]["position_mm"], s["coronal"]["position_mm"], s["axial"]["position_mm"]) == \
        pytest.approx(tuple(p), abs=1e-3)
    assert r.warnings == []
    for v in SLICES:
        assert changed(r0.images[v], r.images[v]), v
    assert r.state["camera"] == r0.state["camera"]

    assert r.state["focus_ras"] == pytest.approx(p, abs=1e-3)

    r = bridge.call("jump_to_point", {"ras": [1e6, p[1], -1e6]})
    s, rng = r.state["slices"], ci["slice_position_range_mm"]
    assert s["sagittal"]["position_mm"] == pytest.approx(rng["sagittal"][1], abs=1e-3)
    assert s["axial"]["position_mm"] == pytest.approx(rng["axial"][0], abs=1e-3)
    assert _clamp_warned(r.warnings, "ras[0]") and _clamp_warned(r.warnings, "ras[2]"), r.warnings
    with pytest.raises(BridgeError):
        bridge.call("jump_to_point", {"ras": [1, 2]})


def test_capture_slices_subset(bridge, fresh):
    r0, _ = fresh
    r = bridge.call("capture", capture=("coronal", "axial"))
    assert set(r.images) == {"coronal", "axial"}
    assert r.state == r0.state


@pytest.mark.parametrize("tool,args", [
    ("set_slice", {"view": "oblique", "position_mm": 0}),
    ("set_slice", {"view": "axial"}),
    ("set_slice", {"view": "axial", "position_mm": True}),
    ("scroll_slice", {"view": "Axial", "delta_slices": 1}),
])
def test_bad_args_rejected(bridge, fresh, tool, args):
    with pytest.raises(BridgeError) as e:
        bridge.call(tool, args)
    assert e.value.type == "invalid_argument"


def test_non_finite_rejected_by_bridge(bridge, fresh):
    """requests refuses NaN, so post a raw body: the bridge's json accepts NaN, argcheck must reject it."""
    env = json.dumps({"request": "nan-test", "token": bridge.token, "launch": bridge.launch_id,
                      "episode": bridge.episode_id})
    r = bridge.session.post(f"{bridge.url}/bridge/point_at",
                            data='{"args": {"view": "axial", "x": NaN, "y": 0}, "capture": [], "slicer_mcp": ' + env + '}',
                            timeout=30)
    d = r.json()
    assert not d["ok"] and d["error"]["type"] == "invalid_argument", d


def test_slice_range_is_voxel_centres_and_edges_are_not_blank(bridge, fresh):
    """The slice range is the voxel-centre range (half a voxel inside the RAS bounds), so a slice at the edge
    still shows data (before: a black frame)."""
    r0, ci = fresh
    for view, axis in (("axial", "S"), ("sagittal", "R"), ("coronal", "A")):
        lo, hi = ci["slice_position_range_mm"][view]
        blo, bhi = ci["ras_bounds_mm"][axis]
        half = r0.state["slices"][view]["spacing_mm"] / 2
        assert lo == pytest.approx(blo + half, abs=0.01) and hi == pytest.approx(bhi - half, abs=0.01), view
    r = bridge.call("set_slice", {"view": "axial", "position_mm": 1e6}, capture=("axial",))
    assert _clamp_warned(r.warnings, "position_mm")
    import io

    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(r.images["axial"])).convert("L"))
    assert (a > 30).mean() > 0.05, "edge slice is (almost) blank"


def test_point_at(bridge, fresh, protocol):
    """State changed (the other two slices move through the pixel), images changed, clamping warns."""
    r0, ci = fresh
    n = protocol["observation"]["size_px"]
    ax0 = r0.state["slices"]["axial"]["position_mm"]
    r = bridge.call("point_at", {"view": "axial", "x": 0.3 * n, "y": 0.4 * n}, capture=SLICES)
    ras = r.result["ras"]
    s = r.state["slices"]
    assert s["axial"]["position_mm"] == pytest.approx(ax0, abs=1e-3)
    assert (s["sagittal"]["position_mm"], s["coronal"]["position_mm"]) == pytest.approx(tuple(ras[:2]), abs=0.01)
    assert r.state["focus_ras"] == pytest.approx(ras, abs=0.01)
    assert not changed(r0.images["axial"], r.images["axial"])
    for v in ("sagittal", "coronal"):
        assert changed(r0.images[v], r.images[v]), v
    assert r.warnings == []
    r = bridge.call("point_at", {"view": "coronal", "x": -10, "y": 5 * n})
    assert _clamp_warned(r.warnings, "x") and _clamp_warned(r.warnings, "y"), r.warnings
    for bad in ({"view": "3d", "x": 1, "y": 1}, {"view": "axial", "x": "1", "y": 1}, {"view": "axial", "x": 1}):
        with pytest.raises(BridgeError) as e:
            bridge.call("point_at", bad)
        assert e.value.type == "invalid_argument"


@pytest.mark.parametrize("view,x,y,expect", [
    ("axial", 0.1, 0.1, (+1, +1, 0)), ("axial", 0.9, 0.9, (-1, -1, 0)),
    ("sagittal", 0.1, 0.1, (0, +1, +1)), ("sagittal", 0.9, 0.9, (0, -1, -1)),
    ("coronal", 0.1, 0.1, (+1, 0, +1)), ("coronal", 0.9, 0.9, (-1, 0, -1)),
])
def test_point_at_orientation(bridge, fresh, protocol, view, x, y, expect):
    r0, ci = fresh
    n = protocol["observation"]["size_px"]
    c = ci["center_ras_mm"]
    centre = bridge.call("point_at", {"view": view, "x": (n - 1) / 2, "y": (n - 1) / 2}).result["ras"]
    assert centre == pytest.approx(c, abs=0.5)
    ras = bridge.call("point_at", {"view": view, "x": x * n, "y": y * n}).result["ras"]
    for i, sign in enumerate(expect):
        if sign:
            assert (ras[i] - c[i]) * sign > 10, (view, i, ras, c)
