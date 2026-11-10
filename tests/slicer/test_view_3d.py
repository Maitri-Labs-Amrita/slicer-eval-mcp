"""Live 3D tests: camera, render preset and clipping tools.
Each tool: state changed as expected, 3D frame changed, out-of-range args clamped with a warning."""

import math

import pytest

from slicer_mcp.bridge_client import BridgeError
from slicer_mcp.imaging import changed, compare

pytestmark = pytest.mark.slicer

TOL = 1e-3


@pytest.fixture()
def fresh(bridge, case_id):
    r = bridge.call("reset_episode", {"case_id": case_id}, capture=("3d",))
    return r, bridge.call("get_case_info").result


def _clamp_warned(warnings, name):
    return any(w.startswith(f"{name}=") and "clamped" in w for w in warnings)


def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _unit(v):
    n = _norm(v)
    return [x / n for x in v]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _from_dir(cam):
    """Unit vector from the focal point to the camera."""
    return _unit(_sub(cam["position"], cam["focal_point"]))


def test_orbit(bridge, fresh):
    r0, _ = fresh
    c0 = r0.state["camera"]
    r = bridge.call("orbit", {"azimuth_deg": 90, "elevation_deg": 0}, capture=("3d",))
    c = r.state["camera"]
    assert _from_dir(c) == pytest.approx([-1, 0, 0], abs=TOL)
    assert c["focal_point"] == c0["focal_point"]
    assert c["distance_mm"] == pytest.approx(c0["distance_mm"], abs=0.01)
    assert changed(r0.images["3d"], r.images["3d"])
    assert r.result["args_after_validation"] == {"azimuth_deg": 90.0, "elevation_deg": 0.0, "roll_deg": 0.0}

    r = bridge.call("orbit", {"azimuth_deg": 500, "elevation_deg": -300, "roll_deg": 1000})
    assert r.result["args_after_validation"] == {"azimuth_deg": 180.0, "elevation_deg": -90.0, "roll_deg": 180.0}
    for n in ("azimuth_deg", "elevation_deg", "roll_deg"):
        assert _clamp_warned(r.warnings, n), r.warnings


@pytest.mark.parametrize("el", [90, -90, 89.9])
def test_orbit_elevation_not_degenerate(bridge, fresh, el):
    r0, _ = fresh
    r = bridge.call("orbit", {"elevation_deg": el}, capture=("3d",))
    c = r.state["camera"]
    assert abs(_dot(_from_dir(c), c["view_up"])) < 1e-3
    assert _norm(c["view_up"]) == pytest.approx(1, abs=TOL)
    assert changed(r0.images["3d"], r.images["3d"])
    if el == 90:
        assert _from_dir(c) == pytest.approx([0, 0, 1], abs=TOL)


def test_orbit_roundtrip(bridge, fresh):
    r0, _ = fresh
    bridge.call("orbit", {"azimuth_deg": 40})
    r = bridge.call("orbit", {"azimuth_deg": -40}, capture=("3d",))
    assert r.state["camera"]["position"] == pytest.approx(r0.state["camera"]["position"], abs=0.01)
    assert compare(r0.images["3d"], r.images["3d"])["ssim"] > 0.99


def test_set_view_direction_all_six(bridge, fresh, protocol):
    r0, _ = fresh
    dirs = protocol["camera"]["view_directions"]
    bridge.call("zoom", {"factor": 1.5})
    moved = bridge.call("pan_to", {"ras": [r0.state["camera"]["focal_point"][0] + 20,
                                           *r0.state["camera"]["focal_point"][1:]]})
    focal, dist = moved.state["camera"]["focal_point"], moved.state["camera"]["distance_mm"]
    prev = bridge.call("capture", capture=("3d",))
    for name in ("posterior", "left", "right", "superior", "inferior", "anterior"):
        r = bridge.call("set_view_direction", {"name": name}, capture=("3d",))
        c = r.state["camera"]
        assert _from_dir(c) == pytest.approx(dirs[name]["from"], abs=TOL), name
        assert c["view_up"] == pytest.approx(dirs[name]["view_up"], abs=TOL), name
        assert c["focal_point"] == pytest.approx(focal, abs=TOL), name
        assert c["distance_mm"] == pytest.approx(dist, abs=0.01), name
        assert c["zoom"] == pytest.approx(1.5)
        assert changed(prev.images["3d"], r.images["3d"]), name
        prev = r
    with pytest.raises(BridgeError) as e:
        bridge.call("set_view_direction", {"name": "LAO30"})
    assert e.value.type == "invalid_argument"


def test_view_direction_is_history_independent(bridge, fresh):
    r0, _ = fresh
    bridge.call("orbit", {"azimuth_deg": 73, "elevation_deg": 21, "roll_deg": 17})
    r = bridge.call("set_view_direction", {"name": "anterior"}, capture=("3d",))
    assert r.state["camera"] == r0.state["camera"]
    c = compare(r0.images["3d"], r.images["3d"])
    assert c["identical"] or c["ssim"] > 0.99, c


def test_zoom(bridge, fresh):
    r0, _ = fresh
    d0 = r0.state["camera"]["distance_mm"]
    r = bridge.call("zoom", {"factor": 2}, capture=("3d",))
    c = r.state["camera"]
    assert c["zoom"] == 2.0 and c["distance_mm"] == pytest.approx(d0 / 2, abs=0.01)
    assert _from_dir(c) == pytest.approx(_from_dir(r0.state["camera"]), abs=TOL)
    assert c["focal_point"] == r0.state["camera"]["focal_point"]
    assert changed(r0.images["3d"], r.images["3d"])

    r = bridge.call("zoom", {"factor": 4}, capture=("3d",))
    assert r.state["camera"]["zoom"] == 4.0
    assert r.state["camera"]["distance_mm"] == pytest.approx(d0 / 4, abs=0.01)
    assert any("cumulative zoom" in w for w in r.warnings), r.warnings

    r = bridge.call("zoom", {"factor": 0.01})
    assert r.result["args_after_validation"]["factor"] == 0.25 and r.state["camera"]["zoom"] == 1.0
    assert _clamp_warned(r.warnings, "factor"), r.warnings
    r = bridge.call("zoom", {"factor": 0.5}, capture=("3d",))
    assert r.state["camera"]["distance_mm"] == pytest.approx(d0 * 2, abs=0.01)
    assert changed(r0.images["3d"], r.images["3d"])
    with pytest.raises(BridgeError):
        bridge.call("zoom", {})


def test_pan_to(bridge, fresh):
    r0, ci = fresh
    c0 = r0.state["camera"]
    target = [c0["focal_point"][0] + 30, c0["focal_point"][1], c0["focal_point"][2] + 40]
    r = bridge.call("pan_to", {"ras": target}, capture=("3d",))
    c = r.state["camera"]
    assert c["focal_point"] == pytest.approx(target, abs=TOL)
    assert _sub(c["position"], c["focal_point"]) == pytest.approx(_sub(c0["position"], c0["focal_point"]), abs=0.01)
    assert c["view_up"] == c0["view_up"] and c["zoom"] == c0["zoom"]
    assert changed(r0.images["3d"], r.images["3d"])

    b = ci["ras_bounds_mm"]
    r = bridge.call("pan_to", {"ras": [1e5, target[1], -1e5]})
    assert r.state["camera"]["focal_point"][0] == pytest.approx(b["R"][1] + 50, abs=TOL)
    assert r.state["camera"]["focal_point"][2] == pytest.approx(b["S"][0] - 50, abs=TOL)
    assert _clamp_warned(r.warnings, "ras[0]") and _clamp_warned(r.warnings, "ras[2]"), r.warnings


def test_set_render_preset_each(bridge, fresh, protocol):
    r0, _ = fresh
    prev = r0
    order = [p for p in protocol["render"]["presets"] if p != protocol["render"]["default_preset"]]
    order.append(protocol["render"]["default_preset"])
    for name in order:
        r = bridge.call("set_render_preset", {"name": name}, capture=("3d",))
        assert r.state["render"] == {"preset": name, "visible": True}
        assert changed(prev.images["3d"], r.images["3d"]), name
        prev = r
    back = compare(r0.images["3d"], prev.images["3d"])
    assert back["identical"] or back["ssim"] > 0.99, back
    with pytest.raises(BridgeError) as e:
        bridge.call("set_render_preset", {"name": "CT-AAA"})
    assert e.value.type == "invalid_argument"


def test_render_preset_does_not_touch_slices(bridge, fresh):
    r0 = bridge.call("capture", capture=("axial",))
    r = bridge.call("set_render_preset", {"name": "CT-Bone"}, capture=("axial",))
    assert not changed(r0.images["axial"], r.images["axial"])


def test_set_clip_box(bridge, fresh):
    r0, ci = fresh
    ctr = ci["center_ras_mm"]
    r = bridge.call("set_clip_box", {"center_ras": ctr, "size_mm": [80, 80, 80]}, capture=("3d",))
    assert r.state["clip"] == {"type": "box", "center_ras": pytest.approx(ctr, abs=TOL), "size_mm": [80.0, 80.0, 80.0]}
    assert changed(r0.images["3d"], r.images["3d"])
    b = ci["ras_bounds_mm"]
    r = bridge.call("set_clip_box", {"center_ras": [1e4, ctr[1], ctr[2]], "size_mm": [1, 80, 1e5]})
    assert r.state["clip"]["center_ras"][0] == pytest.approx(b["R"][1], abs=TOL)
    assert r.state["clip"]["size_mm"][0] == 5.0
    assert r.state["clip"]["size_mm"][2] == pytest.approx(2 * (b["S"][1] - b["S"][0]), abs=TOL)
    for n in ("center_ras[0]", "size_mm[0]", "size_mm[2]"):
        assert _clamp_warned(r.warnings, n), r.warnings


def test_clip_box_keeps_inside_only(bridge, fresh):
    """A small box at the centre leaves much less rendered than the full volume (mean brightness drops)."""
    r0, ci = fresh
    r = bridge.call("set_clip_box", {"center_ras": ci["center_ras_mm"], "size_mm": [40, 40, 40]}, capture=("3d",))
    from slicer_mcp.imaging import load_gray
    assert load_gray(r.images["3d"]).mean() < 0.5 * load_gray(r0.images["3d"]).mean()


@pytest.mark.parametrize("az,cut_side,other_side", [(0, "anterior", "posterior"), (180, "posterior", "anterior")])
def test_clip_plane_removes_the_named_side(bridge, fresh, az, cut_side, other_side):
    """The half-space the (az, el) direction points into is removed: seen from that side the image changes a
    lot (the interior is exposed), seen from the opposite side much less."""
    ssim = {}
    for view in (cut_side, other_side):
        bridge.call("clear_clip")
        a = bridge.call("set_view_direction", {"name": view}, capture=("3d",))
        b = bridge.call("set_clip_plane", {"azimuth_deg": az, "elevation_deg": 0, "offset_mm": 0}, capture=("3d",))
        ssim[view] = compare(a.images["3d"], b.images["3d"])["ssim"]
    assert ssim[cut_side] < ssim[other_side] - 0.05, ssim


def test_set_clip_plane_state_and_clamp(bridge, fresh):
    r0, ci = fresh
    r = bridge.call("set_clip_plane", {"azimuth_deg": 90, "elevation_deg": 0, "offset_mm": 20}, capture=("3d",))
    k = r.state["clip"]
    assert k["type"] == "plane" and k["normal"] == pytest.approx([-1, 0, 0], abs=TOL)
    assert k["origin_ras"] == pytest.approx([ci["center_ras_mm"][0] - 20, *ci["center_ras_mm"][1:]], abs=TOL)
    assert changed(r0.images["3d"], r.images["3d"])
    r = bridge.call("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 90, "offset_mm": 0})
    assert r.state["clip"]["normal"] == pytest.approx([0, 0, 1], abs=TOL)
    r = bridge.call("set_clip_plane", {"azimuth_deg": -400, "elevation_deg": 100, "offset_mm": 999})
    assert r.result["args_after_validation"] == {"azimuth_deg": -180.0, "elevation_deg": 90.0, "offset_mm": 200.0}
    for n in ("azimuth_deg", "elevation_deg", "offset_mm"):
        assert _clamp_warned(r.warnings, n), r.warnings


def test_one_clip_at_a_time(bridge, fresh):
    r0, ci = fresh
    bridge.call("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 0})
    r = bridge.call("set_clip_box", {"center_ras": ci["center_ras_mm"], "size_mm": [100, 100, 100]}, capture=("3d",))
    assert r.state["clip"]["type"] == "box" and any("replaced" in w for w in r.warnings)
    only_box = bridge.call("reset_view")
    only_box = bridge.call("set_clip_box", {"center_ras": ci["center_ras_mm"], "size_mm": [100, 100, 100]}, capture=("3d",))
    assert not changed(only_box.images["3d"], r.images["3d"])


def test_clear_clip(bridge, fresh, case_id):
    r0, ci = fresh
    for tool, args in (("set_clip_box", {"center_ras": ci["center_ras_mm"], "size_mm": [60, 60, 60]}),
                       ("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 0})):
        clipped = bridge.call(tool, args, capture=("3d",))
        r = bridge.call("clear_clip", capture=("3d",))
        assert r.state["clip"] is None and r.state == r0.state
        assert changed(clipped.images["3d"], r.images["3d"])
        c = compare(r0.images["3d"], r.images["3d"])
        assert c["identical"] or c["ssim"] > 0.99, (tool, c)
    inv = bridge.call("reset_episode", {"case_id": case_id}).result["node_inventory"]
    assert "vtkMRMLClipNode" not in inv and "vtkMRMLMarkupsPlaneNode" not in inv


def test_clip_leaves_slices_alone(bridge, fresh):
    r0 = bridge.call("capture", capture=("axial", "coronal"))
    r = bridge.call("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0, "offset_mm": 0},
                    capture=("axial", "coronal"))
    for v in ("axial", "coronal"):
        assert not changed(r0.images[v], r.images[v]), v


@pytest.mark.parametrize("tool,args", [
    ("orbit", {"azimuth_deg": "left"}),
    ("pan_to", {"ras": [0, 0]}),
    ("set_clip_box", {"center_ras": [0, 0, 0]}),
    ("set_clip_plane", {"azimuth_deg": 0, "elevation_deg": 0}),
    ("set_view_direction", {}),
])
def test_bad_args_rejected(bridge, fresh, tool, args):
    with pytest.raises(BridgeError) as e:
        bridge.call(tool, args)
    assert e.value.type == "invalid_argument"


def test_pan_to_and_clip_box_default_to_the_focus(bridge, fresh):
    r0, ci = fresh
    pt = bridge.call("point_at", {"view": "axial", "x": 420, "y": 300})
    focus = pt.state["focus_ras"]
    r = bridge.call("pan_to", {}, capture=("3d",))
    assert r.state["camera"]["focal_point"] == pytest.approx(focus, abs=TOL)
    assert r.result["args_after_validation"]["ras"] == pytest.approx(focus, abs=TOL)
    assert changed(r0.images["3d"], r.images["3d"])
    r = bridge.call("set_clip_box", {"size_mm": [90, 80, 90]}, capture=("3d",))
    assert r.state["clip"]["center_ras"] == pytest.approx(focus, abs=TOL)
    assert r.result["args_after_validation"]["center_ras"] == pytest.approx(focus, abs=TOL)


def test_reset_framing_fits_the_largest_extent(bridge, fresh, protocol):
    import math
    r0, ci = fresh
    b = ci["ras_bounds_mm"]
    e = max(b[k][1] - b[k][0] for k in "RAS")
    half = math.radians(protocol["camera"]["view_angle_deg"] / 2)
    want = (e / 2 / math.tan(half) + e / 2) * protocol["camera"]["reset_fit_margin"]
    assert r0.state["camera"]["distance_mm"] == pytest.approx(want, abs=0.01)
