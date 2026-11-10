"""Slice tools in the live bridge: zoom view=slices, set_slice slab_mm, the HU in point_at /
jump_to_point, and measure, on the session's case. Each: the state changed, the frame changed, out-of-range values
clamped with a warning; refusals change nothing. Points come from the case's own geometry, not from its labels."""

import colorsys
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from slicer_mcp.bridge_client import BridgeError
from slicer_mcp.imaging import changed, compare, load_rgb

pytestmark = pytest.mark.slicer
SLICES = ("axial", "sagittal", "coronal")
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
YELLOW = (0.98, 0.84, 0.0)


def _geometry(nrrd):
    """(HU array (k, j, i), IJK->RAS 4x4) of a NRRD, read with SimpleITK (no Slicer)."""
    sitk = pytest.importorskip("SimpleITK")
    img = sitk.ReadImage(str(nrrd))
    m = np.eye(4)
    m[:3, :3] = np.array(img.GetDirection()).reshape(3, 3) @ np.diag(img.GetSpacing())
    m[:3, 3] = img.GetOrigin()
    return sitk.GetArrayFromImage(img), LPS @ m


@pytest.fixture(scope="module")
def case(bridge, runtime, case_id):
    ct, m = _geometry(Path(runtime["cases_dir"]) / f"{case_id}.nrrd")
    r = bridge.call("reset_episode", {"case_id": case_id}, capture=())
    info = bridge.call("get_case_info").result
    c = np.array(info["center_ras_mm"])
    return {"case": case_id, "ct": ct, "m": m, "center": [float(x) for x in c],
            "other": [float(x) for x in c + np.array([15.0, -10.0, 5.0])], "reset_state": r.state}


def _reset(bridge, case):
    return bridge.call("reset_episode", {"case_id": case["case"]}, capture=SLICES)


def _rejected(bridge, name, args, match):
    before = bridge.call("capture", {}, capture=SLICES)
    with pytest.raises(BridgeError) as e:
        bridge.call(name, args)
    assert e.value.type == "invalid_argument" and match in e.value.message, e.value.message
    after = bridge.call("capture", {}, capture=SLICES)
    assert after.state == before.state
    assert not any(changed(before.images[v], after.images[v]) for v in SLICES)


def _offline_hu(case, ras, r=1.0):
    """The mean CT value within r mm of ras, from the NRRD (what the bridge's HU readout must give)."""
    mi = np.linalg.inv(case["m"])
    ct = case["ct"]
    sp = np.linalg.norm(case["m"][:3, :3], axis=0)
    c = np.rint(mi @ np.r_[ras, 1.0])[:3].astype(int)
    h = np.ceil(r / sp).astype(int) + 1
    vals = []
    for i in range(c[0] - h[0], c[0] + h[0] + 1):
        for j in range(c[1] - h[1], c[1] + h[1] + 1):
            for k in range(c[2] - h[2], c[2] + h[2] + 1):
                if np.linalg.norm((case["m"] @ [i, j, k, 1.0])[:3] - ras) <= r:
                    vals.append(ct[k, j, i])
    return float(np.mean(vals))


def _hsv(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert("HSV"), dtype=float) / 255.0


def _yellow_mask(png):
    hsv = _hsv(png)
    h = colorsys.rgb_to_hsv(*YELLOW)[0]
    d = np.abs(hsv[..., 0] - h)
    return (np.minimum(d, 1 - d) < 0.03) & (hsv[..., 1] > 0.4) & (hsv[..., 2] > 0.3)


def test_slice_zoom_about_the_focus(bridge, case):
    r0 = _reset(bridge, case)
    t0 = r0.state["slice_tools"]
    assert (t0["zoom"], t0["planes"], t0["measure"], t0["slab_mm"]) == (1.0, dict.fromkeys(SLICES), None,
                                                                        dict.fromkeys(SLICES, 0.0))
    bridge.call("jump_to_point", {"ras": case["center"]})
    before = bridge.call("capture", {}, capture=("3d", *SLICES))
    r = bridge.call("zoom", {"factor": 2.5, "view": "slices"}, capture=("3d", *SLICES))
    assert r.result["slice_zoom"] == 2.5 and r.state["slice_tools"]["zoom"] == 2.5
    for v in SLICES:
        assert np.allclose(r.state["slices"][v]["fov_mm"], np.array(before.state["slices"][v]["fov_mm"]) / 2.5,
                           rtol=1e-3)
        assert changed(before.images[v], r.images[v])
        assert r.state["slices"][v]["position_mm"] == before.state["slices"][v]["position_mm"]
    assert r.state["camera"] == before.state["camera"] and not changed(before.images["3d"], r.images["3d"])
    for p in (case["center"], case["other"]):
        bridge.call("jump_to_point", {"ras": p})
        f = bridge.call("capture", {}).state["focus_ras"]
        for v in SLICES:
            c = bridge.call("point_at", {"view": v, "x": 255.5, "y": 255.5}).result["ras"]
            assert np.allclose(c, f, atol=0.5), (v, c, f)
            bridge.call("jump_to_point", {"ras": f})
    a0 = bridge.call("capture", {}).state["slice_tools"]["centers_ras"]
    sc = bridge.call("scroll_slice", {"view": "sagittal", "delta_slices": 40}).state["slice_tools"]["centers_ras"]
    assert sc["axial"] == a0["axial"] and sc["coronal"] == a0["coronal"] and sc["sagittal"] != a0["sagittal"]
    r = bridge.call("zoom", {"factor": 4, "view": "slices"})
    assert r.result["slice_zoom"] == 3.0 and any("slice zoom 10 clamped to 3" in w for w in r.warnings)
    r = bridge.call("zoom", {"factor": 0.25, "view": "slices"}, capture=SLICES)
    assert r.result["slice_zoom"] == 1.0 and any("clamped to 1" in w for w in r.warnings)
    for v in SLICES:
        assert np.allclose(r.state["slices"][v]["fov_mm"], before.state["slices"][v]["fov_mm"], rtol=1e-6)
    bridge.call("zoom", {"factor": 2, "view": "slices"})
    r = bridge.call("reset_view", {})
    assert r.state["slice_tools"]["zoom"] == 1.0
    _rejected(bridge, "zoom", {"factor": 2, "view": "axial"}, "view must be one of")


def test_slab_mip(bridge, case):
    _reset(bridge, case)
    pos = case["center"][1]
    thin = bridge.call("set_slice", {"view": "coronal", "position_mm": pos}, capture=("coronal",))
    r = bridge.call("set_slice", {"view": "coronal", "position_mm": pos, "slab_mm": 10}, capture=("coronal",))
    assert r.result["args_after_validation"]["slab_mm"] == 10.0 and r.state["slice_tools"]["slab_mm"]["coronal"] == 10
    assert changed(thin.images["coronal"], r.images["coronal"])
    g0, g10 = load_rgb(thin.images["coronal"]).mean(-1), load_rgb(r.images["coronal"]).mean(-1)
    assert (g10 < g0 - 3).mean() < 0.01 and (g10 > g0 + 3).any()
    r20 = bridge.call("set_slice", {"view": "coronal", "position_mm": pos, "slab_mm": 20}, capture=("coronal",))
    g20 = load_rgb(r20.images["coronal"]).mean(-1)
    assert (g20 < g10 - 3).mean() < 0.01
    s = bridge.call("scroll_slice", {"view": "coronal", "delta_slices": 3})
    assert s.state["slice_tools"]["slab_mm"]["coronal"] == 20.0
    r = bridge.call("set_slice", {"view": "coronal", "position_mm": pos, "slab_mm": 50})
    assert r.result["args_after_validation"]["slab_mm"] == 20.0 and any("slab_mm=50 clamped to 20" in w
                                                                         for w in r.warnings)
    r = bridge.call("set_slice", {"view": "coronal", "position_mm": pos, "slab_mm": 0}, capture=("coronal",))
    assert r.state["slice_tools"]["slab_mm"]["coronal"] == 0.0
    assert compare(thin.images["coronal"], r.images["coronal"])["max_abs_diff"] <= 1


def test_hu_at_the_point(bridge, case):
    _reset(bridge, case)
    for p in (case["center"], case["other"]):
        res = bridge.call("jump_to_point", {"ras": p}).result
        assert abs(res["hu"] - _offline_hu(case, res["args_after_validation"]["ras"])) <= 1, (p, res["hu"])
    p = bridge.call("point_at", {"view": "axial", "x": 256, "y": 256}).result
    assert isinstance(p["hu"], int)


def test_measure(bridge, case):
    _reset(bridge, case)
    bridge.call("jump_to_point", {"ras": case["center"]})
    before = bridge.call("capture", {}, capture=("3d", *SLICES))
    r = bridge.call("measure", {"view": "axial", "x1": 156, "y1": 300, "x2": 356, "y2": 300},
                    capture=("3d", *SLICES))
    fov = before.state["slices"]["axial"]["fov_mm"][0]
    assert abs(r.result["distance_mm"] - 200 * fov / 512) < 0.2
    assert r.state["slice_tools"]["measure"]["view"] == "axial"
    assert changed(before.images["axial"], r.images["axial"])
    band = _yellow_mask(r.images["axial"]) & ~_yellow_mask(before.images["axial"])
    assert 150 <= band.sum() < 4000, band.sum()
    ys, xs = np.nonzero(band)
    assert ys.min() >= 290 and ys.max() <= 310 and xs.min() >= 145 and xs.max() <= 367, (ys.min(), ys.max())
    off = bridge.call("scroll_slice", {"view": "axial", "delta_slices": 10}, capture=("axial",))
    assert not (_yellow_mask(off.images["axial"]) & (np.abs(np.arange(512) - 300) <= 10)[:, None]).sum() > 20
    for v in ("3d", "sagittal", "coronal"):
        assert not changed(before.images[v], r.images[v]), v
    r = bridge.call("measure", {"view": "axial", "x1": -5, "y1": 300, "x2": 600, "y2": 300})
    assert sum("clamped" in w for w in r.warnings) == 2
    r = bridge.call("reset_view", {}, capture=SLICES)
    assert r.state["slice_tools"]["measure"] is None
    bridge.call("jump_to_point", {"ras": case["center"]})
    _rejected(bridge, "measure", {"view": "axial", "x1": 10, "y1": 10, "x2": 10, "y2": 10}, "same pixel")
