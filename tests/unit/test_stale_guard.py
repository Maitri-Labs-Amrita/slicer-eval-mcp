"""Frozen-frame guard: the pure checks, then EnvSession / the MCP server over a fake bridge that can freeze."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from slicer_mcp import server as server_mod
from slicer_mcp.bridge_client import BridgeError
from slicer_mcp.session import EnvSession
from slicer_mcp.stale import DEFAULTS, StaleFrameGuard, changes, featureless
from tests.unit.fakes import FakeClient, png


def textured(seed, size=64):
    """A non-featureless frame: random 16 x 16 blocks (compresses well at 512 x 512)."""
    tile = np.random.default_rng(seed).integers(0, 255, (size // 16, size // 16, 3), dtype=np.uint8)
    b = io.BytesIO()
    Image.fromarray(tile.repeat(16, 0).repeat(16, 1)).save(b, "PNG")
    return b.getvalue()


def state(pos=0.0, spacing=0.4, wl=(350.0, 40.0), cam=(0.0, -500.0, 0.0), focal=(0.0, 0.0, 0.0), up=(0, 0, 1),
          preset="CT-Coronary-Arteries-2", clip=None):
    sl = {"position_mm": pos, "fov_mm": [200.0, 200.0], "spacing_mm": spacing}
    return {"case_loaded": True, "slices": {"axial": dict(sl), "sagittal": dict(sl), "coronal": dict(sl)},
            "wl": {"window": wl[0], "level": wl[1], "preset": None},
            "camera": {"position": list(cam), "focal_point": list(focal), "view_up": list(up), "view_angle_deg": 30.0},
            "render": {"preset": preset, "visible": True}, "clip": clip}


@pytest.mark.parametrize("view,after,expected", [
    ("axial", state(pos=0.4), ["position"]),
    ("axial", state(pos=0.1), []),
    ("axial", state(wl=(600.0, 170.0)), ["window_level"]),
    ("sagittal", state(cam=(100.0, -500.0, 0.0)), []),
    ("3d", state(cam=(0.0, -506.0, 0.0)), ["camera_position"]),
    ("3d", state(cam=(0.0, -502.0, 0.0)), []),
    ("3d", state(up=(0, 0.1, 1)), ["view_up"]),
    ("3d", state(preset="CT-Bone"), ["render"]),
    ("3d", state(clip={"type": "plane", "offset_mm": 200.0}), []),
    ("3d", state(pos=10.0), []),
])
def test_changes_per_view(view, after, expected):
    assert changes(view, state(), after, DEFAULTS) == expected


def test_preset_change_behind_a_hidden_rendering_is_not_significant():
    hidden = state()
    hidden["render"]["visible"] = False
    after = state(preset="CT-Bone")
    after["render"]["visible"] = False
    assert changes("3d", hidden, after, DEFAULTS) == []
    assert changes("3d", hidden, state(), DEFAULTS) == ["render"]


def test_featureless():
    b = io.BytesIO()
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    a[:5, :5] = 255
    Image.fromarray(a).save(b, "PNG")
    assert featureless(b.getvalue(), 0.99)
    assert not featureless(textured(1), 0.99)


def test_guard_flags_only_identical_frames_whose_state_changed():
    g = StaleFrameGuard()
    f1, f2 = textured(1), textured(2)
    assert g.check({"3d": f1}, state()) == []
    assert g.check({"3d": f1}, state()) == []
    assert g.check({"3d": f2}, state(cam=(0.0, -520.0, 0.0))) == []
    stale = g.check({"3d": f2}, state(cam=(0.0, -560.0, 0.0)))
    assert len(stale) == 1 and stale[0].startswith("3d (camera_position")


def test_guard_exempts_featureless_frames_and_can_be_disabled():
    blank = png(0, 64)
    g = StaleFrameGuard()
    g.check({"coronal": blank}, state())
    assert g.check({"coronal": blank}, state(pos=40.0)) == []
    off = StaleFrameGuard({"enabled": False})
    off.check({"3d": textured(1)}, state())
    assert off.check({"3d": textured(1)}, state(cam=(0.0, -600.0, 0.0))) == []


class FreezableClient(FakeClient):
    """FakeClient with textured frames that depend on the state, and a `frozen` switch that keeps returning the
    last frames (what Slicer does when the observation window is closed or the display sleeps)."""

    def __init__(self, protocol, size=512):
        super().__init__(protocol, size)
        self.frozen = False
        self.last_frames = {}

    def call(self, name, args=None, capture=(), episode_id=None):
        r = super().call(name, args, capture, episode_id)
        r.state["camera"].update(position=[0.0, -500.0 - 10 * self.n, 0.0], focal_point=[0.0, 0.0, 0.0],
                                 view_up=[0, 0, 1], view_angle_deg=30.0)
        r.state["slices"] = {v: {"position_mm": 0.4 * self.n, "fov_mm": [200.0, 200.0], "spacing_mm": 0.4}
                             for v in ("axial", "sagittal", "coronal")}
        for v in capture:
            if not (self.frozen and v in self.last_frames):
                self.last_frames[v] = textured(self.n * 10 + len(v), self.size)
            r.images[v] = self.last_frames[v]
        return r


def test_env_session_raises_stale_render_and_logs_it(protocol, tmp_path, seg_profile):
    fake = FreezableClient(protocol)
    sess = EnvSession(fake, seg_profile, tmp_path)
    sess.reset("case_test")
    assert sess.call("orbit", {"azimuth_deg": 10.0}).ok
    fake.frozen = True
    with pytest.raises(BridgeError) as ei:
        sess.call("orbit", {"azimuth_deg": 10.0})
    assert ei.value.type == "stale_render" and "3d (camera_position" in ei.value.message
    steps = [json.loads(x) for x in (sess.trace_dir / "steps.jsonl").read_text().splitlines()]
    assert steps[-1]["tool"] == "orbit" and steps[-1]["error"]["type"] == "stale_render"
    assert not steps[-1]["frames"]


def test_bookmark_of_an_unchanged_view_is_not_stale(protocol, tmp_path, seg_profile):
    fake = FreezableClient(protocol)
    sess = EnvSession(fake, seg_profile, tmp_path)
    sess.reset("case_test")
    assert sess.call("bookmark_view", {"label": "a"}).ok
    assert sess.call("bookmark_view", {"label": "b"}).ok


def test_frozen_reset_is_refused_before_a_trace_is_written(protocol, tmp_path, seg_profile):
    fake = FreezableClient(protocol)
    sess = EnvSession(fake, seg_profile, tmp_path)
    sess.reset("case_test")
    sess.call("scroll_slice", {"view": "axial", "delta_slices": 3})
    fake.frozen = True
    n_before = len(list(tmp_path.iterdir()))
    with pytest.raises(BridgeError) as ei:
        sess.reset("case_test")
    assert ei.value.type == "stale_render"
    assert len(list(tmp_path.iterdir())) == n_before and sess.logger is None


def test_server_reports_stale_render_as_environment(protocol, monkeypatch, runtime_file):
    fake = FreezableClient(protocol)
    monkeypatch.setattr(server_mod, "BridgeClient", lambda *a, **k: fake)
    srv = server_mod.build("ct_segments", runtime=str(runtime_file))
    assert not srv.handle("reset_episode", {"case_id": "case_test"}).is_error
    fake.frozen = True
    res = srv.handle("scroll_slice", {"view": "coronal", "delta_slices": 5})
    body = json.loads(res.content[0].text)
    assert res.is_error and body["error"]["type"] == "environment"
    assert body["error"]["message"].startswith("stale_render: frozen frame: coronal (position changed)")


def test_slice_tool_changes_count_for_the_slice_views():
    """A slab, a cardiac plane or a pan of a zoomed view changes what a slice view shows, though its position,
    field of view and window are unchanged."""
    tol = dict(DEFAULTS)

    def st(slab=0.0, plane=None, center=(0.0, 0.0, 0.0)):
        s = state()
        s["slice_tools"] = {"zoom": 2.0, "planes": {"axial": plane, "sagittal": None, "coronal": None},
                            "slab_mm": {"axial": slab, "sagittal": 0.0, "coronal": 0.0},
                            "centers_ras": {"axial": list(center), "sagittal": [0, 0, 0], "coronal": [0, 0, 0]},
                            "measure": None}
        return s
    assert changes("axial", st(), st(), tol) == []
    assert changes("axial", st(), st(slab=10.0), tol) == ["slab"]
    assert changes("axial", st(), st(plane={"name": "lvot", "normal": [0, 0.5, 0.866], "origin_ras": [0, 0, 0]}),
                   tol) == ["plane"]
    assert changes("axial", st(plane={"name": "cross_section", "normal": [0, 0, 1], "origin_ras": [0, 0, 0]}),
                   st(), tol) == []
    assert changes("axial", st(), st(center=(5.0, 0.0, 0.0)), tol) == ["pan"]
    assert changes("axial", st(), st(center=(0.1, 0.0, 0.0)), tol) == []
    assert changes("sagittal", st(), st(slab=10.0), tol) == []
    assert changes("axial", state(), state(), tol) == []
