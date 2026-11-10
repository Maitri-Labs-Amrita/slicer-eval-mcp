"""A deterministic stand-in for BridgeClient (no Slicer), shared by the unit tests."""

import io

import numpy as np
from PIL import Image

from slicer_mcp.bridge_client import BridgeError, BridgeResponse

SIZE = 512
HANDLERS = ["capture", "clear_clip", "jump_to_point", "orbit", "pan_to", "ping", "point_at", "reset_episode",
            "reset_view", "scroll_slice", "set_clip_box", "set_clip_plane", "set_render_preset", "set_slice",
            "set_view_direction", "set_window_preset", "zoom", "set_segment_visibility", "set_segment_opacity",
            "set_3d_content", "set_slice_outlines", "center_on_segments", "measure", "release_episode"]
_png_cache = {}


def png(v, size=SIZE):
    """A solid grey PNG (cached: solid frames compress to a few hundred bytes)."""
    key = (v % 256, size)
    if key not in _png_cache:
        b = io.BytesIO()
        Image.fromarray(np.full((size, size, 3), key[0], dtype=np.uint8)).save(b, format="PNG")
        _png_cache[key] = b.getvalue()
    return _png_cache[key]


class FakeClient:
    """state = the real top-level keys; camera = {"n": number of state-changing calls, "last": last handler},
    slices = {"n"}; frames encode n."""

    url = "fake://bridge"

    def __init__(self, protocol, size=SIZE):
        self.protocol = protocol
        self.size = size
        self.n = 0
        self.last = None
        self.calls = []
        self.launch_id = "fakelaunch"
        self.episode_id = None
        self.seq = 0
        self.case_id = None

    def ping(self):
        return {"config_sha256": self.protocol.sha256, "bridge_version": "fake", "presets_sha256": "p",
                "slicer_version": "5.10.0", "slicer_revision": "34045", "obs_size": self.size,
                "device_pixel_ratio": 2.0, "handlers": list(HANDLERS), "launch": "eval", "launch_id": self.launch_id,
                "vtk": "9.4.20250513", "mesa": "24.2.8",
                "opengl": {"vendor": "Mesa", "renderer": "llvmpipe (LLVM 19.1.1, 256 bits)",
                           "version": "4.5 (Core Profile) Mesa 24.2.8-1ubuntu1~24.04.1"}}

    def identity(self):
        """The bridge's identity block (as SlicerMCPBridgeLib/guard.py)."""
        self.seq += 1
        return {"episode_id": self.episode_id, "case_id": self.case_id,
                "volume_sha256": None if self.case_id is None else "v" * 64, "seq": self.seq}

    def call(self, name, args=None, capture=(), episode_id=None):
        args = args or {}
        self.calls.append((name, args, tuple(capture)))
        if name == "reset_episode":
            self.episode_id, self.case_id = episode_id, args.get("case_id")
        if name not in HANDLERS:
            raise BridgeError(name, {"type": "not_found", "message": f"no bridge handler {name!r}"})
        if name == "set_slice" and args.get("view") == "oblique":
            raise BridgeError(name, {"type": "invalid_argument", "message": "bad view"})
        if name == "orbit" and args.get("azimuth_deg") == 123.0:
            raise BridgeError(name, {"type": "internal", "message": "boom", "traceback": "..."})
        warnings, result = [], {}
        if name == "set_segment_visibility" and "XX" in (args.get("show") or []):
            raise BridgeError(name, {"type": "invalid_argument", "message": "XX is not segmented in this case"})
        if name == "reset_episode":
            self.n = 0
            self.segments = bool(args.get("with_segments"))
            result = {"node_inventory": {"x": 1}}
            if self.segments:
                result["segments_manifest_sha256"] = "s" * 64
        elif name not in ("capture", "release_episode"):
            self.n += 1
            self.last = name
            a = dict(args)
            if name == "zoom" and a.get("factor", 1) > 4:
                a["factor"] = 4.0
                warnings.append("factor clamped")
            result = {"args_after_validation": a}
        state = {"case_loaded": True, "camera": {"n": self.n, "last": self.last}, "render": {}, "clip": None,
                 "slices": {"n": self.n}, "focus_ras": [0.0, 100.0, 0.0], "wl": {}}
        if getattr(self, "segments", False):
            state["segments"] = {"mode": "shell_only"}
        if name == "reset_episode":
            result["case_info"] = {"dims": [10, 10, 10], "spacing_mm": [1.0, 1.0, 1.0], "volume_sha256": "v" * 64}
        if name == "reset_episode":
            result["case_info"].update({"slice_position_range_mm": {"axial": [-100.0, 100.0], "sagittal": [-50.0, 50.0],
                                                                    "coronal": [0.0, 200.0]},
                                        "center_ras_mm": [0.0, 100.0, 0.0]})
        return BridgeResponse(name, result, state, {v: png(self.n * 10, self.size) for v in capture}, warnings,
                              {"handler": 1.0, "capture": 1.0, "total": 2.0}, 3.0, {}, self.identity(),
                              {v: {"before": self.seq, "frame": self.seq + 1} for v in capture})


def answer_report(evidence=(), answer="no finding"):
    """A report for the example plugin (examples/report_plugin.py): a free-text answer and its evidence."""
    return {"answer": answer, "evidence": list(evidence)}
