"""Frozen-frame guard (stale-render guard).

When Slicer stops drawing the observation window (window closed, display asleep), every capture returns the last
frame drawn while the state keeps changing, and the model is shown frozen images. The guard remembers each view's
last frame and the state it was captured in. A new frame that is byte-identical to the previous one although the
state that view depends on changed significantly is stale: EnvSession reports it as an environment error and the
runner aborts the episode.

What counts as a significant change for a view (tolerances in runtime.json `stale_guard`, not protocol):
  slice view : its position moved >= `slice_spacings` x the slice spacing, its field of view changed, or the
               window / level changed; the slice tools: its slab changed, or its image centre moved
               >= `slice_spacings` x the slice spacing (a pan of a zoomed view); a non-standard plane (always null
               in this tool set) counts by its geometry (normal, shift along it), not its name
  3d         : the camera position or focal point moved >= `camera_mm`, the view-up turned >= `view_up_deg`,
               the view angle changed, or the render preset / visibility changed (a preset change while the
               volume rendering is hidden, as in a segmentation's shell_only mode, does not count)
The clip is left out on purpose: a plane or box that removes nothing (offset beyond the volume) legitimately
leaves the frame unchanged. A frame that is (almost) one colour (air, empty space; `featureless_fraction` of its
pixels share one value) is exempt, since two such frames are identical whatever the state.

Render counts: the bridge counts the renders VTK finished per view and sends, per captured view,
{"before": count when the request came in, "frame": count right after the capture}. A frame whose count did not go
up was not drawn after the call, whatever changed (shell rebuilds, opacity, seeds and clipping included, and the
episode's first frame): it is stale. This check needs no previous frame and no state; the byte check above covers
bridges that send no counts.
"""

import io
import math

import numpy as np
from PIL import Image

DEFAULTS = {"enabled": True, "camera_mm": 5.0, "view_up_deg": 2.0, "slice_spacings": 1.0,
            "featureless_fraction": 0.99}
SLICES = ("axial", "sagittal", "coronal")


def _dist(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _angle_deg(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 180.0
    c = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


STANDARD_NORMALS = {"axial": [0.0, 0.0, 1.0], "sagittal": [1.0, 0.0, 0.0], "coronal": [0.0, 1.0, 0.0]}


def _standard_plane(view: str, sl: dict) -> dict | None:
    """The plane a standard (no cardiac plane: None in the state) slice view shows, at its slice position."""
    if sl.get("position_mm") is None:
        return None
    n = STANDARD_NORMALS[view]
    return {"normal": n, "origin_ras": [float(sl["position_mm"]) * k for k in n]}


def _plane_moved(a, b, spacing: float, tol: dict) -> bool:
    """A slice plane moved when its normal turned or it shifted along the normal; its name alone does not count
    (set_slice_plane standard on a sagittal already at normal [1, 0, 0] after set_oblique_plane renames it, same image)."""
    if not (a and b and a.get("normal") and b.get("normal") and a.get("origin_ras") and b.get("origin_ras")):
        return a != b
    if _angle_deg(a["normal"], b["normal"]) > 0.1:
        return True
    n = a["normal"]
    nn = math.sqrt(sum(float(x) ** 2 for x in n)) or 1.0
    shift = abs(sum((float(x) - float(y)) * float(k) for x, y, k in zip(a["origin_ras"], b["origin_ras"], n))) / nn
    return shift >= tol["slice_spacings"] * spacing > 0 or (spacing <= 0 and shift > 1e-6)


def changes(view: str, before: dict, after: dict, tol: dict) -> list[str]:
    """The significant changes (names) between two states for one view; [] when nothing that view shows changed."""
    out = []
    if view in SLICES:
        b, a = (before.get("slices") or {}).get(view) or {}, (after.get("slices") or {}).get(view) or {}
        spacing = float(a.get("spacing_mm") or b.get("spacing_mm") or 0.0)
        if (a.get("position_mm") is not None and b.get("position_mm") is not None
                and abs(float(a["position_mm"]) - float(b["position_mm"])) >= tol["slice_spacings"] * spacing > 0):
            out.append("position")
        if _dist(a.get("fov_mm", []), b.get("fov_mm", [])) > 1e-6:
            out.append("fov")
        wb, wa = before.get("wl") or {}, after.get("wl") or {}
        if (wa.get("window"), wa.get("level")) != (wb.get("window"), wb.get("level")):
            out.append("window_level")
        tb, ta = before.get("slice_tools") or {}, after.get("slice_tools") or {}
        if ta or tb:
            pa, pb = (ta.get("planes") or {}).get(view), (tb.get("planes") or {}).get(view)
            if (pa is None) != (pb is None):
                pa, pb = pa or _standard_plane(view, a), pb or _standard_plane(view, b)
            if _plane_moved(pa, pb, spacing, tol):
                out.append("plane")
            cb, ca = (tb.get("centers_ras") or {}).get(view), (ta.get("centers_ras") or {}).get(view)
            if ca and cb and _dist(ca, cb) >= tol["slice_spacings"] * spacing > 0 and "position" not in out:
                out.append("pan")
            if (ta.get("slab_mm") or {}).get(view) != (tb.get("slab_mm") or {}).get(view):
                out.append("slab")
    elif view == "3d":
        b, a = before.get("camera") or {}, after.get("camera") or {}
        for key, name in (("position", "camera_position"), ("focal_point", "focal_point")):
            if a.get(key) and b.get(key) and _dist(a[key], b[key]) >= tol["camera_mm"]:
                out.append(name)
        if a.get("view_up") and b.get("view_up") and _angle_deg(a["view_up"], b["view_up"]) >= tol["view_up_deg"]:
            out.append("view_up")
        if a.get("view_angle_deg") != b.get("view_angle_deg"):
            out.append("view_angle")
        ra, rb = after.get("render") or {}, before.get("render") or {}
        if ra != rb and (ra.get("visible", True) or rb.get("visible", True)):
            out.append("render")
    return out


def featureless(png: bytes, fraction: float) -> bool:
    """True when at least `fraction` of the pixels share one RGB value (air, empty background)."""
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint32)
    packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    _, counts = np.unique(packed, return_counts=True)
    return counts.max() >= fraction * packed.size


class StaleFrameGuard:
    def __init__(self, settings: dict | None = None):
        self.tol = {**DEFAULTS, **(settings or {})}
        self.enabled = bool(self.tol["enabled"])
        self.last = {}

    def check(self, frames: dict, state: dict, render: dict | None = None) -> list[str]:
        """Record the new frames; return a description of each stale one (empty when all are fine). `render` = the
        bridge's per-view render counts, or None."""
        stale = []
        for view, png in frames.items():
            if view not in ("3d", *SLICES):
                continue
            prev = self.last.get(view)
            self.last[view] = (png, state)
            rc = (render or {}).get(view) if self.enabled else None
            if rc is not None:
                try:
                    drawn = int(rc["frame"]) > int(rc["before"])
                except (KeyError, TypeError, ValueError):
                    drawn = False
                if not drawn:
                    stale.append(f"{view} (not rendered after the call: render count {rc})")
                    continue
            if not self.enabled or prev is None or png != prev[0]:
                continue
            ch = changes(view, prev[1], state, self.tol)
            if ch and not featureless(png, self.tol["featureless_fraction"]):
                stale.append(f"{view} ({', '.join(ch)} changed)")
        return stale
