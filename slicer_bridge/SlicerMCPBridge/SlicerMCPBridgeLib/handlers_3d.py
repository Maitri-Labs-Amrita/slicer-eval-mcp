"""3D camera / rendering / clipping handlers.

Conventions:
- orbit: vtkCamera Azimuth (about view-up) -> Elevation (about the camera's right axis; VTK 9.5 rotates the
  view-up with it, so +-90 is not degenerate) -> Roll (about the view direction), all through the focal point.
  From the anterior view, azimuth +90 moves the camera to the patient's left.
- set_view_direction keeps the current focal point and distance (zoom); only the direction and view-up snap.
- zoom is cumulative relative to the reset distance: distance = reset_distance / zoom.
- Clip direction (az, el): az 0 = anterior (+A), +90 = patient left (-R), -90 = right (+R), +-180 = posterior;
  el +90 = superior (+S). Same sense as orbit.
- set_clip_plane REMOVES the half-space the direction points into, beyond the plane at offset_mm from the
  volume centre along that direction. (Slicer names the KEPT space: ClipNegativeSpace keeps -normal.)
- One clip at a time: a box replaces a plane and vice versa (a warning says so).
- pan_to and set_clip_box default their point to the focus (env.focus(): where the three slice views cross).
"""

import math

import slicer
import vtk

from . import argcheck, limits
from .env import _r
from .registry import handler


def _finish_camera(env):
    cn = env.obs.camera_node()
    cn.GetCamera().OrthogonalizeViewUp()
    cn.ResetClippingRange()
    env.obs.renderer().ResetCameraClippingRange()


def _direction(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return [-math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)]


@handler("orbit")
def h_orbit(env, args, warnings):
    env.require_case()
    az = limits.clamp(argcheck.number(args, "azimuth_deg", 0.0), *limits.ORBIT_AZIMUTH_RANGE, "azimuth_deg", warnings)
    el = limits.clamp(argcheck.number(args, "elevation_deg", 0.0), *limits.ORBIT_ELEVATION_RANGE, "elevation_deg", warnings)
    roll = limits.clamp(argcheck.number(args, "roll_deg", 0.0), *limits.ORBIT_ROLL_RANGE, "roll_deg", warnings)
    cn = env.obs.camera_node()
    cam = cn.GetCamera()
    cam.Azimuth(az)
    cam.Elevation(el)
    cam.Roll(roll)
    _finish_camera(env)
    return {"args_after_validation": {"azimuth_deg": az, "elevation_deg": el, "roll_deg": roll}}


@handler("set_view_direction")
def h_set_view_direction(env, args, warnings):
    env.require_case()
    name = argcheck.choice(args, "name", limits.VIEW_DIRECTIONS)
    cn = env.obs.camera_node()
    env.look_from(name, focal=list(cn.GetFocalPoint()), distance=env.camera_distance())
    return {"args_after_validation": {"name": name}}


@handler("zoom")
def h_zoom(env, args, warnings):
    """factor > 1 zooms in (camera moves closer). The step is clamped to zoom_step_range, and the cumulative
    zoom (relative to the reset view) to zoom_cumulative_range."""
    env.require_case()
    f = limits.clamp(argcheck.number(args, "factor"), *limits.ZOOM_STEP_RANGE, "factor", warnings)
    view = argcheck.choice(args, "view", ("3d", "slices")) if args.get("view") is not None else "3d"
    if view == "slices":
        return _zoom_slices(env, f, warnings)
    lo, hi = limits.ZOOM_CUMULATIVE_RANGE
    target = env.zoom * f
    new = min(max(target, lo), hi)
    if new != target:
        warnings.append(f"cumulative zoom {target:g} clamped to {new:g} (allowed [{lo:g}, {hi:g}])")
    cn = env.obs.camera_node()
    p, fp = cn.GetPosition(), cn.GetFocalPoint()
    d = [p[i] - fp[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    dist = env.reset_distance / new
    cn.SetPosition(*[fp[i] + d[i] / n * dist for i in range(3)])
    env.zoom = new
    _finish_camera(env)
    return {"args_after_validation": {"factor": f}, "zoom": _r(new)}


def _zoom_slices(env, f, warnings):
    """Total slice zoom clamped to slice_tools.zoom_range; the views are refitted and centred on the focus (at 1,
    the plain full view)."""
    lo, hi = limits.SLICE_ZOOM_RANGE
    target = env.slice_zoom * f
    new = min(max(target, lo), hi)
    if new != target:
        warnings.append(f"slice zoom {target:g} clamped to {new:g} (allowed [{lo:g}, {hi:g}])")
    env.slice_zoom = new
    for name in limits.SLICE_VIEWS:
        env.fit_slice(name)
    return {"args_after_validation": {"factor": f, "view": "slices"}, "slice_zoom": _r(new)}


@handler("pan_to")
def h_pan_to(env, args, warnings):
    """Move the focal point to `ras` (default: the focus; clamped to the volume bounds +- pan_margin_mm); the camera translates
    with it, so direction, distance and zoom are unchanged."""
    env.require_case()
    ras = argcheck.vec3(args, "ras") if args.get("ras") is not None else env.focus()
    b, m = env.ras_bounds(), limits.PAN_MARGIN_MM
    ras = [limits.clamp(ras[i], b[2 * i] - m, b[2 * i + 1] + m, f"ras[{i}]", warnings) for i in range(3)]
    cn = env.obs.camera_node()
    p, fp = cn.GetPosition(), cn.GetFocalPoint()
    cn.SetFocalPoint(*ras)
    cn.SetPosition(*[p[i] + ras[i] - fp[i] for i in range(3)])
    _finish_camera(env)
    return {"args_after_validation": {"ras": ras}}


@handler("set_render_preset")
def h_set_render_preset(env, args, warnings):
    env.require_case()
    name = argcheck.choice(args, "name", limits.RENDER_PRESETS)
    env.set_render_preset(name)
    return {"args_after_validation": {"name": name}}


def _replace_clip(env, new_type, warnings):
    if env.clip is not None and env.clip["type"] != new_type:
        warnings.append(f"the active clip {env.clip['type']} was replaced (one clip at a time)")
    env.remove_clip_nodes()
    env.clip = None


def seg_box_clip(env):
    """Cut the shells with the active clip box: a clip node on the VR crop ROI (ClipNegativeSpace keeps the
    inside). Reuses the box's clip node when there is one."""
    roi = env.vr_display.GetMarkupsROINode()
    for nid in env.clip_node_ids:
        n = slicer.mrmlScene.GetNodeByID(nid)
        if n is not None and n.IsA("vtkMRMLClipNode"):
            env.segs.clip_with(nid)
            return
    clip = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLClipNode", "SlicerMCPSegBoxClip")
    env.clip_node_ids.append(clip.GetID())
    clip.AddAndObserveClippingNodeID(roi.GetID())
    clip.SetClippingNodeState(roi.GetID(), clip.ClipNegativeSpace)
    env.segs.clip_with(clip.GetID())


@handler("set_clip_box")
def h_set_clip_box(env, args, warnings):
    """Keep only the axis-aligned box (RAS) centred at center_ras (default: the focus) with edge lengths size_mm. The centre is
    clamped to the volume bounds, each size to [box_min_size_mm, 2 x volume extent] on that axis."""
    env.require_case()
    c = argcheck.vec3(args, "center_ras") if args.get("center_ras") is not None else env.focus()
    s = argcheck.vec3(args, "size_mm")
    b = env.ras_bounds()
    c = [limits.clamp(c[i], b[2 * i], b[2 * i + 1], f"center_ras[{i}]", warnings) for i in range(3)]
    s = [limits.clamp(s[i], limits.CLIP_BOX_MIN_SIZE_MM, 2 * (b[2 * i + 1] - b[2 * i]), f"size_mm[{i}]", warnings)
         for i in range(3)]
    _replace_clip(env, "box", warnings)
    roi = env.vr_display.GetMarkupsROINode()
    roi.SetAndObserveObjectToNodeMatrix(vtk.vtkMatrix4x4())
    roi.SetCenter(c)
    roi.SetSize(s)
    roi.GetDisplayNode().SetVisibility(False)
    env.vr_display.SetCroppingEnabled(True)
    if env.segs is not None:
        seg_box_clip(env)
    env.clip = {"type": "box", "center_ras": _r(c), "size_mm": _r(s)}
    return {"args_after_validation": {"center_ras": c, "size_mm": s}}


@handler("set_clip_plane")
def h_set_clip_plane(env, args, warnings):
    """Remove everything on the (az, el) side of a plane at offset_mm from the volume centre along (az, el).
    offset 0 cuts through the centre; a positive offset moves the cut outward (removes less)."""
    env.require_case()
    az = limits.clamp(argcheck.number(args, "azimuth_deg"), *limits.CLIP_PLANE_AZIMUTH_RANGE, "azimuth_deg", warnings)
    el = limits.clamp(argcheck.number(args, "elevation_deg"), *limits.CLIP_PLANE_ELEVATION_RANGE, "elevation_deg", warnings)
    off = limits.clamp(argcheck.number(args, "offset_mm"), *limits.CLIP_PLANE_OFFSET_RANGE, "offset_mm", warnings)
    d = _direction(az, el)
    ctr = env.center()
    origin = [ctr[i] + off * d[i] for i in range(3)]
    _replace_clip(env, "plane", warnings)
    scene = slicer.mrmlScene
    plane = scene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", "SlicerMCPClipPlane")
    env.clip_node_ids.append(plane.GetID())
    plane.CreateDefaultDisplayNodes()
    plane.GetDisplayNode().SetVisibility(False)
    plane.SetOrigin(origin)
    plane.SetNormal(d)
    clip = scene.AddNewNodeByClass("vtkMRMLClipNode", "SlicerMCPClip")
    env.clip_node_ids.append(clip.GetID())
    clip.AddAndObserveClippingNodeID(plane.GetID())
    clip.SetClippingNodeState(plane.GetID(), clip.ClipNegativeSpace)
    env.vr_display.SetAndObserveClipNodeID(clip.GetID())
    env.vr_display.SetClipping(True)
    if env.segs is not None:
        env.segs.clip_with(clip.GetID())
    env.clip = {"type": "plane", "azimuth_deg": _r(az), "elevation_deg": _r(el), "offset_mm": _r(off),
                "normal": _r(d), "origin_ras": _r(origin)}
    return {"args_after_validation": {"azimuth_deg": az, "elevation_deg": el, "offset_mm": off}}


@handler("clear_clip")
def h_clear_clip(env, args, warnings):
    env.require_case()
    env.remove_clip_nodes()
    env.clip = None
    return {"args_after_validation": {}}
