"""Segmentation shells: the five segment tools.

A case prepared with labels (`slicer-mcp case prepare --labels`, tools/prepare_shell.py) has <case>.shell.nrrd
(hollow shells, one label per class), <case>.labels.nrrd (the filled label map) and <case>.shell.json (manifest)
next to its NRRD. reset_episode loads them only with with_segments=true (sent by the server when the profile has a
segmentation), so a profile without one has the plain scene.

- The shell label map becomes one vtkMRMLSegmentationNode (one segment per class, id and name = the class name,
  the config colour), shown as a closed surface in the observation 3D view only. The imported
  label-map node is removed again.
- The slice outlines come from a second segmentation node built from the *filled* labels (`slice_outline`, binary
  label maps only, the three observation slice views only): one line at the edge of each structure's blood, where
  the hollow shell drew two (its inner and outer surface, 1 mm apart). Where two labels touch, both
  outlines run along the contact (the annotator's boundary). Clips do not cut it.
- The filled label map is also kept in memory (numpy + its RAS->IJK matrix) for inside_segment.
- 3D content mode: ct_only (volume rendering only), shell_only (shells only), ct_and_shell (both). The 3D display
  goes through hooks (_after_import, apply_visibility, apply_opacity, set_mode, clip_with, remove) that a subclass
  may override.
- Clipping (handlers_3d) cuts the shells with the same clip as the volume rendering: the plane's clip node, or a
  clip node on the crop ROI (ClipNegativeSpace keeps the inside of the ROI, the sense of VR cropping).
- A class name that is not one of the protocol's classes, or not segmented in this case, is an error and nothing
  changes.
"""

import hashlib
import json
import math
import os

import numpy as np
import slicer
import vtk

from . import argcheck, limits
from .segcheck import ManifestMismatch, check_manifest, file_sha256
from .env import _r
from .registry import BridgeArgError, handler

ALL = "all"


def files_for(cases_dir, case_id):
    return {k: os.path.join(cases_dir, f"{case_id}{s}")
            for k, s in (("shell", ".shell.nrrd"), ("labels", ".labels.nrrd"), ("manifest", ".shell.json"))}


def outline_segmentation(env, labels, ras_to_ijk, names, colors, node_name):
    """A segmentation node drawn only as outlines on the three observation slice views: labels (k, j, i) holds
    1..n for the n `names` in order (0 = none), ras_to_ijk its grid. Returns (node, display node)."""
    scene = slicer.mrmlScene
    lab = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "slicer_mcp_outline_import")
    slicer.util.updateVolumeFromArray(lab, np.ascontiguousarray(labels.astype(np.uint8)))
    m = vtk.vtkMatrix4x4()
    inv = np.linalg.inv(np.asarray(ras_to_ijk, dtype=float))
    for r in range(4):
        for c in range(4):
            m.SetElement(r, c, float(inv[r][c]))
    lab.SetIJKToRASMatrix(m)
    seg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", node_name)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(env.volume)
    s = seg.GetSegmentation()
    ids = vtk.vtkStringArray()
    for name in names:
        s.AddEmptySegment(name, name, colors[name])
        ids.InsertNextValue(name)
    if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg, ids):
        scene.RemoveNode(lab)
        raise RuntimeError("slice outline import failed")
    scene.RemoveNode(lab)
    seg.CreateDefaultDisplayNodes()
    dn = seg.GetDisplayNode()
    dn.RemoveAllViewNodeIDs()
    for name in limits.SLICE_VIEWS:
        dn.AddViewNodeID(env.obs.slice_nodes[name].GetID())
    dn.SetVisibility3D(False)
    dn.SetVisibility2DFill(False)
    dn.SetVisibility2DOutline(True)
    dn.SetSliceIntersectionThickness(limits.SEG_OUTLINE_THICKNESS_PX)
    for name in names:
        dn.SetSegmentOpacity2DOutline(name, 1.0)
    return seg, dn


class ShellSet:
    """The displayed shells of one episode: a segmentation node (segment id = name), its display, the filled masks
    for inside(), and the boxes for center_on_segments. Segments builds it from the prepared files."""

    modes = None

    def __init__(self, env, names, colors, known, lab, filled, filled_ras_to_ijk, value_to_name, bbox_ras):
        """lab: a label-map node holding the shells as 1..n in the order of `names` (removed after the import);
        filled: (k, j, i) array of filled labels sampled by inside(); bbox_ras: {name: {min, max, center}}."""
        self.env = env
        self.names = list(names)
        self.known = list(known)
        scene = slicer.mrmlScene
        seg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", "shell")
        seg.SetReferenceImageGeometryParameterFromVolumeNode(env.volume)
        s = seg.GetSegmentation()
        for name in self.names:
            s.AddEmptySegment(name, name, colors[name])
        ids = vtk.vtkStringArray()
        for name in self.names:
            ids.InsertNextValue(name)
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg, ids):
            raise RuntimeError("shell import failed")
        scene.RemoveNode(lab)
        s.SetConversionParameter("Smoothing factor", f"{limits.SEG_SURFACE_SMOOTHING:g}")
        seg.CreateClosedSurfaceRepresentation()
        seg.CreateDefaultDisplayNodes()
        dn = seg.GetDisplayNode()
        dn.RemoveAllViewNodeIDs()
        dn.AddViewNodeID(env.obs.view_node.GetID())
        dn.SetVisibility2D(False)
        self.node, self.display = seg, dn
        self.labels = filled
        self.ras_to_ijk = np.asarray(filled_ras_to_ijk, dtype=float)
        self.value_to_name = dict(value_to_name)
        self.outline_node, self.outline_display = self._slice_outline(colors)
        self.bbox_ras = dict(bbox_ras)
        self.mode = self.visible = self.opacity = self.slice_outlines = None
        self.modes = list(self.modes or limits.SEG_MODES)
        self._after_import()
        self.reset()

    def _slice_outline(self, colors):
        """The filled labels as a segmentation shown only as outlines on the three observation slice views:
        one line at each structure's blood edge. Binary label maps only (no closed surface)."""
        remap = np.zeros(int(self.labels.max()) + 1, dtype=np.uint8)
        for i, name in enumerate(self.names, start=1):
            for v, n in self.value_to_name.items():
                if n == name and v < len(remap):
                    remap[v] = i
        return outline_segmentation(self.env, remap[self.labels], self.ras_to_ijk, self.names, colors,
                                    "slice_outline")

    def _after_import(self):
        """Hook: extra display nodes, built before the first reset (the segmentation node exists)."""

    def remove(self):
        """Remove the segmentation nodes."""
        for node in (self.node, self.outline_node):
            if slicer.mrmlScene.GetNodeByID(node.GetID()) is not None:
                slicer.mrmlScene.RemoveNode(node)

    def reset(self):
        self.visible = {n: True for n in self.names}
        self.opacity = {n: limits.SEG_DEFAULT_OPACITY for n in self.names}
        self.apply_visibility()
        for n in self.names:
            self.apply_opacity(n)
        self.set_slice_outlines(limits.SEG_DEFAULT_SLICE_OUTLINES)
        self.set_mode(limits.SEG_DEFAULT_MODE)

    def apply_visibility(self):
        for n in self.names:
            self.display.SetSegmentVisibility(n, self.visible[n])
            self.outline_display.SetSegmentVisibility(n, self.visible[n])

    def apply_opacity(self, name):
        self.display.SetSegmentOpacity3D(name, self.opacity[name])

    def set_mode(self, mode):
        self.mode = mode
        self.env.vr_display.SetVisibility(mode != "shell_only")
        self.display.SetVisibility3D(mode != "ct_only")
        self.display.SetVisibility(True)
        slicer.app.processEvents()
        self.env.obs.camera_node().ResetClippingRange()
        self.env.obs.renderer().ResetCameraClippingRange()

    def set_slice_outlines(self, on):
        self.slice_outlines = bool(on)
        self.outline_display.SetVisibility2D(self.slice_outlines)

    def state(self):
        return {"mode": self.mode, "slice_outlines": self.slice_outlines,
                "visible": dict(self.visible), "opacity": {n: _r(v) for n, v in self.opacity.items()}}

    def clip_with(self, clip_node_id):
        self.display.SetAndObserveClipNodeID(clip_node_id)
        self.display.SetClipping(clip_node_id is not None)

    def names_arg(self, value, key, allow_all=True):
        """A list of names from an argument (a name or a list of names; 'all' expands). Unknown or absent names
        are errors."""
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise BridgeArgError(f"{key} must be a list of structure names")
        out = []
        for v in value:
            if v == ALL and allow_all:
                out += [n for n in self.names if n not in out]
            elif v not in self.known:
                raise BridgeArgError(f"unknown structure {v!r} in {key}; the structures are "
                                     f"{', '.join(self.known)}" + (" or 'all'" if allow_all else ""))
            elif v not in self.names:
                raise BridgeArgError(f"{v} is not segmented in this case; available: {', '.join(self.names)}")
            elif v not in out:
                out.append(v)
        return out

    def inside(self, ras):
        """The structure whose filled label contains the RAS point, or None."""
        ijk = self.ras_to_ijk @ np.array([ras[0], ras[1], ras[2], 1.0])
        i, j, k = (int(round(x)) for x in ijk[:3])
        kk, jj, ii = self.labels.shape
        if not (0 <= i < ii and 0 <= j < jj and 0 <= k < kk):
            return None
        return self.value_to_name.get(int(self.labels[k, j, i]))

    def bbox(self, names):
        boxes = [self.bbox_ras[n] for n in names if self.bbox_ras.get(n)]
        lo = [min(b["min"][a] for b in boxes) for a in range(3)]
        hi = [max(b["max"][a] for b in boxes) for a in range(3)]
        return lo, hi


class Segments(ShellSet):
    """The protocol's classes present in the prepared files."""

    def __init__(self, env, case_id):
        f = files_for(env.cases_dir, case_id)
        missing = [k for k, p in f.items() if not os.path.isfile(p)]
        if missing:
            raise BridgeArgError(f"case {case_id!r} has no segmentation shells (prepare it with "
                                 f"slicer-mcp case prepare --labels)")
        with open(f["manifest"], "rb") as fh:
            raw = fh.read()
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest.get("shell_thickness_mm") != limits.SEG_SHELL_THICKNESS_MM:
            raise BridgeArgError("the case's shells were built with another shell_thickness_mm: rebuild them")
        hashes = {k: file_sha256(f[k]) for k in ("shell", "labels")}
        try:
            check_manifest(self.manifest, hashes, limits.SEG_LABELS)
        except ManifestMismatch as e:
            raise BridgeArgError(f"case {case_id!r}: {e}") from None
        names = [c for c in limits.SEG_CLASS_NAMES if c in self.manifest["classes"]]
        scene = slicer.mrmlScene

        lab = slicer.util.loadLabelVolume(f["shell"], {"show": False, "name": "slicer_mcp_shell_import"})
        arr = slicer.util.arrayFromVolume(lab)
        lv = slicer.util.loadLabelVolume(f["labels"], {"show": False, "name": "slicer_mcp_labels_import"})
        labels = slicer.util.arrayFromVolume(lv).copy()
        try:
            check_manifest(self.manifest, hashes, limits.SEG_LABELS, shell_values=np.unique(arr), label_values=np.unique(labels))
        except ManifestMismatch as e:
            scene.RemoveNode(lab)
            scene.RemoveNode(lv)
            raise BridgeArgError(f"case {case_id!r}: {e}") from None
        remap = np.zeros(256, dtype=np.uint8)
        for i, name in enumerate(names, start=1):
            remap[limits.SEG_LABELS[name]] = i
        arr = remap[arr]
        if sorted(int(v) for v in np.unique(arr) if v) != list(range(1, len(names) + 1)):
            scene.RemoveNode(lab)
            scene.RemoveNode(lv)
            raise RuntimeError("the shell label map does not hold every class of its manifest: rebuild it")
        slicer.util.updateVolumeFromArray(lab, arr)

        m = vtk.vtkMatrix4x4()
        lv.GetRASToIJKMatrix(m)
        ras_to_ijk = [[m.GetElement(r, c) for c in range(4)] for r in range(4)]
        scene.RemoveNode(lv)
        super().__init__(env, names, limits.SEG_COLORS, limits.SEG_CLASS_NAMES, lab, labels, ras_to_ijk,
                         {limits.SEG_LABELS[n]: n for n in names}, self.manifest["bbox_ras"])

    def info(self):
        return {"classes": list(self.names),
                "colors": {n: limits.SEG_COLOR_NAMES[n] for n in self.names},
                "shell_thickness_mm": limits.SEG_SHELL_THICKNESS_MM}


def _segs(env):
    env.require_case()
    if env.segs is None:
        raise BridgeArgError("no segmentation is loaded in this episode")
    return env.segs


@handler("set_segment_visibility")
def h_set_segment_visibility(env, args, warnings):
    """show is applied first, then hide (show 'all' + hide ['RA'] = everything but RA). A name in both is an
    error. Showing a shown structure or hiding a hidden one is a warning."""
    sg = _segs(env)
    show = sg.names_arg(args.get("show") or [], "show")
    hide = sg.names_arg(args.get("hide") or [], "hide")
    if not show and not hide:
        raise BridgeArgError("give at least one structure in show or hide")
    explicit_show = [v for v in (args.get("show") or []) if v != ALL] if isinstance(args.get("show"), list) else []
    both = [n for n in hide if n in explicit_show]
    if both:
        raise BridgeArgError(f"{', '.join(both)} cannot be both shown and hidden")
    for n in show:
        if sg.visible[n] and n not in hide:
            warnings.append(f"{n} was already shown")
        sg.visible[n] = True
    for n in hide:
        if not sg.visible[n] and n not in show:
            warnings.append(f"{n} was already hidden")
        sg.visible[n] = False
    sg.apply_visibility()
    return {"args_after_validation": {"show": show, "hide": hide},
            "visible": [n for n in sg.names if sg.visible[n]]}


@handler("set_segment_opacity")
def h_set_segment_opacity(env, args, warnings):
    sg = _segs(env)
    v = args.get("name")
    if not isinstance(v, str):
        raise BridgeArgError("name must be a structure name or 'all'")
    names = sg.names_arg([v], "name")
    o = limits.clamp(argcheck.number(args, "opacity"), *limits.SEG_OPACITY_RANGE, "opacity", warnings)
    for n in names:
        sg.opacity[n] = o
        sg.apply_opacity(n)
    hidden = [n for n in names if not sg.visible[n]]
    if hidden:
        warnings.append(f"{', '.join(hidden)} hidden: the opacity applies when shown")
    return {"args_after_validation": {"name": v, "opacity": o}}


@handler("set_3d_content")
def h_set_3d_content(env, args, warnings):
    sg = _segs(env)
    mode = argcheck.choice(args, "mode", sg.modes)
    if mode == sg.mode:
        warnings.append(f"the 3D view already shows {mode}")
    sg.set_mode(mode)
    return {"args_after_validation": {"mode": mode}}


@handler("set_slice_outlines")
def h_set_slice_outlines(env, args, warnings):
    sg = _segs(env)
    v = args.get("visible")
    if not isinstance(v, bool):
        raise BridgeArgError("visible must be true or false")
    if v == sg.slice_outlines:
        warnings.append(f"the outlines were already {'on' if v else 'off'}")
    sg.set_slice_outlines(v)
    return {"args_after_validation": {"visible": v}}


def fit_camera(env, lo, hi, warnings):
    """Focal point -> the box's centre; distance fits its largest extent E the way reset fits the volume
    ((E/2)/tan(view_angle/2) + E/2) x center_fit_margin; the viewing direction is kept. The zoom is the resulting
    reset_distance / distance, clamped to zoom_cumulative_range (with a warning). Returns (centre, zoom)."""
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]
    e = max(hi[i] - lo[i] for i in range(3))
    half = math.radians(limits.VIEW_ANGLE_DEG / 2)
    dist = (e / 2 / math.tan(half) + e / 2) * limits.SEG_CENTER_FIT_MARGIN
    zlo, zhi = limits.ZOOM_CUMULATIVE_RANGE
    zoom = env.reset_distance / dist
    if not zlo <= zoom <= zhi:
        z = min(max(zoom, zlo), zhi)
        warnings.append(f"zoom {zoom:g} needed to fit clamped to {z:g} (allowed [{zlo:g}, {zhi:g}])")
        zoom = z
    cn = env.obs.camera_node()
    p, fp = cn.GetPosition(), cn.GetFocalPoint()
    d = [p[i] - fp[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    cn.SetFocalPoint(*c)
    cn.SetPosition(*[c[i] + d[i] / n * env.reset_distance / zoom for i in range(3)])
    env.zoom = zoom
    cam = cn.GetCamera()
    cam.OrthogonalizeViewUp()
    cn.ResetClippingRange()
    env.obs.renderer().ResetCameraClippingRange()
    return c, zoom


@handler("center_on_segments")
def h_center_on_segments(env, args, warnings):
    """The structures' bounding box fills the view (fit_camera)."""
    sg = _segs(env)
    if args.get("names") is None:
        names = [n for n in sg.names if sg.visible[n]]
        if not names:
            raise BridgeArgError("no structure is shown: give names")
    else:
        names = sg.names_arg(args["names"], "names")
        if not names:
            raise BridgeArgError("names must not be empty")
    lo, hi = sg.bbox(names)
    c, zoom = fit_camera(env, lo, hi, warnings)
    return {"args_after_validation": {"names": names}, "center_ras": _r(c), "zoom": _r(zoom)}
