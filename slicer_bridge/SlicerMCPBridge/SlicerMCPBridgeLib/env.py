"""Environment core: the episode state held by the bridge, deterministic reset, and state readout.

State dicts contain geometry and settings only: no node names, file paths, descriptions or header text.
"""

import math
import os
import re

import slicer
import vtk

from . import limits
from .registry import BridgeArgError, handler

def voxel_sha256(arr) -> str:
    """The sha256 of a voxel array's bytes in (k, j, i) C order and its own dtype (numpy; as SimpleITK's
    GetArrayFromImage of the same NRRD gives it)."""
    import hashlib
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _r(v):
    """Round for state dicts (config observation.state_decimals) to hide float noise."""
    if isinstance(v, (list, tuple)):
        return [_r(x) for x in v]
    return round(float(v), limits.STATE_DECIMALS) + 0.0


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class Env:
    def __init__(self, obs, presets, cases_dir):
        self.obs = obs
        self.presets = presets
        self.cases_dir = cases_dir
        self.volume = None
        self.volume_sha256 = None
        self.identity = None
        self.vr_display = None
        self.render_preset = None
        self.wl = None
        self.zoom = 1.0
        self.reset_distance = None
        self.clip = None
        self.clip_node_ids = []
        self.case_loaded = False
        self.segs = None
        self.slice_zoom = 1.0
        self.slabs = {name: 0.0 for name in limits.SLICE_VIEWS}
        self.measure = None
        self.measure_node_id = None

    def require_case(self):
        if not self.case_loaded or self.volume is None or slicer.mrmlScene.GetNodeByID(self.volume.GetID()) is None:
            raise BridgeArgError("no case loaded: call reset_episode first")

    def ras_bounds(self):
        b = [0.0] * 6
        self.volume.GetRASBounds(b)
        return b

    def center(self):
        b = self.ras_bounds()
        return [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]

    def slice_normal(self, name):
        m = self.obs.slice_nodes[name].GetSliceToRAS()
        return [m.GetElement(0, 2), m.GetElement(1, 2), m.GetElement(2, 2)]

    def voxel_center_corners(self):
        m = vtk.vtkMatrix4x4()
        self.volume.GetIJKToRASMatrix(m)
        dims = self.volume.GetImageData().GetDimensions()
        pts = []
        for i in (0, dims[0] - 1):
            for j in (0, dims[1] - 1):
                for k in (0, dims[2] - 1):
                    pts.append(m.MultiplyPoint((i, j, k, 1))[:3])
        return pts

    def voxel_center_bounds(self):
        """[Rmin, Rmax, Amin, Amax, Smin, Smax] of the voxel CENTRES (GetRASBounds gives the voxel edges, half a
        voxel further out: a slice there lies outside the data and renders black)."""
        pts = self.voxel_center_corners()
        out = []
        for a in range(3):
            out += [min(p[a] for p in pts), max(p[a] for p in pts)]
        return out

    def slice_position_range(self, name):
        """Slice positions (RAS coordinate along the view axis, mm) that lie on the data: voxel centres, first to
        last."""
        ax = limits.SLICE_AXIS[name]
        b = self.voxel_center_bounds()
        return [b[2 * ax], b[2 * ax + 1]]

    def focus(self):
        """The point where the three slice views cross: [R, A, S] = [sagittal, coronal, axial] positions."""
        out = [0.0, 0.0, 0.0]
        for name, ax in limits.SLICE_AXIS.items():
            out[ax] = self.get_slice_position(name)
        return out

    def view_normal(self, name):
        """Unit vector along which the view's position is measured: its axis."""
        n = [0.0, 0.0, 0.0]
        n[limits.SLICE_AXIS[name]] = 1.0
        return n

    def view_coordinate(self, name, ras):
        """The position the view must take to pass through `ras`."""
        n = self.view_normal(name)
        return sum(n[i] * ras[i] for i in range(3))

    def pixel_to_ras(self, name, x, y):
        """RAS (mm) of pixel (x, y) of slice view `name`'s frame (origin top-left, x right, y down). Frames are
        the slice node's N x N pixels with the rows flipped (VTK's origin is bottom-left)."""
        sn = self.obs.slice_nodes[name]
        h = sn.GetDimensions()[1]
        return list(sn.GetXYToRAS().MultiplyPoint((float(x), float(h - 1 - y), 0.0, 1.0))[:3])

    def _normal_sign(self, name):
        n = self.slice_normal(name)[limits.SLICE_AXIS[name]]
        if abs(abs(n) - 1.0) > 1e-6:
            raise RuntimeError(f"{name} slice is not axis-aligned (normal component {n})")
        return 1.0 if n > 0 else -1.0

    def get_slice_position(self, name):
        return self.obs.slice_logic(name).GetSliceOffset() * self._normal_sign(name)

    def set_slice_position(self, name, pos):
        self.obs.slice_logic(name).SetSliceOffset(pos * self._normal_sign(name))

    def slice_spacing(self, name):
        return float(self.obs.slice_logic(name).GetLowestVolumeSliceSpacing()[2])

    def fit_slice(self, name):
        """Fit the view to the volume at its current position, then apply the slice zoom."""
        pos = self.get_slice_position(name)
        self.obs.slice_logic(name).FitSliceToAll()
        self.set_slice_position(name, pos)
        if self.slice_zoom != 1.0:
            sn = self.obs.slice_nodes[name]
            f = sn.GetFieldOfView()
            sn.SetFieldOfView(f[0] / self.slice_zoom, f[1] / self.slice_zoom, f[2])
            self.center_slice_on(name, self.focus())

    def center_slice_on(self, name, ras):
        """Pan the view (its XYZ origin, in slice coordinates) so that `ras` is at the image centre."""
        sn = self.obs.slice_nodes[name]
        m = sn.GetSliceToRAS()
        d = [ras[i] - m.GetElement(i, 3) for i in range(3)]
        x = sum(d[i] * m.GetElement(i, 0) for i in range(3))
        y = sum(d[i] * m.GetElement(i, 1) for i in range(3))
        sn.SetXYZOrigin(x, y, 0.0)

    def sync_slice_views(self):
        """After point_at / jump_to_point (server.RECENTRE_AFTER): while zoomed, the slice views are
        re-centred on the focus."""
        if self.case_loaded and self.slice_zoom != 1.0:
            f = self.focus()
            for name in limits.SLICE_VIEWS:
                self.center_slice_on(name, f)

    def set_slab(self, name, mm):
        """A maximum-intensity slab of `mm` through the view (0 = an ordinary thin slice)."""
        sn = self.obs.slice_nodes[name]
        if mm > 0:
            sn.SetSlabReconstructionType(vtk.VTK_IMAGE_SLAB_MAX)
            sn.SetSlabReconstructionThickness(float(mm))
        sn.SetSlabReconstructionEnabled(bool(mm > 0))
        self.slabs[name] = float(mm)

    def remove_measure(self):
        if self.measure_node_id is not None:
            n = slicer.mrmlScene.GetNodeByID(self.measure_node_id)
            if n is not None:
                slicer.mrmlScene.RemoveNode(n)
        self.measure_node_id = None
        self.measure = None

    def hu_at(self, ras):
        """Mean CT value (HU) of the voxels within hu_ball_radius_mm of `ras` (the nearest voxel when none is that
        close); None outside the volume."""
        import numpy as np
        r = limits.HU_BALL_MM
        m = vtk.vtkMatrix4x4()
        self.volume.GetRASToIJKMatrix(m)
        ijk = np.array(m.MultiplyPoint((*[float(x) for x in ras], 1.0))[:3])
        dims = np.array(self.volume.GetImageData().GetDimensions())
        c = np.rint(ijk).astype(int)
        if (c < 0).any() or (c >= dims).any():
            return None
        sp = np.array(self.volume.GetSpacing())
        half = np.ceil(r / sp).astype(int) + 1
        lo, hi = np.maximum(c - half, 0), np.minimum(c + half + 1, dims)
        ii, jj, kk = np.meshgrid(*[np.arange(lo[a], hi[a]) for a in range(3)], indexing="ij")
        im = vtk.vtkMatrix4x4()
        self.volume.GetIJKToRASMatrix(im)
        M = np.array([[im.GetElement(a, b) for b in range(4)] for a in range(4)])
        pts = np.c_[ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)] @ M.T
        d = np.linalg.norm(pts[:, :3] - np.asarray(ras, float), axis=1)
        arr = slicer.util.arrayFromVolume(self.volume)
        vals = arr[kk.ravel(), jj.ravel(), ii.ravel()]
        sel = d <= r
        v = vals[sel].mean() if sel.any() else vals[np.argmin(d)]
        return int(round(float(v)))

    def set_camera(self, position, focal, view_up):
        cam = self.obs.camera_node()
        cam.SetFocalPoint(*focal)
        cam.SetPosition(*position)
        cam.SetViewUp(*view_up)
        cam.GetCamera().OrthogonalizeViewUp()
        cam.SetViewAngle(limits.VIEW_ANGLE_DEG)
        cam.SetParallelProjection(False)
        cam.ResetClippingRange()
        self.obs.renderer().ResetCameraClippingRange()

    def look_from(self, direction_name, focal=None, distance=None):
        d, up = limits.VIEW_DIRECTIONS[direction_name]
        f = focal if focal is not None else self.center()
        dist = distance if distance is not None else self.reset_distance / self.zoom
        pos = [f[i] + d[i] * dist for i in range(3)]
        self.set_camera(pos, f, up)

    def camera_distance(self):
        cam = self.obs.camera_node()
        p, f = cam.GetPosition(), cam.GetFocalPoint()
        return math.sqrt(sum((p[i] - f[i]) ** 2 for i in range(3)))

    def set_wl(self, window, level, preset=None):
        dn = self.volume.GetDisplayNode()
        dn.SetAutoWindowLevel(0)
        dn.SetWindowLevel(window, level)
        self.wl = {"window": float(window), "level": float(level), "preset": preset}

    def set_render_preset(self, name):
        self.presets.apply(self.vr_display, name)
        self.render_preset = name

    def reset_slices(self):
        self.slice_zoom = 1.0
        for name in limits.SLICE_VIEWS:
            self.set_slab(name, 0.0)
        c = self.center()
        for name in limits.SLICE_VIEWS:
            comp = self.obs.slice_logic(name).GetSliceCompositeNode()
            comp.SetBackgroundVolumeID(self.volume.GetID())
            comp.SetForegroundVolumeID(None)
            comp.SetLabelVolumeID(None)
            comp.SetForegroundOpacity(0.0)
        slicer.app.processEvents()
        for name in limits.SLICE_VIEWS:
            sl = self.obs.slice_logic(name)
            self.obs.slice_nodes[name].SetOrientation(limits.SLICE_VIEWS[name][1])
            sl.FitSliceToAll()
            self.set_slice_position(name, c[limits.SLICE_AXIS[name]])

    def reset_view_state(self):
        """Everything the model can change goes back to the reset values (not the loaded volume)."""
        self.remove_clip_nodes()
        self.remove_measure()
        self.clip = None
        self.zoom = 1.0
        self.set_render_preset(limits.DEFAULT_RENDER_PRESET)
        w, l = limits.WINDOW_PRESETS[limits.DEFAULT_WINDOW_PRESET]
        self.set_wl(w, l, limits.DEFAULT_WINDOW_PRESET)
        self.reset_slices()
        self.look_from(limits.DEFAULT_VIEW_DIRECTION, distance=self.reset_distance)
        if self.segs is not None:
            self.segs.reset()

    def remove_clip_nodes(self):
        if self.vr_display is not None:
            self.vr_display.SetCroppingEnabled(False)
            self.vr_display.SetClipping(False)
            self.vr_display.SetAndObserveClipNodeID(None)
        if self.segs is not None:
            self.segs.clip_with(None)
        for nid in self.clip_node_ids:
            n = slicer.mrmlScene.GetNodeByID(nid)
            if n is not None:
                slicer.mrmlScene.RemoveNode(n)
        self.clip_node_ids = []

    def reset(self, case_id, with_segments=False):
        """with_segments: also load the case's prepared segmentation shells (the profile has a segmentation)."""
        if not isinstance(case_id, str) or not re.fullmatch(limits.CASE_ID_PATTERN, case_id):
            raise BridgeArgError(f"invalid case_id {case_id!r}")
        if not isinstance(with_segments, bool):
            raise BridgeArgError("with_segments must be true or false")
        if with_segments and not limits.SEGMENTATION:
            raise BridgeArgError("with_segments needs a protocol with a segmentation section")
        path = os.path.join(self.cases_dir, f"{case_id}.nrrd")
        if not os.path.isfile(path):
            raise BridgeArgError(f"unknown case_id {case_id!r}")
        if with_segments:
            from .handlers_seg import files_for
            if not all(os.path.isfile(p) for p in files_for(self.cases_dir, case_id).values()):
                raise BridgeArgError(f"case {case_id!r} has no segmentation shells (prepare it with "
                                     f"slicer-mcp case prepare --labels)")
        scene = slicer.mrmlScene
        self.case_loaded = False
        self.clip_node_ids = []
        self.vr_display = None
        self.segs = None
        self.slice_zoom, self.measure, self.measure_node_id = 1.0, None, None
        self.volume_sha256 = None
        scene.Clear(0)
        vol = slicer.util.loadVolume(path, {"name": case_id, "show": False, "singleFile": True})
        vol.SetName(case_id)
        self.volume = vol
        self.volume_sha256 = voxel_sha256(slicer.util.arrayFromVolume(vol))
        dn = vol.GetDisplayNode()
        dn.SetAndObserveColorNodeID(limits.COLORMAP)
        dn.SetInterpolate(1 if limits.INTERPOLATE else 0)

        vr_logic = slicer.modules.volumerendering.logic()
        vrd = vr_logic.CreateVolumeRenderingDisplayNode(limits.VR_METHOD)
        scene.AddNode(vrd)
        vrd.UnRegister(vr_logic)
        vol.AddAndObserveDisplayNodeID(vrd.GetID())
        vr_logic.UpdateDisplayNodeFromVolumeNode(vrd, vol)
        vrd.SetFollowVolumeDisplayNode(False)
        vrd.RemoveAllViewNodeIDs()
        vrd.AddViewNodeID(self.obs.view_node.GetID())
        vrd.SetCroppingEnabled(False)
        vrd.SetVisibility(True)
        self.vr_display = vrd

        self.obs.apply_view_settings()
        b = self.ras_bounds()
        e = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        half = math.radians(limits.VIEW_ANGLE_DEG / 2)
        self.reset_distance = (e / 2 / math.tan(half) + e / 2) * limits.RESET_FIT_MARGIN
        self.case_loaded = True
        if with_segments:
            from .handlers_seg import Segments
            self.segs = Segments(self, case_id)
        self.reset_view_state()
        slicer.app.processEvents()
        self.obs.check_bound()
        out = {"node_inventory": self.node_inventory(), "case_info": self.case_info()}
        if self.segs is not None:
            out["segments_manifest_sha256"] = self.segs.manifest_sha256
        return out

    def node_inventory(self):
        """Class -> count for every non-singleton node (the 'no leftover nodes' check)."""
        inv = {}
        scene = slicer.mrmlScene
        for i in range(scene.GetNumberOfNodes()):
            n = scene.GetNthNode(i)
            if n.GetSingletonTag():
                continue
            inv[n.GetClassName()] = inv.get(n.GetClassName(), 0) + 1
        return dict(sorted(inv.items()))

    def state(self):
        if not self.case_loaded:
            return {"case_loaded": False}
        cam = self.obs.camera_node()
        slices = {}
        for name in limits.SLICE_VIEWS:
            sn = self.obs.slice_nodes[name]
            fov = sn.GetFieldOfView()
            slices[name] = {
                "position_mm": _r(self.get_slice_position(name)),
                "fov_mm": _r(fov[:2]),
                "spacing_mm": _r(self.slice_spacing(name)),
            }
        return {
            "case_loaded": True,
            "camera": {
                "position": _r(cam.GetPosition()),
                "focal_point": _r(cam.GetFocalPoint()),
                "view_up": _r(cam.GetViewUp()),
                "view_angle_deg": _r(cam.GetViewAngle()),
                "distance_mm": _r(self.camera_distance()),
                "zoom": _r(self.zoom),
            },
            "render": {"preset": self.render_preset, "visible": bool(self.vr_display.GetVisibility())},
            "clip": self.clip,
            "slices": slices,
            "focus_ras": _r(self.focus()),
            "wl": {"window": _r(self.wl["window"]), "level": _r(self.wl["level"]), "preset": self.wl["preset"]},
            **self._segments_state(),
            **self._slice_tools_state(),
        }

    def _slice_tools_state(self):
        """The slice tools. `planes` is always null per view: the key is part of the state contract the model
        reads (this tool set has no plane tools)."""
        h = (self.obs.size_px - 1) / 2.0
        centers = {n: _r(self.pixel_to_ras(n, h, h)) for n in limits.SLICE_VIEWS}
        return {"slice_tools": {"zoom": _r(self.slice_zoom), "planes": {n: None for n in limits.SLICE_VIEWS},
                                "slab_mm": {n: _r(v) for n, v in self.slabs.items()}, "centers_ras": centers,
                                "measure": self.measure}}

    def _segments_state(self):
        return {"segments": self.segs.state()} if self.segs is not None else {}

    def case_info(self):
        self.require_case()
        b = self.ras_bounds()
        return {
            "dims": list(self.volume.GetImageData().GetDimensions()),
            "volume_sha256": self.volume_sha256,
            "volume_dtype": str(slicer.util.arrayFromVolume(self.volume).dtype),
            "spacing_mm": _r(self.volume.GetSpacing()),
            "ras_bounds_mm": {"R": _r(b[0:2]), "A": _r(b[2:4]), "S": _r(b[4:6])},
            "center_ras_mm": _r(self.center()),
            "slice_position_range_mm": {n: _r(self.slice_position_range(n)) for n in limits.SLICE_VIEWS},
            "slice_axes": {"axial": "S", "sagittal": "R", "coronal": "A"},
            **({"segments": self.segs.info()} if self.segs is not None else {}),
        }


@handler("reset_episode")
def h_reset_episode(env, args, warnings):
    return env.reset(args.get("case_id"), args.get("with_segments", False))


@handler("get_state")
def h_get_state(env, args, warnings):
    return {}


@handler("get_case_info")
def h_get_case_info(env, args, warnings):
    return env.case_info()


@handler("reset_view")
def h_reset_view(env, args, warnings):
    env.require_case()
    env.reset_view_state()
    return {}


@handler("capture")
def h_capture(env, args, warnings):
    """No state change; the frames are selected by the request's `capture` list."""
    env.require_case()
    return {}
