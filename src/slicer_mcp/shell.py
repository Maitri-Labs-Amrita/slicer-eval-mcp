"""Hollow shells of a segmentation.

create_shell: the union of all labels is dilated by shell_thickness_mm (SimpleITK BinaryDilate, ball,
radius max(1, round(t / spacing)) voxels per axis) and the union subtracted, which leaves a wall of that
thickness around the OUTSIDE of all labelled structures together; each class is dilated and subtracted the same
way, and the wall keeps the class whose own wall covers it (Maximum: the higher label wins). Boundaries between
touching classes lie inside the union, so the shell has no internal walls.

Runs in the server's venv (unit tests) and inside Slicer's Python (tools/prepare_shell.py, which uses Slicer's
bundled SimpleITK): stdlib, numpy and SimpleITK only.
"""

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class ShellError(RuntimeError):
    pass


def classes_from(protocol) -> list[dict]:
    """The segmentation classes of a protocol config (or its data dict), in label order."""
    seg = protocol["segmentation"]
    return sorted(seg["classes"], key=lambda c: c["label"])


def radius_voxels(spacing, shell_thickness_mm):
    """The dilation radius in voxels: max(1, int(round(t / s))) per axis (spacing in SimpleITK x, y, z order)."""
    return [max(1, int(round(shell_thickness_mm / s))) for s in spacing]


def _require_sitk():
    if sitk is None:
        raise ShellError("SimpleITK is required for shell creation")


def _hollow(inner_seg, shell_thickness_mm):
    array = sitk.GetArrayFromImage(inner_seg)
    inner_bin = sitk.GetImageFromArray((array > 0).astype(np.uint8))
    inner_bin.CopyInformation(inner_seg)
    r = radius_voxels(inner_seg.GetSpacing(), shell_thickness_mm)
    dilated = sitk.BinaryDilate(inner_bin, r, sitk.sitkBall, True)
    hollow_mask = sitk.Subtract(dilated, inner_bin)
    return sitk.BinaryThreshold(hollow_mask, 1, 255, 1, 0)


def _classwise_shell(inner_seg, shell_thickness_mm):
    r = radius_voxels(inner_seg.GetSpacing(), shell_thickness_mm)
    combined = sitk.Image(inner_seg.GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(inner_seg)
    array = sitk.GetArrayFromImage(inner_seg)
    for label in [v for v in np.unique(array) if v != 0]:
        bin_mask = sitk.GetImageFromArray((array == label).astype(np.uint8))
        bin_mask.CopyInformation(inner_seg)
        dilated = sitk.BinaryDilate(bin_mask, r, sitk.sitkBall, True)
        shell = sitk.Subtract(dilated, bin_mask)
        shell = sitk.BinaryThreshold(shell, 1, 255, int(label), 0)
        combined = sitk.Maximum(combined, shell)
    return combined


def create_shell(inner_seg, shell_thickness_mm=1.0):
    """Label image -> shell label image (same geometry, uint8), or None when there is no non-zero label."""
    _require_sitk()
    if not np.any(sitk.GetArrayViewFromImage(inner_seg)):
        return None
    bloodpool_shell = _hollow(inner_seg, shell_thickness_mm)
    return sitk.Mask(_classwise_shell(inner_seg, shell_thickness_mm), bloodpool_shell)


def shell_array(labels, spacing_xyz, shell_thickness_mm=1.0):
    """numpy label array in SimpleITK order (k, j, i) with spacing (x, y, z) -> shell label array (uint8)."""
    _require_sitk()
    img = sitk.GetImageFromArray(np.ascontiguousarray(labels, dtype=np.uint8))
    img.SetSpacing([float(s) for s in spacing_xyz])
    out = create_shell(img, shell_thickness_mm)
    if out is None:
        raise ShellError("no non-zero label: nothing to make a shell of")
    return sitk.GetArrayFromImage(out)


def combine_masks(masks, shape) -> np.ndarray:
    """[(label, bool mask)] -> one uint8 label array; masks are painted in ascending label order, so where two
    classes overlap the higher label wins (the same rule as the shell's Maximum)."""
    out = np.zeros(shape, dtype=np.uint8)
    for label, m in sorted(masks, key=lambda x: x[0]):
        out[np.asarray(m, dtype=bool)] = label
    return out


def registration_report(ct, labels, classes, check, outside=None) -> dict:
    """Is the segmentation on this CT? Per class: voxel count and mean CT value inside the filled label (reported
    only: an unenhanced scan's blood is ~0-50 HU and is still a valid case); with `outside` ({name: fraction}
    of a mesh's points outside the CT), failures lists every class with too much of its mesh outside the CT, and prep
    refuses the case when it is not empty. A label map has no mesh: no failures."""
    per, failures = {}, []
    for c in classes:
        m = labels == c["label"]
        n = int(m.sum())
        if n == 0:
            continue
        mean = float(ct[m].mean())
        frac = None if outside is None else outside.get(c["name"])
        per[c["name"]] = {"voxels": n, "mean_hu": round(mean, 1),
                          **({} if frac is None else {"outside_fraction": round(frac, 4)})}
        if frac is not None and frac > check["max_outside_fraction"]:
            failures.append(f"{c['name']}: {frac:.1%} of the mesh lies outside the CT")
    return {"classes": per, "failures": failures}


def bbox_ras(mask, ijk_to_ras) -> dict | None:
    """Bounding box of a (k, j, i) mask's voxel centres in RAS mm: {min, max, center}, or None if empty."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    m = np.asarray(ijk_to_ras, dtype=float).reshape(4, 4)
    corners = []
    for k in (idx[:, 0].min(), idx[:, 0].max()):
        for j in (idx[:, 1].min(), idx[:, 1].max()):
            for i in (idx[:, 2].min(), idx[:, 2].max()):
                corners.append((m @ np.array([i, j, k, 1.0]))[:3])
    c = np.array(corners)
    lo, hi = c.min(axis=0), c.max(axis=0)
    return {"min": [round(float(x), 2) for x in lo], "max": [round(float(x), 2) for x in hi],
            "center": [round(float(x), 2) for x in (lo + hi) / 2]}
