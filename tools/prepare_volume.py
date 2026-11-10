"""A volume file (NIfTI, NRRD, MHA, ...) -> anonymised, geometry-only NRRD. Runs INSIDE a separate headless Slicer
(Slicer's bundled SimpleITK; nothing is installed):

    /Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --no-main-window \
        --disable-settings --ignore-slicerrc --python-script tools/prepare_volume.py <job.json>

job.json (written by slicer_mcp.cases): {"volume", "case_id", "out_dir", "spacing_mm" (null or [x, y, z]),
"lps_identity" (bool), "intensity_offset" (number)}.

- Reads the volume with SimpleITK. The voxels keep the file's array order (k, j, i); intensity_offset is added and
  the values are rounded to int16 (refused when they do not fit).
- Geometry: the header's (SimpleITK spacing, direction and origin, LPS -> RAS), unless overridden: spacing_mm
  replaces the spacing, and lps_identity replaces the direction and origin with an LPS identity at 0 (i -> patient
  left, j -> posterior, k -> superior), i.e. IJK -> RAS = diag(-sx, -sy, sz). For a dataset whose headers are wrong
  (e.g. 1 mm spacing in every header, the true spacing kept elsewhere) the caller passes both.
- A fresh scalar volume node named <case_id>; no names, no attributes. NRRD compression follows config/runtime.json
  case_prep.nrrd_compression; the header is re-read and the file removed if any key outside the geometry whitelist
  (the one of prepare_case.py) is present. The NRRD is read back and must hold exactly these voxels and geometry.
- If SLICER_MCP_PREP_RESULT is set, writes the case info as JSON to that path.
- Called only by slicer_mcp.cases (temporary cases); it keeps no registry of cases.

The source file is only read.
"""

import datetime
import hashlib
import json
import os
import sys

import numpy as np
import slicer
import vtk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_JSON = os.environ.get("SLICER_MCP_RUNTIME") or os.path.join(REPO, "config", "runtime.json")
NRRD_ALLOWED_KEYS = {
    "type", "dimension", "space", "sizes", "space directions", "kinds", "endian", "encoding",
    "space origin", "space units", "measurement frame",
}
NRRD_ALLOWED_COMMENTS = {
    "# Complete NRRD file format specification at:",
    "# http://teem.sourceforge.net/nrrd/format.html",
}
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def nrrd_compression(path):
    """runtime case_prep.nrrd_compression. ValueError naming the file when the key is missing."""
    with open(path) as f:
        rt = json.load(f)
    cp = rt.get("case_prep") if isinstance(rt, dict) else None
    if not isinstance(cp, dict) or "nrrd_compression" not in cp:
        raise ValueError(f"runtime config {path} lacks required keys: case_prep.nrrd_compression")
    return bool(cp["nrrd_compression"])


def read_nrrd_header(path):
    header = {}
    with open(path, "rb") as f:
        if not f.readline().decode("ascii").startswith("NRRD"):
            raise RuntimeError(f"not a NRRD file: {path}")
        for raw in f:
            line = raw.decode("ascii", errors="replace").rstrip("\n")
            if line == "":
                break
            if line.startswith("#"):
                header.setdefault("#comments", []).append(line)
                continue
            k, v = line.split(":=", 1) if ":=" in line else line.split(":", 1)
            header[k.strip()] = v.strip()
    return header


def check_header(path):
    h = read_nrrd_header(path)
    extra = set(k for k in h if k != "#comments") - NRRD_ALLOWED_KEYS
    bad = set(h.get("#comments", [])) - NRRD_ALLOWED_COMMENTS
    if extra or bad:
        os.remove(path)
        raise RuntimeError(f"NRRD header has non-geometry content, removed {path}: keys={extra} comments={bad}")
    return h


def ijk_to_ras(img, spacing_mm=None, lps_identity=False):
    """The 4 x 4 IJK -> RAS matrix of the volume: the header's geometry, or the overrides."""
    sp = np.array(spacing_mm if spacing_mm is not None else img.GetSpacing(), dtype=float)
    if lps_identity:
        direction, origin = np.eye(3), np.zeros(3)
    else:
        direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
        origin = np.array(img.GetOrigin(), dtype=float)
    m = np.eye(4)
    m[:3, :3] = LPS_TO_RAS @ direction @ np.diag(sp)
    m[:3, 3] = LPS_TO_RAS @ origin
    return (m + 0.0).tolist()


def load(job):
    """(voxels (k, j, i) int16, IJK -> RAS 4x4) of the job's volume."""
    import SimpleITK as sitk
    img = sitk.ReadImage(job["volume"])
    if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError("the volume must be a 3D scalar image")
    raw = sitk.GetArrayFromImage(img)
    off = job.get("intensity_offset") or 0
    vals = raw.astype(np.int32) + int(off) if np.issubdtype(raw.dtype, np.integer) and float(off).is_integer() \
        else np.rint(raw.astype(np.float64) + float(off))
    if vals.min() < -32768 or vals.max() > 32767:
        raise RuntimeError(f"values {vals.min()}..{vals.max()} (after the offset {off}) do not fit int16")
    return vals.astype(np.int16), ijk_to_ras(img, job.get("spacing_mm"), bool(job.get("lps_identity")))


def main(job_path):
    compress = nrrd_compression(RUNTIME_JSON)
    with open(job_path) as f:
        job = json.load(f)
    out_dir = os.path.abspath(job["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    case_id = job["case_id"]
    out_path = os.path.join(out_dir, f"{case_id}.nrrd")
    vox, m_ras = load(job)
    print(f"[prepare_volume] shape (k, j, i) {vox.shape}")

    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", case_id)
    slicer.util.updateVolumeFromArray(node, np.ascontiguousarray(vox, dtype=np.int16))
    m = vtk.vtkMatrix4x4()
    for r, row in enumerate(m_ras):
        for c, v in enumerate(row):
            m.SetElement(r, c, v)
    node.SetIJKToRASMatrix(m)
    if not slicer.util.saveNode(node, out_path, {"useCompression": 1 if compress else 0}):
        raise RuntimeError("saveNode failed")
    header = check_header(out_path)

    back = slicer.util.loadVolume(out_path, {"show": False, "singleFile": True})
    arr = slicer.util.arrayFromVolume(back)
    if arr.shape != vox.shape or not np.array_equal(arr, vox):
        os.remove(out_path)
        raise RuntimeError("read-back of the NRRD differs from the source voxels")
    got = vtk.vtkMatrix4x4()
    back.GetIJKToRASMatrix(got)
    if not np.allclose([[got.GetElement(r, c) for c in range(4)] for r in range(4)], m_ras, atol=1e-4):
        os.remove(out_path)
        raise RuntimeError("read-back of the NRRD has another geometry than the source")
    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    info = {
        "case_id": case_id,
        "sha256": sha,
        "dims": [int(x) for x in vox.shape[::-1]],
        "spacing_mm": [round(float(s), 6) for s in back.GetSpacing()],
        "scalar_range": [float(arr.min()), float(arr.max())],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if os.environ.get("SLICER_MCP_PREP_RESULT"):
        with open(os.environ["SLICER_MCP_PREP_RESULT"], "w") as f:
            json.dump(info, f, indent=2)
    print("[prepare_volume] header:", json.dumps(header, indent=1))
    print("[prepare_volume] ok:", json.dumps(info))


try:
    main(sys.argv[1])
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    slicer.util.exit(1)
else:
    slicer.util.exit(0)
