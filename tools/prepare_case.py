"""DICOM folder -> anonymised, geometry-only NRRD. Runs INSIDE a separate headless Slicer.

    /Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --no-main-window \
        --disable-settings --ignore-slicerrc \
        --python-script tools/prepare_case.py <dicom_dir> <case_id> <out_dir> [<series_uid>]

- Imports the folder into a temporary DICOM database (the user's database and Slicer.ini are not touched).
- Picks the main CT series: the explicit <series_uid> if given, otherwise the CT series with the most files.
- Casts voxels to int16 when the value range fits (lossless).
- Copies ONLY voxels + IJK->RAS geometry into a fresh scalar volume node named <case_id> (no attributes,
  no description, no DICOM UIDs), then writes <out_dir>/<case_id>.nrrd.
- NRRD compression follows config/runtime.json case_prep.nrrd_compression (raw loads ~6x faster).
- Re-reads the NRRD header and fails if any key outside a geometry whitelist is present.
- If SLICER_MCP_PREP_RESULT is set, writes the case info (including source_dir and series_uid) as JSON to that path.
- Called only by slicer_mcp.cases (temporary cases; there are no stored cases). It keeps no
  registry of cases.

The source DICOM folder is only read.
"""

import datetime
import hashlib
import json
import os
import tempfile
import sys

import slicer
import vtk
from DICOMLib import DICOMUtils

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


def nrrd_compression(path):
    """runtime case_prep.nrrd_compression. ValueError naming the file when the key is missing."""
    with open(path) as f:
        rt = json.load(f)
    cp = rt.get("case_prep") if isinstance(rt, dict) else None
    if not isinstance(cp, dict) or "nrrd_compression" not in cp:
        raise ValueError(f"runtime config {path} lacks required keys: case_prep.nrrd_compression")
    return bool(cp["nrrd_compression"])


def pick_series(db, series_uid=None):
    candidates = []
    for patient in db.patients():
        for study in db.studiesForPatient(patient):
            for series in db.seriesForStudy(study):
                files = db.filesForSeries(series)
                if not files:
                    continue
                modality = db.fileValue(files[0], "0008,0060")
                candidates.append((series, modality, len(files)))
    if series_uid:
        match = [c for c in candidates if c[0] == series_uid]
        if not match:
            raise RuntimeError(f"series {series_uid} not found")
        return match[0]
    ct = [c for c in candidates if c[1] == "CT"]
    if not ct:
        raise RuntimeError("no CT series found")
    return max(ct, key=lambda c: c[2])


def read_nrrd_header(path):
    header = {}
    with open(path, "rb") as f:
        magic = f.readline().decode("ascii").strip()
        if not magic.startswith("NRRD"):
            raise RuntimeError(f"not a NRRD file: {magic}")
        for raw in f:
            line = raw.decode("ascii", errors="replace").rstrip("\n")
            if line == "":
                break
            if line.startswith("#"):
                header.setdefault("#comments", []).append(line)
                continue
            if ":=" in line:
                k, v = line.split(":=", 1)
                header[k.strip()] = v
            else:
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip()
    return header


def main(dicom_dir, case_id, out_dir, series_uid=None):
    compress = nrrd_compression(RUNTIME_JSON)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{case_id}.nrrd")
    with DICOMUtils.TemporaryDICOMDatabase(tempfile.mkdtemp(prefix="slicer_mcp_dicomdb_")) as db:
        DICOMUtils.importDicom(dicom_dir, db)
        uid, modality, n_files = pick_series(db, series_uid)
        print(f"[prepare_case] series {uid} modality={modality} files={n_files}")
        loaded_ids = DICOMUtils.loadSeriesByUID([uid])
        vols = [slicer.mrmlScene.GetNodeByID(i) for i in loaded_ids]
        vols = [v for v in vols if v is not None and v.IsA("vtkMRMLScalarVolumeNode")]
        if len(vols) != 1:
            raise RuntimeError(f"expected exactly 1 scalar volume, got {len(vols)}")
        src = vols[0]

        image = vtk.vtkImageData()
        image.DeepCopy(src.GetImageData())
        lo, hi = image.GetScalarRange()
        if image.GetScalarType() != vtk.VTK_SHORT and -32768 <= lo and hi <= 32767:
            cast = vtk.vtkImageCast()
            cast.SetInputData(image)
            cast.SetOutputScalarTypeToShort()
            cast.Update()
            image = vtk.vtkImageData()
            image.DeepCopy(cast.GetOutput())
        ijk_to_ras = vtk.vtkMatrix4x4()
        src.GetIJKToRASMatrix(ijk_to_ras)
        dst = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", case_id)
        dst.SetAndObserveImageData(image)
        dst.SetIJKToRASMatrix(ijk_to_ras)
        if not slicer.util.saveNode(dst, out_path, {"useCompression": 1 if compress else 0}):
            raise RuntimeError("saveNode failed")

    header = read_nrrd_header(out_path)
    extra = set(k for k in header if k != "#comments") - NRRD_ALLOWED_KEYS
    bad_comments = set(header.get("#comments", [])) - NRRD_ALLOWED_COMMENTS
    if extra or bad_comments:
        os.remove(out_path)
        raise RuntimeError(f"NRRD header has non-geometry content, removed file: keys={extra} "
                           f"comments={bad_comments}")

    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    dims = image.GetDimensions()
    spacing = dst.GetSpacing()
    arr = slicer.util.arrayFromVolume(dst)
    info = {
        "case_id": case_id,
        "nrrd": os.path.relpath(out_path, os.path.dirname(out_dir.rstrip("/"))),
        "sha256": sha,
        "source_dir": os.path.abspath(dicom_dir),
        "series_uid": uid,
        "n_files": n_files,
        "dims": list(dims),
        "spacing_mm": [round(s, 6) for s in spacing],
        "scalar_range": [float(arr.min()), float(arr.max())],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if os.environ.get("SLICER_MCP_PREP_RESULT"):
        with open(os.environ["SLICER_MCP_PREP_RESULT"], "w") as f:
            json.dump(info, f, indent=2)
    print("[prepare_case] header:", json.dumps({k: v for k, v in header.items()}, indent=1))
    print("[prepare_case] ok:", json.dumps({k: v for k, v in info.items()
                                           if k not in ("source_dir", "series_uid")}))


try:
    main(*sys.argv[1:5])
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    slicer.util.exit(1)
else:
    slicer.util.exit(0)
