"""A label map on a prepared case's voxel grid -> hollow-shell label maps. Runs INSIDE a
separate headless Slicer (Slicer's bundled SimpleITK; nothing is installed):

    /Applications/Slicer.app/Contents/MacOS/Slicer --no-splash --no-main-window \
        --disable-settings --ignore-slicerrc --python-script tools/prepare_shell.py <job.json>

job.json (written by slicer_mcp.cases): {"labels", "case_id", "cases_dir", "protocol", "label_map"}.

- labels: a label map (NIfTI, NRRD, ...) on the case's own voxel grid: read with SimpleITK in its array order, the
  order prepare_volume.py keeps, so it must have the case's shape (no resampling). label_map maps its codes to the
  protocol's segmentation class names ({"1": "LV", ...}); every other code is ignored.
- Needs <cases_dir>/<case_id>.nrrd (prepare_volume.py or prepare_case.py). Classes are painted in label order, so on
  overlap the higher label wins.
- Registration report (protocol segmentation.registration_check): the mean CT value inside each filled label,
  reported only.
- The shell: slicer_mcp.shell.create_shell, shell_thickness_mm from the protocol.
- Both maps are cropped to the segmentation's bounding box plus a margin, and written with only voxels and
  geometry (gzip; the same NRRD header whitelist as prepare_case):
      <case_id>.shell.nrrd   the shell label map (what the model sees)
      <case_id>.labels.nrrd  the filled label map (sampled for inside_segment; never displayed)
      <case_id>.shell.json   manifest: classes present, voxel counts, RAS boxes, sha256s, registration numbers
  No file or node name, path or header text from the source reaches these files.
- If SLICER_MCP_PREP_RESULT is set, writes the manifest plus the ignored codes and the source there as JSON.
- Called only by slicer_mcp.cases (temporary cases); it keeps no registry of cases.

The label map is only read.
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
sys.path.insert(0, os.path.join(REPO, "src"))
from slicer_mcp import shell  # noqa: E402  (stdlib + numpy + SimpleITK only)

CROP_MARGIN_VOX = 8
NRRD_ALLOWED_KEYS = {
    "type", "dimension", "space", "sizes", "space directions", "kinds", "endian", "encoding",
    "space origin", "space units", "measurement frame",
}
NRRD_ALLOWED_COMMENTS = {
    "# Complete NRRD file format specification at:",
    "# http://teem.sourceforge.net/nrrd/format.html",
}


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


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_labelmap(arr, ijk_to_ras, path):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "shell_out")
    slicer.util.updateVolumeFromArray(node, np.ascontiguousarray(arr, dtype=np.uint8))
    node.SetIJKToRASMatrix(ijk_to_ras)
    if not slicer.util.saveNode(node, path, {"useCompression": 1}):
        raise RuntimeError(f"saveNode failed: {path}")
    slicer.mrmlScene.RemoveNode(node)
    check_header(path)


def main(job_path):
    with open(job_path) as f:
        job = json.load(f)
    src, case_id, cases_dir = os.path.abspath(job["labels"]), job["case_id"], os.path.abspath(job["cases_dir"])
    with open(job["protocol"]) as f:
        protocol = json.load(f)
    seg_cfg = protocol["segmentation"]
    classes = shell.classes_from(protocol)
    if not os.path.isfile(src):
        raise RuntimeError(f"label map not found: {src}")

    ct_path = os.path.join(cases_dir, f"{case_id}.nrrd")
    if not os.path.isfile(ct_path):
        raise RuntimeError(f"case {case_id} not prepared: {ct_path} missing")
    ct = slicer.util.loadVolume(ct_path, {"show": False, "singleFile": True})
    ct_arr = slicer.util.arrayFromVolume(ct).astype(np.float32)
    ijk_to_ras = vtk.vtkMatrix4x4()
    ct.GetIJKToRASMatrix(ijk_to_ras)

    import SimpleITK as sitk
    lab = sitk.GetArrayFromImage(sitk.ReadImage(src))
    if lab.shape != ct_arr.shape:
        raise RuntimeError(f"label map shape {lab.shape} != CT {ct_arr.shape} (not this case's grid)")
    code_of = {name: int(code) for code, name in job["label_map"].items()}
    unknown = sorted(n for n in code_of if n not in {c["name"] for c in classes})
    if unknown:
        raise RuntimeError(f"label_map names {unknown} are not protocol classes {[c['name'] for c in classes]}")
    found, masks = {}, []
    for c in classes:
        code = code_of.get(c["name"])
        m = (lab == code) if code is not None else None
        if m is not None and m.any():
            found[c["name"]] = src
            masks.append((c["label"], m))
    codes = sorted(int(v) for v in np.unique(lab) if v)
    ignored = [v for v in codes if v not in code_of.values()]
    if not found:
        raise RuntimeError(f"no class label in {src} (label_map {job['label_map']}, codes {codes})")
    print(f"[prepare_shell] label map classes {sorted(found)}; ignored codes {ignored}")
    labels = shell.combine_masks(masks, ct_arr.shape)
    return _finish(labels, found, ignored, classes, seg_cfg, ct, ct_arr, ijk_to_ras, case_id, cases_dir, src)


def _finish(labels, found, ignored, classes, seg_cfg, ct, ct_arr, ijk_to_ras, case_id, cases_dir, src):
    report = shell.registration_report(ct_arr, labels, classes, seg_cfg["registration_check"])
    print("[prepare_shell] registration:", json.dumps(report))
    if report["failures"]:
        raise RuntimeError("registration check failed: " + "; ".join(report["failures"]))

    idx = np.argwhere(labels > 0)
    lo = np.maximum(idx.min(axis=0) - CROP_MARGIN_VOX, 0)
    hi = np.minimum(idx.max(axis=0) + CROP_MARGIN_VOX + 1, labels.shape)
    crop = tuple(slice(int(a), int(z)) for a, z in zip(lo, hi))
    labels_c = labels[crop]
    spacing = ct.GetSpacing()
    shell_c = shell.shell_array(labels_c, spacing, seg_cfg["shell_thickness_mm"])
    m = np.array([[ijk_to_ras.GetElement(r, c) for c in range(4)] for r in range(4)])
    m_c = m.copy()
    m_c[:3, 3] = (m @ np.array([lo[2], lo[1], lo[0], 1.0]))[:3]
    crop_mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            crop_mat.SetElement(r, c, m_c[r, c])

    out = {k: os.path.join(cases_dir, f"{case_id}.{k}.nrrd") for k in ("shell", "labels")}
    write_labelmap(shell_c, crop_mat, out["shell"])
    write_labelmap(labels_c, crop_mat, out["labels"])
    manifest = {
        "format": "slicer-mcp-shell-1",
        "case_id": case_id,
        "classes": [c["name"] for c in classes if c["name"] in found],
        "labels": {c["name"]: c["label"] for c in classes if c["name"] in found},
        "shell_thickness_mm": seg_cfg["shell_thickness_mm"],
        "dims": [int(x) for x in labels_c.shape[::-1]],
        "voxels": {c["name"]: int((labels_c == c["label"]).sum()) for c in classes if c["name"] in found},
        "shell_voxels": {c["name"]: int((shell_c == c["label"]).sum()) for c in classes if c["name"] in found},
        "bbox_ras": {c["name"]: shell.bbox_ras(labels_c == c["label"], m_c) for c in classes if c["name"] in found},
        "registration": report["classes"],
        "sha256": {k: sha256(p) for k, p in out.items()},
    }
    man_path = os.path.join(cases_dir, f"{case_id}.shell.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    private = {"shell_source": src, "shell_ignored_codes": ignored,
               "shell_created": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if os.environ.get("SLICER_MCP_PREP_RESULT"):
        with open(os.environ["SLICER_MCP_PREP_RESULT"], "w") as f:
            json.dump({**manifest, **private}, f, indent=2)
    print("[prepare_shell] ok:", json.dumps({k: manifest[k] for k in ("classes", "dims", "shell_voxels")}))


try:
    main(sys.argv[1])
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    slicer.util.exit(1)
else:
    slicer.util.exit(0)
