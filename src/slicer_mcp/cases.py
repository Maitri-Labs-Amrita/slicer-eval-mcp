"""Temporary, anonymised cases: a DICOM folder or a volume file, optionally with a
label map for the segment tools, converted in a separate headless Slicer into <cases_dir>/<case_id>.nrrd.

    slicer-mcp case prepare --dicom <dir> [--series UID] [--labels L --label-map 1=LV,...] [--profile P] [--runtime R]
    slicer-mcp case prepare --volume <file> [--spacing X Y Z] [--lps-identity] [--intensity-offset N]
                            [--case-key KEY] [--labels L --label-map 1=LV,...] [--profile P] [--runtime R]
    slicer-mcp case remove <case_id> [--lease TOKEN] [--runtime R]

The bridge loads only <cases_dir>/<case_id>.nrrd (geometry and voxels; it never opens DICOM, so no header text can
reach the scene), so cases_dir here must be the bridge's (both read runtime.json). `prepare` prints one JSON object:
case_id, nrrd, sha256, dims, spacing_mm, scalar_range, lease (and, with labels, `shell`: the shell manifest).
`remove` deletes the case's files once no other run holds a lease on it. Only ids case_dcm_<10 hex> / case_nii_<10
hex> are ever written or deleted here; leftovers from a killed run are removed by a later `prepare`.

Case ids say nothing about the case:
- DICOM: case_dcm_ + sha256(series UID)[:10] (tools/prepare_case.py picks the CT series with the most files).
- Volume: case_nii_ + sha256(case key)[:10]; the key is --case-key (e.g. "mydataset:0001": the same case always gets
  the same id) or, without one, "file:" + the file's sha256.

The volume's geometry is its header's (tools/prepare_volume.py), unless overridden: --spacing replaces the spacing,
--lps-identity replaces the direction and origin with an LPS identity at 0. --intensity-offset is added to every
value (e.g. -1024 for a dataset that stores HU + 1024). Values are kept as int16.

--labels: a label map on the case's own voxel grid (same shape, same array order), --label-map its codes -> the
profile's segmentation class names; tools/prepare_shell.py builds <case>.shell.nrrd, .labels.nrrd and .shell.json
for the segment tools. In Python, `temporary()` prepares a case and always removes it afterwards.
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from slicer_mcp.config import REPO, load_protocol, load_runtime
from slicer_mcp.jobenv import job_env

DCM_PREFIX = "case_dcm_"
VOL_PREFIX = "case_nii_"
PREFIXES = (DCM_PREFIX, VOL_PREFIX)
ID_RE = re.compile(r"^case_(dcm|nii)_[0-9a-f]{10}$")
DICOM_SCRIPT = REPO / "tools" / "prepare_case.py"
VOLUME_SCRIPT = REPO / "tools" / "prepare_volume.py"
SHELL_SCRIPT = REPO / "tools" / "prepare_shell.py"
SHELL_SUFFIXES = (".shell.nrrd", ".labels.nrrd", ".shell.json", ".shell.source.json")
TMP_DCM_ID = "case_dcm_converting"
TMP_VOL_ID = "case_nii_converting"
LEFTOVER_MIN_AGE_S = 6 * 3600
LEASES = ".leases"
LEASE_TTL_S = 48 * 3600
SLICER_FLAGS = ("--no-splash", "--no-main-window", "--disable-settings", "--ignore-slicerrc")


class CaseError(RuntimeError):
    pass


def take_lease(cases_dir, case_id) -> str:
    """Record that one more run uses <case_id>: remove() deletes the files only when no other run's lease is
    left, and remove_leftovers never deletes a leased case. Returns the lease token."""
    d = Path(cases_dir) / LEASES
    d.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    (d / f"{case_id}.{token}").write_text(json.dumps({"pid": os.getppid(), "created": time.time()}))
    return token


def live_leases(cases_dir, case_id, ttl_s=LEASE_TTL_S) -> list:
    """The case's leases younger than ttl_s, oldest first."""
    d = Path(cases_dir) / LEASES
    now, out = time.time(), []
    for p in d.glob(f"{case_id}.*") if d.is_dir() else ():
        try:
            age = now - p.stat().st_mtime
        except FileNotFoundError:
            continue
        if age < ttl_s:
            out.append((p.stat().st_mtime, p))
    return [p for _, p in sorted(out)]


def tmp_id(prefix=TMP_VOL_ID) -> str:
    """A conversion's own temporary case id: <prefix>_<8 hex>."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_id_for_series(series_uid: str) -> str:
    return DCM_PREFIX + hashlib.sha256(series_uid.encode()).hexdigest()[:10]


def case_id_for_key(key: str) -> str:
    return VOL_PREFIX + hashlib.sha256(key.encode()).hexdigest()[:10]


def slicer_executable(rt: dict) -> str:
    p = os.environ.get("SLICER_MCP_SLICER_EXECUTABLE") or rt.get("slicer_executable")
    if not p or not os.path.isfile(p) or not os.access(p, os.X_OK):
        raise CaseError(f"Slicer executable not found: {p!r} (runtime.json slicer_executable)")
    return p


def dicom_command(slicer, dicom_dir, out_dir, series_uid=None, tmp=TMP_DCM_ID) -> list:
    """The headless Slicer command line (pure: unit-tested)."""
    cmd = [str(slicer), *SLICER_FLAGS, "--python-script", str(DICOM_SCRIPT), str(dicom_dir), tmp, str(out_dir)]
    return cmd + ([series_uid] if series_uid else [])


def script_command(slicer, script, job_path) -> list:
    """The headless Slicer command line for a job-file converter (pure: unit-tested)."""
    return [str(slicer), *SLICER_FLAGS, "--python-script", str(script), str(job_path)]


def parse_label_map(text) -> dict:
    """'1=LV,2=RV' -> {"1": "LV", "2": "RV"} (codes 1..255, names unique)."""
    out = {}
    for part in [p for p in (text or "").split(",") if p.strip()]:
        code, sep, name = part.partition("=")
        if not sep or not code.strip().isdigit() or not 0 < int(code) < 256 or not name.strip():
            raise CaseError(f"--label-map: {part!r} is not <code 1..255>=<class name>")
        out[str(int(code))] = name.strip()
    if not out or len(set(out.values())) != len(out):
        raise CaseError("--label-map needs one or more codes with distinct class names")
    return out


def case_files(cases_dir, case_id) -> list:
    """Every file of a case in cases_dir: the NRRD and the shell maps."""
    return [Path(cases_dir) / f"{case_id}{s}" for s in (".nrrd",) + SHELL_SUFFIXES]


def remove_leftovers(cases_dir, min_age_s=LEFTOVER_MIN_AGE_S) -> list:
    """Delete case_dcm_* / case_nii_* files (NRRDs, shell maps) left by a run that was killed before its cleanup:
    only files not modified for min_age_s, so a case another run on the same cases_dir is converting or about to load
    is kept. Returns the removed names."""
    gone, now = [], time.time()
    for prefix in PREFIXES:
        for pat in ("*.nrrd", "*.shell.json"):
            for p in Path(cases_dir).glob(prefix + pat):
                cid = p.name.split(".", 1)[0]
                if ID_RE.fullmatch(cid) and live_leases(cases_dir, cid):
                    continue
                try:
                    if now - p.stat().st_mtime < min_age_s:
                        continue
                    p.unlink()
                except FileNotFoundError:
                    continue
                gone.append(p.name)
    return sorted(gone)


def refuse_leased_change(cases_dir, case_id, src, sha256) -> None:
    """Refuse to replace an existing case that another run holds a lease on with different content: the converted
    file `src` is removed and CaseError names the case and the lease. Identical content is allowed."""
    dst = Path(cases_dir) / f"{case_id}.nrrd"
    if not dst.exists():
        return
    held = live_leases(cases_dir, case_id)
    if not held or file_sha256(dst) == sha256:
        return
    Path(src).unlink(missing_ok=True)
    lease = held[0].name.split(".", 1)[1]
    raise CaseError(f"case {case_id} is in use (lease {lease}) and this preparation would change its content: "
                    "not replaced. Remove the case when its run is done, or use another --case-key")


def _place_checked(src, dst, sha256) -> None:
    """Move a converted NRRD into place and check that it is still the file the converter hashed. An existing case
    with a live lease is replaced only by identical content."""
    refuse_leased_change(Path(dst).parent, Path(dst).name.removesuffix(".nrrd"), src, sha256)
    Path(src).replace(dst)
    got = file_sha256(dst)
    if got != sha256:
        Path(dst).unlink(missing_ok=True)
        raise CaseError(f"{Path(dst).name}: sha256 {got[:12]} != the converter's {sha256[:12]}; another "
                        "conversion overwrote it: the case is refused")


def _run_slicer(cmd, td, runtime, timeout_s, what) -> dict:
    """Run a converter in a headless Slicer with its own HOME / tmp; returns the JSON it wrote."""
    result = Path(td) / "result.json"
    env = {**job_env(os.environ, Path(td) / "slicer"), "SLICER_MCP_PREP_RESULT": str(result)}
    if runtime:
        env["SLICER_MCP_RUNTIME"] = str(Path(runtime).resolve())
    try:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise CaseError(f"{what} timed out after {timeout_s} s") from e
    if p.returncode != 0 or not result.exists():
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
        raise CaseError(f"{what} failed (exit {p.returncode}):\n{tail}")
    return json.loads(result.read_text())


def _cases_dir(rt, remove_stale) -> Path:
    cases_dir = Path(rt["cases_dir"])
    cases_dir.mkdir(parents=True, exist_ok=True)
    if remove_stale:
        remove_leftovers(cases_dir)
    return cases_dir


def _place(cases_dir, info, case_id) -> dict:
    """Move the converted NRRD (the conversion's own temporary file) to <case_id>.nrrd, checked by its sha256."""
    dst = Path(cases_dir) / f"{case_id}.nrrd"
    _place_checked(Path(cases_dir) / f"{info['case_id']}.nrrd", dst, info["sha256"])
    return {"case_id": case_id, "nrrd": str(dst), "lease": take_lease(cases_dir, case_id)}


def shell_source(labels, label_map) -> dict:
    """What a case's shells are built from: the label map file's sha256 and the code mapping."""
    return {"labels_sha256": file_sha256(Path(labels).expanduser()), "label_map": label_map}


def leased_shell(cases_dir, case_id, lease, source) -> dict | None:
    """The existing shell manifest when another run holds a lease on the case and its shells were built from the same
    source (they are reused, not rebuilt); None when no other run holds the case. CaseError when another run holds
    shells built from other labels: rebuilding them would change that run's case."""
    others = [p for p in live_leases(cases_dir, case_id) if p.name != f"{case_id}.{lease}"]
    manifest = Path(cases_dir) / f"{case_id}.shell.json"
    if not others or not manifest.exists():
        return None
    rec = Path(cases_dir) / f"{case_id}.shell.source.json"
    try:
        held = json.loads(rec.read_text())
    except (OSError, ValueError):
        held = None
    if held != source:
        token = others[0].name.split(".", 1)[1]
        raise CaseError(f"case {case_id} is in use (lease {token}) with shells from other labels: not rebuilt. "
                        "Remove the case when its run is done, or use another --case-key")
    info = json.loads(manifest.read_text())
    return {k: v for k, v in info.items() if k not in ("shell_source", "shell_created")}


def _with_labels(out, labels, label_map, profile, rt, runtime, timeout_s) -> dict:
    """Build the shell maps for a placed case (or reuse identical ones another run holds); on failure the case is
    removed again."""
    if labels is None:
        return out
    cases_dir = Path(rt["cases_dir"])
    try:
        source = shell_source(labels, label_map)
        shell = leased_shell(cases_dir, out["case_id"], out["lease"], source)
        if shell is None:
            shell = prepare_shell(labels, label_map, out["case_id"], rt, profile, runtime, timeout_s)
            (cases_dir / f"{out['case_id']}.shell.source.json").write_text(json.dumps(source))
        out["shell"] = shell
    except CaseError:
        remove(out["case_id"], runtime, out["lease"])
        raise
    return out


def prepare_dicom(dicom_dir, series_uid=None, runtime=None, timeout_s=900, labels=None, label_map=None,
                  profile=None, remove_stale=True) -> dict:
    """A DICOM folder (tools/prepare_case.py) -> a temporary case_dcm_* case."""
    rt = load_runtime(runtime)
    dicom_dir = Path(dicom_dir).expanduser().resolve()
    if not dicom_dir.is_dir():
        raise CaseError(f"not a folder: {dicom_dir}")
    _check_labels(labels, label_map)
    cases_dir = _cases_dir(rt, remove_stale)
    tmp = tmp_id(TMP_DCM_ID)
    with tempfile.TemporaryDirectory(prefix="slicer_mcp_dicom_") as td:
        try:
            info = _run_slicer(dicom_command(slicer_executable(rt), dicom_dir, cases_dir, series_uid, tmp), td,
                               runtime, timeout_s, "DICOM conversion")
        except CaseError:
            (cases_dir / f"{tmp}.nrrd").unlink(missing_ok=True)
            raise
    case_id = case_id_for_series(info["series_uid"])
    dst = cases_dir / f"{case_id}.nrrd"
    _place_checked(cases_dir / f"{tmp}.nrrd", dst, info["sha256"])
    out = {**info, "case_id": case_id, "nrrd": str(dst), "lease": take_lease(cases_dir, case_id)}
    return _with_labels(out, labels, label_map, profile, rt, runtime, timeout_s)


def prepare_volume(volume, runtime=None, timeout_s=900, spacing_mm=None, lps_identity=False, intensity_offset=0,
                   case_key=None, labels=None, label_map=None, profile=None, remove_stale=True) -> dict:
    """A volume file (tools/prepare_volume.py) -> a temporary case_nii_* case. The file is checked here first, so a
    wrong path fails before Slicer starts. remove_stale=False keeps other temporary cases."""
    rt = load_runtime(runtime)
    volume = Path(volume).expanduser().resolve()
    if not volume.is_file():
        raise CaseError(f"volume not found: {volume}")
    if spacing_mm is not None and (len(spacing_mm) != 3 or not all(float(s) > 0 for s in spacing_mm)):
        raise CaseError("--spacing needs three positive numbers (x, y, z mm)")
    _check_labels(labels, label_map)
    case_id = case_id_for_key(case_key or f"file:{file_sha256(volume)}")
    cases_dir = _cases_dir(rt, remove_stale)
    tid = tmp_id(TMP_VOL_ID)
    with tempfile.TemporaryDirectory(prefix="slicer_mcp_volume_") as td:
        job = Path(td) / "job.json"
        job.write_text(json.dumps({"volume": str(volume), "case_id": tid, "out_dir": str(cases_dir),
                                   "spacing_mm": [float(s) for s in spacing_mm] if spacing_mm is not None else None,
                                   "lps_identity": bool(lps_identity), "intensity_offset": intensity_offset}))
        try:
            info = _run_slicer(script_command(slicer_executable(rt), VOLUME_SCRIPT, job), td, runtime, timeout_s,
                               "volume conversion")
        except CaseError:
            (cases_dir / f"{tid}.nrrd").unlink(missing_ok=True)
            raise
    out = {**info, **_place(cases_dir, info, case_id)}
    return _with_labels(out, labels, label_map, profile, rt, runtime, timeout_s)


def _check_labels(labels, label_map):
    if (labels is None) != (label_map is None):
        raise CaseError("--labels and --label-map go together")
    if labels is not None and not Path(labels).expanduser().is_file():
        raise CaseError(f"label map not found: {labels}")


def prepare_shell(labels, label_map, case_id, rt, profile=None, runtime=None, timeout_s=900) -> dict:
    """Shell maps for an already converted case (tools/prepare_shell.py), with the profile's segmentation classes.
    Returns the manifest."""
    protocol = load_protocol(profile)
    if protocol.get("segmentation") is None:
        raise CaseError(f"profile {protocol.path.parent.name} has no segmentation: no shells to build")
    with tempfile.TemporaryDirectory(prefix="slicer_mcp_shell_") as td:
        job = Path(td) / "job.json"
        job.write_text(json.dumps({"labels": str(Path(labels).expanduser().resolve()), "case_id": case_id,
                                   "cases_dir": str(rt["cases_dir"]), "protocol": str(protocol.path),
                                   "label_map": label_map}))
        info = _run_slicer(script_command(slicer_executable(rt), SHELL_SCRIPT, job), td, runtime, timeout_s,
                           "shell preparation")
    return {k: v for k, v in info.items() if k not in ("shell_source", "shell_created")}


def remove(case_id: str, runtime=None, lease=None) -> bool:
    """Drop this run's lease on the case (the given token, else its oldest) and delete the case's files unless another
    run still holds a lease (the same case prepared by two jobs on one cases_dir). True when files were deleted."""
    if not ID_RE.fullmatch(case_id):
        raise CaseError(f"refusing to remove {case_id!r}: not a temporary case id")
    cases_dir = load_runtime(runtime)["cases_dir"]
    held = live_leases(cases_dir, case_id)
    mine = [p for p in held if p.name == f"{case_id}.{lease}"] if lease else held[:1]
    for p in mine:
        p.unlink(missing_ok=True)
    if [p for p in held if p not in mine]:
        return False
    for p in (Path(cases_dir) / LEASES).glob(f"{case_id}.*") if (Path(cases_dir) / LEASES).is_dir() else ():
        p.unlink(missing_ok=True)
    try:
        (Path(cases_dir) / LEASES).rmdir()
    except OSError:
        pass
    gone = False
    for p in case_files(cases_dir, case_id):
        if p.exists():
            p.unlink()
            gone = True
    return gone


@contextlib.contextmanager
def temporary(*, dicom=None, volume=None, runtime=None, **kw):
    """prepare_dicom(dicom, ...) or prepare_volume(volume, ...) for a with-block; the case's files are removed on
    exit, also after an error. Yields the info."""
    if (dicom is None) == (volume is None):
        raise CaseError("temporary() needs exactly one of dicom= or volume=")
    info = prepare_dicom(dicom, runtime=runtime, **kw) if dicom is not None else \
        prepare_volume(volume, runtime=runtime, **kw)
    try:
        yield info
    finally:
        remove(info["case_id"], runtime, info.get("lease"))


def voxel_sha256(nrrd_path) -> str:
    """sha256 of a prepared case's voxels in (k, j, i) C order and their own dtype, as the bridge reports it at reset
    (case_info / _meta slicer_mcp/identity volume_sha256): a runner checks that Slicer loaded this case. Needs
    SimpleITK (the dev group)."""
    import numpy as np
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nrrd_path)))
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slicer-mcp case", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dicom", help="a DICOM folder (only read)")
    g.add_argument("--volume", help="a volume file: NIfTI, NRRD, MHA, ... (only read)")
    p.add_argument("--series", help="with --dicom: the series UID (default: the CT series with the most files)")
    p.add_argument("--spacing", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="with --volume: the voxel spacing in mm, instead of the header's")
    p.add_argument("--lps-identity", action="store_true",
                   help="with --volume: direction and origin = an LPS identity at 0, instead of the header's")
    p.add_argument("--intensity-offset", type=float, default=0, help="with --volume: added to every value")
    p.add_argument("--case-key", help="with --volume: the case id is derived from this key (default: the file's "
                                      "sha256)")
    p.add_argument("--labels", help="a label map on the case's voxel grid: also build the segmentation shells")
    p.add_argument("--label-map", help="with --labels: its codes -> the profile's class names, e.g. 1=LV,2=RV")
    p.add_argument("--profile", help="the profile whose segmentation classes the shells use (default: "
                                     "$SLICER_MCP_PROFILE, else ct_basic)")
    r = sub.add_parser("remove")
    r.add_argument("case_id")
    r.add_argument("--lease", help="the lease token prepare printed (default: the case's oldest lease)")
    for s in (p, r):
        s.add_argument("--runtime")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            if a.dicom and (a.spacing or a.lps_identity or a.intensity_offset or a.case_key):
                ap.error("--spacing, --lps-identity, --intensity-offset and --case-key are for --volume")
            if a.volume and a.series:
                ap.error("--series is for --dicom")
            label_map = parse_label_map(a.label_map) if a.label_map else None
            off = int(a.intensity_offset) if float(a.intensity_offset).is_integer() else a.intensity_offset
            if a.dicom:
                out = prepare_dicom(a.dicom, a.series, a.runtime, labels=a.labels, label_map=label_map,
                                    profile=a.profile)
            else:
                out = prepare_volume(a.volume, a.runtime, spacing_mm=a.spacing, lps_identity=a.lps_identity,
                                     intensity_offset=off, case_key=a.case_key, labels=a.labels,
                                     label_map=label_map, profile=a.profile)
            print(json.dumps(out))
        else:
            print(json.dumps({"removed": remove(a.case_id, a.runtime, a.lease)}))
    except CaseError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
