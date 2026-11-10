"""Temporary cases (slicer_mcp.cases): ids, command lines, label maps, cleanup, leases and the CLI. A fake Slicer
stands in for the real one (it runs whichever converter script it is given, in miniature)."""

import json
import os
import re
import stat
import sys
import time

import pytest

from slicer_mcp import cases as cs
from slicer_mcp.config import DEFAULT_RUNTIME, load_protocol
from slicer_mcp.schemas.tools import CASE_ID_PATTERN

FAKE_SLICER = """
import hashlib, json, os, sys
script = sys.argv[sys.argv.index("--python-script") + 1]
a = sys.argv[sys.argv.index("--python-script") + 2:]
result = os.environ["SLICER_MCP_PREP_RESULT"]
home = {"home": os.environ["HOME"], "tmpdir": os.environ["TMPDIR"]}
if script.endswith("prepare_shell.py"):
    job = json.load(open(a[0]))
    out_dir, case_id = job["cases_dir"], job["case_id"]
    assert os.path.exists(os.path.join(out_dir, case_id + ".nrrd"))
    if os.environ.get("FAKE_SHELL_FAIL"):
        open(os.path.join(out_dir, case_id + ".shell.nrrd"), "w").write("partial")
        print("label map shape differs"); sys.exit(1)
    for s in (".shell.nrrd", ".labels.nrrd", ".shell.json"):
        open(os.path.join(out_dir, case_id + s), "w").write("x")
    json.dump({"format": "slicer-mcp-shell-1", "classes": sorted(set(job["label_map"].values())),
               "shell_source": job["labels"], "shell_created": "now", "job": job}, open(result, "w"))
    sys.exit(0)
if script.endswith("prepare_volume.py"):
    job = json.load(open(a[0]))
    case_id, out_dir = job["case_id"], job["out_dir"]
    assert case_id.startswith("case_nii_converting_")  # its own tmp id
    if os.environ.get("FAKE_FAIL"):
        open(os.path.join(out_dir, case_id + ".nrrd"), "w").write("partial")
        print("bad header"); sys.exit(1)
    body = "nrrd" + (json.dumps(job["spacing_mm"]) if job["spacing_mm"] else "")
    open(os.path.join(out_dir, case_id + ".nrrd"), "w").write(body)
    json.dump({"case_id": case_id, "sha256": hashlib.sha256(body.encode()).hexdigest(), "dims": [6, 5, 4],
               "spacing_mm": job["spacing_mm"] or [1, 1, 1], "job": job, **home}, open(result, "w"))
    sys.exit(0)
dicom_dir, case_id, out_dir = a[:3]
assert case_id.startswith("case_dcm_converting_")
if os.environ.get("FAKE_FAIL"):
    open(os.path.join(out_dir, case_id + ".nrrd"), "w").write("partial")
    print("boom"); sys.exit(1)
open(os.path.join(out_dir, case_id + ".nrrd"), "w").write("nrrd")
json.dump({"case_id": case_id, "sha256": hashlib.sha256(b"nrrd").hexdigest(), "source_dir": dicom_dir,
           "series_uid": a[3] if len(a) > 3 else "1.2.3", "n_files": 3, "dims": [1, 1, 3], "spacing_mm": [1, 1, 1],
           **home}, open(result, "w"))
"""
HEADLESS = ["--no-splash", "--no-main-window", "--disable-settings", "--ignore-slicerrc"]


def old(path, text="stale"):
    """A leftover from a run killed hours ago: remove_leftovers removes only files older than 6 h."""
    path.write_text(text)
    t = time.time() - 7 * 3600
    os.utime(path, (t, t))


def names(d):
    return sorted(p.name for p in d.iterdir() if p.name != cs.LEASES)


@pytest.fixture
def rt(tmp_path, monkeypatch):
    (tmp_path / "fake_slicer.py").write_text(FAKE_SLICER)
    slicer = tmp_path / "Slicer"
    slicer.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_slicer.py"}" "$@"\n')
    slicer.chmod(slicer.stat().st_mode | stat.S_IEXEC)
    r = json.loads(DEFAULT_RUNTIME.read_text())
    r.update(cases_dir=str(tmp_path / "cases"), slicer_executable=str(slicer))
    p = tmp_path / "runtime.json"
    p.write_text(json.dumps(r))
    monkeypatch.delenv("SLICER_MCP_SLICER_EXECUTABLE", raising=False)
    (tmp_path / "dicom").mkdir()
    (tmp_path / "ct.nii.gz").write_bytes(b"volume")
    (tmp_path / "labels.nii.gz").write_bytes(b"labels")
    return p


def test_case_ids_are_stable_anonymous_and_valid():
    a, k = cs.case_id_for_series("1.2.840.1"), cs.case_id_for_key("example:0001")
    assert a == cs.case_id_for_series("1.2.840.1") != cs.case_id_for_series("1.2.840.2")
    assert k == cs.case_id_for_key("example:0001") != cs.case_id_for_key("example:0002")
    for cid in (a, k):
        assert cs.ID_RE.match(cid) and re.match(CASE_ID_PATTERN, cid)
    assert a.startswith("case_dcm_") and k.startswith("case_nii_") and "840" not in a and "0001" not in k


def test_case_ids_are_pinned():
    assert cs.case_id_for_key("example:0001") == "case_nii_09a45f3875"


def test_commands_are_headless_with_temporary_settings():
    cmd = cs.dicom_command("/S", "/d", "/out", "9.9", "case_dcm_converting_1a2b3c4d")
    assert cmd[:5] == ["/S", *HEADLESS]
    assert cmd[-5:] == [str(cs.DICOM_SCRIPT), "/d", "case_dcm_converting_1a2b3c4d", "/out", "9.9"]
    assert cs.dicom_command("/S", "/d", "/out")[-1] == "/out"
    assert cs.script_command("/S", cs.VOLUME_SCRIPT, "/j.json") == ["/S", *HEADLESS, "--python-script",
                                                                     str(cs.VOLUME_SCRIPT), "/j.json"]


def test_each_conversion_has_its_own_temporary_id():
    a, b = cs.tmp_id(), cs.tmp_id(cs.TMP_DCM_ID)
    assert a.startswith("case_nii_converting_") and b.startswith("case_dcm_converting_") and cs.tmp_id() != a


@pytest.mark.parametrize("text,want", [("1=kidney_left,2=kidney_right", {"1": "kidney_left", "2": "kidney_right"}),
                                       (" 01 = liver , 6=aorta,", {"1": "liver", "6": "aorta"})])
def test_label_map_parsed(text, want):
    assert cs.parse_label_map(text) == want


@pytest.mark.parametrize("text", ["", "kidney_left", "0=kidney_left", "256=kidney_left", "x=kidney_left", "1=", "1=kidney_left,2=kidney_left"])
def test_bad_label_map_refused(text):
    with pytest.raises(cs.CaseError):
        cs.parse_label_map(text)


def test_prepare_dicom_names_by_series_and_removes_leftovers(rt, tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    old(cases / "case_dcm_0000000000.nrrd")
    (cases / "other_case.nrrd").write_text("other")
    info = cs.prepare_dicom(tmp_path / "dicom", series_uid="7.7", runtime=rt)
    assert info["case_id"] == cs.case_id_for_series("7.7") and info["lease"]
    assert names(cases) == sorted(["other_case.nrrd", info["case_id"] + ".nrrd"])
    assert cs.remove(info["case_id"], rt) is True and cs.remove(info["case_id"], rt) is False
    assert (cases / "other_case.nrrd").exists()


def test_failed_conversion_leaves_nothing(rt, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(cs.CaseError, match="boom"):
        cs.prepare_dicom(tmp_path / "dicom", runtime=rt)
    with pytest.raises(cs.CaseError, match="bad header"):
        cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt)
    assert list((tmp_path / "cases").iterdir()) == []


def test_prepare_refuses_a_missing_source_before_slicer(rt, tmp_path):
    with pytest.raises(cs.CaseError, match="not a folder"):
        cs.prepare_dicom(tmp_path / "nope", runtime=rt)
    with pytest.raises(cs.CaseError, match="volume not found"):
        cs.prepare_volume(tmp_path / "nope.nii.gz", runtime=rt)
    with pytest.raises(cs.CaseError, match="--spacing"):
        cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt, spacing_mm=[1, 0, 1])
    with pytest.raises(cs.CaseError, match="go together"):
        cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt, labels=tmp_path / "labels.nii.gz")
    with pytest.raises(cs.CaseError, match="label map not found"):
        cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt, labels=tmp_path / "nope", label_map={"1": "kidney_left"})
    assert not (tmp_path / "cases").exists()


def test_remove_refuses_other_ids(rt):
    for bad in ("other_case", "case_dcm_../x", "case_nii_converting", "LEAK_TEST", "case_nii_0123456789\n"):
        with pytest.raises(cs.CaseError):
            cs.remove(bad, rt)


def test_prepare_volume_passes_the_geometry_overrides_and_names_by_key(rt, tmp_path):
    info = cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt, spacing_mm=[0.3, 0.3, 0.5], lps_identity=True,
                             intensity_offset=-1024, case_key="example:0001")
    assert info["case_id"] == "case_nii_09a45f3875"
    assert info["nrrd"] == str(tmp_path / "cases" / "case_nii_09a45f3875.nrrd")
    job = info["job"]
    assert job["volume"] == str((tmp_path / "ct.nii.gz").resolve()) and job["out_dir"] == str(tmp_path / "cases")
    assert job["spacing_mm"] == [0.3, 0.3, 0.5] and job["lps_identity"] is True and job["intensity_offset"] == -1024
    assert names(tmp_path / "cases") == ["case_nii_09a45f3875.nrrd"]
    assert cs.remove(info["case_id"], rt, info["lease"]) is True


def test_volume_without_a_key_is_named_by_its_content(rt, tmp_path):
    (tmp_path / "same.nii.gz").write_bytes(b"volume")
    a = cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt)
    b = cs.prepare_volume(tmp_path / "same.nii.gz", runtime=rt)
    assert a["case_id"] == b["case_id"] == cs.case_id_for_key(f"file:{cs.file_sha256(tmp_path / 'ct.nii.gz')}")
    assert a["job"]["spacing_mm"] is None and a["job"]["lps_identity"] is False and a["job"]["intensity_offset"] == 0
    assert cs.remove(a["case_id"], rt, a["lease"]) is False and cs.remove(b["case_id"], rt, b["lease"]) is True


def test_temporary_removes_the_case_also_after_an_error(rt, tmp_path):
    for kw in ({"dicom": tmp_path / "dicom", "series_uid": "7.7"}, {"volume": tmp_path / "ct.nii.gz"}):
        with pytest.raises(RuntimeError, match="episode broke"):
            with cs.temporary(runtime=rt, **kw) as info:
                assert (tmp_path / "cases" / f"{info['case_id']}.nrrd").exists()
                raise RuntimeError("episode broke")
        assert list((tmp_path / "cases").iterdir()) == []
    with pytest.raises(cs.CaseError, match="exactly one"):
        with cs.temporary(runtime=rt):
            pass


def test_labels_build_shell_maps_with_the_profiles_classes(rt, tmp_path):
    info = cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt, labels=tmp_path / "labels.nii.gz",
                             label_map={"1": "kidney_left", "2": "kidney_right"}, profile="ct_segments")
    cid = info["case_id"]
    shell = info["shell"]
    assert shell["format"] == "slicer-mcp-shell-1" and shell["classes"] == ["kidney_left", "kidney_right"]
    assert "shell_source" not in shell and "shell_created" not in shell
    assert shell["job"]["protocol"] == str(load_protocol("ct_segments").path)
    assert shell["job"]["label_map"] == {"1": "kidney_left", "2": "kidney_right"} and shell["job"]["cases_dir"] == str(tmp_path / "cases")
    assert names(tmp_path / "cases") == sorted(cid + s for s in (".nrrd", ".shell.nrrd", ".labels.nrrd", ".shell.json",
                                                                 ".shell.source.json"))
    assert cs.remove(cid, rt) is True and list((tmp_path / "cases").iterdir()) == []


def test_failed_shell_leaves_nothing(rt, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SHELL_FAIL", "1")
    with pytest.raises(cs.CaseError, match="label map shape differs"):
        cs.prepare_dicom(tmp_path / "dicom", runtime=rt, labels=tmp_path / "labels.nii.gz",
                         label_map={"1": "kidney_left"}, profile="ct_segments")
    assert list((tmp_path / "cases").iterdir()) == []


def test_labels_need_a_profile_with_a_segmentation(rt, tmp_path):
    with pytest.raises(cs.CaseError, match="no segmentation"):
        cs.prepare_volume(tmp_path / "ct.nii.gz", runtime=rt, labels=tmp_path / "labels.nii.gz",
                          label_map={"1": "kidney_left"}, profile="ct_basic")
    assert list((tmp_path / "cases").iterdir()) == []


def test_leftovers_of_both_prefixes_and_their_shells_are_removed(tmp_path):
    for n in ("case_nii_converting.nrrd", "case_nii_0123456789.nrrd", "case_dcm_0123456789.nrrd", "other.nrrd",
              "case_dcm_0123456789.shell.nrrd", "case_dcm_0123456789.shell.json", "other_case.shell.json"):
        old(tmp_path / n, "x")
    assert cs.remove_leftovers(tmp_path) == ["case_dcm_0123456789.nrrd", "case_dcm_0123456789.shell.json",
                                             "case_dcm_0123456789.shell.nrrd", "case_nii_0123456789.nrrd",
                                             "case_nii_converting.nrrd"]
    assert names(tmp_path) == ["other.nrrd", "other_case.shell.json"]


def test_fresh_cases_of_another_run_are_not_removed(tmp_path):
    old(tmp_path / "case_nii_0123456789.nrrd")
    (tmp_path / "case_nii_abcdefabcd.nrrd").write_text("in use")
    (tmp_path / "case_nii_converting_1a2b3c4d.nrrd").write_text("converting")
    assert cs.remove_leftovers(tmp_path) == ["case_nii_0123456789.nrrd"]
    assert names(tmp_path) == ["case_nii_abcdefabcd.nrrd", "case_nii_converting_1a2b3c4d.nrrd"]


def test_a_case_overwritten_before_placement_is_refused(tmp_path):
    src, dst = tmp_path / "case_nii_converting_1a2b3c4d.nrrd", tmp_path / "case_nii_0123456789.nrrd"
    src.write_text("this case")
    sha = cs.file_sha256(src)
    src.write_text("another case")
    with pytest.raises(cs.CaseError, match="overwrote"):
        cs._place_checked(src, dst, sha)
    assert not dst.exists()


def test_a_case_two_runs_prepared_is_kept_until_both_removed(tmp_path):
    """The same case prepared twice on one cases_dir: the first remove keeps the files for the second
    run; remove_leftovers never deletes a leased case, however old its file."""
    rtp = tmp_path / "runtime.json"
    r = json.loads(DEFAULT_RUNTIME.read_text())
    r.update(cases_dir=str(tmp_path / "cases"))
    rtp.write_text(json.dumps(r))
    cases = tmp_path / "cases"
    cases.mkdir()
    cid = "case_nii_0123456789"
    old(cases / f"{cid}.nrrd")
    a, b = cs.take_lease(cases, cid), cs.take_lease(cases, cid)
    assert cs.remove_leftovers(cases) == []
    assert cs.remove(cid, str(rtp), a) is False and (cases / f"{cid}.nrrd").exists()
    assert [p.name for p in cs.live_leases(cases, cid)] == [f"{cid}.{b}"]
    assert cs.remove(cid, str(rtp), b) is True and not (cases / f"{cid}.nrrd").exists()
    assert cs.live_leases(cases, cid) == []


def test_a_leased_case_is_not_replaced_by_other_content(rt, tmp_path):
    """Preparing the same key again with other options while the first case is leased leaves the first case's file
    unchanged; identical content is still accepted, and once the lease is gone the case may be replaced."""
    vol = tmp_path / "ct.nii.gz"
    first = cs.prepare_volume(vol, runtime=rt, case_key="k:1", spacing_mm=[1, 1, 1])
    nrrd = tmp_path / "cases" / f"{first['case_id']}.nrrd"
    before = nrrd.read_bytes()
    with pytest.raises(cs.CaseError) as e:
        cs.prepare_volume(vol, runtime=rt, case_key="k:1", spacing_mm=[0.5, 0.5, 0.5])
    assert first["case_id"] in str(e.value) and first["lease"] in str(e.value)
    assert nrrd.read_bytes() == before
    assert names(tmp_path / "cases") == [nrrd.name]
    again = cs.prepare_volume(vol, runtime=rt, case_key="k:1", spacing_mm=[1, 1, 1])
    assert again["case_id"] == first["case_id"] and nrrd.read_bytes() == before
    cs.remove(first["case_id"], rt, first["lease"])
    cs.remove(again["case_id"], rt, again["lease"])
    other = cs.prepare_volume(vol, runtime=rt, case_key="k:1", spacing_mm=[0.5, 0.5, 0.5])
    assert other["case_id"] == first["case_id"] and nrrd.read_bytes() != before
    cs.remove(other["case_id"], rt, other["lease"])


def test_a_leased_case_keeps_its_shells_when_other_labels_come(rt, tmp_path):
    """Preparing a leased case again with the same volume but other labels is refused and leaves its shells as they
    were; the same labels reuse them."""
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    a.write_bytes(b"labels a")
    b.write_bytes(b"labels b")
    vol, lm = tmp_path / "ct.nii.gz", {"1": "kidney_left"}
    first = cs.prepare_volume(vol, runtime=rt, case_key="k:2", labels=a, label_map=lm, profile="ct_segments")
    cid = first["case_id"]
    shell = tmp_path / "cases" / f"{cid}.shell.nrrd"
    shell.write_text("first run's shells")
    (tmp_path / "cases" / f"{cid}.shell.json").write_text(json.dumps({"classes": ["kidney_left"]}))
    with pytest.raises(cs.CaseError, match="shells from other labels") as e:
        cs.prepare_volume(vol, runtime=rt, case_key="k:2", labels=b, label_map=lm, profile="ct_segments")
    assert first["lease"] in str(e.value) and shell.read_text() == "first run's shells"
    again = cs.prepare_volume(vol, runtime=rt, case_key="k:2", labels=a, label_map=lm, profile="ct_segments")
    assert again["case_id"] == cid and shell.read_text() == "first run's shells"
    assert again["shell"] == {"classes": ["kidney_left"]}
    cs.remove(cid, rt, again["lease"])
    cs.remove(cid, rt, first["lease"])
    assert list((tmp_path / "cases").glob(f"{cid}*")) == []


def test_an_expired_lease_does_not_protect(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    cid = "case_dcm_0123456789"
    old(cases / f"{cid}.nrrd")
    tok = cs.take_lease(cases, cid)
    p = cases / cs.LEASES / f"{cid}.{tok}"
    t = time.time() - cs.LEASE_TTL_S - 10
    os.utime(p, (t, t))
    assert cs.remove_leftovers(cases) == [f"{cid}.nrrd"]


def test_converters_run_with_their_own_home_and_tmp(rt, tmp_path):
    """Every converter Slicer gets its own HOME / XDG / TMPDIR (inside its temporary folder)."""
    for info in (cs.prepare_dicom(tmp_path / "dicom", runtime=rt), cs.prepare_volume(tmp_path / "ct.nii.gz",
                                                                                      runtime=rt)):
        assert info["home"].endswith("/slicer/home") and info["tmpdir"].endswith("/slicer/tmp")
        assert info["home"] != os.environ.get("HOME") and info["tmpdir"] != os.environ.get("TMPDIR")
        assert cs.remove(info["case_id"], rt, info["lease"]) is True


def test_cli_prepare_and_remove(rt, tmp_path, capsys):
    cs.main(["prepare", "--volume", str(tmp_path / "ct.nii.gz"), "--case-key", "example:0001", "--intensity-offset",
             "-1024", "--spacing", "0.3", "0.3", "0.5", "--lps-identity", "--labels", str(tmp_path / "labels.nii.gz"),
             "--label-map", "1=liver,2=spleen,3=kidney_left,4=kidney_right,6=aorta,7=pancreas", "--profile", "ct_segments", "--runtime", str(rt)])
    out = json.loads(capsys.readouterr().out)
    assert out["case_id"] == "case_nii_09a45f3875" and out["job"]["intensity_offset"] == -1024
    assert isinstance(out["job"]["intensity_offset"], int)
    assert out["shell"]["classes"] == ["aorta", "kidney_left", "kidney_right", "liver", "pancreas", "spleen"]
    cs.main(["remove", out["case_id"], "--lease", out["lease"], "--runtime", str(rt)])
    assert json.loads(capsys.readouterr().out) == {"removed": True}


@pytest.mark.parametrize("bad", [["prepare"], ["prepare", "--dicom", "D", "--volume", "V"],
                                 ["prepare", "--dicom", "D", "--spacing", "1", "1", "1"],
                                 ["prepare", "--dicom", "D", "--case-key", "k"],
                                 ["prepare", "--volume", "V", "--series", "1.2"],
                                 ["prepare", "--volume", "V", "--label-map", "1=kidney_left"],
                                 ["prepare", "--volume", "V", "--labels", "L", "--label-map", "kidney_left"],
                                 ["remove", "other_case"]])
def test_cli_argument_rules(rt, tmp_path, bad):
    args = [str(tmp_path / "dicom") if x == "D" else str(tmp_path / "ct.nii.gz") if x == "V"
            else str(tmp_path / "labels.nii.gz") if x == "L" else x for x in bad]
    with pytest.raises(SystemExit) as e:
        cs.main(args + ["--runtime", str(rt)])
    assert e.value.code in (1, 2)
    assert not (tmp_path / "cases").exists() or names(tmp_path / "cases") == []
