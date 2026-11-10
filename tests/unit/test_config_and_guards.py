"""Config consistency, bridge limits loadable without Slicer, loopback-only runtime, no dev exec anywhere."""

import hashlib
import importlib.util
import json
import re

import pytest

from slicer_mcp.config import PROFILES, REPO, load_protocol, load_runtime

BRIDGE = REPO / "slicer_bridge" / "SlicerMCPBridge"
LIMITS_PY = BRIDGE / "SlicerMCPBridgeLib" / "limits.py"
PROFILE_NAMES = sorted(p.name for p in PROFILES.iterdir() if (p / "protocol.json").is_file())


def load_bridge_limits(path=None):
    spec = importlib.util.spec_from_file_location("slicer_mcp_bridge_limits", LIMITS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if path is not None:
        mod.load(str(path))
    return mod


def test_shipped_profiles():
    assert PROFILE_NAMES == ["ct_basic", "ct_segments"]


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_bridge_and_server_read_the_same_config(name):
    cfg = load_protocol(name)
    lim = load_bridge_limits(cfg.path)
    assert lim.CONFIG_SHA256 == cfg.sha256
    assert lim.RENDER_PRESETS == cfg["render"]["presets"]
    assert lim.DEFAULT_OBS_SIZE == cfg["observation"]["size_px"]
    assert lim.SEGMENTATION == (cfg.get("segmentation") is not None)


def test_bridge_default_is_ct_basic():
    assert load_bridge_limits().CONFIG_SHA256 == load_protocol("ct_basic").sha256


def test_ct_segments_protocol_values():
    lim = load_bridge_limits(load_protocol("ct_segments").path)
    assert set(lim.VIEW_DIRECTIONS) == {"anterior", "posterior", "left", "right", "superior", "inferior"}
    assert set(lim.WINDOW_PRESETS) == {"CT-Soft-Tissue", "CT-Lung", "CT-Bone", "CT-Angio", "CT-Brain"}
    assert lim.SLICE_ZOOM_RANGE == (1.0, 3.0) and lim.SLAB_MM_RANGE == (0.0, 20.0)
    assert lim.RENDER_PRESETS == ["CT-Chest-Contrast-Enhanced", "CT-Bone", "CT-Soft-Tissue", "CT-Coronary-Arteries-2"]
    assert lim.ZOOM_CUMULATIVE_RANGE == (0.25, 4.0)
    assert lim.DEFAULT_RENDER_PRESET in lim.RENDER_PRESETS
    assert set(lim.SEG_CLASS_NAMES) == {"kidney_left", "kidney_right", "liver", "spleen", "aorta", "pancreas"}


@pytest.mark.parametrize("change", [
    lambda c: c["window_level"].__setitem__("default_preset", "nope"),
    lambda c: c["render"].__setitem__("default_preset", "nope"),
    lambda c: c["render"].__setitem__("extra_presets_xml", "presets.xml"),
    lambda c: c["segmentation"]["classes"].append(dict(c["segmentation"]["classes"][0], name="dup")),
    lambda c: c["segmentation"].__setitem__("default_mode", "nope"),
])
def test_bad_config_rejected(tmp_path, change):
    cfg = json.loads((PROFILES / "ct_segments" / "protocol.json").read_text())
    change(cfg)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_bridge_limits(p)


def test_vendored_stock_presets_hash_pinned():
    lim = load_bridge_limits()
    xml = BRIDGE / "presets" / "presets.xml"
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == lim.STOCK_PRESETS_SHA256
    names = set(re.findall(r'name="([^"]+)"', xml.read_text()))
    assert set(load_protocol("ct_basic")["render"]["presets"]) <= names


def test_a_profile_can_bring_its_own_presets(tmp_path):
    """A profile's presets.xml (one stock preset under a new name here) is named and hash-pinned in its
    protocol.json, and its presets join the stock ones."""
    stock_xml = (BRIDGE / "presets" / "presets.xml").read_text()
    prop = re.search(r'<VolumeProperty [^>]*name="CT-Bone"[^>]*/?>', stock_xml).group(0)
    xml = tmp_path / "presets.xml"
    xml.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<MRML>\n  '
                   + prop.replace('name="CT-Bone"', 'name="Example-Preset"') + "\n</MRML>\n")
    cfg = json.loads((PROFILES / "ct_basic" / "protocol.json").read_text())
    cfg["render"].update(extra_presets_xml="presets.xml",
                         extra_presets_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                         presets=cfg["render"]["presets"] + ["Example-Preset"])
    (tmp_path / "protocol.json").write_text(json.dumps(cfg))
    lim = load_bridge_limits(tmp_path / "protocol.json")
    assert "Example-Preset" in lim.RENDER_PRESETS and lim.EXTRA_PRESETS_SHA256 == cfg["render"]["extra_presets_sha256"]


def test_runtime_refuses_non_loopback(tmp_path):
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    rt["bridge_host"] = "0.0.0.0"
    p = tmp_path / "rt.json"
    p.write_text(json.dumps(rt))
    with pytest.raises(ValueError):
        load_runtime(p)
    lim = load_bridge_limits()
    with pytest.raises(ValueError):
        lim.load_runtime(str(p))


def test_server_code_never_uses_dev_exec():
    """The MCP server and the bridge must never import devexec or use the stock exec endpoint."""
    offenders = []
    for root in (REPO / "src", REPO / "slicer_bridge"):
        for f in root.rglob("*.py"):
            t = f.read_text()
            if re.search(r"devexec|dev_exec|/slicer/exec|:2016", t):
                offenders.append(str(f.relative_to(REPO)))
    assert offenders == ["src/slicer_mcp/launch.py"]


def test_bridge_has_no_exec_route():
    t = (BRIDGE / "SlicerMCPBridgeLib" / "server.py").read_text()
    assert "SlicerRequestHandler" not in t
    assert re.search(r"\bexec\(|\beval\(", "".join(p.read_text() for p in (REPO / "slicer_bridge").rglob("*.py"))) is None


def _segcheck():
    spec = importlib.util.spec_from_file_location("slicer_mcp_bridge_segcheck", BRIDGE / "SlicerMCPBridgeLib" /
                                                  "segcheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_segmentation_manifest_is_checked_against_files_and_labels(tmp_path):
    sc = _segcheck()
    shell, labels = tmp_path / "c.shell.nrrd", tmp_path / "c.labels.nrrd"
    shell.write_bytes(b"shell")
    labels.write_bytes(b"labels")
    class_labels = {"liver": 1, "spleen": 2, "aorta": 6}
    hashes = {"shell": sc.file_sha256(shell), "labels": sc.file_sha256(labels)}
    man = {"classes": ["liver", "aorta"], "labels": {"liver": 1, "aorta": 6}, "sha256": dict(hashes)}
    sc.check_manifest(man, hashes, class_labels, shell_values=[0, 1, 6], label_values=[6, 1, 0])
    bad = [
        ({**man, "sha256": {**hashes, "shell": "0" * 64}}, {}, "shell file's sha256"),
        ({**man, "sha256": {**hashes, "labels": "0" * 64}}, {}, "labels file's sha256"),
        ({**man, "classes": ["liver", "heart"]}, {}, "does not have"),
        ({**man, "labels": {"liver": 2, "aorta": 6}}, {}, "differ from the profile"),
        (man, {"shell_values": [0, 1]}, "shell file holds labels [1]"),
        (man, {"label_values": [0, 1, 2, 6]}, "labels file holds labels [1, 2, 6]"),
    ]
    for m, kw, needle in bad:
        with pytest.raises(sc.ManifestMismatch, match=re.escape(needle)):
            sc.check_manifest(m, hashes, class_labels, **kw)
    labels.write_bytes(b"changed")
    with pytest.raises(sc.ManifestMismatch, match="labels file's sha256"):
        sc.check_manifest(man, {"shell": sc.file_sha256(shell), "labels": sc.file_sha256(labels)}, class_labels)


def test_a_profile_report_plugin_must_stay_inside_the_profile(tmp_path):
    from slicer_mcp.config import load_profile
    plugin = (REPO / "examples" / "report_plugin.py").read_text()
    cfg = json.loads((PROFILES / "ct_basic" / "protocol.json").read_text())
    prof = tmp_path / "prof"
    prof.mkdir()
    (tmp_path / "outside.py").write_text(plugin)
    (prof / "report.py").write_text(plugin)
    for name in ("../outside.py", str(tmp_path / "outside.py")):
        (prof / "protocol.json").write_text(json.dumps({**cfg, "mcp": {**cfg["mcp"], "report": name}}))
        with pytest.raises(ValueError, match="outside the profile folder"):
            load_profile(prof)
    (prof / "link.py").symlink_to(tmp_path / "outside.py")
    (prof / "protocol.json").write_text(json.dumps({**cfg, "mcp": {**cfg["mcp"], "report": "link.py"}}))
    with pytest.raises(ValueError, match="outside the profile folder"):
        load_profile(prof)
    (prof / "protocol.json").write_text(json.dumps({**cfg, "mcp": {**cfg["mcp"], "report": "report.py"}}))
    assert load_profile(prof).report.SCHEMA_VERSION == "example-answer-v1"
    assert load_profile(prof, report=str(tmp_path / "outside.py")).report.SCHEMA_VERSION == "example-answer-v1"


def test_runtime_reports_every_missing_key_with_the_file(tmp_path):
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    for k in ("traces_dir", "client_timeout_s"):
        del rt[k]
    del rt["case_prep"]["nrrd_compression"]
    p = tmp_path / "rt.json"
    p.write_text(json.dumps(rt))
    with pytest.raises(ValueError) as e:
        load_runtime(p)
    msg = str(e.value)
    assert str(p) in msg and all(k in msg for k in ("traces_dir", "client_timeout_s", "case_prep.nrrd_compression"))
    del rt["bridge_port"]
    p.write_text(json.dumps(rt))
    with pytest.raises(ValueError, match="bridge_port, traces_dir"):
        load_bridge_limits().load_runtime(str(p))


@pytest.mark.parametrize("script", ["prepare_volume.py", "prepare_case.py"])
def test_converter_scripts_report_the_missing_compression_key(tmp_path, script):
    """The converters run inside Slicer; their runtime check is tested on its own source."""
    import ast
    src = (REPO / "tools" / script).read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "nrrd_compression")
    ns = {"json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), script, "exec"), ns)
    p = tmp_path / "rt.json"
    p.write_text(json.dumps({"case_prep": {}}))
    with pytest.raises(ValueError, match=re.escape(f"{p} lacks required keys: case_prep.nrrd_compression")):
        ns["nrrd_compression"](str(p))
    p.write_text(json.dumps({"case_prep": {"nrrd_compression": 1}}))
    assert ns["nrrd_compression"](str(p)) is True
