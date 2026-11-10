"""Integration fixtures: talk to the live Slicer MCP bridge (SLICER_URL or config/runtime.json).

The profile is the shipped one whose protocol.json the bridge runs (its sha256 in /bridge/ping); the segment tests
need a profile with a segmentation (`slicer-mcp launch --profile ct_segments`) and skip on a profile without a segmentation.

The suite runs on a temporary case prepared at the start of the session and removed at the end, from your own CT:
    SLICER_MCP_TEST_VOLUME     the volume file (required unless SLICER_MCP_TEST_CASE is set)
    SLICER_MCP_TEST_PREPARE    optional JSON of extra slicer_mcp.cases.prepare_volume arguments, e.g.
                               '{"spacing_mm": [0.3, 0.3, 0.75], "intensity_offset": -1024, "lps_identity": true}'
    SLICER_MCP_TEST_LABELS     a label map on the volume's grid (only used with a segmentation profile), with
    SLICER_MCP_TEST_LABEL_MAP  its codes -> the profile's class names, e.g. 1=liver,2=spleen,3=kidney_left,4=kidney_right,5=aorta,6=pancreas
SLICER_MCP_TEST_CASE=<id> uses an already prepared case instead.
"""

import json
import os
from pathlib import Path

import pytest

from slicer_mcp.bridge_client import BridgeClient
from slicer_mcp.config import PROFILES, REPO, load_profile, load_runtime

TEST_VOLUME = os.environ.get("SLICER_MCP_TEST_VOLUME", "")
TEST_LABELS = os.environ.get("SLICER_MCP_TEST_LABELS", "")
RESULTS_DIR = REPO / "data" / "dev" / "test_results"


def profile_of(sha256):
    for d in sorted(PROFILES.iterdir()):
        if (d / "protocol.json").is_file():
            p = load_profile(d.name)
            if p.protocol.sha256 == sha256:
                return p
    return None


_CLIENTS = []


@pytest.fixture(scope="session")
def bridge():
    client = BridgeClient(connect_retries=0)
    try:
        info = client.ping()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Slicer MCP bridge not reachable at {client.url}: {e}")
    prof = profile_of(info["config_sha256"])
    assert prof is not None, f"the bridge runs protocol {info['config_sha256'][:12]}, not a shipped profile's"
    client.obs_size = prof.protocol["observation"]["size_px"]
    client.profile = prof
    _CLIENTS.append(client)
    return client


@pytest.fixture(scope="session")
def profile(bridge):
    return bridge.profile


@pytest.fixture(scope="session")
def protocol(profile):
    return profile.protocol


@pytest.fixture(scope="session")
def make_report():
    """make_report(view_ref=None) -> a valid report for the server's report: the built-in JSON report (the live
    server runs without --report) takes any JSON object, so this one also fits the example plugin."""
    from tests.unit.fakes import answer_report

    def _make(ref=None):
        return answer_report([ref] if ref else [])
    return _make


@pytest.fixture(scope="session")
def runtime():
    return load_runtime()


@pytest.fixture(autouse=True)
def _release_episode(monkeypatch):
    """An episode owns the bridge until it is released (or idle for the 900 s lease). Release the shared
    client's episode, and that of every in-process server a test built (its own client), after each test, so the next
    test (or a slicer-mcp serve subprocess) may reset it."""
    from slicer_mcp import server as server_mod
    real, built = server_mod.build, []

    def build(*a, **k):
        srv = real(*a, **k)
        built.append(srv.session.client)
        return srv
    monkeypatch.setattr(server_mod, "build", build)
    yield
    for c in _CLIENTS + built:
        c.release()


@pytest.fixture(scope="session")
def case_id(bridge, profile, runtime):
    """The session's case: a temporary case from TEST_VOLUME (+ TEST_LABELS shells), removed at the end."""
    if os.environ.get("SLICER_MCP_TEST_CASE"):
        yield os.environ["SLICER_MCP_TEST_CASE"]
        return
    from slicer_mcp import cases
    if not TEST_VOLUME or not Path(TEST_VOLUME).is_file():
        pytest.skip(f"no test volume {TEST_VOLUME!r}: set SLICER_MCP_TEST_VOLUME or SLICER_MCP_TEST_CASE")
    kw = json.loads(os.environ.get("SLICER_MCP_TEST_PREPARE") or "{}")
    label_map = (cases.parse_label_map(os.environ["SLICER_MCP_TEST_LABEL_MAP"])
                 if os.environ.get("SLICER_MCP_TEST_LABEL_MAP") else None)
    if profile.segmentation and label_map and TEST_LABELS and Path(TEST_LABELS).is_file():
        kw.update(labels=TEST_LABELS, label_map=label_map, profile=profile.dir)
    with cases.temporary(volume=TEST_VOLUME, remove_stale=False, **kw) as info:
        yield info["case_id"]


@pytest.fixture(scope="session")
def alias_cases(runtime, case_id):
    """Same voxels (and shells, when the case has them) under two names (case_alias and LEAK_TEST) for the name-leak
    test. Symlinks in data/."""
    d = Path(runtime["cases_dir"])
    made = []
    for name in ("case_alias", "LEAK_TEST"):
        for suffix in (".nrrd", ".shell.nrrd", ".labels.nrrd", ".shell.json"):
            p = d / f"{name}{suffix}"
            if (d / f"{case_id}{suffix}").exists() and not p.exists():
                p.symlink_to(f"{case_id}{suffix}")
                made.append(p)
    yield ("case_alias", "LEAK_TEST")
    for p in made:
        p.unlink()


@pytest.fixture(scope="session")
def record():
    """Write a named JSON result under data/dev/test_results/ (the logged numbers for checkpoints)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def _write(name, obj):
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2))
        print(f"\n[{name}] {json.dumps(obj)}")
    return _write
