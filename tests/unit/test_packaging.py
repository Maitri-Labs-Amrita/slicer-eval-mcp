"""The installed package carries every file the server runs outside its own Python (the bridge, the case converters,
the headless setup), the example profiles and the default runtime, under slicer_mcp/_root/ with the checkout's layout."""

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

from slicer_mcp import cases, config, launch
from slicer_mcp.config import REPO

PYPROJECT = REPO / "pyproject.toml"


def _force_include():
    data = tomllib.loads(PYPROJECT.read_text())
    return data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]


def test_a_checkout_is_detected_and_resolves_relative_paths_against_itself():
    assert config.FROM_CHECKOUT and config.base_dir() == REPO


def test_every_force_included_source_exists_and_lands_under_root_with_the_same_layout():
    inc = _force_include()
    assert inc
    for src, dst in inc.items():
        assert (REPO / src).exists(), src
        assert dst == f"slicer_mcp/_root/{src}", (src, dst)


def test_the_files_the_code_runs_are_shipped():
    shipped = [REPO / src for src in _force_include()]
    needed = [launch.START_BRIDGE, cases.DICOM_SCRIPT, cases.VOLUME_SCRIPT, cases.SHELL_SCRIPT,
              REPO / "tools" / "setup_headless.sh", REPO / "tools" / "xvfb_session.py",
              config.DEFAULT_PROFILE / "protocol.json", config.DEFAULT_RUNTIME,
              REPO / "slicer_bridge" / "SlicerMCPBridge" / "presets" / "presets.xml"]
    for f in needed:
        assert f.exists(), f
        assert any(f == s or f.is_relative_to(s) for s in shipped), f"{f} is not in the wheel's force-include"
