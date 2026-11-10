"""docs/tools/<profile>.md is generated from the code (slicer-mcp tools-md) and must not go stale or leak case data."""

import re

import pytest

from slicer_mcp import tools_md
from slicer_mcp.config import REPO, load_profile
from slicer_mcp.session import RUNNER_TOOLS

PROFILES = tools_md.shipped_profiles()


@pytest.fixture(scope="module")
def generated():
    return {name: tools_md.render(name) for name in PROFILES}


@pytest.mark.parametrize("name", PROFILES)
def test_tools_md_is_up_to_date(generated, name):
    out = REPO / "docs" / "tools" / f"{name}.md"
    assert out.read_text() == generated[name], f"{out.name} is stale: run uv run slicer-mcp tools-md"


def test_check_flag(capsys):
    tools_md.main(["--check"])
    assert "up to date" in capsys.readouterr().out


@pytest.mark.parametrize("name", PROFILES)
def test_every_tool_of_the_profile_is_documented(generated, name):
    headings = re.findall(r"^### `(\w+)`", generated[name], re.M)
    assert headings == list(RUNNER_TOOLS + load_profile(name).tools)


@pytest.mark.parametrize("name", PROFILES)
def test_runner_only_tools_are_marked(generated, name):
    text = generated[name]
    for t in RUNNER_TOOLS:
        assert re.search(rf"^### `{t}` \(runner only\)$", text, re.M)
    assert text.count("runner-only: yes") == len(RUNNER_TOOLS)
    for t in load_profile(name).tools:
        assert f"### `{t}` (runner only)" not in text


@pytest.mark.parametrize("name", PROFILES)
def test_no_case_ids_or_file_paths(generated, name):
    text = generated[name]
    assert not re.search(r"case_(\d+|nii_|dcm_)", text)
    assert not re.search(r"(/Users/|/home/|/private/|/tmp/|~/|[A-Za-z]:\\)", text)
    assert str(REPO) not in text
