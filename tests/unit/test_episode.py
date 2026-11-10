import pytest

from slicer_mcp.config import load_profile
from slicer_mcp.episode import EpisodeError, EpisodeSession

from tests.unit.conftest import EXAMPLE_REPORT, SEG
from tests.unit.fakes import answer_report


def test_bookmark_refs():
    ep = EpisodeSession("case_test")
    b1 = ep.bookmark("liver dome", 3, {"camera": {}}, {"3d": b"png"})
    b2 = ep.bookmark("left kidney", 4, {}, {})
    assert (b1.view_ref, b2.view_ref) == ("v001", "v002")
    with pytest.raises(EpisodeError):
        ep.bookmark("", 7, {}, {})


def test_submit_report_validates_and_ends():
    ep = EpisodeSession("case_test", report_model=load_profile(SEG, report=EXAMPLE_REPORT).report.Model)
    ep.bookmark("axial overview", 1, {}, {})
    with pytest.raises(EpisodeError):
        ep.submit_report({**answer_report(), "extra": 1})
    assert not ep.done
    with pytest.raises(EpisodeError, match="unknown view_refs"):
        ep.submit_report(answer_report(["v002"]))
    r = ep.submit_report(answer_report(["v001"]))
    assert ep.done and r.evidence == ["v001"]
    with pytest.raises(EpisodeError):
        ep.bookmark("late", 9, {}, {})


def test_no_report_model_refuses():
    with pytest.raises(EpisodeError, match="takes no report"):
        EpisodeSession("case_test").submit_report({"answer": "x"})
