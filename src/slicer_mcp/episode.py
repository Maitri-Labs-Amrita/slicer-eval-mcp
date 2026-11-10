"""Server-side episode bookkeeping: bookmarks (view_refs) and the submitted report.

None of this touches the Slicer scene. bookmark_view stores the current state and frames, which the caller fetches
from the bridge. submit_report validates the report with the profile's report plugin (reports.py).
"""

from dataclasses import dataclass, field

from pydantic import ValidationError

from slicer_mcp.reports import format_errors

MAX_LABEL = 100


class EpisodeError(ValueError):
    pass


@dataclass
class Bookmark:
    view_ref: str
    label: str
    step: int
    state: dict
    frames: dict


@dataclass
class EpisodeSession:
    case_id: str
    report_model: type | None = None
    bookmarks: dict = field(default_factory=dict)
    report: object | None = None

    @property
    def done(self) -> bool:
        return self.report is not None

    def _check_open(self):
        if self.done:
            raise EpisodeError("episode already ended (report submitted)")

    def bookmark(self, label: str, step: int, state: dict, frames: dict) -> Bookmark:
        self._check_open()
        label = (label or "").strip()
        if not label or len(label) > MAX_LABEL:
            raise EpisodeError(f"label must be 1..{MAX_LABEL} characters")
        ref = f"v{len(self.bookmarks) + 1:03d}"
        b = Bookmark(ref, label, step, state, frames)
        self.bookmarks[ref] = b
        return b

    def validate_report(self, report: dict):
        """The validated report (a report_model instance); the episode stays open. Raises EpisodeError."""
        self._check_open()
        if self.report_model is None:
            raise EpisodeError("this profile takes no report")
        try:
            return self.report_model.model_validate(report, context={"view_refs": set(self.bookmarks)})
        except ValidationError as e:
            raise EpisodeError(f"invalid report: {format_errors(e)}") from None

    def end(self, report):
        """Accept a validated report: the episode is done. The session calls this only after the report plugin
        accepted it and its trace files are written."""
        self._check_open()
        self.report = report

    def submit_report(self, report: dict):
        """Validate and accept the report (ends the episode). Raises EpisodeError, and the episode stays open."""
        r = self.validate_report(report)
        self.end(r)
        return r
