"""An example report plugin (slicer_mcp/reports.py): a short free-text answer with optional evidence, schema
example-answer-v1. A task harness writes its own and passes it with `slicer-mcp serve --report <file>`; the unit tests
use this one.

    uv run slicer-mcp serve --profile ct_basic --report examples/report_plugin.py
"""

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

SCHEMA_VERSION = "example-answer-v1"
VIEW_REF_PATTERN = r"^v\d{3,}$"
DESCRIPTION = (
    "Submit the final answer. An accepted report ends the episode: no further tool calls are accepted. An invalid "
    "report is rejected with every reason at once and the episode continues, so you can correct it and submit "
    "again. answer is your answer to the task in plain text (at most 4000 characters); evidence (optional) lists "
    "view_refs from bookmark_view.")
ARGUMENT_DESCRIPTION = "the answer and, optionally, its evidence (see the schema)"


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=4000, description="the answer to the task")
    evidence: list[str] = Field(default_factory=list, max_length=12, description="view_refs from bookmark_view")

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v, info: ValidationInfo):
        bad = sorted({r for r in v if not re.match(VIEW_REF_PATTERN, r)})
        known = (info.context or {}).get("view_refs")
        unknown = sorted({r for r in v if known is not None and r not in known and r not in bad})
        if bad or unknown:
            raise ValueError("; ".join(([f"bad view_refs {bad}"] if bad else [])
                                       + ([f"cites unknown view_refs {unknown}"] if unknown else [])))
        return v


Model = Answer


def accept(rep: Answer):
    """The result the model reads, and report.json for the trace."""
    return {"accepted": True}, {"report.json": {**rep.model_dump(mode="json"), "report_schema_version": SCHEMA_VERSION}}
