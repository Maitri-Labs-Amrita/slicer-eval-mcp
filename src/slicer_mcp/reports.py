"""Reports: what submit_report takes, and what an accepted report writes into the trace.

The report's format belongs to the task, so the harness supplies it: `slicer-mcp serve --report <plugin.py>`, or the
profile's mcp.report (a file in the profile folder). Without either, submit_report takes the built-in JSON report
(BUILTIN): any JSON object, written to the trace as is. A plugin is a Python file with:

    SCHEMA_VERSION        str: recorded in the trace (meta.json report_schema_version)
    DESCRIPTION           str: submit_report's description (the model reads it)
    ARGUMENT_DESCRIPTION  str: the description of submit_report's `report` argument
    Model                 a pydantic BaseModel: the report. It is validated with context={"view_refs": <the
                          episode's bookmarked view_refs>}, so its validators may check cited evidence.
    accept(report)        (result, files) for an accepted report (a Model instance): `result` is the tool result the
                          model reads, `files` maps trace file names ("report.json", ...) to JSON objects.

An invalid report is rejected with every reason in one line (format_errors) and the episode continues; an accepted
one ends it. The plugin owns the task (classes, rules); the server only validates, logs and ends the episode. A
plugin is code: load only plugins you trust. Rendering a report (a PDF, a page) is the harness's job, never the server's.
"""

import json
import re
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

VIEW_REF_PATTERN = r"^v\d{3,}$"
FILE_RE = re.compile(r"^[a-z0-9_]{1,64}\.json$")
RESERVED = ("meta.json", "replay.json")
REQUIRED = ("SCHEMA_VERSION", "DESCRIPTION", "ARGUMENT_DESCRIPTION", "Model", "accept")


def check_plugin(mod):
    """The module, or ValueError naming what it lacks."""
    missing = [k for k in REQUIRED if not hasattr(mod, k)]
    if missing:
        raise ValueError(f"report plugin {getattr(mod, '__file__', mod)} lacks {missing}")
    if not (isinstance(mod.Model, type) and issubclass(mod.Model, BaseModel)):
        raise ValueError("report plugin Model must be a pydantic BaseModel")
    return mod


def check_files(files: dict) -> dict:
    bad = [k for k in files if not FILE_RE.fullmatch(k) or k in RESERVED]
    if bad:
        raise ValueError(f"report plugin trace files must be <name>.json (not {list(RESERVED)}), got {bad}")
    return files


def format_errors(e: ValidationError) -> str:
    """All of a report's problems as one model-facing line: 'field: message; field: message'. No pydantic
    boilerplate (URLs, input echoes, error counts)."""
    parts = []
    for err in e.errors(include_url=False, include_input=False):
        loc = ".".join(str(p) for p in err["loc"]) or "report"
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            msg = "unknown field"
        elif err["type"] == "missing":
            msg = "required field missing"
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


JSON_REPORT_MAX_BYTES = 65536


class JSONReport(BaseModel):
    """Any JSON object of at most JSON_REPORT_MAX_BYTES (compact UTF-8)."""
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _size(cls, v):
        if isinstance(v, dict) and len(json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode()) > \
                JSON_REPORT_MAX_BYTES:
            raise ValueError(f"the report is larger than {JSON_REPORT_MAX_BYTES} bytes")
        return v


def _accept_json(rep: JSONReport):
    return {"accepted": True}, {"report.json": {"report_schema_version": BUILTIN.SCHEMA_VERSION,
                                                "report": rep.model_dump(mode="json")}}


BUILTIN = SimpleNamespace(
    SCHEMA_VERSION="json-object-v1",
    DESCRIPTION=(
        "Submit the final report: one JSON object, with the fields the task asks for. An accepted report ends the "
        f"episode: no further tool calls are accepted. A report that is not a JSON object, or is larger than "
        f"{JSON_REPORT_MAX_BYTES // 1024} KB, is rejected and the episode continues, so you can submit again."),
    ARGUMENT_DESCRIPTION="the report: a JSON object with the fields the task asks for",
    Model=JSONReport,
    accept=_accept_json,
)
