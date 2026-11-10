"""Generate docs/tools/<profile>.md, the model-facing tool reference of each shipped profile, from code (never edited
by hand).

    slicer-mcp tools-md            # write docs/tools/<profile>.md for every profile in profiles/
    slicer-mcp tools-md --check    # exit 1 if any is stale (tests/unit/test_tools_md.py)

Everything protocol-shaped comes from the code and the profile: tool lists (the profile's mcp.tools), descriptions,
titles and server instructions (tools/registry.py), argument schemas (schemas/tools.py and the report plugin),
default frames (session.BRIDGE_TOOLS), error types (literals in server.py / session.py) and the state keys (the
bridge's state builder). The only prose kept here is SEMANTICS and ERROR_MEANINGS; render() raises when either is
out of step with the code, so they cannot drift silently.
"""

import argparse
import ast
import difflib
import sys

from slicer_mcp.config import PROFILES, REPO, load_profile
from slicer_mcp.session import BRIDGE_TOOLS, RUNNER_TOOLS, SLICES, TOOL_SCHEMA_VERSION
from slicer_mcp.tools.registry import MODEL_TOOLS, RUNNER_ONLY_META, instructions, tool_specs

OUT_DIR = REPO / "docs" / "tools"
SERVER_PY = REPO / "src" / "slicer_mcp" / "server.py"
SESSION_PY = REPO / "src" / "slicer_mcp" / "session.py"
BRIDGE_ENV_PY = REPO / "slicer_bridge" / "SlicerMCPBridge" / "SlicerMCPBridgeLib" / "env.py"
VIEWS = ("3d", "axial", "sagittal", "coronal")

SEMANTICS = {
    "reset_episode": "runner only: load the case, restore the deterministic initial view, start a new trace (step 0); "
                     "an open episode is closed as abandoned",
    "end_episode": "runner only: closes the open episode as abandoned, before the runner disconnects",
    "observe": "runner only: every view as it is now, nothing changed, logged as a step",
    "reset_view": "restores everything the model can change",
    "bookmark_view": "saves the current view as evidence (view_ref v001, v002, ...) and returns every view",
    "submit_report": "validates the report with the profile's report plugin; accepted -> the episode ends; rejected "
                     "-> every reason, and the episode continues",
    "set_slice": "RAS coordinate along the view axis (axial->S, sagittal->R, coronal->A), clamped to the volume; "
                 "slab_mm = a maximum-intensity slab, kept until set to 0",
    "scroll_slice": "whole slices; stops at the volume edge (warning)",
    "set_window_preset": "grey-scale window/level of the slice views only",
    "jump_to_point": "all three slices through the point (clamped to the voxel-centre range); sets the focus; "
                     "result.hu (mean HU in a small ball)",
    "point_at": "the other two slices through the pixel's RAS point; sets the focus; result.hu (mean HU in a small "
                "ball)",
    "orbit": "trackball about the focal point, relative; from anterior, azimuth +90 -> camera at patient left",
    "set_view_direction": "one of the axis views; keeps focal point and zoom",
    "zoom": "relative factor, > 1 = closer; the cumulative zoom is clamped too; view=slices zooms the slice "
            "views about the focus, which then follow the focus",
    "pan_to": "focal point -> ras (default: the focus), within the volume bounds plus a margin",
    "set_render_preset": "volume-rendering preset; 3D only",
    "set_clip_box": "keeps only the axis-aligned box (default centre: the focus)",
    "set_clip_plane": "removes the side (azimuth, elevation) points to; offset > 0 removes less",
    "clear_clip": "removes the clip; one clip at a time (box and plane replace each other, with a warning)",
    "set_segment_visibility": "shows / hides structure shells (show first, then hide); unknown or absent names are "
                              "errors",
    "set_segment_opacity": "3D opacity of one structure shell or all, clamped to the protocol's range",
    "set_3d_content": "shell_only / ct_and_shell / ct_only: shells, volume rendering and shells, or volume rendering",
    "set_slice_outlines": "draws the shown structures' outlines on the slice views, or not",
    "center_on_segments": "focal point -> the structures' box centre, zoom to fit it; direction kept",
    "measure": "distance in mm between two pixels of one slice view's last frame; a line without text on that view",
}

STATE_SPLATS = {"_segments_state": "segments (a profile with a segmentation)", "_slice_tools_state": "slice_tools"}

ERROR_MEANINGS = {
    "invalid_argument": "the model's fault (bad arguments, a tool outside the profile, an invalid report, a call "
                        "after the report); logged as a step, nothing changes, the episode continues",
    "not_found": "the bridge could not find what the call named; logged as a step, the episode continues",
    "no_episode": "no episode is running (the runner must call reset_episode first)",
    "episode_closed": "the episode has ended (report submitted); no further calls are accepted",
    "environment": "bridge internal or connection error; the runner should abort the episode",
    "stale_render": "internal (EnvSession, logged in the trace): a frozen frame, i.e. byte-identical to the view's "
                    "previous frame while its state changed; the model-facing type is `environment`, so the "
                    "runner aborts",
    "identity_mismatch": "internal (EnvSession, logged in the trace): the bridge answered for another episode or "
                         "another loaded volume; the model-facing type is `environment`, so the runner aborts",
}

SESSION_FRAMES = {"reset_episode": "all", "bookmark_view": "all", "submit_report": "none", "end_episode": "none"}


def _error_types() -> set:
    """error.type literals: server.py `self._error("<type>", ...)`, and in session.py the bridge error types
    passed through (`e.type not in (...)`) and the `{"type": "<literal>"}` errors it builds."""
    found = set()
    for node in ast.walk(ast.parse(SERVER_PY.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_error"
                and node.args and isinstance(node.args[0], ast.Constant)):
            found.add(node.args[0].value)
    for node in ast.walk(ast.parse(SESSION_PY.read_text())):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) and node.left.attr == "type"
                and isinstance(node.comparators[0], ast.Tuple)):
            found |= {e.value for e in node.comparators[0].elts if isinstance(e, ast.Constant)}
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "type" and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    found.add(v.value)
    return found


def _state_keys() -> list:
    """Top-level keys of the bridge's full state dict (read from its source; the bridge runs inside Slicer); a
    `**self._<x>_state()` splat is listed as STATE_SPLATS names it."""
    for node in ast.walk(ast.parse(BRIDGE_ENV_PY.read_text())):
        if (isinstance(node, ast.Dict) and len(node.keys) > 1 and node.keys and isinstance(node.keys[0], ast.Constant)
                and node.keys[0].value == "case_loaded"):
            keys = []
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    keys.append(k.value)
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute):
                    keys.append(STATE_SPLATS[v.func.attr])
            return keys
    raise RuntimeError(f"state dict (first key case_loaded) not found in {BRIDGE_ENV_PY.name}")


def _check_prose():
    tools = set(RUNNER_TOOLS) | set(MODEL_TOOLS)
    for name, d in (("SEMANTICS", SEMANTICS), ("SESSION_FRAMES", {**SESSION_FRAMES, **dict.fromkeys(BRIDGE_TOOLS)})):
        missing, extra = sorted(tools - set(d)), sorted(set(d) - tools)
        if missing or extra:
            raise RuntimeError(f"tools_md.py {name} is out of date: missing {missing}, no longer tools {extra}")
    errs = _error_types()
    missing, extra = sorted(errs - set(ERROR_MEANINGS)), sorted(set(ERROR_MEANINGS) - errs)
    if missing or extra:
        raise RuntimeError(f"tools_md.py ERROR_MEANINGS is out of date: missing {missing}, gone {extra}")


def _cell(s) -> str:
    return str(s).replace("\n", " ").replace("|", "\\|")


def _code(s) -> str:
    return f"`{s}`"


def _num(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _type(s: dict) -> str:
    if "anyOf" in s:
        parts = [_type(x) for x in s["anyOf"]]
        return " or ".join(parts)
    t = s.get("type", "any")
    if t == "array":
        item = _type(s.get("items", {}))
        n_lo, n_hi = s.get("minItems"), s.get("maxItems")
        if n_lo is not None and n_lo == n_hi:
            return f"{item}[{n_lo}]"
        return f"{item}[]"
    if t == "object" and isinstance(s.get("additionalProperties"), dict):
        return f"object of {_type(s['additionalProperties'])}"
    return t


def _bounds(s: dict) -> str:
    out = []
    for x in s.get("anyOf", [s]):
        if "minimum" in x or "maximum" in x:
            out.append(f"{_num(x.get('minimum', '-inf'))} .. {_num(x.get('maximum', 'inf'))}")
        if "minLength" in x or "maxLength" in x:
            out.append(f"length {x.get('minLength', 0)} .. {x.get('maxLength', 'any')}")
        if "pattern" in x:
            out.append(f"pattern {_code(x['pattern'])}")
        if x.get("type") == "array" and x.get("maxItems") is not None and x.get("minItems") != x.get("maxItems"):
            out.append(f"at most {x['maxItems']} items")
    return "; ".join(out)


def _rows(schema: dict, prefix="") -> list:
    """Argument table rows, recursing into nested objects (the report) and arrays of objects."""
    rows = []
    req = set(schema.get("required", ()))
    for name, s in schema.get("properties", {}).items():
        full = prefix + name
        default = _code(repr_json(s["default"])) if "default" in s else ""
        choices = ", ".join(_code(e) for e in s.get("enum", ()))
        rows.append([_code(full), _type(s), "yes" if name in req else "no", _bounds(s), choices, default,
                     s.get("description", "")])
        if "properties" in s:
            rows += _rows(s, full + ".")
        if s.get("type") == "array" and "properties" in s.get("items", {}):
            rows += _rows(s["items"], full + "[].")
    return rows


def repr_json(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return _num(v)


def _table(header, rows) -> list:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return out


def _fenced(text) -> list:
    return ["```text", text, "```"]


def _views(vs) -> str:
    return ", ".join(vs) if vs else "none"


def _frames(spec) -> str:
    if spec.name in BRIDGE_TOOLS:
        frames_for = BRIDGE_TOOLS[spec.name][1]
        per_view = {v: tuple(frames_for({"view": v})) for v in SLICES}
        if "view" in spec.input_schema.get("properties", {}) and all(per_view[v] == (v,) for v in SLICES):
            base = "the slice view named by `view`"
        else:
            base = _views(tuple(frames_for({})))
    else:
        base = {"all": f"every view ({_views(VIEWS)})", "none": "none"}[SESSION_FRAMES[spec.name]]
    if "composite" in spec.input_schema.get("properties", {}):
        base += f"; with `composite: true`, every view ({_views(VIEWS)})"
    return base


def render(profile) -> str:
    _check_prose()
    p = load_profile(profile)
    protocol = p.protocol
    specs = tool_specs(p)
    size = protocol["observation"]["size_px"]
    report = p.report
    L = [f"# Slicer MCP tools: profile `{p.name}` (model-facing reference)", "",
         "<!-- generated by slicer-mcp tools-md from the code and the profile: do not edit. -->",
         "<!-- regenerate: uv run slicer-mcp tools-md (tests/unit/test_tools_md.py checks it) -->", "",
         f"- Tool schema version: `{TOOL_SCHEMA_VERSION}`",
         f"- Protocol config: `{protocol.version}`, sha256 `{protocol.sha256[:12]}`",
         f"- Report: " + ("none (no submit_report)" if report is None else
                          f"`{report.SCHEMA_VERSION}` (built in; a harness passes its own with `serve --report`)"
                          if p.report_sha256 is None else
                          f"`{report.SCHEMA_VERSION}` (plugin `{protocol['mcp']['report']}`, sha256 "
                          f"`{p.report_sha256[:12]}`)"),
         f"- Segmentation: " + (", ".join(c["name"] for c in sorted(protocol["segmentation"]["classes"],
                                                                    key=lambda c: c["label"]))
                                if p.segmentation else "none"), "",
         "Descriptions, instructions and schemas below are exactly what the MCP server sends to the model.", "",
         "## Overview", ""]
    L += _table(["Tools (model)", "Runner-only", "Live views", "Bridge state keys (traced)"],
                [[str(len(p.tools)), ", ".join(_code(t) for t in RUNNER_TOOLS), _views(VIEWS),
                  ", ".join(_code(k) for k in _state_keys())]])
    L += ["", f"Runner-only tools carry the tool `_meta` `{{\"{RUNNER_ONLY_META}\": true}}`; the client must hide "
          "them from the model.", ""]
    errs = _error_types()
    L += ["## Result format", "",
          "Every call returns MCP content: `content[0]` is one text block holding the JSON object "
          "`{step_index, state, result, warnings, error, frames}`; `content[1:]` are the images named in `frames`, "
          f"in that order, each a PNG of {size} x {size} pixels. `state` is the bridge state as is (the trace keeps "
          "the same); `result` holds the tool's own output; `warnings` lists every clamp (`name=v clamped to c "
          "(allowed [lo, hi])`) and edge stop. On any error `is_error` is true, the JSON has the same shape, and "
          "`error` is `{type, message}`.", "", "### Error types", ""]
    L += _table(["error.type", "Meaning"], [[_code(e), ERROR_MEANINGS[e]] for e in ERROR_MEANINGS if e in errs])
    L += ["", "## Server instructions", ""] + _fenced(instructions(p))
    L += ["", "## Tools"]
    for s in specs:
        flags = [f"title: {s.title}", f"read-only: {'yes' if s.read_only else 'no'}",
                 f"runner-only: {'yes' if s.runner_only else 'no'}"]
        L += ["", f"### `{s.name}`" + (" (runner only)" if s.runner_only else ""), "",
              "- " + "\n- ".join(flags), f"- returns frames: {_frames(s)}", f"- semantics: {SEMANTICS[s.name]}", "",
              "Description:", ""] + _fenced(s.description) + [""]
        rows = _rows(s.input_schema)
        if rows:
            L += _table(["Argument", "Type", "Required", "Bounds", "Choices", "Default", "Description"], rows)
        else:
            L += ["No arguments."]
    return "\n".join(L) + "\n"


def shipped_profiles() -> list:
    return sorted(d.name for d in PROFILES.iterdir() if (d / "protocol.json").is_file())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slicer-mcp tools-md", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a docs/tools/<profile>.md differs from the "
                                                        "generated text")
    a = ap.parse_args(argv)
    stale = []
    for name in shipped_profiles():
        out = OUT_DIR / f"{name}.md"
        text = render(name)
        if a.check:
            old = out.read_text() if out.exists() else ""
            if old != text:
                diff = list(difflib.unified_diff(old.splitlines(), text.splitlines(), str(out.relative_to(REPO)),
                                                 "generated", lineterm="", n=0))
                print("\n".join(diff[:40]) + ("\n..." if len(diff) > 40 else ""), file=sys.stderr)
                stale.append(name)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out.relative_to(REPO)} ({len(text)} bytes)")
    if a.check:
        if stale:
            print(f"stale: {stale}; run uv run slicer-mcp tools-md", file=sys.stderr)
            sys.exit(1)
        print("docs/tools/*.md are up to date")


if __name__ == "__main__":
    main()
