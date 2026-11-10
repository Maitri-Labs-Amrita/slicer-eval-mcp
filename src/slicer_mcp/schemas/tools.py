"""Pydantic argument models for every tool, built from the protocol config bounds (never hard-coded).

Validation rules, the same as the bridge's:
- a wrong argument name, a wrong type (bool is not a number, "3" is not a number), NaN/inf, an unknown
  preset / view / direction name, or a missing required argument -> rejected (ArgsRejected);
- a number that is merely out of range -> clamped to the config bound, with a warning in the bridge's words.
Bounds that depend on the loaded volume (slice positions, RAS points, box centre and maximum size, the
cumulative zoom) are clamped by the bridge, which reports its own warnings.
"""

import math
from typing import Annotated, Any, Literal

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, ValidationError, ValidationInfo,
                      WithJsonSchema, create_model)


CASE_ID_PATTERN = r"^[A-Za-z0-9_]{1,64}$"
SLICE_VIEWS = ("axial", "sagittal", "coronal")
MAX_LABEL = 100


class ArgsRejected(ValueError):
    """The arguments cannot be used (wrong name / type / choice). The message is shown to the model."""


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _clamp_note(name, v, c, lo, hi):
    lo_s = "-inf" if lo is None else f"{lo:g}"
    hi_s = "inf" if hi is None else f"{hi:g}"
    return f"{name}={v:g} clamped to {c:g} (allowed [{lo_s}, {hi_s}])"


def _clamp(lo, hi):
    """AfterValidator: clamp a float to [lo, hi] (either may be None), noting a warning in the context."""
    def fn(v: float, info: ValidationInfo) -> float:
        c = v if lo is None else max(v, lo)
        c = c if hi is None else min(c, hi)
        if c != v and info.context is not None:
            info.context["warnings"].append(_clamp_note(info.field_name, v, c, lo, hi))
        return c
    return fn


def _clamp_each(lo, hi):
    def fn(v: list, info: ValidationInfo) -> list:
        out = []
        for i, x in enumerate(v):
            c = x if lo is None else max(x, lo)
            c = c if hi is None else min(c, hi)
            if c != x and info.context is not None:
                info.context["warnings"].append(_clamp_note(f"{info.field_name}[{i}]", x, c, lo, hi))
            out.append(c)
        return out
    return fn


def _integral(v: float) -> int:
    if v != int(v):
        raise ValueError("must be a whole number of slices")
    return int(v)


def Num(lo=None, hi=None, desc="", default=...):
    """A finite number clamped to [lo, hi]; the bounds are shown in the JSON schema."""
    extra = {k: b for k, b in (("minimum", lo), ("maximum", hi)) if b is not None}
    return (Annotated[float, AfterValidator(_clamp(lo, hi))],
            Field(default, description=desc, json_schema_extra=extra or None))


def OptNum(lo, hi, desc):
    """An optional Num (None when omitted)."""
    extra = {k: b for k, b in (("minimum", lo), ("maximum", hi)) if b is not None}
    return (Annotated[float, AfterValidator(_clamp(lo, hi))] | None,
            Field(None, description=desc, json_schema_extra=extra or None))


def Vec3(desc, lo=None, required=True):
    t = Annotated[list[float], AfterValidator(_clamp_each(lo, None))]
    if required:
        return t, Field(..., min_length=3, max_length=3, description=desc)
    return t | None, Field(None, min_length=3, max_length=3, description=desc)


def Choice(names, desc, default=...):
    return Literal[tuple(names)], Field(default, description=desc)


def Text(desc, max_length, pattern=None, required=True):
    if required:
        return str, Field(..., min_length=1, max_length=max_length, pattern=pattern, description=desc)
    return str | None, Field(None, max_length=max_length, pattern=pattern, description=desc)


COMPOSITE = (bool, Field(False, description="also return the frames of all the other views"))


def build_arg_models(profile) -> dict[str, type[ToolArgs]]:
    """tool name -> its argument model, with bounds and choices from the profile's protocol config. The segment
    tools take the segmentation's classes; submit_report takes the report plugin's Model."""
    protocol = profile.protocol
    wl, cam, clip = protocol["window_level"], protocol["camera"], protocol["clip"]
    px = protocol["observation"]["size_px"] - 1
    m = protocol["slices"]["scroll_max_slices"]
    slice_view = Choice(SLICE_VIEWS, "slice view: axial (position along S), sagittal (R) or coronal (A)")
    ras = Vec3("point [R, A, S] in mm (patient coordinates; geometry in the reset result's case_info)")
    focus_ras = Vec3("point [R, A, S] in mm; omit it to use the focus (where the three slice views cross)",
                     required=False)

    def model(tool, /, __base__=ToolArgs, **fields):
        return create_model(f"{tool}_args", __base__=__base__, **fields)

    st = protocol["slice_tools"]
    slab = {"slab_mm": OptNum(*st["slab_mm_range"], "thickness in mm of a maximum-intensity slab through the view "
                                                     "(0 = an ordinary thin slice); kept while scrolling")}
    zoom_view = {"view": (Literal["3d", "slices"] | None,
                          Field(None, description="'3d' (default) zooms the 3D view; 'slices' zooms the three slice "
                                                  "views about the focus"))}
    delta_desc = f"whole slices to move (+ = towards S / R / A); clamped to +-{m}"
    delta = (Annotated[float, AfterValidator(_integral), AfterValidator(_clamp(-m, m)),
                       WithJsonSchema({"type": "integer", "minimum": -m, "maximum": m})],
             Field(..., description=delta_desc))
    out = {
        "reset_episode": model("reset_episode",
                               case_id=Text("anonymised case id", 64, CASE_ID_PATTERN),
                               client_meta=(dict[str, Any] | None, Field(None, description=(
                                   "episode metadata for meta.json: model_id, model_version, system_prompt_sha256, "
                                   "seed, and anything else")))),
        "end_episode": model("end_episode"),
        "observe": model("observe"),
        "reset_view": model("reset_view"),
        "bookmark_view": model("bookmark_view", label=Text("short label for this view", MAX_LABEL)),
        "set_slice": model("set_slice", view=slice_view,
                           position_mm=Num(desc="position in mm along the view's axis (RAS coordinate); clamped to "
                                                "the volume"),
                           **slab, composite=COMPOSITE),
        "scroll_slice": model("scroll_slice", view=slice_view, delta_slices=delta, composite=COMPOSITE),
        "set_window_preset": model("set_window_preset", name=Choice(list(wl["presets"]), "window preset"),
                                   composite=COMPOSITE),
        "jump_to_point": model("jump_to_point", ras=ras, composite=COMPOSITE),
        "point_at": model("point_at", view=slice_view,
                          x=Num(0, px, "pixel column in that view's last frame (0 = left edge)"),
                          y=Num(0, px, "pixel row in that view's last frame (0 = top edge)"),
                          composite=COMPOSITE),
        "orbit": model("orbit",
                       azimuth_deg=Num(*cam["orbit_azimuth_range"], "rotation about the screen's vertical axis",
                                       default=0.0),
                       elevation_deg=Num(*cam["orbit_elevation_range"], "rotation about the screen's horizontal axis",
                                         default=0.0),
                       roll_deg=Num(*cam["orbit_roll_range"], "rotation about the viewing direction", default=0.0),
                       composite=COMPOSITE),
        "set_view_direction": model("set_view_direction",
                                    name=Choice(cam["view_directions"], "side of the patient to look from"),
                                    composite=COMPOSITE),
        "zoom": model("zoom", factor=Num(*cam["zoom_step_range"], desc="> 1 zooms in, < 1 zooms out"),
                      **zoom_view, composite=COMPOSITE),
        "pan_to": model("pan_to", ras=focus_ras, composite=COMPOSITE),
        "set_render_preset": model("set_render_preset",
                                   name=Choice(protocol["render"]["presets"], "volume-rendering preset"),
                                   composite=COMPOSITE),
        "set_clip_box": model("set_clip_box",
                              center_ras=Vec3("box centre [R, A, S] in mm; omit it to centre the box on the focus "
                                              "(where the three slice views cross)", required=False),
                              size_mm=Vec3("box edge lengths [R, A, S] in mm", lo=clip["box_min_size_mm"]),
                              composite=COMPOSITE),
        "set_clip_plane": model("set_clip_plane",
                                azimuth_deg=Num(*clip["plane_azimuth_range"],
                                                "side to remove: 0 anterior, 90 patient left, -90 patient right, "
                                                "180 posterior"),
                                elevation_deg=Num(*clip["plane_elevation_range"],
                                                  "tilt of that side: 90 superior, -90 inferior"),
                                offset_mm=Num(*clip["plane_offset_range_mm"],
                                              "plane distance from the volume centre towards the removed side"),
                                composite=COMPOSITE),
        "clear_clip": model("clear_clip", composite=COMPOSITE),
        "measure": model(
            "measure", view=slice_view,
            x1=Num(0, px, "first point: pixel column in that view's last frame (0 = left edge)"),
            y1=Num(0, px, "first point: pixel row (0 = top edge)"),
            x2=Num(0, px, "second point: pixel column"),
            y2=Num(0, px, "second point: pixel row"),
            composite=COMPOSITE),
    }
    if profile.report is not None:
        report = (Annotated[dict[str, Any], WithJsonSchema(_inlined(profile.report.Model.model_json_schema()))],
                  Field(..., description=profile.report.ARGUMENT_DESCRIPTION))
        out["submit_report"] = model("submit_report", report=report)
    seg = protocol.get("segmentation")
    if seg is not None:
        seg_names = [c["name"] for c in sorted(seg["classes"], key=lambda c: c["label"])]
        seg_name = Literal[tuple(seg_names)]
        seg_or_all = Literal[tuple(seg_names) + ("all",)]
        out.update({
            "set_segment_visibility": model("set_segment_visibility",
                                            show=(list[seg_or_all], Field([], max_length=len(seg_names) + 1,
                                                                          description="structures to show ('all' = every one)")),
                                            hide=(list[seg_or_all], Field([], max_length=len(seg_names) + 1,
                                                                          description="structures to hide ('all' = every one); "
                                                                                      "applied after show")),
                                            composite=COMPOSITE),
            "set_segment_opacity": model("set_segment_opacity",
                                         name=(seg_or_all, Field(..., description="structure, or 'all'")),
                                         opacity=Num(*seg["opacity_range"], "0 = invisible, 1 = opaque"),
                                         composite=COMPOSITE),
            "set_3d_content": model("set_3d_content", mode=Choice(seg["modes"], "what the 3D view shows"),
                                    composite=COMPOSITE),
            "set_slice_outlines": model("set_slice_outlines",
                                        visible=(bool, Field(..., description="true = draw the shells on the slices")),
                                        composite=COMPOSITE),
            "center_on_segments": model("center_on_segments",
                                        names=(list[seg_name] | None, Field(None, min_length=1,
                                                                            max_length=len(seg_names),
                                                                            description="structures to fit; omit for every "
                                                                                        "shown structure")),
                                        composite=COMPOSITE),
        })
    return out


def _inlined(s: dict) -> dict:
    """A JSON schema with its $defs inlined (no $ref left) and no auto-generated titles."""
    defs = s.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline({**defs[node["$ref"].rsplit("/", 1)[1]],
                               **{k: v for k, v in node.items() if k != "$ref"}})
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node
    return _strip_titles(inline(s))


def _strip_titles(node):
    if isinstance(node, dict):
        for k in list(node):
            if k == "title" and isinstance(node[k], str):
                del node[k]
            else:
                _strip_titles(node[k])
    elif isinstance(node, list):
        for x in node:
            _strip_titles(x)
    return node


def input_schema(model: type[ToolArgs]) -> dict:
    """JSON schema for the MCP tool listing (object root, no auto-generated titles)."""
    s = _inlined(model.model_json_schema())
    s.setdefault("properties", {})
    s["additionalProperties"] = False
    return s


def validate_args(model: type[ToolArgs], raw: Any) -> tuple[dict, list]:
    """(args_after_validation, warnings) or raise ArgsRejected with a message for the model.
    `composite` is dropped when it was not requested, and an omitted point (pan_to, set_clip_box)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ArgsRejected("arguments must be a JSON object")
    ctx = {"warnings": []}
    try:
        m = model.model_validate(raw, context=ctx)
    except ValidationError as e:
        raise ArgsRejected("; ".join(_fmt(err) for err in e.errors(include_url=False))) from None
    out = m.model_dump()
    if not out.get("composite"):
        out.pop("composite", None)
    for k in ("ras", "center_ras", "names", "slab_mm", "view"):
        if k in out and out[k] is None:
            del out[k]
    for v in out.values():
        if isinstance(v, float) and not math.isfinite(v):
            raise ArgsRejected("numbers must be finite")
    return out, ctx["warnings"]


def _fmt(err) -> str:
    loc = ".".join(str(p) for p in err["loc"]) or "arguments"
    msg = err["msg"]
    if err["type"] == "extra_forbidden":
        msg = "unknown argument"
    elif msg.startswith("Value error, "):
        msg = msg[len("Value error, "):]
    return f"{loc}: {msg}"
