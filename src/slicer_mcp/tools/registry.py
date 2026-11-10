"""Tool registry: MCP tool definitions from the profile's protocol config.

The descriptions are part of the task protocol (the model reads them). Values (bounds, preset lists, view
directions, the segmentation's classes and colours) come from the profile's protocol.json, and submit_report's text
from its report plugin. Any change to a name, schema or description needs a TOOL_SCHEMA_VERSION bump. Nothing here
may name a case: descriptions are identical for every case.
"""

from dataclasses import dataclass

from slicer_mcp.schemas.tools import build_arg_models, input_schema
from slicer_mcp.session import RUNNER_TOOLS, TOOL_SCHEMA_VERSION

RUNNER_ONLY_META = "slicer_mcp/runner_only"
READ_ONLY = set()
TITLES = {
    "reset_episode": "Reset episode (runner only)", "end_episode": "End episode (runner only)",
    "observe": "Observe (runner only)",
    "reset_view": "Reset view", "bookmark_view": "Bookmark view",
    "submit_report": "Submit report", "set_slice": "Set slice position",
    "scroll_slice": "Scroll slice", "set_window_preset": "Set window preset",
    "jump_to_point": "Jump slices to point", "point_at": "Point at pixel", "orbit": "Orbit camera",
    "set_view_direction": "Set view direction", "zoom": "Zoom", "pan_to": "Pan to point",
    "set_render_preset": "Set render preset", "set_clip_box": "Clip to box", "set_clip_plane": "Clip with plane",
    "clear_clip": "Clear clip",
    "set_segment_visibility": "Show or hide structures", "set_segment_opacity": "Set structure opacity",
    "set_3d_content": "Set 3D content", "set_slice_outlines": "Slice outlines",
    "center_on_segments": "Centre on structures",
    "measure": "Measure distance",
}
MODEL_TOOLS = ("reset_view", "bookmark_view", "submit_report", "set_slice", "scroll_slice", "set_window_preset",
               "jump_to_point", "point_at", "orbit", "set_view_direction", "zoom", "pan_to", "set_render_preset",
               "set_clip_box", "set_clip_plane", "clear_clip", "set_segment_visibility", "set_segment_opacity",
               "set_3d_content", "set_slice_outlines", "center_on_segments", "measure")
SEG_TOOLS = ("set_segment_visibility", "set_segment_opacity", "set_3d_content", "set_slice_outlines",
             "center_on_segments")
_AXIS_WORD = {(1, 0, 0): "patient right", (-1, 0, 0): "patient left", (0, 1, 0): "anterior",
              (0, -1, 0): "posterior", (0, 0, 1): "superior", (0, 0, -1): "inferior"}
VIEWS_PHRASE = "the 3D view and the axial, sagittal and coronal slice views"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_schema: dict
    model: type
    read_only: bool
    runner_only: bool


def _g(x) -> str:
    return f"{x:g}"


def describe(profile) -> dict[str, str]:
    """tool -> model-facing description for this profile."""
    protocol = profile.protocol
    wl, cam, clip, render = protocol["window_level"], protocol["camera"], protocol["clip"], protocol["render"]
    views = VIEWS_PHRASE
    m = protocol["slices"]["scroll_max_slices"]
    px = protocol["observation"]["size_px"] - 1
    az, el, ro = cam["orbit_azimuth_range"], cam["orbit_elevation_range"], cam["orbit_roll_range"]
    zs, zc = cam["zoom_step_range"], cam["zoom_cumulative_range"]
    ups = []
    for name, d in cam["view_directions"].items():
        up = _AXIS_WORD.get(tuple(round(x) for x in d["view_up"]), str(d["view_up"]))
        ups.append(f"{name} ({up} up)")
    st = protocol["slice_tools"]
    wl_list = ", ".join(f"{n} (W {_g(w)} / L {_g(l)})" for n, (w, l) in wl["presets"].items())
    wl_note = f"{wl['note']} " if wl.get("note") else ""
    sz, sl = st["zoom_range"], st["slab_mm_range"]
    hu_note = f" result.hu is the mean CT value (HU) within {_g(st['hu_ball_radius_mm'])} mm of the point."
    zoom_note = (" While the slices are zoomed (zoom view slices), every slice view is re-centred on the new focus, "
                 "so point at the new frames.")
    has_bookmarks = "bookmark_view" in profile.tools
    seg = protocol.get("segmentation")
    has_seg = seg is not None
    d = {
        "reset_episode": (
            "RUNNER ONLY: never offer this tool to the model. Loads the anonymised case, brings the viewer to its "
            "deterministic initial state and starts a new episode and trace (step 0). An open episode is closed as "
            "abandoned. client_meta (model_id, model_version, system_prompt_sha256, seed, ...) goes to meta.json."
            + (f" Returns {views} at the initial view, and the volume geometry in result.case_info: voxel "
               "dimensions and spacing (mm), RAS bounds (mm), the centre point, and each slice view's position "
               "range (axial along S, sagittal along R, coronal along A).")
            + (" With segments (a profile with a segmentation): also loads the case's segmentation shells; "
               "case_info.segments lists the structures and their colours." if has_seg else "")),
        "end_episode": (
            "RUNNER ONLY: never offer this tool to the model. Closes the open episode as abandoned (a submitted one "
            "is already closed). Call it before disconnecting. Returns the last trace "
            "directory and status in result."),
        "observe": (
            "RUNNER ONLY: never offer this tool to the model. Captures " + views + " as they are now, "
            "changing nothing, and logs them as a step (the first observation after the runner's own calls). "
            "Returns the current state and images."),
        "reset_view": (
            "Restore the initial view: camera, zoom, render preset, clipping, slice positions and window/level, "
            "the slice zoom, slabs and measure line"
            + (", and the structures (all shown, opaque, the 3D content and slice outlines at their start values). "
               if has_seg else ". ")
            + f"Returns {views}."),
        "bookmark_view": (
            "Save the current view as evidence and return its view_ref (v001, v002, ...). Cite view_refs in the "
            "report's evidence. Returns the current images of " + views + ", so it also shows you every view at "
            "once."),
        "set_slice": (
            "Move one slice view to a position in mm along its own axis, in patient (RAS) coordinates: axial -> S "
            "(larger = more superior), sagittal -> R (larger = further to the patient's right), coronal -> A "
            "(larger = more anterior). The position is clamped to the volume (ranges in the reset result's "
            "case_info)."
            + f" slab_mm ({_g(sl[0])} to {_g(sl[1])}) turns the view into a maximum-intensity slab of that "
              f"thickness: each pixel shows the brightest value through the slab. The slab stays on that view while "
              f"you scroll or move it, until you set slab_mm 0; omit slab_mm to keep it as it is."
            + " Returns that slice view."),
        "scroll_slice": (
            f"Scroll one slice view by a whole number of slices (positive = towards S / R / A; one slice = the "
            f"view's slice spacing). At most +-{m} slices per call (clamped). The scroll stops at the volume edge, "
            f"with a warning. Returns that slice view and how far it moved (moved_slices, moved_mm)."),
        "set_window_preset": (
            f"Set the grey-scale window/level of the slice views to a preset: {wl_list}. "
            + "Slice views only; the 3D rendering is not affected. "
            + wl_note
            + "Returns the three slice views."),
        "jump_to_point": None,
        "point_at": None,
        "orbit": (
            f"Rotate the 3D camera about its focal point, relative to the current view, in degrees: azimuth about "
            f"the screen's vertical axis, then elevation about the screen's horizontal axis, then roll about the "
            f"viewing direction. It is a trackball, so going over the top is fine. From the anterior view, azimuth "
            f"+90 moves the camera to the patient's left and elevation +90 looks from above. Per call: azimuth "
            f"+-{_g(az[1])}, elevation +-{_g(el[1])}, roll +-{_g(ro[1])} (clamped). Returns the 3D view."),
        "set_view_direction": (
            "Look at the patient from one side: " + ", ".join(ups) + ". Keeps the current focal point and zoom; "
            "only the viewing direction and the screen's up direction change. Returns the 3D view."),
        "zoom": (
            f"Zoom the 3D view by a factor relative to the current zoom (> 1 = closer, < 1 = farther), "
            f"{_g(zs[0])} to {_g(zs[1])} per call. The total zoom relative to the initial view stays within "
            f"[{_g(zc[0])}, {_g(zc[1])}] (clamped, with a warning). Returns the 3D view."
            + f" With view 'slices' it zooms the three slice views instead, about the focus (where they cross): "
              f"the total slice zoom stays within [{_g(sz[0])}, {_g(sz[1])}] (clamped, with a warning), the slice "
              f"views are re-centred on the focus whenever point_at or jump_to_point moves it "
              f"(set_slice and scroll_slice do not pan them), and reset_view (or zooming back to 1) shows the whole "
              f"slice again. Returns the three slice views."),
        "pan_to": (
            f"Move the 3D camera's focal point (the image centre and the centre of rotation) to a point [R, A, S] "
            f"in mm, or, with no point, to the focus (where the three slice views cross); the viewing direction "
            f"and zoom are kept. Clamped to the volume bounds +-"
            f"{_g(cam['pan_margin_mm'])} mm. Returns the 3D view."),
        "set_render_preset": (
            "Change the 3D volume-rendering preset (the opacity and colour mapping of CT values): "
            + ", ".join(render["presets"]) + ". 3D view only. Returns the 3D view."),
        "set_clip_box": None,
        "set_clip_plane": None,
        "clear_clip": "Remove the active clip box or clip plane. Returns the 3D view.",
        "measure": (
            f"Measure the distance between two pixels of one slice view's most recent image (x1, y1, x2, y2: "
            f"columns and rows, 0 to {_g(px)}, as in point_at). result.distance_mm is the distance in mm between "
            f"their patient positions. A {st['measure_color_name']} line joins them on that view (it has no text) "
            f"until the next measure or reset_view; it disappears when the view moves off it. Returns that slice "
            f"view."),
    }
    seg_clip = " It cuts the structure shells too." if has_seg else ""
    seg_point = ""
    if has_seg:
        seg_classes = sorted(seg["classes"], key=lambda c: c["label"])
        seg_names = ", ".join(c["name"] for c in seg_classes)
        seg_point = (" result.inside_segment names the structure (" + seg_names + ") whose segmented blood pool "
                     "contains the point, or null.")
    d["jump_to_point"] = (
        "Move all three slice views through a point [R, A, S] in mm: sagittal takes R, coronal takes A, axial "
        "takes S. Each coordinate is clamped to the volume. The point becomes the focus (where the three "
        "slice views cross)." + seg_point + hu_note + zoom_note + " Returns the three slice views.")
    d["point_at"] = (
        f"Point at a pixel of a slice view's most recent image: view, then x (column, 0 = left edge) and y "
        f"(row, 0 = top edge), in that image's pixels (0 to {_g(px)}). The pixel's patient position becomes "
        f"the focus: the other two slice views move through it, and the view you pointed at stays where it "
        f"is. result.ras is the point "
        f"[R, A, S] in mm." + seg_point + hu_note + zoom_note
        + " pan_to and set_clip_box use the focus when you give them no point. "
        + "Returns the three slice views.")
    d["set_clip_box"] = (
        f"Crop the 3D rendering to an axis-aligned box: keep only what lies inside the box centred at "
        f"center_ras [R, A, S] with edge lengths size_mm [R, A, S], in mm. Omit center_ras to centre the box on "
        f"the focus (where the three slice views cross). The centre is clamped to the volume; "
        f"each size to at least {_g(clip['box_min_size_mm'])} mm and at most twice the volume's extent. One "
        f"clip at a time: this replaces a clip plane (with a warning). 3D view only; the slice views are not "
        f"cut.{seg_clip} Returns the 3D view.")
    d["set_clip_plane"] = (
        f"Cut away one side of the 3D rendering with a plane. azimuth_deg and elevation_deg name the side "
        f"that is REMOVED: azimuth 0 = anterior, +90 = patient left, -90 = patient right, +-180 = posterior; "
        f"elevation +90 = superior, -90 = inferior (tilting the side towards S or I). The plane is "
        f"perpendicular to that direction, at offset_mm from the volume centre along it: 0 cuts through the "
        f"centre, a positive offset moves the cut outward (removes less), a negative one inward (removes more). "
        f"Ranges: azimuth +-{_g(clip['plane_azimuth_range'][1])}, elevation +-"
        f"{_g(clip['plane_elevation_range'][1])}, offset +-{_g(clip['plane_offset_range_mm'][1])} mm (clamped). "
        f"One clip at a time: this replaces a clip box (with a warning). 3D view only.{seg_clip} Returns the 3D "
        f"view.")
    if profile.report is not None:
        d["submit_report"] = profile.report.DESCRIPTION if has_bookmarks else (
            profile.report.DESCRIPTION + " (bookmark_view is not offered: leave any evidence empty.)")
    if has_seg:
        legend = ", ".join(f"{c['name']} {c['color_name']}" for c in seg_classes)
        lo_o, hi_o = seg["opacity_range"]
        first = seg_classes[0]["name"]
        example = seg_classes[1]["name"] if len(seg_classes) > 1 else first
        bad_name = ("A name that is not one of " + seg_names + ", or not segmented in this case (the reset result's "
                    "case_info.segments lists them), is an error and changes nothing. ")
        d.update({
            "set_segment_visibility": (
                f"Show or hide individual structure shells ({legend}). show is applied first, then hide, so show "
                f"['all'] with hide ['{example}'] shows everything but {example}; a name in both is an error. Hidden "
                f"structures disappear from the 3D view and from the slice outlines. Showing a shown structure or "
                f"hiding a hidden one gives a warning. " + bad_name + "Returns the 3D view and the three slice views."),
            "set_segment_opacity": (
                f"Set the 3D opacity of one structure shell, or of all ('all'), from {_g(lo_o)} (invisible) to "
                f"{_g(hi_o)} (opaque) (clamped, with a warning): a see-through shell shows what lies behind or inside "
                f"it. The slice outlines are not affected. " + bad_name + "Returns the 3D view."),
            "set_3d_content": (
                "Choose what the 3D view shows: shell_only (the structure shells alone"
                + (f": {seg['shell_only_note']}" if seg.get("shell_only_note") else "")
                + "), ct_and_shell (the shells inside the CT volume rendering) or ct_only (the volume rendering "
                "alone, as without segments). The camera, clipping and render preset are kept. Returns the 3D view."),
            "set_slice_outlines": (
                "Turn the structure outlines on the axial, sagittal and coronal slices on or off. Each shown "
                "structure's segmented blood is outlined in its colour where the slice cuts it: one line at the edge "
                "of its blood (where two structures touch, both lines run along the boundary between them). Returns "
                "the three slice views."),
            "center_on_segments": (
                "Centre the 3D view on structures: the focal point moves to the centre of their bounding box and the "
                "camera moves along the current viewing direction until the box fills the view (the total zoom is "
                f"clamped to [{_g(zc[0])}, {_g(zc[1])}], with a warning). names omitted = every shown structure. "
                + bad_name + "result.center_ras and result.zoom give the new centre and zoom. Returns the 3D view."),
        })
    return d


def instructions(profile) -> str:
    """Server instructions (MCP initialize): the tool contract only. How to use the viewer in an examination is the
    client's system prompt; the tool descriptions give each tool's arguments, limits and refusals."""
    return (f"{profile.instructions_title}\n\n"
            "How the tools work\n"
            "- Coordinates are RAS (patient Right, Anterior, Superior) in millimetres; angles in degrees.\n"
            "- Every tool returns one JSON text block {step_index, state, result, warnings, error, frames} followed "
            "by the images named in frames, in that order.\n"
            "- Out-of-range numbers are clamped and reported in warnings; invalid arguments return an error and "
            "change nothing.\n"
            "- Each tool's description gives its arguments, defaults, limits and refusals.")


def tool_specs(profile) -> list[ToolSpec]:
    """The profile's tools: the runner-only ones first, then the profile's allowlist in its order."""
    models = build_arg_models(profile)
    desc = describe(profile)
    return [ToolSpec(t, TITLES[t], desc[t], input_schema(models[t]), models[t], t in READ_ONLY, t in RUNNER_TOOLS)
            for t in RUNNER_TOOLS + tuple(profile.tools)]


__all__ = ["ToolSpec", "tool_specs", "describe", "instructions", "RUNNER_ONLY_META", "TOOL_SCHEMA_VERSION",
           "MODEL_TOOLS", "SEG_TOOLS"]
