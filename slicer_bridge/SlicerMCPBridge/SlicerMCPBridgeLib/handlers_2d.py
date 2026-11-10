"""2D slice handlers.

Slice positions are RAS coordinates along each view's axis (axial -> S, sagittal -> R, coronal -> A), always
through env.get_slice_position / env.set_slice_position (never raw SetSliceOffset: sagittal sign trap).
Every handler returns args_after_validation (the tracer logs it; replay re-issues it).
Slice positions are clamped to the voxel-centre range (env.slice_position_range), so a slice never lies outside the data.
set_slice takes slab_mm; point_at and jump_to_point return the HU at the point.
"""

from . import argcheck, limits
from .env import _r
from .registry import handler


def _clamp_position(env, view, pos, name, warnings):
    lo, hi = env.slice_position_range(view)
    return limits.clamp(pos, lo, hi, name, warnings)


@handler("set_slice")
def h_set_slice(env, args, warnings):
    env.require_case()
    view = argcheck.choice(args, "view", limits.SLICE_VIEWS)
    pos = _clamp_position(env, view, argcheck.number(args, "position_mm"), "position_mm", warnings)
    slab = None
    if args.get("slab_mm") is not None:
        slab = limits.clamp(argcheck.number(args, "slab_mm"), *limits.SLAB_MM_RANGE, "slab_mm", warnings)
    env.set_slice_position(view, pos)
    out = {"view": view, "position_mm": pos}
    if slab is not None:
        env.set_slab(view, slab)
        out["slab_mm"] = slab
    return {"args_after_validation": out}


@handler("scroll_slice")
def h_scroll_slice(env, args, warnings):
    """Relative scroll by whole slices (spacing = the view's slice spacing). The step count is clamped to
    +-scroll_max_slices; the resulting position is clamped to the volume, so the actual move can be shorter."""
    env.require_case()
    view = argcheck.choice(args, "view", limits.SLICE_VIEWS)
    delta = argcheck.integer(args, "delta_slices")
    m = limits.SCROLL_MAX_SLICES
    delta = int(limits.clamp(delta, -m, m, "delta_slices", warnings))
    spacing = env.slice_spacing(view)
    start = env.get_slice_position(view)
    lo, hi = env.slice_position_range(view)
    target = start + delta * spacing
    pos = min(max(target, lo), hi)
    if pos != target:
        warnings.append(f"scroll stopped at the volume edge: position {pos:g} mm "
                        f"(requested {target:g} mm, allowed [{lo:g}, {hi:g}])")
    env.set_slice_position(view, pos)
    return {"args_after_validation": {"view": view, "delta_slices": delta},
            "moved_mm": round(pos - start, 4), "moved_slices": round((pos - start) / spacing, 3),
            "spacing_mm": round(spacing, 4)}


@handler("set_window_preset")
def h_set_window_preset(env, args, warnings):
    env.require_case()
    name = argcheck.choice(args, "name", limits.WINDOW_PRESETS)
    w, l = limits.WINDOW_PRESETS[name]
    env.set_wl(w, l, name)
    return {"args_after_validation": {"name": name}}


def _inside(env, ras):
    """With segments: the structure whose filled label contains the point (None outside every structure); and the
    mean HU around the point."""
    out = {"inside_segment": env.segs.inside(ras)} if env.segs is not None else {}
    out["hu"] = env.hu_at(ras)
    return out


def _move_through(env, ras, warnings, name="ras", skip=None):
    """Move the slice views (all, or all but `skip`) through `ras`, each component clamped to its view's
    position range. Returns the clamped point."""
    out = list(ras)
    for view, axis in limits.SLICE_AXIS.items():
        lo, hi = env.slice_position_range(view)
        out[axis] = limits.clamp(ras[axis], lo, hi, f"{name}[{axis}]", warnings)
        if view != skip:
            env.set_slice_position(view, out[axis])
    return out


@handler("jump_to_point")
def h_jump_to_point(env, args, warnings):
    """Move all three slice views through the RAS point (each view takes its own axis component).
    Each component is clamped to that view's position range. Main-window slice views are not touched."""
    env.require_case()
    ras = _move_through(env, argcheck.vec3(args, "ras"), warnings)
    return {"args_after_validation": {"ras": ras}, **_inside(env, ras)}


@handler("point_at")
def h_point_at(env, args, warnings):
    """Point at pixel (x, y) of a slice view's frame (origin top-left; clamped to [0, N-1]). The pixel's RAS
    point becomes the focus: the other two slice views move through it; the pointed view stays where it is."""
    env.require_case()
    view = argcheck.choice(args, "view", limits.SLICE_VIEWS)
    n = env.obs.size_px - 1
    x = limits.clamp(argcheck.number(args, "x"), 0, n, "x", warnings)
    y = limits.clamp(argcheck.number(args, "y"), 0, n, "y", warnings)
    ras = _move_through(env, env.pixel_to_ras(view, x, y), warnings, name="point", skip=view)
    return {"args_after_validation": {"view": view, "x": x, "y": y}, "ras": _r(ras), **_inside(env, ras)}
