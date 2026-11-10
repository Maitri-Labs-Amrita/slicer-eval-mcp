"""Protocol settings for the bridge, loaded from the profile's protocol.json. Pure Python (stdlib only).

The protocol values (bounds, preset lists, defaults, view directions, the segmentation classes) live in
<profile>/protocol.json, which the MCP server reads too. Its sha256 is reported by /bridge/ping and recorded in
traces. Only non-protocol internals (node tags, window title, id pattern) are constants here.

Config path: load(path), else $SLICER_MCP_PROTOCOL, else <repo>/profiles/ct_basic/protocol.json.
The `segmentation` section is optional: without it an episode cannot load segments.
"""

import hashlib
import json
import os

BRIDGE_VERSION = "0.1.0"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

SLICE_VIEWS = {
    "axial": ("SlicerMCPObsAxial", "Axial"),
    "sagittal": ("SlicerMCPObsSagittal", "Sagittal"),
    "coronal": ("SlicerMCPObsCoronal", "Coronal"),
}
SLICE_AXIS = {"axial": 2, "sagittal": 0, "coronal": 1}
THREED_VIEW = "SlicerMCPObs3D"
OWNER_TAG = "SlicerMCPObsOwner"
WINDOW_TITLE = "Slicer MCP Observation"
CASE_ID_PATTERN = r"^[A-Za-z0-9_]{1,64}$"

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_CONFIG_PATH = os.path.join(_REPO, "profiles", "ct_basic", "protocol.json")
DEFAULT_RUNTIME_PATH = os.path.join(_REPO, "config", "runtime.json")
STOCK_PRESETS_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets", "presets.xml")
STOCK_PRESETS_SHA256 = "5b9a6de6f4b4ab9f25c8d064d055adf8c3391581addf24db052cd03e929e65a0"

CONFIG = None
CONFIG_PATH = None
CONFIG_SHA256 = None


def load(path=None):
    """(Re)load the protocol config and set the module-level names used by the handlers."""
    global CONFIG, CONFIG_PATH, CONFIG_SHA256
    global DEFAULT_OBS_SIZE, BACKGROUND_RGB, VR_QUALITY, EXPECTED_FPS, OVERSAMPLING
    global PNG_COMPRESSION_LEVEL, STATE_DECIMALS, RENDER_MODE, RAYCAST_TECHNIQUE, SURFACE_SMOOTHING
    global RULER, ORIENTATION_MARKER, COLORMAP, INTERPOLATE, VR_METHOD
    global WINDOW_PRESETS, DEFAULT_WINDOW_PRESET
    global RENDER_PRESETS, DEFAULT_RENDER_PRESET, EXTRA_PRESETS_XML, EXTRA_PRESETS_SHA256, SCROLL_MAX_SLICES
    global VIEW_ANGLE_DEG, RESET_FIT_MARGIN, ORBIT_AZIMUTH_RANGE, ORBIT_ELEVATION_RANGE, ORBIT_ROLL_RANGE
    global ZOOM_CUMULATIVE_RANGE, ZOOM_STEP_RANGE, PAN_MARGIN_MM, VIEW_DIRECTIONS, DEFAULT_VIEW_DIRECTION
    global CLIP_BOX_MIN_SIZE_MM, CLIP_PLANE_OFFSET_RANGE, CLIP_PLANE_AZIMUTH_RANGE, CLIP_PLANE_ELEVATION_RANGE
    global SEGMENTATION, SEG_CLASS_NAMES, SEG_LABELS, SEG_COLORS, SEG_COLOR_NAMES, SEG_SHELL_THICKNESS_MM, SEG_MODES
    global SEG_DEFAULT_MODE, SEG_DEFAULT_OPACITY, SEG_OPACITY_RANGE, SEG_DEFAULT_SLICE_OUTLINES
    global SEG_OUTLINE_THICKNESS_PX, SEG_SURFACE_SMOOTHING, SEG_CENTER_FIT_MARGIN
    global SLICE_ZOOM_RANGE, SLAB_MM_RANGE, MEASURE_COLOR, MEASURE_LINE_THICKNESS, HU_BALL_MM

    path = path or os.environ.get("SLICER_MCP_PROTOCOL") or DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        raw = f.read()
    cfg = json.loads(raw.decode("utf-8"))
    CONFIG, CONFIG_PATH, CONFIG_SHA256 = cfg, os.path.abspath(path), hashlib.sha256(raw).hexdigest()

    o, wl, r, s, c, k = cfg["observation"], cfg["window_level"], cfg["render"], cfg["slices"], cfg["camera"], cfg["clip"]
    DEFAULT_OBS_SIZE = int(o["size_px"])
    BACKGROUND_RGB = tuple(o["background_rgb"])
    VR_QUALITY = o["vr_quality"]
    EXPECTED_FPS = float(o["expected_fps"])
    OVERSAMPLING = float(o["oversampling"])
    PNG_COMPRESSION_LEVEL = int(o["png_compression_level"])
    STATE_DECIMALS = int(o["state_decimals"])
    RENDER_MODE = o["render_mode"]
    RAYCAST_TECHNIQUE = o["raycast_technique"]
    SURFACE_SMOOTHING = bool(o["surface_smoothing"])
    RULER = o["ruler"]
    ORIENTATION_MARKER = {k: dict(v) for k, v in o["orientation_marker"].items()}
    COLORMAP = wl["colormap"]
    INTERPOLATE = bool(wl["interpolate"])
    VR_METHOD = r["method"]

    WINDOW_PRESETS = {n: tuple(v) for n, v in wl["presets"].items()}
    DEFAULT_WINDOW_PRESET = wl["default_preset"]

    RENDER_PRESETS = list(r["presets"])
    DEFAULT_RENDER_PRESET = r["default_preset"]
    extra = r.get("extra_presets_xml")
    EXTRA_PRESETS_XML = os.path.join(os.path.dirname(CONFIG_PATH), extra) if extra else None
    EXTRA_PRESETS_SHA256 = r.get("extra_presets_sha256")

    SCROLL_MAX_SLICES = int(s["scroll_max_slices"])

    t = cfg["slice_tools"]
    SLICE_ZOOM_RANGE = tuple(float(x) for x in t["zoom_range"])
    SLAB_MM_RANGE = tuple(float(x) for x in t["slab_mm_range"])
    MEASURE_COLOR = tuple(t["measure_color_rgb"])
    MEASURE_LINE_THICKNESS = float(t["measure_line_thickness"])
    HU_BALL_MM = float(t["hu_ball_radius_mm"])

    VIEW_ANGLE_DEG = float(c["view_angle_deg"])
    RESET_FIT_MARGIN = float(c["reset_fit_margin"])
    ORBIT_AZIMUTH_RANGE = tuple(c["orbit_azimuth_range"])
    ORBIT_ELEVATION_RANGE = tuple(c["orbit_elevation_range"])
    ORBIT_ROLL_RANGE = tuple(c["orbit_roll_range"])
    ZOOM_CUMULATIVE_RANGE = tuple(c["zoom_cumulative_range"])
    ZOOM_STEP_RANGE = tuple(c["zoom_step_range"])
    PAN_MARGIN_MM = float(c["pan_margin_mm"])
    VIEW_DIRECTIONS = {n: (tuple(v["from"]), tuple(v["view_up"])) for n, v in c["view_directions"].items()}
    DEFAULT_VIEW_DIRECTION = c["default_view_direction"]

    CLIP_BOX_MIN_SIZE_MM = float(k["box_min_size_mm"])
    CLIP_PLANE_OFFSET_RANGE = tuple(k["plane_offset_range_mm"])
    CLIP_PLANE_AZIMUTH_RANGE = tuple(k["plane_azimuth_range"])
    CLIP_PLANE_ELEVATION_RANGE = tuple(k["plane_elevation_range"])

    g = cfg.get("segmentation")
    SEGMENTATION = g is not None
    g = g or {"classes": [], "shell_thickness_mm": 1.0, "modes": ["ct_only"], "default_mode": "ct_only",
              "default_opacity": 1.0, "opacity_range": [0.0, 1.0], "default_slice_outlines": False,
              "outline_thickness_px": 1, "surface_smoothing_factor": 0.0, "center_fit_margin": 1.0}
    classes = sorted(g["classes"], key=lambda x: x["label"])
    SEG_CLASS_NAMES = [x["name"] for x in classes]
    SEG_LABELS = {x["name"]: int(x["label"]) for x in classes}
    SEG_COLORS = {x["name"]: tuple(x["color_rgb"]) for x in classes}
    SEG_COLOR_NAMES = {x["name"]: x["color_name"] for x in classes}
    SEG_SHELL_THICKNESS_MM = float(g["shell_thickness_mm"])
    SEG_MODES = list(g["modes"])
    SEG_DEFAULT_MODE = g["default_mode"]
    SEG_DEFAULT_OPACITY = float(g["default_opacity"])
    SEG_OPACITY_RANGE = tuple(g["opacity_range"])
    SEG_DEFAULT_SLICE_OUTLINES = bool(g["default_slice_outlines"])
    SEG_OUTLINE_THICKNESS_PX = int(g["outline_thickness_px"])
    SEG_SURFACE_SMOOTHING = float(g["surface_smoothing_factor"])
    SEG_CENTER_FIT_MARGIN = float(g["center_fit_margin"])

    _check()
    return CONFIG


def _check():
    if DEFAULT_WINDOW_PRESET not in WINDOW_PRESETS:
        raise ValueError(f"default window preset {DEFAULT_WINDOW_PRESET!r} not in presets")
    if DEFAULT_RENDER_PRESET not in RENDER_PRESETS:
        raise ValueError(f"default render preset {DEFAULT_RENDER_PRESET!r} not in presets")
    if DEFAULT_VIEW_DIRECTION not in VIEW_DIRECTIONS:
        raise ValueError(f"default view direction {DEFAULT_VIEW_DIRECTION!r} not in view_directions")
    for name, (w, l) in WINDOW_PRESETS.items():
        if not (w > 0 and -1024 <= l <= 3071):
            raise ValueError(f"window preset {name}: window must be > 0 and level in [-1024, 3071] HU")
    if (EXTRA_PRESETS_XML is None) != (EXTRA_PRESETS_SHA256 is None):
        raise ValueError("render extra_presets_xml and extra_presets_sha256 go together")
    if SEG_DEFAULT_MODE not in SEG_MODES:
        raise ValueError(f"segmentation default_mode {SEG_DEFAULT_MODE!r} not in modes")
    if len(set(SEG_LABELS.values())) != len(SEG_LABELS) or not all(0 < v < 256 for v in SEG_LABELS.values()):
        raise ValueError("segmentation labels must be distinct values in 1..255")
    if "all" in SEG_CLASS_NAMES:
        raise ValueError("'all' cannot be a segmentation class name")
    if not (SLICE_ZOOM_RANGE[0] == 1.0 < SLICE_ZOOM_RANGE[1]) or not (SLAB_MM_RANGE[0] == 0.0 < SLAB_MM_RANGE[1]):
        raise ValueError("slice_tools zoom_range must start at 1 and slab_mm_range at 0")
    if HU_BALL_MM < 0:
        raise ValueError("slice_tools hu_ball_radius_mm must be >= 0")
    for name, (d, up) in VIEW_DIRECTIONS.items():
        if abs(sum(d[i] * up[i] for i in range(3))) > 1e-6:
            raise ValueError(f"view direction {name}: view_up not orthogonal to direction")


def load_runtime(path=None):
    """Deployment settings (host, port, paths). Not protocol: not hashed into traces.
    Relative paths are resolved against the repo root. A non-loopback host is refused."""
    path = path or os.environ.get("SLICER_MCP_RUNTIME") or DEFAULT_RUNTIME_PATH
    with open(path) as f:
        rt = json.load(f)
    missing = [k for k in ("bridge_host", "bridge_port", "cases_dir", "traces_dir") if k not in rt]
    if missing:
        raise ValueError(f"runtime config {path} lacks required keys: {', '.join(missing)}")
    if rt["bridge_host"] not in LOOPBACK_HOSTS:
        raise ValueError(f"bridge_host {rt['bridge_host']!r} is not loopback; refusing")
    for k in ("cases_dir", "traces_dir"):
        if not os.path.isabs(rt[k]):
            rt[k] = os.path.join(_REPO, rt[k])
    return rt


def clamp(value, lo, hi, name, warnings):
    """Clamp value to [lo, hi]; append a warning when it changed."""
    v = float(value)
    c = min(max(v, lo), hi)
    if c != v:
        warnings.append(f"{name}={v:g} clamped to {c:g} (allowed [{lo:g}, {hi:g}])")
    return c


load()
