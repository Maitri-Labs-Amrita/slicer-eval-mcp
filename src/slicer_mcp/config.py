"""Load a task profile (its protocol.json is hashed into traces) and the runtime config (not hashed).

A profile is a folder holding protocol.json and, optionally, the files it names: render.extra_presets_xml (extra
volume-rendering presets, read by the bridge) and mcp.report (a report plugin, a Python file inside the profile
folder; see reports.py). The same protocol.json is read by the Slicer bridge (SlicerMCPBridgeLib/limits.py); at
startup the server compares its sha256 with the one the bridge reports and refuses to run on a mismatch.

protocol.json's `mcp` section is the server's part of the profile:
    instructions_title  the first line of the server instructions (e.g. "CT viewer.")
    tools               the model tools this profile offers (the per-task allowlist), in this order
    report              optional: a report plugin file in the profile folder; without it (and without
                        `serve --report`) submit_report takes the built-in JSON report

`observation.views` (optional) names the live views the model is shown, among VIEWS; default all four. It must hold the
three slice views; without "3d" no frame of the 3D view is ever captured, and the profile may not offer a 3D-only tool
(tools/registry.py THREED_TOOLS).
"""

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PKG = Path(__file__).resolve().parent
_CHECKOUT = PKG.parents[1]
# Run from a source checkout (an editable install), or installed as a package. The bridge, tools/, profiles/ and
# config/runtime.json are the checkout's own files, or their copies shipped in the package under slicer_mcp/_root/.
FROM_CHECKOUT = (_CHECKOUT / "slicer_bridge").is_dir() and (_CHECKOUT / "src" / "slicer_mcp").is_dir()
REPO = _CHECKOUT if FROM_CHECKOUT else PKG / "_root"
PROFILES = REPO / "profiles"
DEFAULT_PROFILE = PROFILES / "ct_basic"
DEFAULT_RUNTIME = REPO / "config" / "runtime.json"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
VIEWS = ("3d", "axial", "sagittal", "coronal")  # every live view of the observation window, in capture order
SLICE_VIEWS = ("axial", "sagittal", "coronal")
TOKEN_ENV = "SLICER_MCP_BRIDGE_TOKEN"
REQUIRED_RUNTIME_KEYS = ("bridge_host", "bridge_port", "cases_dir", "traces_dir", "client_timeout_s",
                         "case_prep.nrrd_compression")
DEFAULT_STATE_FILE = "data/dev/eval_slicer.json"


@dataclass(frozen=True)
class ProtocolConfig:
    data: dict
    path: Path
    sha256: str

    @property
    def version(self) -> str:
        return self.data["config_version"]

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)


@dataclass(frozen=True)
class Profile:
    """A task profile: the protocol, the tool allowlist, the server instructions' title and the report plugin."""
    dir: Path
    protocol: ProtocolConfig
    tools: tuple
    instructions_title: str
    report: object | None
    report_sha256: str | None
    views: tuple = VIEWS  # the live views the model is shown (observation.views), in VIEWS order

    @property
    def has_3d(self) -> bool:
        return "3d" in self.views

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def segmentation(self) -> bool:
        return self.protocol.get("segmentation") is not None


def protocol_path(profile=None) -> Path:
    """A profile folder or its protocol.json; default $SLICER_MCP_PROFILE, else profiles/ct_basic. A bare name is
    looked up in the repo's profiles/ folder."""
    p = Path(profile or os.environ.get("SLICER_MCP_PROFILE") or DEFAULT_PROFILE)
    if not p.exists() and not p.is_absolute() and len(p.parts) == 1 and (PROFILES / p).is_dir():
        p = PROFILES / p
    return p / "protocol.json" if p.is_dir() else p


def load_protocol(path=None) -> ProtocolConfig:
    p = protocol_path(path)
    raw = p.read_bytes()
    return ProtocolConfig(json.loads(raw), p.resolve(), hashlib.sha256(raw).hexdigest())


def load_profile(profile=None, report=None) -> Profile:
    """The profile, with its report: the plugin file `report` (the harness's, e.g. `serve --report`), else the
    profile's mcp.report, else the built-in JSON report; None when the profile does not offer submit_report.
    Raises ValueError for a profile that names an unknown tool, a missing report plugin, segment tools without a
    segmentation, a report plugin without submit_report, an observation.views list that is not distinct names among
    VIEWS holding the three slice views, or a 3D-only tool without the "3d" view."""
    from slicer_mcp import reports
    from slicer_mcp.tools.registry import MODEL_TOOLS, SEG_TOOLS, THREED_TOOLS
    protocol = load_protocol(profile)
    mcp = protocol.get("mcp") or {}
    tools = tuple(mcp.get("tools") or ())
    unknown = [t for t in tools if t not in MODEL_TOOLS]
    if unknown or len(set(tools)) != len(tools):
        raise ValueError(f"profile tools must be distinct names among {list(MODEL_TOOLS)}; unknown {unknown}")
    if protocol.get("segmentation") is None and any(t in SEG_TOOLS for t in tools):
        raise ValueError(f"segment tools {[t for t in tools if t in SEG_TOOLS]} need a segmentation section")
    views = _views(protocol)
    if "3d" not in views and any(t in THREED_TOOLS for t in tools):
        raise ValueError(f"3D tools {[t for t in tools if t in THREED_TOOLS]} need the '3d' view in observation.views")
    path = Path(report) if report else (profile_file(protocol.path.parent, mcp["report"]) if mcp.get("report")
                                        else None)
    if path is not None and "submit_report" not in tools:
        raise ValueError(f"a report plugin ({path}) needs submit_report in the profile's tools")
    report, report_sha = None, None
    if path is not None:
        raw = path.read_bytes()
        report_sha = hashlib.sha256(raw).hexdigest()
        report = reports.check_plugin(_import_file(path, f"slicer_mcp_report_{report_sha[:12]}"))
    elif "submit_report" in tools:
        report = reports.BUILTIN
    return Profile(protocol.path.parent, protocol, tools, mcp.get("instructions_title") or "3D Slicer viewer.",
                   report, report_sha, views)


def _views(protocol) -> tuple:
    """observation.views in VIEWS order (all four when absent); ValueError for anything else than distinct names among
    VIEWS that include the three slice views."""
    views = (protocol.get("observation") or {}).get("views")
    if views is None:
        return VIEWS
    if (not isinstance(views, list) or any(v not in VIEWS for v in views) or len(set(views)) != len(views)
            or not set(SLICE_VIEWS) <= set(views)):
        raise ValueError(f"observation.views must be distinct names among {list(VIEWS)} including "
                         f"{list(SLICE_VIEWS)}; got {views!r}")
    return tuple(v for v in VIEWS if v in views)


def profile_file(profile_dir, name) -> Path:
    """A file a profile names, resolved inside the profile folder. ValueError for a path that leaves it."""
    root = Path(profile_dir).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"profile file {name!r} is outside the profile folder {root}")
    return p


def _import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def base_dir() -> Path:
    """The folder that relative runtime paths (cases, traces, state file, logs) are resolved against: the checkout,
    or for an installed package the current working directory."""
    return REPO if FROM_CHECKOUT else Path.cwd()


def missing_keys(data: dict, keys) -> list:
    """The keys (dotted for nested ones, e.g. "case_prep.nrrd_compression") that data lacks."""
    out = []
    for key in keys:
        d = data
        for part in key.split("."):
            if not isinstance(d, dict) or part not in d:
                out.append(key)
                break
            d = d[part]
    return out


def load_runtime(path=None) -> dict:
    """The runtime config. ValueError naming the file and every missing required key."""
    p = Path(path or os.environ.get("SLICER_MCP_RUNTIME") or DEFAULT_RUNTIME)
    rt = json.loads(p.read_text())
    missing = missing_keys(rt, REQUIRED_RUNTIME_KEYS)
    if missing:
        raise ValueError(f"runtime config {p} lacks required keys: {', '.join(missing)}")
    if rt["bridge_host"] not in LOOPBACK_HOSTS:
        raise ValueError(f"bridge_host {rt['bridge_host']!r} is not loopback; refusing")
    for k in ("cases_dir", "traces_dir"):
        if not os.path.isabs(rt[k]):
            rt[k] = str(base_dir() / rt[k])
    return rt


def launch_state_path(rt: dict) -> Path:
    """The launch state file named by runtime eval_launch.state_file (relative paths: base_dir())."""
    p = Path(os.path.expanduser(str(rt.get("eval_launch", {}).get("state_file", DEFAULT_STATE_FILE))))
    return p if p.is_absolute() else base_dir() / p


def read_launch_state(rt: dict) -> dict | None:
    """The running launch's state file, or None when there is none."""
    try:
        return json.loads(launch_state_path(rt).read_text())
    except (OSError, ValueError):
        return None


def launch_token(runtime=None) -> str | None:
    """The bridge token: $SLICER_MCP_BRIDGE_TOKEN, else the launch state file's, else None. `runtime` is a runtime
    dict or a runtime file path (default: the default runtime)."""
    if os.environ.get(TOKEN_ENV):
        return os.environ[TOKEN_ENV]
    try:
        rt = runtime if isinstance(runtime, dict) else load_runtime(runtime)
    except (OSError, ValueError):
        return None
    state = read_launch_state(rt) or {}
    tok = state.get("token")
    return tok if isinstance(tok, str) and tok else None


def bridge_url(runtime: dict | None = None) -> str:
    """SLICER_URL wins (one server process per Slicer instance); else runtime.json host/port."""
    env = os.environ.get("SLICER_URL")
    if env:
        return env.rstrip("/")
    rt = runtime or load_runtime()
    return f"http://{rt['bridge_host']}:{rt['bridge_port']}"
