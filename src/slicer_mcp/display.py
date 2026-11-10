"""Display power check (stale-render guard).

When the display sleeps, Slicer stops drawing the observation window and every capture returns the last frame
drawn. The server checks at start (and before every reset) that the display is awake, and warns when nothing
prevents it from sleeping during a run (`caffeinate -d`; `slicer-mcp launch` starts one for its Slicer).

Supported on macOS: CoreGraphics `CGDisplayIsAsleep(CGMainDisplayID())` for the display state, and
`pmset -g assertions` / `pmset -g` for what keeps it awake. Elsewhere the status is unknown and the check only
warns (a Linux / Xvfb launch has no display to sleep).

Modes (runtime.json `display_check`): "require" (default: an asleep display refuses), "warn", "off".
"""

import ctypes
import ctypes.util
import re
import subprocess
import sys

MODES = ("require", "warn", "off")


class DisplayAsleep(RuntimeError):
    pass


def _cg_asleep():
    """True / False from CoreGraphics, or None when it cannot be asked."""
    try:
        cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayIsAsleep.argtypes = [ctypes.c_uint32]
        cg.CGDisplayIsAsleep.restype = ctypes.c_bool
        return bool(cg.CGDisplayIsAsleep(cg.CGMainDisplayID()))
    except (OSError, AttributeError, TypeError):
        return None


def _run(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def parse_assertions(text):
    """`pmset -g assertions` -> True when PreventUserIdleDisplaySleep is held (system-wide count > 0)."""
    if text is None:
        return None
    m = re.search(r"^\s*PreventUserIdleDisplaySleep\s+(\d+)\s*$", text, re.M)
    return None if m is None else int(m.group(1)) > 0


def parse_displaysleep(text):
    """`pmset -g` -> the display-sleep timer in minutes (0 = never), or None."""
    if text is None:
        return None
    m = re.search(r"^\s*displaysleep\s+(\d+)", text, re.M)
    return None if m is None else int(m.group(1))


def display_status(asleep_only=False) -> dict:
    """{"platform", "asleep", "prevent_display_sleep", "displaysleep_min"}; None where unknown."""
    st = {"platform": sys.platform, "asleep": None, "prevent_display_sleep": None, "displaysleep_min": None}
    if sys.platform != "darwin":
        return st
    st["asleep"] = _cg_asleep()
    if not asleep_only:
        st["prevent_display_sleep"] = parse_assertions(_run(["pmset", "-g", "assertions"]))
        st["displaysleep_min"] = parse_displaysleep(_run(["pmset", "-g"]))
    return st


def evaluate(status: dict, mode: str = "require") -> tuple[list, list]:
    """(errors, warnings) for a status. Errors only in "require" mode; "off" returns nothing."""
    if mode not in MODES:
        raise ValueError(f"display_check {mode!r}; expected one of {MODES}")
    errors, warnings = [], []
    if mode == "off":
        return errors, warnings
    if status.get("asleep") is True:
        msg = "the display is asleep: Slicer does not draw, so every frame would be stale"
        (errors if mode == "require" else warnings).append(msg)
    elif status.get("asleep") is None:
        warnings.append(f"display state unknown on {status.get('platform')}: cannot check for display sleep")
    if status.get("prevent_display_sleep") is False and status.get("displaysleep_min") not in (0, None):
        warnings.append(f"nothing prevents display sleep (timer {status['displaysleep_min']} min): run "
                        "`caffeinate -d &` or launch Slicer with `slicer-mcp launch`")
    return errors, warnings
