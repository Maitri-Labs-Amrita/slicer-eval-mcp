"""Argument checks shared by the handlers (stdlib only).

Wrong names / types / missing args raise BridgeArgError (ok=false). Numeric values that are merely out of
range are clamped by the handlers with limits.clamp(), which appends a warning.
"""

import math

from .registry import BridgeArgError


def number(args, key, default=None):
    """A finite real number (bool is not a number here)."""
    if key not in args or args[key] is None:
        if default is not None:
            return float(default)
        raise BridgeArgError(f"missing argument {key!r}")
    v = args[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BridgeArgError(f"{key} must be a number, got {type(v).__name__}")
    if not math.isfinite(v):
        raise BridgeArgError(f"{key} must be finite")
    return float(v)


def integer(args, key):
    """An integer; a float with an integral value (3.0) is accepted, 2.5 is not."""
    v = number(args, key)
    if v != int(v):
        raise BridgeArgError(f"{key} must be an integer, got {args[key]!r}")
    return int(v)


def choice(args, key, allowed):
    v = args.get(key)
    if not isinstance(v, str) or v not in allowed:
        raise BridgeArgError(f"{key} must be one of {sorted(allowed)}, got {v!r}")
    return v


def vec3(args, key):
    v = args.get(key)
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise BridgeArgError(f"{key} must be a list of 3 numbers [R, A, S]")
    return [number({key: x}, key) for x in v]
