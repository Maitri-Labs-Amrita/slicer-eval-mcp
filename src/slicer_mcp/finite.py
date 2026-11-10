"""Non-finite numbers. JSON has no inf or nan, and the trace is strict JSON, but Python's decoder
reads 1e400 as inf (and NaN / Infinity literals as such). Tool arguments, runner metadata and bridge responses are
checked with nonfinite_paths before they reach the episode or the trace."""

import math


def nonfinite_paths(obj, path="") -> list:
    """The paths in obj (parsed JSON) that hold inf or nan, e.g. ["report.n", "m[1]"]."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return [path or "value"]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in nonfinite_paths(v, f"{path}.{k}" if path else str(k))]
    if isinstance(obj, (list, tuple)):
        return [p for i, v in enumerate(obj) for p in nonfinite_paths(v, f"{path}[{i}]")]
    return []


def finite_copy(obj):
    """obj with every inf / nan replaced by its name as a string ("inf", "-inf", "nan"), so it can be logged."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: finite_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_copy(v) for v in obj]
    return obj
