"""Replay a trace against the live bridge and diff the regenerated frames against the logged ones.

    slicer-mcp replay traces/<episode_id> [--ssim-min 0.99] [--force]

Resets the same case, then re-issues every successful bridge step with its logged args_after_validation
(bookmark_view re-captures all four views; submit_report and failed steps have no live frames
and are skipped). Each frame passes when it is pixel-identical or SSIM >= ssim-min. The logged state_after
must match too. Refuses to run when the bridge's protocol config differs from the trace's (unless --force).
Writes <trace>/replay.json and exits non-zero on any failure.
"""

import argparse
import json
import sys
from pathlib import Path

from slicer_mcp.bridge_client import BridgeClient
from slicer_mcp.imaging import compare
from slicer_mcp.session import BRIDGE_TOOLS, SESSION_ARGS
from slicer_mcp.tracing.logger import read_trace, utc_now

SSIM_MIN = 0.99


def _state_equal(a, b, tol=1e-3):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_state_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_state_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol
    return a == b


def _bridge_call(step):
    """(handler, args, capture) to reproduce a logged step, or None when the step has nothing to replay."""
    tool = step["tool"]
    views = tuple(Path(f).stem.split("_", 1)[1] for f in step["frames"])
    if step.get("error"):
        return None
    if tool == "reset_episode":
        return "reset_episode", step["args_after_validation"], views
    if tool == "bookmark_view":
        return "capture", {}, views
    if tool in BRIDGE_TOOLS:
        args = {k: v for k, v in (step["args_after_validation"] or {}).items() if k not in SESSION_ARGS}
        return BRIDGE_TOOLS[tool][0], args, views
    return None


def replay(trace_dir, client=None, ssim_min=SSIM_MIN, force=False) -> dict:
    trace_dir = Path(trace_dir)
    meta, steps = read_trace(trace_dir)
    client = client or BridgeClient(obs_size=meta["image"]["size_px"])
    info = client.ping()
    if "config_sha256" not in info:
        raise SystemExit("the bridge answered only a liveness ping: no valid bridge token (set SLICER_MCP_RUNTIME to "
                         "the launch's runtime config, or SLICER_MCP_BRIDGE_TOKEN)")
    if info["config_sha256"] != meta["protocol"]["config_sha256"] and not force:
        raise SystemExit(f"bridge protocol config {info['config_sha256'][:12]} != trace "
                         f"{meta['protocol']['config_sha256'][:12]}; use --force to replay anyway")
    if not steps or steps[0]["tool"] != "reset_episode":
        raise SystemExit("trace does not start with reset_episode")
    rows, ok = [], True
    for s in steps:
        call = _bridge_call(s)
        if call is None:
            rows.append({"step": s["step"], "tool": s["tool"], "skipped": True})
            continue
        handler, args, views = call
        r = client.call(handler, args, capture=views)
        frames = {}
        for rel in s["frames"]:
            view = Path(rel).stem.split("_", 1)[1]
            c = compare((trace_dir / rel).read_bytes(), r.images[view])
            c["pass"] = bool(c["identical"] or c["ssim"] >= ssim_min)
            frames[view] = c
        state_ok = _state_equal(s["state_after"], r.state)
        row_ok = state_ok and all(f["pass"] for f in frames.values())
        ok &= row_ok
        rows.append({"step": s["step"], "tool": s["tool"], "pass": row_ok, "state_equal": state_ok, "frames": frames})
    n_frames = sum(len(r.get("frames", {})) for r in rows)
    report = {
        "episode_id": meta["episode_id"],
        "replayed_at": utc_now(),
        "ssim_min": ssim_min,
        "pass": ok,
        "n_steps": len(steps),
        "n_frames": n_frames,
        "n_identical": sum(f["identical"] for r in rows for f in r.get("frames", {}).values()),
        "min_ssim": min((f["ssim"] for r in rows for f in r.get("frames", {}).values()), default=None),
        "bridge": {"version": info["bridge_version"], "config_sha256": info["config_sha256"]},
        "steps": rows,
    }
    (trace_dir / "replay.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slicer-mcp replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir")
    ap.add_argument("--ssim-min", type=float, default=SSIM_MIN)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    rep = replay(a.trace_dir, ssim_min=a.ssim_min, force=a.force)
    for r in rep["steps"]:
        if r.get("skipped"):
            print(f"  step {r['step']:3d} {r['tool']:20s} skipped (no frames)")
            continue
        fr = " ".join(f"{v}:{'=' if c['identical'] else round(c['ssim'], 4)}" for v, c in r["frames"].items())
        print(f"  step {r['step']:3d} {r['tool']:20s} {'PASS' if r['pass'] else 'FAIL'}"
              f"{'' if r['state_equal'] else ' (state differs)'}  {fr}")
    print(f"{'PASS' if rep['pass'] else 'FAIL'}: {rep['n_frames']} frames, {rep['n_identical']} identical, "
          f"min SSIM {rep['min_ssim']}")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
