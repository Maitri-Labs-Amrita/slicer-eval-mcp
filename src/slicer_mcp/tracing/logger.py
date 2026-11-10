"""Episode trace writer. Traces are also training data: keep the raw material, never summarise.

    traces/<episode_id>/
      meta.json        case, profile, model, versions + hashes, times, reset latency
      steps.jsonl      one JSON object per step (step 0 = the runner's reset_episode)
      frames/<step>_<view>.png
      report.json ...  the accepted report: the files the profile's report plugin names (reports.py)

Every line is flushed as it is written, so a crashed episode still leaves a readable trace.
"""

import json
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from slicer_mcp.config import REPO

STEP_FIELDS = ("step", "t", "model_reasoning", "tool", "args", "args_after_validation", "state_after", "frames",
               "view_ref", "result", "latency_ms", "tokens", "warnings", "error")
STEP_EXTRA_FIELDS = ("identity", "render", "case_info")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_info(repo: Path = REPO) -> dict:
    """Commit of slicer-mcp and whether the working tree had uncommitted changes."""
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=10)
    try:
        head = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def new_episode_id(case_id: str, profile: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}Z_{case_id}_{profile}_{secrets.token_hex(3)}"


class TraceLogger:
    def __init__(self, traces_dir, episode_id: str):
        self.dir = Path(traces_dir) / episode_id
        self.episode_id = episode_id
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self._steps = open(self.dir / "steps.jsonl", "a", encoding="utf-8")
        self.meta = {}

    def write_meta(self, meta: dict):
        self.meta = {"episode_id": self.episode_id, **meta}
        self._dump("meta.json", self.meta)

    def update_meta(self, **kv):
        self.meta.update(kv)
        self._dump("meta.json", self.meta)

    def write_frames(self, step: int, frames: dict) -> list:
        """Write PNG frames for a step; returns their paths relative to the episode dir, in view order."""
        rel = []
        for view, png in frames.items():
            name = f"frames/{step:03d}_{view}.png"
            (self.dir / name).write_bytes(png)
            rel.append(name)
        return rel

    def log_step(self, step: int, tool: str, *, args=None, args_after_validation=None, state_after=None,
                 frames=None, view_ref=None, result=None, latency_ms=None, model_reasoning=None, tokens=None,
                 warnings=None, error=None, extra=None) -> dict:
        line = {
            "step": step,
            "t": utc_now(),
            "model_reasoning": model_reasoning,
            "tool": tool,
            "args": args or {},
            "args_after_validation": args_after_validation,
            "state_after": state_after,
            "frames": self.write_frames(step, frames or {}),
            "view_ref": view_ref,
            "result": result,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "warnings": list(warnings or []),
            "error": error,
            **{k: v for k, v in (extra or {}).items() if v is not None},
        }
        self._steps.write(json.dumps(line, allow_nan=False) + "\n")
        self._steps.flush()
        return line

    def write_report_files(self, files: dict):
        """The accepted report as the plugin's files: {name: JSON object}, e.g. {"report.json": ...}."""
        texts = {name: json.dumps(obj, indent=2, allow_nan=False) for name, obj in files.items()}
        tmps, done = {name: self.dir / f".{name}.tmp" for name in texts}, []
        try:
            for name, text in texts.items():
                tmps[name].write_text(text)
            for name in texts:
                tmps[name].replace(self.dir / name)
                done.append(self.dir / name)
        except BaseException:
            for f in [*tmps.values(), *done]:
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def close(self):
        if not self._steps.closed:
            self._steps.close()

    def _dump(self, name, obj):
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
        tmp.replace(self.dir / name)


def read_trace(path) -> tuple[dict, list]:
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines() if line.strip()]
    return meta, steps
