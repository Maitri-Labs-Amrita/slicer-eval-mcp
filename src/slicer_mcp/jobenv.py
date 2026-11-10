"""A Slicer process's own HOME, XDG_* folders and TMPDIR. No Slicer.

With --disable-settings every Slicer of one user shares one Slicer-tmp.ini, and Slicer's temporary folder (and the DICOM
database a converter opens there) is per user, so two Slicers started at once by parallel jobs can collide. Every
Slicer the server starts (the eval launch, slicer-mcp launch; the case converters, slicer_mcp.cases) gets its own
folders through job_env().
"""

import os
from pathlib import Path

DIRS = {"HOME": "home", "XDG_CONFIG_HOME": "config", "XDG_CACHE_HOME": "cache", "XDG_DATA_HOME": "data",
        "XDG_RUNTIME_DIR": "runtime", "TMPDIR": "tmp"}


def job_env(environ, root) -> dict:
    """`environ` with HOME, XDG_* and TMPDIR / TMP / TEMP pointing into `root` (created)."""
    root = Path(root)
    env = dict(environ)
    for key, sub in DIRS.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        env[key] = str(d)
    os.chmod(root / DIRS["XDG_RUNTIME_DIR"], 0o700)
    env["TMP"] = env["TEMP"] = env["TMPDIR"]
    return env
