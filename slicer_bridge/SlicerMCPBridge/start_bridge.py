"""Entry point for a launched Slicer instance (evaluation and training runs). Normally run by `slicer-mcp launch`:

    Slicer --no-splash [--no-main-window] --ignore-slicerrc --disable-settings \
        --python-script <repo>/slicer_bridge/SlicerMCPBridge/start_bridge.py -- \
        [--config <profile>/protocol.json] [--runtime config/runtime.json] [--port N] [--size N] [--cases DIR]
        [--port-file PATH]

Only the Slicer MCP bridge is started (no stock WebServer, so no exec route). The SlicerMCPBridge scripted module
itself is not needed: this script puts SlicerMCPBridgeLib on sys.path. Defaults come from the config files; the flags
override single values (e.g. one port per instance). If the bridge cannot start, Slicer exits with code 3, so
the launcher fails fast instead of waiting for a ping that never comes.

--port 0 binds a free port chosen by the OS; --port-file then gets {"port", "launch_id", "pid"} (written
atomically after the bind), so the launcher learns the port and checks it is its own launch ($SLICER_MCP_LAUNCH_NONCE).
"""

import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import slicer  # noqa: E402

from SlicerMCPBridgeLib.server import Bridge  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None, help="protocol config (default: $SLICER_MCP_PROTOCOL or profiles/ct_basic/protocol.json)")
ap.add_argument("--runtime", default=None, help="runtime config (default: $SLICER_MCP_RUNTIME or config/runtime.json)")
ap.add_argument("--port", type=int, default=None)
ap.add_argument("--size", type=int, default=None, help="override observation.size_px")
ap.add_argument("--cases", default=None)
ap.add_argument("--port-file", default=None, help="write {port, launch_id, pid} here once the bridge is bound")
argv = sys.argv[1:]
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
a, _ = ap.parse_known_args(argv)
try:
    slicer.slicer_mcp_bridge = Bridge(port=a.port, obs_size=a.size, cases_dir=a.cases, config_path=a.config,
                               runtime_path=a.runtime, launch="eval")
    print("[slicer-mcp-bridge]", slicer.slicer_mcp_bridge.start(), flush=True)
    if a.port_file:
        b = slicer.slicer_mcp_bridge
        tmp = f"{a.port_file}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"port": b.port, "launch_id": b.guard.launch_id, "pid": os.getpid()}, f)
        os.replace(tmp, a.port_file)
except Exception:  # noqa: BLE001
    print("[slicer-mcp-bridge] FAILED TO START\n" + traceback.format_exc(), file=sys.stderr, flush=True)
    slicer.util.exit(3)
