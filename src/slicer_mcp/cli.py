"""The slicer-mcp command.

    slicer-mcp serve   [--profile P] [--runtime R] [--slicer-url U]   the MCP server (stdio; an MCP client starts it)
    slicer-mcp launch  [--profile P] [--runtime R] [--no-main-window] [--check | --stop]   the Slicer that serves it
    slicer-mcp case    prepare ... | remove <case_id>                  temporary anonymised cases (cases.py)
    slicer-mcp tools-md [--check]                                      the tool reference (docs/tools/<profile>.md)
    slicer-mcp replay  <trace dir>                                     replay a trace against the live bridge
    slicer-mcp setup-headless [options]                                Linux without a screen: Xvfb, Slicer, libraries
    slicer-mcp xvfb-session [options] -- <command>                     run a command on its own virtual display
"""

import sys

COMMANDS = ("serve", "launch", "case", "tools-md", "replay", "setup-headless", "xvfb-session")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    cmd, rest = argv[0], argv[1:]
    if cmd in ("setup-headless", "xvfb-session"):
        sys.exit(_script(cmd, rest))
    if cmd == "serve":
        from slicer_mcp.server import main as run
    elif cmd == "launch":
        from slicer_mcp.launch import main as run
    elif cmd == "case":
        from slicer_mcp.cases import main as run
    elif cmd == "tools-md":
        from slicer_mcp.tools_md import main as run
    else:
        from slicer_mcp.tracing.replay import main as run
    run(rest)


def _script(cmd, rest) -> int:
    """tools/setup_headless.sh and tools/xvfb_session.py, from the checkout or from their copies in the package."""
    import os
    import subprocess
    from slicer_mcp.config import REPO
    if cmd == "setup-headless":
        env = {**os.environ, "SLICER_MCP_PYTHON": sys.executable}
        return subprocess.call(["bash", str(REPO / "tools" / "setup_headless.sh"), *rest], env=env)
    import runpy
    sys.argv = [str(REPO / "tools" / "xvfb_session.py"), *rest]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


if __name__ == "__main__":
    main()
