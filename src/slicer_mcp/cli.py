"""The slicer-mcp command.

    slicer-mcp serve   [--profile P] [--runtime R] [--slicer-url U]   the MCP server (stdio; an MCP client starts it)
    slicer-mcp launch  [--profile P] [--runtime R] [--no-main-window] [--check | --stop]   the Slicer that serves it
    slicer-mcp case    prepare ... | remove <case_id>                  temporary anonymised cases (cases.py)
    slicer-mcp tools-md [--check]                                      the tool reference (docs/tools/<profile>.md)
    slicer-mcp replay  <trace dir>                                     replay a trace against the live bridge
"""

import sys

COMMANDS = ("serve", "launch", "case", "tools-md", "replay")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    cmd, rest = argv[0], argv[1:]
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


if __name__ == "__main__":
    main()
