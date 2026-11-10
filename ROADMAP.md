# Roadmap

Planned work, not in a fixed order. Each item is a candidate for a later release, not a promise.

- **Cardiac view planes:** standard cardiac views (for example short axis, two-chamber and four-chamber) for the
  slice views.
- **More slice tools:** further tools for reading the slice views.
- **Compact model state:** a smaller state for the model to read, in place of the full state that every tool
  result carries today.
- **Named toolsets:** named groups of tools on top of the per-profile allowlist.
- **Profile-specific segment wording:** the structure wording in the segment tool descriptions (`jump_to_point`,
  `point_at`) and the shell mode become profile settings, so that a profile with separate structures (for example
  liver and kidneys) gets wording that fits it.

## Rule for tool-surface changes

The 0.1.0 tool surface stays reproducible. Any change to a tool name, a tool description, an argument schema or the
server instructions:

1. bumps `TOOL_SCHEMA_VERSION` (`src/slicer_mcp/session.py`),
2. regenerates `docs/tools/` (`uv run slicer-mcp tools-md`), and
3. ships as a new profile. The existing profiles stay unchanged.
