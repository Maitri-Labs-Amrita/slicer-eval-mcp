"""What renders the frames: VTK and the OpenGL driver (e.g. Mesa llvmpipe under Xvfb), for the trace's
provenance. No Slicer: parse_gl_report is pure and unit-tested; the bridge passes it its render window's
ReportCapabilities() text.
"""

import re

_GL_KEYS = {"vendor": "OpenGL vendor string", "renderer": "OpenGL renderer string", "version": "OpenGL version string"}


def parse_gl_report(text) -> dict:
    """{vendor, renderer, version} from vtkRenderWindow.ReportCapabilities(); a line that is missing gives None."""
    out = {}
    for key, label in _GL_KEYS.items():
        m = re.search(rf"^\s*{re.escape(label)}:\s*(.+?)\s*$", text or "", re.MULTILINE)
        out[key] = m.group(1) if m else None
    return out


def mesa_version(gl: dict):
    """'23.2.1' from an OpenGL version string such as '4.5 (Core Profile) Mesa 23.2.1-1ubuntu3', else None (not Mesa)."""
    m = re.search(r"\bMesa\s+(\d+(?:\.\d+)*)", (gl or {}).get("version") or "")
    return m.group(1) if m else None
