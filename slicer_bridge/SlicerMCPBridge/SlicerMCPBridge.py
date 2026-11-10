"""SlicerMCPBridge: Slicer scripted module wrapper for the Slicer MCP bridge.

The bridge itself lives in SlicerMCPBridgeLib. It is started explicitly (start_bridge.py), never automatically on
module load, so loading this module path has no side effects.
"""

from slicer.ScriptedLoadableModule import ScriptedLoadableModule


class SlicerMCPBridge(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Slicer MCP Bridge"
        parent.categories = ["Developer Tools"]
        parent.dependencies = ["WebServer", "VolumeRendering"]
        parent.contributors = ["MAITRI Labs, Amrita"]
        parent.helpText = "Named-handler HTTP bridge (127.0.0.1, no exec) for the Slicer MCP server."
        parent.acknowledgementText = ""
        parent.hidden = True
