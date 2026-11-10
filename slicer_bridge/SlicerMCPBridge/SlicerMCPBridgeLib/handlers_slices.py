"""measure: the distance between two pixels of one slice view's last frame.

It takes two pixels of one slice view's last frame and returns the distance between their patient positions in mm.
A line in the protocol's measure colour (a locked markups line, shown in that slice view only, no labels: frames
carry no text) joins them until the next measure or a view reset; it disappears from a frame that no longer
contains it. Refusals change nothing: no case, two identical pixels.
"""

import math

import slicer

from . import argcheck, limits
from .env import _r
from .registry import BridgeArgError, handler


@handler("measure")
def h_measure(env, args, warnings):
    env.require_case()
    view = argcheck.choice(args, "view", limits.SLICE_VIEWS)
    n = env.obs.size_px - 1
    xy = [limits.clamp(argcheck.number(args, k), 0, n, k, warnings) for k in ("x1", "y1", "x2", "y2")]
    if xy[0] == xy[2] and xy[1] == xy[3]:
        raise BridgeArgError("the two points are the same pixel")
    p1, p2 = env.pixel_to_ras(view, xy[0], xy[1]), env.pixel_to_ras(view, xy[2], xy[3])
    dist = math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(3)))
    draw_line(env, view, p1, p2)
    env.measure = {"view": view, "p1_ras": _r(p1), "p2_ras": _r(p2), "distance_mm": _r(dist)}
    return {"args_after_validation": {"view": view, "x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3]},
            "distance_mm": round(dist, 1), "p1_ras": _r(p1), "p2_ras": _r(p2)}


def draw_line(env, view, *points, closed=False, color=None):
    """The measure overlay: a locked line through the RAS points in that slice view only, no text; a closed
    curve for an outline. Replaces the previous one."""
    env.remove_measure()
    cls = ("vtkMRMLMarkupsClosedCurveNode" if closed else
           "vtkMRMLMarkupsLineNode" if len(points) == 2 else "vtkMRMLMarkupsCurveNode")
    node = slicer.mrmlScene.AddNewNodeByClass(cls, "slicer_mcp_measure")
    node.SetLocked(True)
    node.CreateDefaultDisplayNodes()
    d = node.GetDisplayNode()
    d.RemoveAllViewNodeIDs()
    d.AddViewNodeID(env.obs.slice_nodes[view].GetID())
    rgb = color or limits.MEASURE_COLOR
    d.SetColor(*rgb)
    d.SetSelectedColor(*rgb)
    d.SetActiveColor(*rgb)
    d.SetPointLabelsVisibility(False)
    d.SetPropertiesLabelVisibility(False)
    d.SetHandlesInteractive(False)
    d.SetSliceProjection(False)
    d.SetLineThickness(limits.MEASURE_LINE_THICKNESS)
    d.SetGlyphScale(1.5)
    if len(points) > 2:
        node.SetCurveTypeToLinear()
        d.SetGlyphScale(0.8 if closed else 1.5)
    for p in points:
        node.AddControlPoint(*[float(x) for x in p])
    for i in range(len(points)):
        node.SetNthControlPointLocked(i, True)
    for i in range(node.GetNumberOfMeasurements()):
        node.GetNthMeasurement(i).SetEnabled(False)
    env.measure_node_id = node.GetID()
    return node
