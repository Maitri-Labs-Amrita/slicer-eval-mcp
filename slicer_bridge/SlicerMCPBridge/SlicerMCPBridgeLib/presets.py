"""Volume-rendering presets: Slicer 5.10's own presets.xml, VENDORED (never Slicer's installed copy), plus the
profile's extra presets (render.extra_presets_xml, e.g. a task's own preset). Both files are hash-checked."""

import hashlib

import slicer

from . import limits


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _import(path):
    """{name: vtkMRMLVolumePropertyNode} from a presets file, in a private MRML scene (kept alive by the caller)."""
    scene = slicer.vtkMRMLScene()
    scene.RegisterNodeClass(slicer.vtkMRMLVolumePropertyNode())
    scene.SetURL(path)
    if not scene.Import():
        raise RuntimeError(f"failed to import presets from {path}")
    nodes = {}
    coll = scene.GetNodesByClass("vtkMRMLVolumePropertyNode")
    for i in range(coll.GetNumberOfItems()):
        n = coll.GetItemAsObject(i)
        nodes[n.GetName()] = n
    return scene, nodes


class PresetStore:
    """Private MRML scenes holding the vendored and the profile's vtkMRMLVolumePropertyNodes."""

    def __init__(self, path=limits.STOCK_PRESETS_XML, extra=None, extra_sha256=None):
        extra = extra if extra is not None else limits.EXTRA_PRESETS_XML
        extra_sha256 = extra_sha256 if extra_sha256 is not None else limits.EXTRA_PRESETS_SHA256
        stock = sha256_file(path)
        if stock != limits.STOCK_PRESETS_SHA256:
            raise RuntimeError(f"vendored presets.xml hash {stock} != expected {limits.STOCK_PRESETS_SHA256}")
        self.scenes = []
        scene, self.nodes = _import(path)
        self.scenes.append(scene)
        parts = [stock]
        if extra:
            got = sha256_file(extra)
            if got != extra_sha256:
                raise RuntimeError(f"profile presets {extra} hash {got} != protocol extra_presets_sha256 {extra_sha256}")
            scene, nodes = _import(extra)
            self.scenes.append(scene)
            clash = sorted(set(nodes) & set(self.nodes))
            if clash:
                raise RuntimeError(f"profile presets {clash} have the names of Slicer presets")
            self.nodes.update(nodes)
            parts.append(got)
        self.sha256 = hashlib.sha256(":".join(parts).encode()).hexdigest()
        missing = [p for p in limits.RENDER_PRESETS if p not in self.nodes]
        if missing:
            raise RuntimeError(f"presets missing: {missing}")

    def apply(self, display_node, name):
        """Copy preset `name` verbatim into the display node's volume property (no shift, no tuning)."""
        if name not in limits.RENDER_PRESETS:
            raise ValueError(f"unknown render preset {name!r}")
        prop = display_node.GetVolumePropertyNode()
        prop.Copy(self.nodes[name])
        prop.SetName(name)
