"""Standalone, fixed-size observation window: one 3D view + three slice views.

The views are owned by a singleton scripted-module node, so the main layout manager ignores them and
their view / camera / slice / composite nodes are singletons that survive scene.Clear(0).
The logical size of each view is N / devicePixelRatio of the screen the window is on, so every render window is exactly
N x N pixels. The ratio is read after the window is shown and again whenever it changes screen, at every reset and
before a capture whose size is wrong: a window created at ratio 2 (a Retina screen) and shown on a
ratio-1 monitor would otherwise render at N/2.
Mouse input is disabled: the window is for watching, and it must not change the state.
"""

import qt
import slicer
import vtk

from . import limits


def app_slice_logics():
    """The application logic's slice-logic collection. With --no-main-window there is no layout manager, which
    is what normally creates it (GetSliceLogics() is None): create it then."""
    app_logic = slicer.app.applicationLogic()
    logics = app_logic.GetSliceLogics()
    if logics is None:
        logics = vtk.vtkCollection()
        app_logic.SetSliceLogics(logics)
    return logics


class ObservationWindow:
    def __init__(self, size_px):
        self.size_px = int(size_px)
        scene = slicer.mrmlScene
        self.widget = qt.QWidget()
        self.widget.setWindowTitle(limits.WINDOW_TITLE)
        self.dpr = self.widget.devicePixelRatioF()
        self.logical = self._logical_for(self.dpr)
        self.refits = 0
        self.fit_error = None

        self.owner = self._owner_node(scene)
        grid = qt.QGridLayout(self.widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.view_node = self._owned_view_node(scene, "vtkMRMLViewNode", limits.THREED_VIEW)
        self.view_node.SetLayoutLabel("3D")
        self.threed_widget = slicer.qMRMLThreeDWidget()
        self.threed_widget.setMRMLScene(scene)
        self.threed_widget.setMRMLViewNode(self.view_node)
        self.threed_widget.threeDController().setVisible(False)
        self.threed_view = self.threed_widget.threeDView()
        self.threed_view.setFixedSize(self.logical, self.logical)
        grid.addWidget(self.threed_widget, 0, 0)

        self.slice_widgets = {}
        self.slice_nodes = {}
        self._slice_logics = []
        cells = {"axial": (0, 1), "sagittal": (1, 0), "coronal": (1, 1)}
        for name, (layout_name, orientation) in limits.SLICE_VIEWS.items():
            sn = self._owned_view_node(scene, "vtkMRMLSliceNode", layout_name)
            sn.SetLayoutLabel(name[0].upper())
            if not sn.SetOrientation(orientation):
                raise RuntimeError(f"slice node {layout_name} has no {orientation} orientation preset")
            sw = slicer.qMRMLSliceWidget()
            sw.setMRMLScene(scene)
            sw.setMRMLSliceNode(sn)
            sw.sliceViewName = layout_name
            sw.sliceViewLabel = name[0].upper()
            app_slice_logics().AddItem(sw.sliceLogic())
            self._slice_logics.append(sw.sliceLogic())
            sw.sliceController().setVisible(False)
            sw.sliceView().setFixedSize(self.logical, self.logical)
            grid.addWidget(sw, *cells[name])
            self.slice_widgets[name] = sw
            self.slice_nodes[name] = sn

        for w in [self.threed_widget, *self.slice_widgets.values()]:
            w.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)
        self.widget.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)
        self.render_counts = {}
        self._render_observers = []
        for vname, view in self.all_views().items():
            self.render_counts[vname] = 0
            rw = view.renderWindow()
            self._render_observers.append((rw, rw.AddObserver(vtk.vtkCommand.EndEvent, self._counter(vname))))
        self.widget.show()
        slicer.app.processEvents()
        self.fit_to_screen()
        self._watch_screen()
        self.check_bound()
        self.apply_view_settings()

    def _logical_for(self, dpr):
        logical = self.size_px / dpr
        if abs(logical - round(logical)) > 1e-9:
            raise RuntimeError(f"obs size {self.size_px} is not divisible by the screen's devicePixelRatio {dpr}")
        return int(round(logical))

    def bad_sizes(self):
        return {k: s for k, s in self.render_sizes().items() if s != [self.size_px, self.size_px]}

    def fit_to_screen(self):
        """Size every view for the devicePixelRatio of the window's current screen and check the render windows
        are N x N. Returns True when the size changed. Slice-node dimensions follow the render windows (square to
        square: the field of view in mm is unchanged)."""
        dpr = self.widget.devicePixelRatioF()
        logical = self._logical_for(dpr)
        changed = logical != self.logical or dpr != self.dpr
        self.dpr, self.logical = dpr, logical
        for v in self.all_views().values():
            v.setFixedSize(logical, logical)
        self.widget.adjustSize()
        slicer.app.processEvents()
        bad = self.bad_sizes()
        if bad:
            self.fit_error = (f"observation views are not {self.size_px}x{self.size_px} device pixels on this screen "
                              f"(devicePixelRatio {dpr}, logical {logical}): {bad}")
            raise RuntimeError(self.fit_error)
        self.fit_error = None
        for name in getattr(self, "slice_widgets", {}):
            w, h = self.slice_view(name).renderWindow().GetSize()
            self.slice_logic(name).ResizeSliceNode(w, h)
        if changed:
            self.refits += 1
        return changed

    def _watch_screen(self):
        """Refit when the window moves to another screen (its devicePixelRatio may differ)."""
        wh = self.widget.windowHandle()
        if wh is not None:
            wh.connect("screenChanged(QScreen*)", lambda *_: qt.QTimer.singleShot(0, self._refit_quietly))

    def _refit_quietly(self):
        try:
            self.fit_to_screen()
        except Exception:  # noqa: BLE001
            pass

    def screen_info(self):
        scr = self.widget.screen() if hasattr(self.widget, "screen") else None
        name = getattr(scr, "name", None)
        return {"device_pixel_ratio": self.dpr, "logical_px": self.logical, "refits": self.refits,
                "screen": (name() if callable(name) else name), "fit_error": self.fit_error}

    def render_stack(self):
        """VTK, OpenGL and Mesa versions of the 3D observation view's render window."""
        from .renderinfo import mesa_version, parse_gl_report
        try:
            gl = parse_gl_report(self.threed_view.renderWindow().ReportCapabilities())
        except Exception:  # noqa: BLE001
            gl = {"vendor": None, "renderer": None, "version": None}
        return {"vtk": vtk.vtkVersion.GetVTKVersion(), "opengl": gl, "mesa": mesa_version(gl)}

    def check_bound(self):
        """Every widget must be bound to an in-scene node (guards against the orphan-node trap above)."""
        scene = slicer.mrmlScene
        nodes = {"3d": self.threed_widget.mrmlViewNode(),
                 **{n: w.mrmlSliceNode() for n, w in self.slice_widgets.items()}}
        bad = [k for k, nd in nodes.items() if nd is None or nd.GetID() is None or scene.GetNodeByID(nd.GetID()) is not nd]
        if bad:
            raise RuntimeError(f"observation views not bound to scene nodes: {bad}")

    def _owned_view_node(self, scene, cls, tag):
        """Singleton view node owned by self.owner.

        Created with its parent layout node set BEFORE AddNode, so the main layout manager's view factory does
        not build a hidden twin widget for it on creation (vtkMRMLSliceLogic.AddSliceNode / vtkMRMLViewLogic.
        AddViewNode add the node first). Never call AddSliceNode on an existing singleton: it returns an orphan.
        KNOWN (Slicer 5.10, accepted): the factory still adopts these nodes whenever a scene
        batch-process state ends (scene.Clear, volume load), ignoring the parent. The twins are hidden, never
        in frames, and their count is stable; ping reports them. Headless launches have no layout manager.
        """
        n = scene.GetSingletonNode(tag, cls)
        if n is None:
            n = scene.CreateNodeByClass(cls)
            n.UnRegister(None)
            n.SetName(tag)
            n.SetLayoutName(tag)
            n.SetSingletonTag(tag)
            n.SetAndObserveParentLayoutNodeID(self.owner.GetID())
            n = scene.AddNode(n)
        n.SetAndObserveParentLayoutNodeID(self.owner.GetID())
        return n

    @staticmethod
    def adopted_by_layout_manager(tag):
        lm = slicer.app.layoutManager()
        if lm is None:
            return False
        if tag in list(lm.sliceViewNames()):
            return True
        return any(lm.threeDWidget(i).mrmlViewNode() is not None
                   and lm.threeDWidget(i).mrmlViewNode().GetSingletonTag() == tag
                   for i in range(lm.threeDViewCount))

    @staticmethod
    def _owner_node(scene):
        owner = scene.GetSingletonNode(limits.OWNER_TAG, "vtkMRMLScriptedModuleNode")
        if owner is None:
            owner = scene.CreateNodeByClass("vtkMRMLScriptedModuleNode")
            owner.UnRegister(None)
            owner.SetSingletonTag(limits.OWNER_TAG)
            owner.SetName(limits.OWNER_TAG)
            owner = scene.AddNode(owner)
        return owner

    def apply_view_settings(self):
        V, A = slicer.vtkMRMLViewNode, slicer.vtkMRMLAbstractViewNode
        vn = self.view_node
        vn.SetBoxVisible(False)
        vn.SetAxisLabelsVisible(False)
        vn.SetBackgroundColor(*limits.BACKGROUND_RGB)
        vn.SetBackgroundColor2(*limits.BACKGROUND_RGB)
        self._marker(vn, limits.ORIENTATION_MARKER["3d"])
        vn.SetRulerType(getattr(A, "RulerType" + limits.RULER))
        vn.SetRenderMode(getattr(V, limits.RENDER_MODE))
        vn.SetAnimationMode(V.Off)
        vn.SetFPSVisible(False)
        vn.SetVolumeRenderingQuality(getattr(V, limits.VR_QUALITY))
        vn.SetVolumeRenderingSurfaceSmoothing(limits.SURFACE_SMOOTHING)
        vn.SetExpectedFPS(limits.EXPECTED_FPS)
        vn.SetVolumeRenderingOversamplingFactor(limits.OVERSAMPLING)
        vn.SetRaycastTechnique(getattr(V, limits.RAYCAST_TECHNIQUE))
        for name, sn in self.slice_nodes.items():
            w, h = self.slice_view(name).renderWindow().GetSize()
            self.slice_logic(name).ResizeSliceNode(w, h)
            if tuple(sn.GetDimensions()[:2]) != (w, h):
                raise RuntimeError(f"{name} slice node dimensions {sn.GetDimensions()} != render window {w}x{h}")
            sn.SetOrientation(limits.SLICE_VIEWS[name][1])
            self._marker(sn, limits.ORIENTATION_MARKER["slices"])
            sn.SetRulerType(getattr(A, "RulerType" + limits.RULER))
            sn.SetSliceVisible(False)
            sn.SetSliceSpacingModeToAutomatic()
            dn = self.slice_logic(name).GetSliceDisplayNode()
            if dn is not None:
                dn.SetIntersectingSlicesVisibility(False)
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        if crosshair is not None:
            crosshair.SetCrosshairMode(slicer.vtkMRMLCrosshairNode.NoCrosshair)

    @staticmethod
    def _marker(node, spec):
        A = slicer.vtkMRMLAbstractViewNode
        node.SetOrientationMarkerType(getattr(A, "OrientationMarkerType" + spec["type"]))
        node.SetOrientationMarkerSize(getattr(A, "OrientationMarkerSize" + spec["size"]))

    def slice_logic(self, name):
        return self.slice_widgets[name].sliceLogic()

    def slice_view(self, name):
        return self.slice_widgets[name].sliceView()

    def camera_node(self):
        cam = slicer.modules.cameras.logic().GetViewActiveCameraNode(self.view_node)
        if cam is None:
            raise RuntimeError("no active camera node for the observation 3D view")
        return cam

    def renderer(self):
        return self.threed_view.renderWindow().GetRenderers().GetFirstRenderer()

    def all_views(self):
        return {"3d": self.threed_view, **{n: self.slice_view(n) for n in self.slice_widgets}}

    def render_sizes(self):
        return {k: list(v.renderWindow().GetSize()) for k, v in self.all_views().items()}

    def clear_annotations(self):
        for v in self.all_views().values():
            ca = v.cornerAnnotation()
            if ca is not None:
                ca.ClearAllTexts()

    def visibility(self):
        return {"visible": bool(self.widget.isVisible()), "minimized": bool(self.widget.isMinimized())}

    def check_visible(self):
        """A closed or minimized window is not rendered: every capture would return the last frame drawn."""
        v = self.visibility()
        if not v["visible"] or v["minimized"]:
            raise RuntimeError("the observation window is closed or minimized, so frames would be stale "
                               "(the next reset_episode reopens it)")

    def ensure_visible(self):
        """Reopen a closed or minimized window (before every reset). Returns the repairs made; raises if the
        window is still not shown at N x N afterwards."""
        repairs = []
        if self.widget.isMinimized():
            self.widget.showNormal()
            repairs.append("restored_from_minimized")
        if not self.widget.isVisible():
            self.widget.show()
            repairs.append("reopened")
        if repairs:
            self.widget.raise_()
            slicer.app.processEvents()
        self.check_visible()
        if self.fit_to_screen():
            repairs.append("refit_to_screen")
        return repairs

    def capture(self, name):
        """Render view `name` and return PNG bytes. Asserts the window is shown and the N x N pixel size."""
        self.check_visible()
        if self.bad_sizes():
            self.fit_to_screen()
        view = self.all_views()[name]
        self.clear_annotations()
        view.forceRender()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(view.renderWindow())
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.ShouldRerenderOff()
        w2i.Update()
        dims = w2i.GetOutput().GetDimensions()
        if (dims[0], dims[1]) != (self.size_px, self.size_px):
            raise RuntimeError(f"capture size {dims[0]}x{dims[1]} != {self.size_px}x{self.size_px} for view {name}")
        writer = vtk.vtkPNGWriter()
        writer.SetCompressionLevel(limits.PNG_COMPRESSION_LEVEL)
        writer.WriteToMemoryOn()
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        return bytes(memoryview(writer.GetResult()))

    def _counter(self, name):
        def count(caller, event):
            self.render_counts[name] = self.render_counts.get(name, 0) + 1
        return count

    def close(self):
        for rw, tag in getattr(self, "_render_observers", []):
            rw.RemoveObserver(tag)
        self._render_observers = []
        logics = app_slice_logics()
        for sl in self._slice_logics:
            logics.RemoveItem(sl)
        self._slice_logics = []
        self.widget.close()
        self.widget.deleteLater()
