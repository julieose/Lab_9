"""Qt выводит готовый CPU-кадр. Модель рисуется только растеризатором."""
import numpy as np

from .qt import QtCore as C, QtGui as G, QtWidgets as W, Signal
from .camera import Camera
from .software_renderer import RenderOptions, render_scene


class Viewport(W.QWidget):
    camera_changed = Signal()
    frame_ready = Signal(str)

    def __init__(self):
        super().__init__()
        self.camera = Camera()
        self.render_options = RenderOptions()
        self.items = []
        self.show_axes = False
        self.drag = None
        self.setMinimumSize(420,420)
        self.setCursor(C.Qt.CursorShape.OpenHandCursor)
        self.setSizePolicy(W.QSizePolicy.Policy.Expanding,W.QSizePolicy.Policy.Expanding)

    def fit(self):
        if self.items:
            self.camera.fit(np.concatenate([o.model.world_vertices for o in self.items]),
                            self.width()/max(1,self.height()))
            self.camera_changed.emit()
            self.update()

    def paintEvent(self, event):
        painter = G.QPainter(self)
        painter.fillRect(self.rect(), G.QColor("#101a29"))
        w,h = self.width(),self.height()
        factor = min(1.,640/w,510/h)
        rw,rh = max(1,int(w*factor)),max(1,int(h*factor))
        result = render_scene(self.items,self.camera,rw,rh,self.render_options)
        image = G.QImage(result.pixels.tobytes(),rw,rh,rw*4,G.QImage.Format.Format_RGBA8888)
        painter.setRenderHint(G.QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.rect(), image)
        if self.show_axes:
            # Миниатюра на экране показывает направление мировых осей.
            v = self.camera.view_matrix()
            painter.setRenderHint(G.QPainter.RenderHint.Antialiasing)
            origin = C.QPointF(56,self.height()-67)
            for i,(label,color) in enumerate(zip("XYZ",("#f68391","#6fdbad","#86b8ff"))):
                end = origin+C.QPointF(29*v[0,i],-29*v[1,i])
                painter.setPen(G.QPen(G.QColor(color),2))
                painter.drawLine(origin,end)
                painter.drawText(end+C.QPointF(4,-3),label)
        painter.end()
        self.frame_ready.emit(f"Граней {result.total_faces}   ·   Отсечено {result.culled_faces}   ·   В растре {result.rendered_faces}")

    @staticmethod
    def _point(event):
        return event.position() if hasattr(event,"position") else event.localPos()

    def mousePressEvent(self,event):
        if event.button()==C.Qt.MouseButton.LeftButton:
            self.drag = self._point(event)
            self.setCursor(C.Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self,event):
        if self.drag is not None:
            p = self._point(event)
            delta = p-self.drag
            self.drag = p
            self.camera.orbit(delta.x()*.42,-delta.y()*.42)
            self.camera_changed.emit()
            self.update()

    def mouseReleaseEvent(self,event):
        self.drag = None
        self.setCursor(C.Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self,event):
        delta = event.angleDelta().y()
        if delta:
            self.camera.zoom(.87**(delta/120))
            self.camera_changed.emit()
            self.update()
