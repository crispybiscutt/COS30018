"""
COS30018 - Drawing Canvas Widget
Canvas for drawing handwritten digits.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
import numpy as np


class DrawingCanvas(QWidget):
    """Canvas widget - white background, black strokes (like paper)."""

    def __init__(self, parent=None, width=560, height=200):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setStyleSheet(
            "border: 2px solid #cbd5e1; border-radius: 8px; background: #ffffff;"
        )

        self.image = QImage(width, height, QImage.Format_RGB32)
        self.image.fill(QColor(255, 255, 255))

        self.pen_color = QColor(30, 41, 59)  # Dark ink
        self.pen_width = 15
        self.drawing = False
        self.last_point = QPoint()

    def clear(self):
        self.image.fill(QColor(255, 255, 255))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(self.pen_color, self.pen_width,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPoint(event.pos())
            painter.end()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(self.pen_color, self.pen_width,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def get_image_array(self):
        """Convert canvas to grayscale numpy array (white bg, black strokes)."""
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        gray = arr[:, :, 0].copy()
        return gray

    def is_empty(self):
        arr = self.get_image_array()
        return np.all(arr > 240)
