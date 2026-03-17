"""
COS30018 - Drawing Canvas Widget
PyQt5 canvas widget that allows users to draw digits with mouse/stylus.
Exports the drawn image as a numpy array for prediction.
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
import numpy as np


class DrawingCanvas(QWidget):
    """
    A canvas widget for drawing handwritten digits.
    White background, black brush strokes.
    """

    def __init__(self, parent=None, width=280, height=280):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setStyleSheet("border: 2px solid #333; background-color: white;")

        # Create QImage as the drawing surface
        self.image = QImage(width, height, QImage.Format_RGB32)
        self.image.fill(QColor(255, 255, 255))

        # Drawing settings
        self.pen_color = QColor(0, 0, 0)
        self.pen_width = 15  # Thick pen for handwriting
        self.drawing = False
        self.last_point = QPoint()

    def set_pen_width(self, width):
        """Change the pen thickness."""
        self.pen_width = width

    def clear(self):
        """Clear the canvas to white."""
        self.image.fill(QColor(255, 255, 255))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            # Draw a dot at the press point
            painter = QPainter(self.image)
            painter.setPen(QPen(self.pen_color, self.pen_width,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPoint(event.pos())
            painter.end()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
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
        """
        Convert the canvas drawing to a numpy array.

        Returns:
            Grayscale numpy array of the canvas content (height x width).
        """
        # Convert QImage to numpy array
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))

        # Convert to grayscale (take one channel since drawing is B&W)
        gray = arr[:, :, 0]  # R channel (same as G and B for grayscale drawing)
        return gray.copy()

    def is_empty(self):
        """Check if canvas has any drawing (not all white)."""
        arr = self.get_image_array()
        return np.all(arr > 240)
