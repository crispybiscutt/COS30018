"""
COS30018 - Recognition Tab
PyQt5 tab for recognizing handwritten digits and numbers.
Features: drawing canvas, image upload, folder loading, prediction display.
"""
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QMessageBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from gui.drawing_canvas import DrawingCanvas
from config import AVAILABLE_MODELS, MODEL_CNN_KERAS, MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN


class RecognitionTab(QWidget):
    """Tab widget for digit/number recognition."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_model = None
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # --- Left Panel: Input ---
        left_panel = QVBoxLayout()

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        model_names = {
            MODEL_CNN_KERAS: "CNN (Keras)",
            MODEL_CNN_PYTORCH: "CNN (PyTorch)",
            MODEL_SVM: "SVM",
            MODEL_KNN: "KNN",
        }
        for model_id in AVAILABLE_MODELS:
            self.model_combo.addItem(model_names.get(model_id, model_id), model_id)
        self.model_combo.currentIndexChanged.connect(self._load_model)
        model_layout.addWidget(self.model_combo)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_btn)
        left_panel.addLayout(model_layout)

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #f44336; font-style: italic;")
        left_panel.addWidget(self.model_status)

        # Drawing canvas
        canvas_group = QGroupBox("Draw Here (Single Digit or Number)")
        canvas_layout = QVBoxLayout()
        self.canvas = DrawingCanvas()
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        # Canvas buttons
        canvas_btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear)
        self.clear_btn.setStyleSheet("padding: 5px 15px;")
        canvas_btn_layout.addWidget(self.clear_btn)

        self.predict_draw_btn = QPushButton("Predict Drawing")
        self.predict_draw_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 5px 15px; border-radius: 3px; }"
        )
        self.predict_draw_btn.clicked.connect(self._predict_drawing)
        canvas_btn_layout.addWidget(self.predict_draw_btn)
        canvas_layout.addLayout(canvas_btn_layout)
        canvas_group.setLayout(canvas_layout)
        left_panel.addWidget(canvas_group)

        # Image loading options
        load_group = QGroupBox("Or Load Image")
        load_layout = QVBoxLayout()

        self.load_image_btn = QPushButton("Load Image File")
        self.load_image_btn.clicked.connect(self._load_image)
        load_layout.addWidget(self.load_image_btn)

        self.load_folder_btn = QPushButton("Create Number from Digit Folder")
        self.load_folder_btn.clicked.connect(self._load_from_folder)
        load_layout.addWidget(self.load_folder_btn)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedHeight(100)
        self.image_label.setStyleSheet("border: 1px solid #ddd; background: #fafafa;")
        load_layout.addWidget(self.image_label)

        self.predict_image_btn = QPushButton("Predict Image")
        self.predict_image_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 5px 15px; border-radius: 3px; }"
        )
        self.predict_image_btn.clicked.connect(self._predict_loaded_image)
        self.predict_image_btn.setEnabled(False)
        load_layout.addWidget(self.predict_image_btn)

        load_group.setLayout(load_layout)
        left_panel.addWidget(load_group)

        layout.addLayout(left_panel, stretch=1)

        # --- Right Panel: Results ---
        right_panel = QVBoxLayout()

        result_group = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout()

        self.result_label = QLabel("?")
        self.result_label.setFont(QFont("Arial", 72, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "color: #1976D2; background: white; border: 2px solid #1976D2; "
            "border-radius: 10px; padding: 20px; min-height: 120px;"
        )
        result_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setFont(QFont("Arial", 14))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.confidence_label)

        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group)

        # Probability distribution
        proba_group = QGroupBox("Probability Distribution")
        self.proba_layout = QGridLayout()
        self.proba_bars = []
        self.proba_labels = []

        for i in range(10):
            label = QLabel(f"{i}:")
            label.setFont(QFont("Arial", 11, QFont.Bold))
            self.proba_layout.addWidget(label, i, 0)

            bar = QFrame()
            bar.setFixedHeight(18)
            bar.setStyleSheet("background-color: #e0e0e0; border-radius: 3px;")
            self.proba_layout.addWidget(bar, i, 1)
            self.proba_bars.append(bar)

            pct = QLabel("0%")
            pct.setFixedWidth(50)
            self.proba_layout.addWidget(pct, i, 2)
            self.proba_labels.append(pct)

        proba_group.setLayout(self.proba_layout)
        right_panel.addWidget(proba_group)

        right_panel.addStretch()
        layout.addLayout(right_panel, stretch=1)

        self.loaded_image = None

    def _load_model(self):
        """Load the selected trained model."""
        model_name = self.model_combo.currentData()
        try:
            from models.model_manager import load_trained_model
            self.current_model = load_trained_model(model_name)
            if self.current_model and self.current_model.is_trained:
                self.model_status.setText(f"Loaded: {self.current_model.name}")
                self.model_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                self.model_status.setText(f"No trained model found for {model_name}. Train first!")
                self.model_status.setStyleSheet("color: #f44336; font-style: italic;")
                self.current_model = None
        except Exception as e:
            self.model_status.setText(f"Error: {e}")
            self.model_status.setStyleSheet("color: #f44336;")

    def _predict_drawing(self):
        """Predict the digit drawn on canvas."""
        if self.current_model is None:
            QMessageBox.warning(self, "No Model", "Please load a trained model first!")
            return

        if self.canvas.is_empty():
            QMessageBox.warning(self, "Empty Canvas", "Please draw a digit first!")
            return

        canvas_img = self.canvas.get_image_array()
        self._recognize_image(canvas_img, is_single_digit=True)

    def _load_image(self):
        """Load an image file for recognition."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            import cv2
            self.loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._show_loaded_image(self.loaded_image)
            self.predict_image_btn.setEnabled(True)

    def _load_from_folder(self):
        """Create number image from a folder of digit images."""
        folder = QFileDialog.getExistingDirectory(self, "Select Digit Folder")
        if folder:
            try:
                from utils.image_utils import create_number_from_folder
                self.loaded_image = create_number_from_folder(folder)
                self._show_loaded_image(self.loaded_image)
                self.predict_image_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _show_loaded_image(self, img):
        """Display loaded image in the preview label."""
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            q_img = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(250, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _predict_loaded_image(self):
        """Predict the loaded image (could be multi-digit)."""
        if self.current_model is None:
            QMessageBox.warning(self, "No Model", "Please load a trained model first!")
            return

        if self.loaded_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first!")
            return

        self._recognize_image(self.loaded_image, is_single_digit=False)

    def _recognize_image(self, image, is_single_digit=False):
        """Run the full recognition pipeline on an image."""
        import cv2
        from preprocessing.preprocessor import preprocess
        from segmentation.segmenter import segment
        from models.model_manager import predict_digit

        try:
            if is_single_digit:
                # Single digit: just preprocess and predict
                from utils.image_utils import canvas_to_mnist_format
                processed = canvas_to_mnist_format(image)
                label, confidence, proba = predict_digit(self.current_model, processed)

                self.result_label.setText(str(label))
                self.confidence_label.setText(f"Confidence: {confidence:.1%}")
                self._update_proba_bars(proba)
            else:
                # Multi-digit: segment then predict each digit
                digits, boxes = segment(image)

                if not digits:
                    QMessageBox.warning(self, "No Digits", "Could not find any digits in the image!")
                    return

                number_str = ""
                for digit_img in digits:
                    processed = preprocess(digit_img)
                    label, conf, proba = predict_digit(self.current_model, processed)
                    number_str += str(label)

                self.result_label.setText(number_str)
                self.confidence_label.setText(f"Detected {len(digits)} digit(s)")
                # Show proba of last digit
                self._update_proba_bars(proba)

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))

    def _update_proba_bars(self, probabilities):
        """Update the probability bar display."""
        max_width = 200
        max_idx = np.argmax(probabilities)

        for i in range(10):
            pct = probabilities[i] * 100
            width = int(probabilities[i] * max_width)

            color = "#4CAF50" if i == max_idx else "#2196F3"
            self.proba_bars[i].setFixedWidth(max(width, 2))
            self.proba_bars[i].setStyleSheet(
                f"background-color: {color}; border-radius: 3px;"
            )
            self.proba_labels[i].setText(f"{pct:.1f}%")
