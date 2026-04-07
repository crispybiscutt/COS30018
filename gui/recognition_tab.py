"""
COS30018 - Recognition Tab
Tab for recognizing handwritten digits, numbers, and arithmetic expressions.
Supports: single digits, multi-digit numbers, and expressions with +, -, *, /
"""
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QMessageBox, QFrame, QCheckBox, QFormLayout, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage, QColor, QPainter
from gui.drawing_canvas import DrawingCanvas
from gui.theme import btn_primary, btn_success, btn_danger, PRIMARY, SUCCESS, DANGER, TEXT_SECONDARY
from config import (
    MODEL_CNN_KERAS, MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN,
    PREPROCESS_BASIC, PREPROCESS_OTSU, PREPROCESS_ADAPTIVE, PREPROCESS_PHOTO,
    SEGMENT_CONTOUR, SEGMENT_CONNECTED, SEGMENT_PROJECTION,
)
from models.model_manager import get_available_models


class RecognitionTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_model = None
        self.expression_model = None
        self._init_ui()
        self._load_model()
        # Also pre-load expression model
        self._load_expression_model()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # === Top: Settings row ===
        settings_row = QHBoxLayout()
        settings_row.setSpacing(12)

        # Model
        settings_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(140)
        names = {
            MODEL_CNN_PYTORCH: "CNN (PyTorch)",
            MODEL_SVM: "SVM",
            MODEL_KNN: "KNN",
            MODEL_CNN_KERAS: "CNN (Keras)",
        }
        for mid in get_available_models():
            self.model_combo.addItem(names.get(mid, mid), mid)
        self.model_combo.currentIndexChanged.connect(self._load_model)
        settings_row.addWidget(self.model_combo)

        # Preprocessing
        settings_row.addWidget(QLabel("Preprocessing:"))
        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItem("Basic", PREPROCESS_BASIC)
        self.preprocess_combo.addItem("Otsu", PREPROCESS_OTSU)
        self.preprocess_combo.addItem("Adaptive", PREPROCESS_ADAPTIVE)
        self.preprocess_combo.addItem("Photo (Camera)", PREPROCESS_PHOTO)
        settings_row.addWidget(self.preprocess_combo)

        # Segmentation
        settings_row.addWidget(QLabel("Segmentation:"))
        self.segment_combo = QComboBox()
        self.segment_combo.addItem("Contour", SEGMENT_CONTOUR)
        self.segment_combo.addItem("Connected Components", SEGMENT_CONNECTED)
        self.segment_combo.addItem("Projection", SEGMENT_PROJECTION)
        settings_row.addWidget(self.segment_combo)

        # Expression mode
        self.expr_mode = QCheckBox("Expression Mode")
        self.expr_mode.setToolTip("Enable to recognize +, -, *, / operators and compute result")
        self.expr_mode.setChecked(True)  # Default ON for more impressive demo
        settings_row.addWidget(self.expr_mode)

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-style: italic;")
        settings_row.addStretch()
        settings_row.addWidget(self.model_status)

        main_layout.addLayout(settings_row)

        # === Middle: Canvas + Buttons ===
        canvas_grp = QGroupBox("Draw Here (digits, numbers, or expressions like 2+3)")
        canvas_layout = QVBoxLayout()
        canvas_layout.setSpacing(8)

        self.canvas = DrawingCanvas()
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        btns = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(btn_danger())
        self.clear_btn.setFixedWidth(120)
        self.clear_btn.clicked.connect(self.canvas.clear)
        btns.addWidget(self.clear_btn)

        self.predict_draw_btn = QPushButton("Recognize")
        self.predict_draw_btn.setStyleSheet(btn_primary())
        self.predict_draw_btn.setFixedWidth(160)
        self.predict_draw_btn.clicked.connect(self._predict_drawing)
        btns.addWidget(self.predict_draw_btn)

        btns.addSpacing(20)

        self.load_image_btn = QPushButton("Upload Photo")
        self.load_image_btn.clicked.connect(self._load_image)
        self.load_image_btn.setFixedWidth(130)
        btns.addWidget(self.load_image_btn)

        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self._load_from_folder)
        self.load_folder_btn.setFixedWidth(120)
        btns.addWidget(self.load_folder_btn)

        btns.addStretch()
        canvas_layout.addLayout(btns)
        canvas_grp.setLayout(canvas_layout)
        main_layout.addWidget(canvas_grp)

        # === Bottom: Results (horizontal) ===
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        # Result display
        result_grp = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout()

        self.result_label = QLabel("?")
        self.result_label.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            f"color: {PRIMARY}; background: #eff6ff; border: 2px solid #bfdbfe; "
            "border-radius: 12px; padding: 12px; min-height: 60px;"
        )
        result_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Draw something and click Recognize")
        self.confidence_label.setFont(QFont("Segoe UI", 10))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        result_layout.addWidget(self.confidence_label)

        # Expression result (= answer)
        self.expr_result_label = QLabel("")
        self.expr_result_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.expr_result_label.setAlignment(Qt.AlignCenter)
        self.expr_result_label.setStyleSheet(
            f"color: {SUCCESS}; background: #f0fdf4; border: 2px solid #bbf7d0; "
            "border-radius: 10px; padding: 10px;"
        )
        self.expr_result_label.setVisible(False)
        result_layout.addWidget(self.expr_result_label)

        result_grp.setLayout(result_layout)
        bottom.addWidget(result_grp, stretch=1)

        # Segmented symbols display
        seg_grp = QGroupBox("Segmented Symbols")
        seg_layout = QVBoxLayout()
        self.segments_container = QHBoxLayout()
        self.segments_container.setSpacing(4)
        self.segments_widget = QWidget()
        self.segments_widget.setLayout(self.segments_container)

        seg_scroll = QScrollArea()
        seg_scroll.setWidget(self.segments_widget)
        seg_scroll.setWidgetResizable(True)
        seg_scroll.setFixedHeight(100)
        seg_scroll.setStyleSheet("border: 1px solid #e2e8f0; border-radius: 6px; background: #f8fafc;")
        seg_layout.addWidget(seg_scroll)

        self.seg_info_label = QLabel("Segments will appear here after recognition")
        self.seg_info_label.setFont(QFont("Segoe UI", 9))
        self.seg_info_label.setAlignment(Qt.AlignCenter)
        self.seg_info_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        seg_layout.addWidget(self.seg_info_label)

        seg_grp.setLayout(seg_layout)
        bottom.addWidget(seg_grp, stretch=1)

        # Probability bars
        proba_grp = QGroupBox("Last Symbol Confidence")
        self.proba_layout = QGridLayout()
        self.proba_layout.setSpacing(2)
        self.proba_bars = []
        self.proba_labels = []

        for i in range(10):
            lbl = QLabel(f"{i}")
            lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
            lbl.setFixedWidth(14)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.proba_layout.addWidget(lbl, i, 0)

            bar = QFrame()
            bar.setFixedHeight(14)
            bar.setStyleSheet("background: #e2e8f0; border-radius: 3px;")
            self.proba_layout.addWidget(bar, i, 1)
            self.proba_bars.append(bar)

            pct = QLabel("0%")
            pct.setFixedWidth(42)
            pct.setFont(QFont("Segoe UI", 8))
            pct.setStyleSheet(f"color: {TEXT_SECONDARY};")
            self.proba_layout.addWidget(pct, i, 2)
            self.proba_labels.append(pct)

        proba_grp.setLayout(self.proba_layout)
        bottom.addWidget(proba_grp, stretch=1)

        main_layout.addLayout(bottom)
        self.loaded_image = None

    def _load_model(self):
        model_name = self.model_combo.currentData()
        try:
            from models.model_manager import load_trained_model
            self.current_model = load_trained_model(model_name)
            if self.current_model and self.current_model.is_trained:
                self.model_status.setText(f"Loaded: {self.current_model.name}")
                self.model_status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
            else:
                self.model_status.setText(f"No trained model for {model_name}")
                self.model_status.setStyleSheet("color: #dc2626; font-style: italic;")
                self.current_model = None
        except Exception as e:
            self.model_status.setText(f"Error: {e}")
            self.model_status.setStyleSheet("color: #dc2626;")

    def _load_expression_model(self):
        if self.expression_model is None:
            try:
                from extension.operator_recognizer import load_expression_model
                self.expression_model = load_expression_model()
            except Exception:
                pass
        return self.expression_model

    def _predict_drawing(self):
        if not self.current_model:
            QMessageBox.warning(self, "No Model", "Please load a trained model first!")
            return
        if self.canvas.is_empty():
            QMessageBox.warning(self, "Empty Canvas", "Please draw something first!")
            return
        # Always use segmentation-based recognition (not single digit)
        self._recognize_image(self.canvas.get_image_array())

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload Photo / Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            import cv2
            # Load in color first (camera photos are color)
            self.loaded_image = cv2.imread(path)
            if self.loaded_image is None:
                QMessageBox.warning(self, "Error", f"Could not load image: {path}")
                return
            # Auto-switch to Photo preprocessing for camera images
            # (detect by checking if image is large - camera photos are typically >200px)
            h, w = self.loaded_image.shape[:2]
            if max(h, w) > 300:
                photo_idx = self.preprocess_combo.findData(PREPROCESS_PHOTO)
                if photo_idx >= 0:
                    self.preprocess_combo.setCurrentIndex(photo_idx)
            self._recognize_image(self.loaded_image)

    def _load_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Digit Folder")
        if folder:
            try:
                from utils.image_utils import create_number_from_folder
                self.loaded_image = create_number_from_folder(folder)
                self._recognize_image(self.loaded_image)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _recognize_image(self, image):
        """Universal recognition: handles single digit, multi-digit, and expressions."""
        import cv2
        from preprocessing.preprocessor import normalize_segmented
        from segmentation.segmenter import segment
        from models.model_manager import predict_digit
        from utils.image_utils import canvas_to_mnist_format

        seg_method = self.segment_combo.currentData()
        preprocess_method = self.preprocess_combo.currentData()
        is_expression = self.expr_mode.isChecked()

        try:
            # Step 1: Segment the image (using selected preprocessing)
            digits, boxes = segment(image, method=seg_method,
                                    preprocess_method=preprocess_method)

            # If segmentation found nothing, try treating as single digit
            if not digits:
                processed = canvas_to_mnist_format(image)
                label, confidence, proba = predict_digit(self.current_model, processed)
                self.result_label.setText(str(label))
                self.confidence_label.setText(f"Confidence: {confidence:.1%}")
                self.expr_result_label.setVisible(False)
                self._update_proba_bars(proba)
                self._clear_segments()
                self.seg_info_label.setText("Single digit detected")
                return

            # Step 2: If only 1 segment and not expression mode, treat as single digit
            if len(digits) == 1 and not is_expression:
                processed = normalize_segmented(digits[0])
                label, confidence, proba = predict_digit(self.current_model, processed)
                self.result_label.setText(str(label))
                self.confidence_label.setText(f"Confidence: {confidence:.1%}")
                self.expr_result_label.setVisible(False)
                self._update_proba_bars(proba)
                self._show_segments(digits, [str(label)])
                self.seg_info_label.setText("1 symbol detected")
                return

            # Step 3: Expression mode - use 16-class ExpressionCNN + dedicated digit model
            if is_expression:
                expr_model = self._load_expression_model()
                if expr_model is not None:
                    from extension.operator_recognizer import classify_symbol
                    from extension.expression_evaluator import _build_expression, _safe_eval

                    symbols = []
                    symbol_labels = []
                    for digit_img in digits:
                        processed = normalize_segmented(digit_img)
                        # Pass digit model for precise digit classification
                        sym_type, value = classify_symbol(
                            processed, expr_model, digit_model=self.current_model
                        )
                        symbols.append((sym_type, value))
                        symbol_labels.append(str(value))

                    expression = _build_expression(symbols)
                    self.result_label.setText(expression or "?")
                    self._show_segments(digits, symbol_labels)
                    self.seg_info_label.setText(
                        f"{len(digits)} symbols: {', '.join(symbol_labels)}"
                    )

                    # Evaluate expression
                    result, error = _safe_eval(expression)
                    self.expr_result_label.setVisible(True)
                    if result is not None:
                        self.expr_result_label.setText(f"= {result}")
                        self.expr_result_label.setStyleSheet(
                            f"color: {SUCCESS}; background: #f0fdf4; border: 2px solid #bbf7d0; "
                            "border-radius: 10px; padding: 10px;"
                        )
                        self.confidence_label.setText(
                            f"Expression recognized with {len(digits)} symbols"
                        )
                    else:
                        self.expr_result_label.setText(f"= ? ({error})")
                        self.expr_result_label.setStyleSheet(
                            f"color: {DANGER}; background: #fef2f2; border: 2px solid #fecaca; "
                            "border-radius: 10px; padding: 10px;"
                        )
                        self.confidence_label.setText(error or "")

                    # Show proba bars for last digit-type symbol
                    self._reset_proba_bars()
                    return

            # Step 4: Digit-only mode with multiple segments
            self.expr_result_label.setVisible(False)
            number_str = ""
            last_proba = None
            symbol_labels = []
            for digit_img in digits:
                processed = normalize_segmented(digit_img)
                label, conf, proba = predict_digit(self.current_model, processed)
                number_str += str(label)
                symbol_labels.append(str(label))
                last_proba = proba

            self.result_label.setText(number_str)
            self.confidence_label.setText(f"Detected {len(digits)} digit(s)")
            self._show_segments(digits, symbol_labels)
            self.seg_info_label.setText(f"{len(digits)} digits detected")
            if last_proba is not None:
                self._update_proba_bars(last_proba)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Prediction Error", str(e))

    def _show_segments(self, digit_images, labels):
        """Display segmented symbol images with their predicted labels."""
        self._clear_segments()

        for i, (img, label) in enumerate(zip(digit_images, labels)):
            frame = QFrame()
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(2, 2, 2, 2)
            frame_layout.setSpacing(2)

            # Convert numpy to QPixmap
            h, w = img.shape[:2]
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("background: white; border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px;")
            frame_layout.addWidget(img_label)

            txt = QLabel(str(label))
            txt.setFont(QFont("Segoe UI", 10, QFont.Bold))
            txt.setAlignment(Qt.AlignCenter)
            txt.setStyleSheet(f"color: {PRIMARY};")
            frame_layout.addWidget(txt)

            self.segments_container.addWidget(frame)

        self.segments_container.addStretch()

    def _clear_segments(self):
        """Remove all segment display widgets."""
        while self.segments_container.count():
            item = self.segments_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_proba_bars(self, probabilities):
        max_width = 140
        max_idx = np.argmax(probabilities)
        for i in range(10):
            pct = probabilities[i] * 100
            width = int(probabilities[i] * max_width)
            if i == max_idx:
                self.proba_bars[i].setFixedWidth(max(width, 4))
                self.proba_bars[i].setStyleSheet("background: #2563eb; border-radius: 3px;")
                self.proba_labels[i].setStyleSheet(f"color: {PRIMARY}; font-weight: bold;")
            else:
                self.proba_bars[i].setFixedWidth(max(width, 2))
                self.proba_bars[i].setStyleSheet("background: #e2e8f0; border-radius: 3px;")
                self.proba_labels[i].setStyleSheet(f"color: {TEXT_SECONDARY};")
            self.proba_labels[i].setText(f"{pct:.1f}%")

    def _reset_proba_bars(self):
        for i in range(10):
            self.proba_bars[i].setFixedWidth(2)
            self.proba_bars[i].setStyleSheet("background: #e2e8f0; border-radius: 3px;")
            self.proba_labels[i].setStyleSheet(f"color: {TEXT_SECONDARY};")
            self.proba_labels[i].setText("0%")
