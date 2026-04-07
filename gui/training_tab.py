"""
COS30018 - Models Tab (Modern Dark Theme)
Displays information about pre-trained models and allows re-training if needed.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout, QFrame, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTextEdit, QFormLayout,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, SAVED_MODELS_DIR,
    MODEL_CNN_KERAS, MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN
)
from models.model_manager import get_available_models
from gui.theme import (
    btn_success, btn_danger, btn_warning,
    SUCCESS, PRIMARY, TEXT_SECONDARY, WARNING
)


class TrainingWorker(QThread):
    """Background thread for model training."""
    progress = pyqtSignal(int, int, dict)
    finished = pyqtSignal(str, dict, float)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_name, epochs, batch_size):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            from models.model_manager import train_model

            def callback(epoch, total, history):
                self.progress.emit(epoch, total, history)
                if "accuracy" in history:
                    acc = history["accuracy"][-1]
                    self.log.emit(f"Epoch {epoch}/{total}  |  acc: {acc:.4f}")

            self.log.emit(f"Starting training: {self.model_name}")
            model, history, train_time = train_model(
                self.model_name,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callback=callback,
            )
            self.finished.emit(self.model_name, history, train_time)
        except Exception as e:
            self.error.emit(str(e))


class TrainingTab(QWidget):
    """Tab showing pre-trained model info with option to re-train."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        # === Model Cards ===
        cards_group = QGroupBox("Pre-trained Models")
        cards_layout = QGridLayout()
        cards_layout.setSpacing(12)

        models_info = [
            (MODEL_CNN_PYTORCH, "CNN (PyTorch)", "99.45%", "413.2s",
             "Conv2D(32)→Conv2D(64)→Conv2D(64)→Dense(128)→Dense(10) + Data Augmentation", PRIMARY),
            (MODEL_SVM, "SVM", "96.31%", "343.7s",
             "RBF kernel, C=10, gamma=scale, 20K samples", "#7c3aed"),
            (MODEL_KNN, "KNN", "93.14%", "7.7s",
             "k=5, distance-weighted, 20K samples", SUCCESS),
        ]

        for col, (model_id, name, acc, time_str, desc, color) in enumerate(models_info):
            card = self._create_model_card(model_id, name, acc, time_str, desc, color)
            cards_layout.addWidget(card, 0, col)

        cards_group.setLayout(cards_layout)
        layout.addWidget(cards_group)

        # === Re-train Section (collapsible) ===
        train_group = QGroupBox("Re-train Model (Optional)")
        train_layout = QVBoxLayout()
        train_layout.setSpacing(10)

        # Settings row
        settings_row = QHBoxLayout()

        self.model_combo = QComboBox()
        model_display_names = {
            MODEL_CNN_PYTORCH: "CNN (PyTorch)",
            MODEL_SVM: "SVM",
            MODEL_KNN: "KNN",
        }
        for model_id in get_available_models():
            self.model_combo.addItem(
                model_display_names.get(model_id, model_id), model_id
            )
        settings_row.addWidget(QLabel("Model:"))
        settings_row.addWidget(self.model_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(EPOCHS)
        settings_row.addWidget(QLabel("Epochs:"))
        settings_row.addWidget(self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 512)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(BATCH_SIZE)
        settings_row.addWidget(QLabel("Batch:"))
        settings_row.addWidget(self.batch_spin)

        settings_row.addStretch()

        self.train_btn = QPushButton("Train")
        self.train_btn.setStyleSheet(btn_success())
        self.train_btn.clicked.connect(self._start_training)
        settings_row.addWidget(self.train_btn)

        train_layout.addLayout(settings_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        train_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setPlaceholderText("Training log will appear here...")
        train_layout.addWidget(self.log_text)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        layout.addStretch()

    def _create_model_card(self, model_id, name, accuracy, train_time, desc, color):
        """Create a styled card widget for a model."""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: #ffffff; border: 1px solid #e2e8f0; "
            f"border-radius: 10px; border-top: 3px solid {color}; }}"
        )

        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(8)
        card_layout.setContentsMargins(16, 14, 16, 14)

        title = QLabel(name)
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet(f"color: {color}; border: none;")
        card_layout.addWidget(title)

        acc_label = QLabel(accuracy)
        acc_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        acc_label.setStyleSheet("color: #1e293b; border: none;")
        acc_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(acc_label)

        acc_sub = QLabel("accuracy")
        acc_sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; border: none;")
        acc_sub.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(acc_sub)

        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #e2e8f0; border: none;")
        card_layout.addWidget(line)

        # Details
        time_label = QLabel(f"Training time: {train_time}")
        time_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; border: none;")
        card_layout.addWidget(time_label)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; border: none;")
        card_layout.addWidget(desc_label)

        # Status
        model_path = os.path.join(SAVED_MODELS_DIR, f"{model_id}.pth" if "cnn" in model_id else f"{model_id}.pkl")
        if os.path.exists(model_path):
            status = QLabel("Ready")
            status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 11px; border: none;")
        else:
            status = QLabel("Not trained")
            status.setStyleSheet(f"color: {WARNING}; font-style: italic; font-size: 11px; border: none;")
        status.setAlignment(Qt.AlignRight)
        card_layout.addWidget(status)

        return card

    def _start_training(self):
        model_name = self.model_combo.currentData()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()

        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.worker = TrainingWorker(model_name, epochs, batch_size)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(lambda msg: self.log_text.append(msg))
        self.worker.start()

    def _on_progress(self, current, total, history):
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_finished(self, model_name, history, train_time):
        self.train_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        acc = history.get("accuracy", [0])[-1]
        self.log_text.append(f"\nDone! Accuracy: {acc:.4f}, Time: {train_time:.1f}s")
        QMessageBox.information(self, "Complete", f"{model_name}: {acc:.4f} in {train_time:.1f}s")

    def _on_error(self, error_msg):
        self.train_btn.setEnabled(True)
        self.log_text.append(f"\nERROR: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
