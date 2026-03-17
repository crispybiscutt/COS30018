"""
COS30018 - Training Tab
PyQt5 tab for training ML models with configurable hyperparameters.
Features: model selection, hyperparameter controls, progress bar, training log.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QProgressBar,
    QTextEdit, QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_CNN_KERAS, MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN
)
from models.model_manager import get_available_models


class TrainingWorker(QThread):
    """Background thread for model training to keep GUI responsive."""
    progress = pyqtSignal(int, int, dict)  # current_epoch, total_epochs, history
    finished = pyqtSignal(str, dict, float)  # model_name, history, train_time
    error = pyqtSignal(str)  # error message
    log = pyqtSignal(str)  # log message

    def __init__(self, model_name, epochs, batch_size):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        try:
            import sys
            import io

            from models.model_manager import train_model

            def callback(epoch, total, history):
                self.progress.emit(epoch, total, history)
                if "accuracy" in history:
                    acc = history["accuracy"][-1]
                    self.log.emit(f"Epoch {epoch}/{total} - accuracy: {acc:.4f}")

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
    """Tab widget for model training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Model Selection ---
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()

        self.model_combo = QComboBox()
        model_display_names = {
            MODEL_CNN_KERAS: "CNN (Keras/TensorFlow)",
            MODEL_CNN_PYTORCH: "CNN (PyTorch)",
            MODEL_SVM: "SVM (scikit-learn)",
            MODEL_KNN: "KNN (scikit-learn)",
        }
        for model_id in get_available_models():
            self.model_combo.addItem(model_display_names.get(model_id, model_id), model_id)

        model_layout.addRow("Model:", self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # --- Hyperparameters ---
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(EPOCHS)
        hyper_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 512)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(BATCH_SIZE)
        hyper_layout.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(LEARNING_RATE)
        hyper_layout.addRow("Learning Rate:", self.lr_spin)

        hyper_group.setLayout(hyper_layout)
        layout.addWidget(hyper_group)

        # --- Train Button & Progress ---
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 14px; padding: 8px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #ccc; }"
        )
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-size: 14px; padding: 8px 20px; border-radius: 4px; }"
        )
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # --- Training Log ---
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

    def _start_training(self):
        """Start training in background thread."""
        model_name = self.model_combo.currentData()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_text.append(f"Training {model_name}...")

        self.worker = TrainingWorker(model_name, epochs, batch_size)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._on_log)
        self.worker.start()

    def _on_progress(self, current, total, history):
        """Update progress bar."""
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)

    def _on_finished(self, model_name, history, train_time):
        """Training completed."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        final_acc = history.get("accuracy", [0])[-1]
        self.log_text.append(f"\nTraining complete!")
        self.log_text.append(f"Final accuracy: {final_acc:.4f}")
        self.log_text.append(f"Training time: {train_time:.1f}s")
        self.log_text.append(f"Model saved successfully.")

        QMessageBox.information(
            self, "Training Complete",
            f"Model: {model_name}\n"
            f"Accuracy: {final_acc:.4f}\n"
            f"Time: {train_time:.1f}s"
        )

    def _on_error(self, error_msg):
        """Training failed."""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append(f"\nERROR: {error_msg}")
        QMessageBox.critical(self, "Training Error", error_msg)

    def _on_log(self, message):
        """Append message to training log."""
        self.log_text.append(message)
