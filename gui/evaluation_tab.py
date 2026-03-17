"""
COS30018 - Evaluation Tab
PyQt5 tab for comparing and evaluating trained models.
Features: model evaluation, confusion matrix, comparison charts.
"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTextEdit, QMessageBox, QCheckBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import AVAILABLE_MODELS, MODEL_CNN_KERAS, MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN


class EvaluationWorker(QThread):
    """Background thread for model evaluation."""
    finished = pyqtSignal(list)  # list of evaluation results
    progress = pyqtSignal(str)   # progress message
    error = pyqtSignal(str)

    def __init__(self, model_names):
        super().__init__()
        self.model_names = model_names

    def run(self):
        try:
            from models.model_manager import load_trained_model, load_mnist
            from evaluation.evaluator import evaluate_model

            self.progress.emit("Loading MNIST test data...")
            _, _, X_test, y_test = load_mnist()

            results = []
            for name in self.model_names:
                self.progress.emit(f"Evaluating {name}...")
                model = load_trained_model(name)
                if model and model.is_trained:
                    result = evaluate_model(model, X_test, y_test)
                    results.append(result)
                    self.progress.emit(
                        f"{name}: {result['accuracy']*100:.2f}% accuracy"
                    )
                else:
                    self.progress.emit(f"{name}: Not trained yet, skipping")

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class EvaluationTab(QWidget):
    """Tab widget for model evaluation and comparison."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- Model Selection ---
        select_group = QGroupBox("Select Models to Evaluate")
        select_layout = QHBoxLayout()

        self.model_checks = {}
        model_names = {
            MODEL_CNN_KERAS: "CNN (Keras)",
            MODEL_CNN_PYTORCH: "CNN (PyTorch)",
            MODEL_SVM: "SVM",
            MODEL_KNN: "KNN",
        }
        for model_id in AVAILABLE_MODELS:
            cb = QCheckBox(model_names.get(model_id, model_id))
            cb.setChecked(True)
            self.model_checks[model_id] = cb
            select_layout.addWidget(cb)

        self.eval_btn = QPushButton("Evaluate All")
        self.eval_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "font-size: 14px; padding: 8px 20px; border-radius: 4px; }"
        )
        self.eval_btn.clicked.connect(self._start_evaluation)
        select_layout.addWidget(self.eval_btn)

        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        # --- Results ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self.log_text)

        # --- Charts ---
        chart_layout = QHBoxLayout()

        # Accuracy comparison chart
        self.accuracy_figure = Figure(figsize=(5, 4))
        self.accuracy_canvas = FigureCanvas(self.accuracy_figure)
        chart_layout.addWidget(self.accuracy_canvas)

        # Confusion matrix chart
        self.cm_figure = Figure(figsize=(5, 4))
        self.cm_canvas = FigureCanvas(self.cm_figure)
        chart_layout.addWidget(self.cm_canvas)

        layout.addLayout(chart_layout)

        # --- Navigation for confusion matrices ---
        nav_layout = QHBoxLayout()
        self.prev_cm_btn = QPushButton("< Previous Model")
        self.prev_cm_btn.clicked.connect(self._show_prev_cm)
        self.prev_cm_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_cm_btn)

        self.cm_model_label = QLabel("No data")
        self.cm_model_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.cm_model_label)

        self.next_cm_btn = QPushButton("Next Model >")
        self.next_cm_btn.clicked.connect(self._show_next_cm)
        self.next_cm_btn.setEnabled(False)
        nav_layout.addWidget(self.next_cm_btn)

        layout.addLayout(nav_layout)

        self.current_cm_index = 0

    def _start_evaluation(self):
        """Start evaluating selected models."""
        selected = [
            model_id for model_id, cb in self.model_checks.items()
            if cb.isChecked()
        ]

        if not selected:
            QMessageBox.warning(self, "No Models", "Please select at least one model!")
            return

        self.eval_btn.setEnabled(False)
        self.log_text.clear()
        self.log_text.append("Starting evaluation...")

        self.worker = EvaluationWorker(selected)
        self.worker.progress.connect(lambda msg: self.log_text.append(msg))
        self.worker.finished.connect(self._on_evaluation_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_evaluation_done(self, results):
        """Display evaluation results."""
        self.eval_btn.setEnabled(True)
        self.results = results

        if not results:
            self.log_text.append("No models were evaluated. Train models first!")
            return

        # Generate text report
        from evaluation.evaluator import generate_evaluation_report
        report = generate_evaluation_report(results)
        self.log_text.append(f"\n{report}")

        # Plot accuracy comparison
        self._plot_accuracy(results)

        # Show first confusion matrix
        self.current_cm_index = 0
        self._show_confusion_matrix(0)

        # Enable navigation
        self.prev_cm_btn.setEnabled(len(results) > 1)
        self.next_cm_btn.setEnabled(len(results) > 1)

    def _on_error(self, error_msg):
        """Handle evaluation error."""
        self.eval_btn.setEnabled(True)
        self.log_text.append(f"\nERROR: {error_msg}")
        QMessageBox.critical(self, "Evaluation Error", error_msg)

    def _plot_accuracy(self, results):
        """Plot accuracy comparison bar chart."""
        self.accuracy_figure.clear()
        ax = self.accuracy_figure.add_subplot(111)

        names = [r["model_name"] for r in results]
        accs = [r["accuracy"] * 100 for r in results]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        bars = ax.bar(range(len(names)), accs, color=colors[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model Accuracy Comparison")
        ax.set_ylim(max(0, min(accs) - 5), 100)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f"{acc:.2f}%", ha="center", va="bottom", fontsize=9)

        self.accuracy_figure.tight_layout()
        self.accuracy_canvas.draw()

    def _show_confusion_matrix(self, index):
        """Display confusion matrix for a specific model."""
        if not self.results or index >= len(self.results):
            return

        result = self.results[index]
        cm = result["confusion_matrix"]

        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)

        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(10), yticklabels=range(10), ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {result['model_name']}")

        self.cm_figure.tight_layout()
        self.cm_canvas.draw()
        self.cm_model_label.setText(
            f"{result['model_name']} ({index+1}/{len(self.results)})"
        )

    def _show_prev_cm(self):
        """Show previous model's confusion matrix."""
        if self.results:
            self.current_cm_index = (self.current_cm_index - 1) % len(self.results)
            self._show_confusion_matrix(self.current_cm_index)

    def _show_next_cm(self):
        """Show next model's confusion matrix."""
        if self.results:
            self.current_cm_index = (self.current_cm_index + 1) % len(self.results)
            self._show_confusion_matrix(self.current_cm_index)
