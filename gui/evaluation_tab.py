"""
COS30018 - Evaluation Tab (Modern Dark Theme)
Displays pre-computed evaluation results with charts.
Loads results from data/evaluation_results/results.json on startup.
"""
import json
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTextEdit, QMessageBox, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.theme import btn_warning, btn_primary, TEXT_SECONDARY, PRIMARY, SUCCESS, DANGER


CHART_STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#e2e8f0",
    "axes.labelcolor": "#1e293b",
    "text.color": "#1e293b",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
}

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(_BASE_DIR, "data", "evaluation_results", "results.json")
MULTI_DIGIT_PATH = os.path.join(_BASE_DIR, "data", "evaluation_results", "multi_digit_results.json")


class EvaluationTab(QWidget):
    """Tab displaying pre-computed evaluation results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []
        self.current_cm_index = 0
        self._init_ui()
        self._load_results()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Summary cards row
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setSpacing(12)
        layout.addLayout(self.cards_layout)

        # Charts
        chart_layout = QHBoxLayout()
        chart_layout.setSpacing(12)

        self.accuracy_figure = Figure(figsize=(5, 3.5))
        self.accuracy_canvas = FigureCanvas(self.accuracy_figure)
        self.accuracy_canvas.setStyleSheet("background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;")
        chart_layout.addWidget(self.accuracy_canvas)

        self.cm_figure = Figure(figsize=(5, 3.5))
        self.cm_canvas = FigureCanvas(self.cm_figure)
        self.cm_canvas.setStyleSheet("background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;")
        chart_layout.addWidget(self.cm_canvas)

        layout.addLayout(chart_layout, stretch=1)

        # Navigation
        nav_layout = QHBoxLayout()

        self.prev_cm_btn = QPushButton("Previous")
        self.prev_cm_btn.clicked.connect(self._show_prev_cm)
        self.prev_cm_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_cm_btn)

        self.cm_model_label = QLabel("Loading results...")
        self.cm_model_label.setAlignment(Qt.AlignCenter)
        self.cm_model_label.setFont(QFont("Segoe UI", 11))
        self.cm_model_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        nav_layout.addWidget(self.cm_model_label, stretch=1)

        self.next_cm_btn = QPushButton("Next")
        self.next_cm_btn.clicked.connect(self._show_next_cm)
        self.next_cm_btn.setEnabled(False)
        nav_layout.addWidget(self.next_cm_btn)

        layout.addLayout(nav_layout)

        # Per-class metrics
        metrics_group = QGroupBox("Per-Class Metrics (Best Model)")
        self.metrics_layout = QGridLayout()
        self.metrics_layout.setSpacing(4)
        metrics_group.setLayout(self.metrics_layout)
        layout.addWidget(metrics_group)

        # Multi-digit evaluation results
        multi_group = QGroupBox("Multi-Digit Sequence Evaluation")
        multi_layout = QVBoxLayout()
        multi_layout.setSpacing(6)

        self.multi_grid = QGridLayout()
        self.multi_grid.setSpacing(4)
        multi_layout.addLayout(self.multi_grid)

        self.multi_summary = QLabel("")
        self.multi_summary.setFont(QFont("Segoe UI", 11))
        self.multi_summary.setAlignment(Qt.AlignLeft)
        multi_layout.addWidget(self.multi_summary)

        multi_group.setLayout(multi_layout)
        layout.addWidget(multi_group)

    def _load_results(self):
        """Load pre-computed results from JSON."""
        if not os.path.exists(RESULTS_PATH):
            self.cm_model_label.setText("No results found. Run evaluation first.")
            return

        with open(RESULTS_PATH, "r") as f:
            self.results = json.load(f)

        if not self.results:
            return

        # Create summary cards
        colors = ["#2563eb", "#7c3aed", "#16a34a", "#f59e0b"]
        for i, result in enumerate(self.results):
            card = self._create_summary_card(result, colors[i % len(colors)])
            self.cards_layout.addWidget(card)

        # Plot charts
        self._plot_accuracy()
        self.current_cm_index = 0
        self._show_confusion_matrix(0)
        self.prev_cm_btn.setEnabled(len(self.results) > 1)
        self.next_cm_btn.setEnabled(len(self.results) > 1)

        # Show per-class metrics for best model
        best = max(self.results, key=lambda r: r["accuracy"])
        self._show_per_class_metrics(best)

        # Load multi-digit results
        self._load_multi_digit_results()

    def _create_summary_card(self, result, color):
        """Create a summary card for a model."""
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: #ffffff; border: 1px solid #e2e8f0; "
            f"border-radius: 10px; border-left: 4px solid {color}; }}"
        )

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)
        card_layout.setSpacing(4)

        name = QLabel(result["model_name"])
        name.setFont(QFont("Segoe UI", 11, QFont.Bold))
        name.setStyleSheet(f"color: {color}; border: none;")
        card_layout.addWidget(name)

        acc = QLabel(f"{result['accuracy']*100:.2f}%")
        acc.setFont(QFont("Segoe UI", 22, QFont.Bold))
        acc.setStyleSheet("color: #1e293b; border: none;")
        acc.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(acc)

        time_val = result.get("total_inference_time", result.get("inference_time", 0))
        info = QLabel(f"Inference: {time_val:.2f}s (10K samples)")
        info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; border: none;")
        card_layout.addWidget(info)

        return card

    def _plot_accuracy(self):
        self.accuracy_figure.clear()

        with matplotlib.rc_context(CHART_STYLE):
            ax = self.accuracy_figure.add_subplot(111)
            ax.set_facecolor("#f8fafc")
            self.accuracy_figure.set_facecolor("#ffffff")

            names = [r["model_name"] for r in self.results]
            accs = [r["accuracy"] * 100 for r in self.results]
            colors = ["#2563eb", "#7c3aed", "#16a34a", "#f59e0b"]

            bars = ax.bar(range(len(names)), accs,
                         color=colors[:len(names)], width=0.6,
                         edgecolor="#e2e8f0", linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_title("Model Comparison", fontsize=13, fontweight="bold", pad=12)
            ax.set_ylim(max(0, min(accs) - 5), 100.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e2e8f0")
            ax.spines["bottom"].set_color("#e2e8f0")
            ax.grid(axis="y", alpha=0.3, color="#e2e8f0")

            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f"{acc:.2f}%", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        self.accuracy_figure.tight_layout()
        self.accuracy_canvas.draw()

    def _show_confusion_matrix(self, index):
        if not self.results or index >= len(self.results):
            return

        result = self.results[index]
        cm = np.array(result["confusion_matrix"])

        self.cm_figure.clear()

        with matplotlib.rc_context(CHART_STYLE):
            ax = self.cm_figure.add_subplot(111)
            ax.set_facecolor("#f8fafc")
            self.cm_figure.set_facecolor("#ffffff")

            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=range(10), yticklabels=range(10), ax=ax,
                        linewidths=0.5, linecolor="#e2e8f0",
                        cbar_kws={"shrink": 0.8})
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_title(f"Confusion Matrix - {result['model_name']}",
                        fontsize=12, fontweight="bold", pad=10)

        self.cm_figure.tight_layout()
        self.cm_canvas.draw()
        self.cm_model_label.setText(
            f"{result['model_name']}  ({index+1}/{len(self.results)})"
        )

    def _show_per_class_metrics(self, result):
        """Display per-class precision/recall/F1 for a model."""
        # Header
        headers = ["Digit", "Precision", "Recall", "F1-Score"]
        for col, h in enumerate(headers):
            label = QLabel(h)
            label.setFont(QFont("Segoe UI", 10, QFont.Bold))
            label.setStyleSheet(f"color: {PRIMARY};")
            label.setAlignment(Qt.AlignCenter)
            self.metrics_layout.addWidget(label, 0, col)

        per_class = result.get("per_class", {})
        for digit in range(10):
            d = per_class.get(str(digit), {})

            digit_label = QLabel(str(digit))
            digit_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
            digit_label.setAlignment(Qt.AlignCenter)
            self.metrics_layout.addWidget(digit_label, digit + 1, 0)

            for col, key in enumerate(["precision", "recall", "f1"], 1):
                val = d.get(key, 0)
                lbl = QLabel(f"{val:.4f}")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFont(QFont("Segoe UI", 10))
                color = SUCCESS if val > 0.99 else "#eaeaea" if val > 0.95 else "#ff9f43"
                lbl.setStyleSheet(f"color: {color};")
                self.metrics_layout.addWidget(lbl, digit + 1, col)

    def _show_prev_cm(self):
        if self.results:
            self.current_cm_index = (self.current_cm_index - 1) % len(self.results)
            self._show_confusion_matrix(self.current_cm_index)

    def _show_next_cm(self):
        if self.results:
            self.current_cm_index = (self.current_cm_index + 1) % len(self.results)
            self._show_confusion_matrix(self.current_cm_index)

    def _load_multi_digit_results(self):
        """Load and display multi-digit evaluation results."""
        if not os.path.exists(MULTI_DIGIT_PATH):
            self.multi_summary.setText("No multi-digit results found. Run train_and_evaluate.py first.")
            return

        with open(MULTI_DIGIT_PATH, "r") as f:
            data = json.load(f)

        sequences = data.get("sequences", [])
        if not sequences:
            return

        # Table headers
        headers = ["Ground Truth", "Predicted", "Segments", "Result"]
        for col, h in enumerate(headers):
            label = QLabel(h)
            label.setFont(QFont("Segoe UI", 10, QFont.Bold))
            label.setStyleSheet(f"color: {PRIMARY};")
            label.setAlignment(Qt.AlignCenter)
            self.multi_grid.addWidget(label, 0, col)

        # Table rows
        for row, seq in enumerate(sequences):
            gt = QLabel(seq["ground_truth"])
            gt.setFont(QFont("Consolas", 11, QFont.Bold))
            gt.setAlignment(Qt.AlignCenter)
            self.multi_grid.addWidget(gt, row + 1, 0)

            pred = QLabel(seq["predicted"])
            pred.setFont(QFont("Consolas", 11))
            pred.setAlignment(Qt.AlignCenter)
            self.multi_grid.addWidget(pred, row + 1, 1)

            seg_str = f"{seq['num_segments']}/{seq['expected_segments']}"
            seg = QLabel(seg_str)
            seg.setFont(QFont("Segoe UI", 10))
            seg.setAlignment(Qt.AlignCenter)
            self.multi_grid.addWidget(seg, row + 1, 2)

            is_correct = seq["correct"]
            result_label = QLabel("CORRECT" if is_correct else "WRONG")
            result_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
            result_label.setAlignment(Qt.AlignCenter)
            result_label.setStyleSheet(f"color: {SUCCESS};" if is_correct else f"color: {DANGER};")
            self.multi_grid.addWidget(result_label, row + 1, 3)

        # Summary
        seq_acc = data.get("sequence_accuracy", 0) * 100
        dig_acc = data.get("digit_accuracy", 0) * 100
        seg_acc = data.get("segmentation_accuracy", 0) * 100
        n = data.get("num_sequences", 0)
        correct_n = int(seq_acc * n / 100)

        self.multi_summary.setText(
            f"Sequence Accuracy: {seq_acc:.1f}% ({correct_n}/{n})    |    "
            f"Digit Accuracy: {dig_acc:.1f}%    |    "
            f"Segmentation Accuracy: {seg_acc:.1f}%"
        )
        self.multi_summary.setStyleSheet(f"color: {PRIMARY}; font-weight: bold; padding: 6px 0;")
