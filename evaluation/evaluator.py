"""
COS30018 - Task 4: Evaluation and Testing
Comprehensive evaluation of ML models including:
- Per-class metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Model comparison charts (accuracy, training time, inference speed)
- Multi-digit number recognition evaluation
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive evaluation of a single model.

    Returns dict with:
        - accuracy: Overall accuracy
        - per_class: Dict with precision, recall, f1 per digit
        - confusion_matrix: 10x10 confusion matrix
        - inference_time: Average time per prediction (seconds)
        - predictions: All predicted labels
    """
    # Predictions and timing
    start = time.time()
    predictions = model.predict(X_test)
    total_time = time.time() - start
    avg_inference = total_time / len(X_test)

    # Overall accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=range(10), zero_division=0
    )

    per_class = {}
    for digit in range(10):
        per_class[digit] = {
            "precision": precision[digit],
            "recall": recall[digit],
            "f1": f1[digit],
            "support": int(support[digit]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=range(10))

    return {
        "model_name": model.name,
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm,
        "inference_time": avg_inference,
        "total_inference_time": total_time,
        "predictions": predictions,
    }


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot a confusion matrix heatmap.

    Args:
        cm: 10x10 confusion matrix
        model_name: Name of the model (for title)
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10), ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_model_comparison(results_list, save_path=None):
    """
    Bar chart comparing accuracy of different models.

    Args:
        results_list: List of evaluation result dicts from evaluate_model()
    """
    names = [r["model_name"] for r in results_list]
    accuracies = [r["accuracy"] * 100 for r in results_list]
    times = [r["total_inference_time"] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    bars1 = ax1.bar(names, accuracies, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Model Accuracy Comparison", fontsize=14)
    ax1.set_ylim(80, 100)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=10)

    # Inference time comparison
    bars2 = ax2.bar(names, times, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax2.set_ylabel("Total Inference Time (s)", fontsize=12)
    ax2.set_title("Inference Speed Comparison", fontsize=14)
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f"{t:.2f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_training_history(history, model_name, save_path=None):
    """
    Plot training loss and accuracy curves over epochs.

    Args:
        history: Dict with 'loss', 'accuracy', 'val_loss', 'val_accuracy' lists
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    # Accuracy plot
    if "accuracy" in history:
        ax1.plot(epochs, history["accuracy"], "b-", label="Training")
    if "val_accuracy" in history:
        ax1.plot(epochs, history["val_accuracy"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{model_name} - Accuracy")
    ax1.legend()

    # Loss plot
    if "loss" in history:
        ax2.plot(epochs, history["loss"], "b-", label="Training")
    if "val_loss" in history:
        ax2.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{model_name} - Loss")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def generate_evaluation_report(results_list):
    """
    Generate a text summary comparing all models.

    Returns:
        Formatted string with comparison table.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Model':<25} {'Accuracy':>10} {'Inference(s)':>15} {'Avg/sample(ms)':>15}")
    lines.append("-" * 70)

    for r in results_list:
        lines.append(
            f"{r['model_name']:<25} {r['accuracy']*100:>9.2f}% "
            f"{r['total_inference_time']:>14.2f} "
            f"{r['inference_time']*1000:>14.3f}"
        )

    lines.append("-" * 70)

    # Find best model
    best = max(results_list, key=lambda r: r["accuracy"])
    lines.append(f"\nBest model: {best['model_name']} ({best['accuracy']*100:.2f}%)")

    return "\n".join(lines)
