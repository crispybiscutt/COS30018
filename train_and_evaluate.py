"""
COS30018 - Train and Evaluate All Models
Retrains all models and saves comprehensive evaluation results.

Usage:
    python train_and_evaluate.py
"""
import os
import sys
import json
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from models.model_manager import train_model, load_mnist, get_available_models
from evaluation.evaluator import (
    evaluate_model, generate_evaluation_report,
    plot_confusion_matrix, plot_model_comparison, plot_training_history
)
from config import SAVED_MODELS_DIR


def main():
    print("=" * 60)
    print("HNRS - Full Training & Evaluation Pipeline")
    print("=" * 60)

    # Load test data
    print("\nLoading MNIST dataset...")
    _, _, X_test, y_test = load_mnist()
    print(f"Test set: {len(X_test)} samples")

    results = []
    histories = {}

    # Train each model
    models_to_train = ["cnn_pytorch", "svm", "knn"]

    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")

        try:
            # Use more epochs for CNN
            epochs = 15 if "cnn" in model_name else None
            model, history, train_time = train_model(model_name, epochs=epochs)

            # Evaluate
            result = evaluate_model(model, X_test, y_test)
            result["train_time"] = train_time

            # Convert numpy types for JSON serialization
            result["confusion_matrix"] = result["confusion_matrix"].tolist()
            del result["predictions"]  # Don't save raw predictions

            # Convert per_class keys to strings
            per_class_str = {}
            for k, v in result["per_class"].items():
                per_class_str[str(k)] = {
                    kk: float(vv) for kk, vv in v.items()
                }
            result["per_class"] = per_class_str

            results.append(result)
            histories[model_name] = history

            print(f"\n{model_name} accuracy: {result['accuracy']*100:.2f}%")
            print(f"Training time: {train_time:.1f}s")

        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_dir = os.path.join("data", "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save charts
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Confusion matrices
    for result in results:
        cm = np.array(result["confusion_matrix"])
        plot_confusion_matrix(
            cm, result["model_name"],
            save_path=os.path.join(charts_dir, f"cm_{result['model_name'].replace(' ', '_').replace('(', '').replace(')', '')}.png")
        )

    # Model comparison
    plot_model_comparison(results, save_path=os.path.join(charts_dir, "model_comparison.png"))

    # Training histories for CNN
    for name, history in histories.items():
        if "accuracy" in history and len(history["accuracy"]) > 1:
            plot_training_history(
                history, name,
                save_path=os.path.join(charts_dir, f"history_{name}.png")
            )

    # Print report
    print("\n")
    print(generate_evaluation_report(results))
    print(f"\nCharts saved to: {charts_dir}")


if __name__ == "__main__":
    main()
