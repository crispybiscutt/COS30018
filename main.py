"""
COS30018 - Handwritten Number Recognition System (HNRS)
Main entry point - launches the PyQt5 GUI application.

Usage:
    python main.py          # Launch GUI
    python main.py --train  # Quick train all models without GUI
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def launch_gui():
    """Launch the PyQt5 GUI application."""
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("HNRS")
    app.setStyle("Fusion")  # Modern cross-platform style

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


def quick_train():
    """Train all models from command line (no GUI)."""
    from config import AVAILABLE_MODELS
    from models.model_manager import train_model, load_mnist
    from evaluation.evaluator import evaluate_model, generate_evaluation_report

    print("=" * 60)
    print("HNRS - Quick Training Mode")
    print("=" * 60)

    # Load test data for evaluation
    _, _, X_test, y_test = load_mnist()

    results = []
    for model_name in AVAILABLE_MODELS:
        try:
            model, history, train_time = train_model(model_name)
            result = evaluate_model(model, X_test, y_test)
            result["train_time"] = train_time
            results.append(result)
        except Exception as e:
            print(f"\nERROR training {model_name}: {e}")

    # Print comparison
    if results:
        print("\n")
        print(generate_evaluation_report(results))


if __name__ == "__main__":
    if "--train" in sys.argv:
        quick_train()
    else:
        launch_gui()
