"""
COS30018 - Model Manager
Unified interface to load MNIST data, train, save, load, and evaluate any model.
Acts as a central controller for all ML model operations.
"""
import os
import time
import numpy as np
from config import (
    SAVED_MODELS_DIR, MODEL_CNN_KERAS, MODEL_CNN_PYTORCH,
    MODEL_SVM, MODEL_KNN, VALIDATION_SPLIT
)


def load_mnist():
    """
    Load MNIST dataset via torchvision.
    Returns: (X_train, y_train, X_test, y_test)
        X: float32 arrays normalized to [0, 1], shape (N, 28, 28)
        y: int arrays of labels 0-9, shape (N,)
    """
    import torchvision
    train_set = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True
    )
    test_set = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True
    )
    X_train = train_set.data.numpy()
    y_train = train_set.targets.numpy()
    X_test = test_set.data.numpy()
    y_test = test_set.targets.numpy()

    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, y_train, X_test, y_test


def split_validation(X_train, y_train, val_split=VALIDATION_SPLIT):
    """Split training data into train and validation sets."""
    n = len(X_train)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (X_train[train_idx], y_train[train_idx],
            X_train[val_idx], y_train[val_idx])


_tf_available = None  # Cache result to avoid repeated DLL loading attempts


def is_tensorflow_available():
    """Check if TensorFlow is properly installed and usable. Result is cached.

    Tests import directly in-process so it detects DLL conflicts
    (e.g. PyQt5 + TF DLL incompatibility on some systems).
    """
    global _tf_available
    if _tf_available is not None:
        return _tf_available
    try:
        import tensorflow as tf
        tf.constant(1)
        _tf_available = True
    except Exception:
        _tf_available = False
    return _tf_available


def get_available_models():
    """Return list of model names that can actually be used on this system."""
    available = [MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN]
    if is_tensorflow_available():
        available.insert(0, MODEL_CNN_KERAS)
    return available


def get_model(model_name):
    """
    Factory function: create a model instance by name.

    Args:
        model_name: One of 'cnn_keras', 'cnn_pytorch', 'svm', 'knn'

    Returns:
        BaseModel instance
    """
    if model_name == MODEL_CNN_KERAS:
        if not is_tensorflow_available():
            raise RuntimeError(
                "TensorFlow is not available on this system. "
                "Use CNN (PyTorch) instead."
            )
        from models.cnn_keras import CNNKeras
        return CNNKeras()
    elif model_name == MODEL_CNN_PYTORCH:
        from models.cnn_pytorch import CNNPyTorch
        return CNNPyTorch()
    elif model_name == MODEL_SVM:
        from models.svm_model import SVMModel
        return SVMModel()
    elif model_name == MODEL_KNN:
        from models.knn_model import KNNModel
        return KNNModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_path(model_name):
    """Get the file path for saving/loading a model."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    extensions = {
        MODEL_CNN_KERAS: ".h5",
        MODEL_CNN_PYTORCH: ".pth",
        MODEL_SVM: ".pkl",
        MODEL_KNN: ".pkl",
    }
    ext = extensions.get(model_name, ".pkl")
    return os.path.join(SAVED_MODELS_DIR, f"{model_name}{ext}")


def train_model(model_name, epochs=None, batch_size=None, callback=None):
    """
    Full training pipeline: load data -> split -> train -> save.

    Args:
        model_name: Name of the model to train
        epochs: Override default epochs (for CNN models)
        batch_size: Override default batch size (for CNN models)
        callback: Progress callback function

    Returns:
        Tuple of (model, training_history, training_time_seconds)
    """
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    # Load data
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Split validation
    X_train, y_train, X_val, y_val = split_validation(X_train, y_train)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create and build model
    model = get_model(model_name)
    model.build()

    # Train with timing
    start_time = time.time()

    kwargs = {}
    if epochs is not None:
        kwargs["epochs"] = epochs
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if callback:
        kwargs["callback"] = callback

    history = model.train(X_train, y_train, X_val, y_val, **kwargs)
    train_time = time.time() - start_time

    # Save model
    model_path = get_model_path(model_name)
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    print(f"Training time: {train_time:.1f} seconds")

    return model, history, train_time


def load_trained_model(model_name):
    """Load a previously trained model from disk."""
    model = get_model(model_name)
    model_path = get_model_path(model_name)

    if os.path.exists(model_path):
        model.load(model_path)
        return model
    else:
        print(f"No saved model found at: {model_path}")
        return None


def predict_digit(model, image):
    """
    Predict a single digit from a preprocessed image.

    Args:
        model: Trained BaseModel instance
        image: Preprocessed image, shape (28, 28), values [0, 1]

    Returns:
        Tuple of (predicted_label, confidence, all_probabilities)
    """
    proba = model.predict_proba(image)
    label = np.argmax(proba[0])
    confidence = proba[0][label]
    return int(label), float(confidence), proba[0]
