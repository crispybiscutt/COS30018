"""
COS30018 - Base Model
Abstract base class that all ML models must implement.
Ensures consistent interface across different model types (CNN, SVM, KNN).
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all digit recognition models."""

    def __init__(self, name):
        self.name = name
        self.is_trained = False
        self.training_history = {}

    @abstractmethod
    def build(self):
        """Build/initialize the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model on given data.

        Args:
            X_train: Training images, shape (N, 28, 28) or (N, 784)
            y_train: Training labels, shape (N,)
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional parameters (epochs, batch_size, etc.)

        Returns:
            Training history dict with 'accuracy', 'loss', etc.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict digit labels for input images.

        Args:
            X: Images to predict, shape (N, 28, 28) or (N, 784)

        Returns:
            Predicted labels as numpy array of shape (N,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict probability distribution over digits 0-9.

        Args:
            X: Images to predict, shape (N, 28, 28) or (N, 784)

        Returns:
            Probability array of shape (N, 10)
        """
        pass

    @abstractmethod
    def save(self, path):
        """Save the trained model to disk."""
        pass

    @abstractmethod
    def load(self, path):
        """Load a trained model from disk."""
        pass

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.

        Returns:
            Dict with 'accuracy', 'predictions', 'probabilities'
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        return {
            "accuracy": accuracy,
            "predictions": predictions,
        }
