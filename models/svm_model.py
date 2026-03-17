"""
COS30018 - Task 3: SVM Model using scikit-learn
Support Vector Machine for handwritten digit recognition.

SVM works by finding the optimal hyperplane that separates different classes.
Uses RBF (Radial Basis Function) kernel which maps data to higher dimensions
where it becomes linearly separable.

Note: SVM requires flattened input (28x28 -> 784 features).
Training on full MNIST (60K samples) can be slow, so we use a subset by default.
"""
import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from config import SVM_KERNEL, SVM_C, SVM_GAMMA


class SVMModel(BaseModel):
    """Support Vector Machine for digit recognition."""

    def __init__(self):
        super().__init__("SVM (scikit-learn)")
        self.model = None
        self.scaler = StandardScaler()

    def build(self, kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA):
        """
        Build SVM classifier.
        kernel='rbf': Non-linear kernel, good for image data
        C=10: Regularization, higher = more complex boundary
        gamma='scale': Kernel coefficient, 'scale' = 1/(n_features * var)
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable predict_proba (slower training but needed)
            verbose=False,
        )
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              max_samples=20000, callback=None, **kwargs):
        """
        Train SVM on MNIST data.
        Uses a subset of data for speed (SVM training is O(n^2) to O(n^3)).

        Args:
            max_samples: Maximum training samples (default 20000 for speed)
        """
        if self.model is None:
            self.build()

        # Flatten images: (N, 28, 28) -> (N, 784)
        X_flat = self._flatten(X_train)

        # Use subset if dataset is too large
        if len(X_flat) > max_samples:
            indices = np.random.choice(len(X_flat), max_samples, replace=False)
            X_flat = X_flat[indices]
            y_train = np.array(y_train)[indices]

        # Scale features to zero mean and unit variance
        X_scaled = self.scaler.fit_transform(X_flat)

        print(f"Training SVM on {len(X_scaled)} samples...")
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        # Compute training accuracy
        train_acc = self.model.score(X_scaled, y_train)
        self.training_history = {"accuracy": [train_acc]}

        print(f"SVM Training accuracy: {train_acc:.4f}")

        if callback:
            callback(1, 1, self.training_history)

        return self.training_history

    def predict(self, X):
        """Predict digit labels."""
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict probability distribution over 10 digits."""
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)

    def save(self, path):
        """Save SVM model and scaler to .pkl file."""
        if self.model:
            joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path):
        """Load SVM model and scaler from .pkl file."""
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = True

    def _flatten(self, X):
        """Flatten images from (N, 28, 28) to (N, 784)."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 2 and X.shape == (28, 28):
            return X.reshape(1, -1)
        elif X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X
