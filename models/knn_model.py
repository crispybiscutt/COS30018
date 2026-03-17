"""
COS30018 - Task 3: KNN Model using scikit-learn
K-Nearest Neighbors classifier for handwritten digit recognition.

KNN is a simple, instance-based learning algorithm. For each test image,
it finds the K most similar training images and assigns the most common
label among them. No explicit training phase - the model stores the
training data and performs computation at prediction time.

Note: KNN requires flattened input (28x28 -> 784 features).
"""
import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel
from config import KNN_N_NEIGHBORS


class KNNModel(BaseModel):
    """K-Nearest Neighbors for digit recognition."""

    def __init__(self):
        super().__init__("KNN (scikit-learn)")
        self.model = None
        self.scaler = StandardScaler()

    def build(self, n_neighbors=KNN_N_NEIGHBORS):
        """
        Build KNN classifier.
        n_neighbors: Number of neighbors to consider (default 5).
        weights='distance': Closer neighbors have more influence.
        algorithm='auto': Automatically choose best algorithm (ball_tree, kd_tree, brute).
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            algorithm="auto",
            n_jobs=-1,  # Use all CPU cores for prediction
        )
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              max_samples=20000, callback=None, **kwargs):
        """
        'Train' KNN (stores training data).
        Uses subset for speed since KNN prediction time scales with dataset size.
        """
        if self.model is None:
            self.build()

        # Flatten images: (N, 28, 28) -> (N, 784)
        X_flat = self._flatten(X_train)

        # Use subset for speed
        if len(X_flat) > max_samples:
            indices = np.random.choice(len(X_flat), max_samples, replace=False)
            X_flat = X_flat[indices]
            y_train = np.array(y_train)[indices]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)

        print(f"Training KNN (k={self.model.n_neighbors}) on {len(X_scaled)} samples...")
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        # Compute training accuracy (on a small subset for speed)
        sample_size = min(2000, len(X_scaled))
        sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        train_acc = self.model.score(X_scaled[sample_idx], np.array(y_train)[sample_idx])
        self.training_history = {"accuracy": [train_acc]}

        print(f"KNN Training accuracy (sampled): {train_acc:.4f}")

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
        """Save KNN model and scaler to .pkl file."""
        if self.model:
            joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path):
        """Load KNN model and scaler from .pkl file."""
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
