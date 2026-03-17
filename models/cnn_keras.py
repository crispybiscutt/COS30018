"""
COS30018 - Task 3: CNN Model using TensorFlow/Keras
Convolutional Neural Network for handwritten digit recognition.

Architecture:
    Input (28x28x1) -> Conv2D(32) -> ReLU -> MaxPool ->
    Conv2D(64) -> ReLU -> MaxPool -> Flatten ->
    Dense(128) -> ReLU -> Dropout(0.5) -> Dense(10) -> Softmax
"""
import os
import numpy as np
from models.base_model import BaseModel
from config import IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE


class CNNKeras(BaseModel):
    """CNN model implemented with TensorFlow/Keras."""

    def __init__(self):
        super().__init__("CNN (Keras/TensorFlow)")
        self.model = None

    def build(self, num_classes=10):
        """Build the CNN architecture using Keras Sequential API."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        self.model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation="relu",
                          input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.MaxPooling2D((2, 2)),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            # Third convolutional block
            layers.Conv2D(64, (3, 3), activation="relu"),

            # Classifier head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=EPOCHS, batch_size=BATCH_SIZE, callback=None):
        """
        Train the Keras CNN model.
        X_train shape: (N, 28, 28) -> reshaped to (N, 28, 28, 1)
        """
        if self.model is None:
            self.build()

        # Reshape for CNN input: (N, 28, 28) -> (N, 28, 28, 1)
        X_train = self._reshape_input(X_train)
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val = self._reshape_input(X_val)
            validation_data = (X_val, y_val)

        callbacks = []
        if callback:
            callbacks.append(callback)

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True
        self.training_history = history.history
        return history.history

    def predict(self, X):
        """Predict digit labels."""
        X = self._reshape_input(X)
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """Predict probability distribution over 10 digits."""
        X = self._reshape_input(X)
        return self.model.predict(X, verbose=0)

    def save(self, path):
        """Save model weights to .h5 file."""
        if self.model:
            self.model.save(path)

    def load(self, path):
        """Load model from .h5 file."""
        import tensorflow as tf
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            self.is_trained = True

    def _reshape_input(self, X):
        """Ensure input is (N, 28, 28, 1) for CNN."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 2:
            # Single image (28, 28) -> (1, 28, 28, 1)
            X = X.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        elif X.ndim == 3:
            # Batch (N, 28, 28) -> (N, 28, 28, 1)
            X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        return X

    def summary(self):
        """Print model architecture summary."""
        if self.model:
            return self.model.summary()
