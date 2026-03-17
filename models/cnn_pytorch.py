"""
COS30018 - Task 3: CNN Model using PyTorch
Convolutional Neural Network for handwritten digit recognition.
Same architecture as the Keras version for fair comparison.

Architecture:
    Input (1x28x28) -> Conv2d(32) -> ReLU -> MaxPool ->
    Conv2d(64) -> ReLU -> MaxPool -> Conv2d(64) -> ReLU ->
    Flatten -> Linear(128) -> ReLU -> Dropout(0.5) -> Linear(10)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.base_model import BaseModel
from config import IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE


class DigitCNN(nn.Module):
    """PyTorch CNN architecture matching the Keras version."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Third convolutional block
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNPyTorch(BaseModel):
    """CNN model implemented with PyTorch."""

    def __init__(self):
        super().__init__("CNN (PyTorch)")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, num_classes=10):
        """Build the CNN architecture."""
        self.model = DigitCNN(num_classes).to(self.device)
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=EPOCHS, batch_size=BATCH_SIZE, callback=None):
        """
        Train the PyTorch CNN with custom training loop.
        Includes training + validation per epoch.
        """
        if self.model is None:
            self.build()

        # Prepare data: (N, 28, 28) -> (N, 1, 28, 28) for PyTorch conv layers
        X_train_t = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            # --- Training phase ---
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            # --- Validation phase ---
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - "
                      f"acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - "
                      f"val_acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - "
                      f"acc: {epoch_acc:.4f}")

            # Call progress callback if provided (for GUI progress bar)
            if callback:
                callback(epoch + 1, epochs, history)

        self.is_trained = True
        self.training_history = history
        return history

    def _validate(self, val_loader, criterion):
        """Run validation and return loss and accuracy."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return val_loss / total, correct / total

    def predict(self, X):
        """Predict digit labels."""
        self.model.eval()
        X_t = torch.FloatTensor(np.array(X, dtype=np.float32))
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(0).unsqueeze(0)
        elif X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)
        X_t = X_t.to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """Predict probability distribution over 10 digits."""
        self.model.eval()
        X_t = torch.FloatTensor(np.array(X, dtype=np.float32))
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(0).unsqueeze(0)
        elif X_t.ndim == 3:
            X_t = X_t.unsqueeze(1)
        X_t = X_t.to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def save(self, path):
        """Save model state dict to .pth file."""
        if self.model:
            torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model state dict from .pth file."""
        if os.path.exists(path):
            if self.model is None:
                self.build()
            self.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self.is_trained = True

    def summary(self):
        """Print model architecture."""
        if self.model:
            print(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\nTotal parameters: {total_params:,}")
