"""
COS30018 - Extension Option 2: Operator Recognition
Recognizes mathematical operators (+, -, *, ÷) and parentheses in handwritten images.

Approach: Train a single CNN on 16 classes (digits 0-9 + 6 operators).
Uses synthetically generated operator images combined with MNIST digits.
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import IMAGE_SIZE, SAVED_MODELS_DIR


# 16-class label mapping: 0-9 = digits, 10-15 = operators
LABEL_MAP = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "+", 11: "-", 12: "*", 13: "/", 14: "(", 15: ")",
}

SYMBOL_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 16


def generate_operator_images(num_per_class=3000):
    """
    Generate synthetic training images for operators.
    Creates varied handwriting-like operator symbols with random perturbations.

    Returns: (images, labels) - images shape (N, 28, 28), labels shape (N,)
    """
    images = []
    labels = []

    for _ in range(num_per_class):
        for op_label, draw_func in [
            (10, _draw_plus),
            (11, _draw_minus),
            (12, _draw_multiply),
            (13, _draw_divide),
            (14, _draw_lparen),
            (15, _draw_rparen),
        ]:
            img = draw_func()
            images.append(img)
            labels.append(op_label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def _add_noise(img, noise_level=0.05):
    """Add random noise and slight rotation for data augmentation."""
    # Random noise
    noise = np.random.randn(*img.shape) * noise_level
    img = np.clip(img + noise, 0, 1)

    # Random slight rotation
    angle = np.random.uniform(-10, 10)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img.astype(np.float32), M, (w, h))

    return img.astype(np.float32)


def _draw_plus():
    """Draw a + symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx, cy = 14 + np.random.randint(-2, 3), 14 + np.random.randint(-2, 3)
    length = np.random.randint(6, 11)
    thickness = np.random.randint(2, 4)

    # Horizontal line
    cv2.line(img, (cx - length, cy), (cx + length, cy), 1.0, thickness)
    # Vertical line
    cv2.line(img, (cx, cy - length), (cx, cy + length), 1.0, thickness)
    return _add_noise(img)


def _draw_minus():
    """Draw a - symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cy = 14 + np.random.randint(-3, 4)
    length = np.random.randint(6, 12)
    thickness = np.random.randint(2, 4)
    cx = 14 + np.random.randint(-2, 3)

    cv2.line(img, (cx - length, cy), (cx + length, cy), 1.0, thickness)
    return _add_noise(img)


def _draw_multiply():
    """Draw a * or x symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx, cy = 14 + np.random.randint(-2, 3), 14 + np.random.randint(-2, 3)
    length = np.random.randint(5, 9)
    thickness = np.random.randint(2, 4)

    # Two diagonal lines (X shape)
    cv2.line(img, (cx - length, cy - length), (cx + length, cy + length), 1.0, thickness)
    cv2.line(img, (cx + length, cy - length), (cx - length, cy + length), 1.0, thickness)
    return _add_noise(img)


def _draw_divide():
    """Draw a / symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    thickness = np.random.randint(2, 4)
    offset = np.random.randint(-2, 3)

    # Diagonal line from bottom-left to top-right
    cv2.line(img, (8 + offset, 22 + offset), (20 + offset, 6 + offset), 1.0, thickness)
    return _add_noise(img)


def _draw_lparen():
    """Draw a ( symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx = 16 + np.random.randint(-2, 3)
    thickness = np.random.randint(2, 3)

    # Draw arc (left parenthesis)
    cv2.ellipse(img, (cx, 14), (6, 10), 0, 120, 240, 1.0, thickness)
    return _add_noise(img)


def _draw_rparen():
    """Draw a ) symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx = 12 + np.random.randint(-2, 3)
    thickness = np.random.randint(2, 3)

    # Draw arc (right parenthesis)
    cv2.ellipse(img, (cx, 14), (6, 10), 0, -60, 60, 1.0, thickness)
    return _add_noise(img)


class ExpressionCNN(nn.Module):
    """CNN for 16-class classification (digits 0-9 + 6 operators)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_expression_model(epochs=8, batch_size=64):
    """
    Train the 16-class expression recognition model.
    Combines MNIST digits with synthetically generated operator images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST digits
    print("Loading MNIST digits...")
    from models.model_manager import load_mnist
    X_train, y_train, _, _ = load_mnist()

    # Subsample MNIST to balance with operators (3000 per digit)
    balanced_X, balanced_y = [], []
    for d in range(10):
        idx = np.where(y_train == d)[0][:3000]
        balanced_X.append(X_train[idx])
        balanced_y.append(y_train[idx])

    # Generate operator images
    print("Generating operator training images...")
    op_X, op_y = generate_operator_images(num_per_class=3000)

    # Combine
    all_X = np.concatenate([np.concatenate(balanced_X)] + [op_X])
    all_y = np.concatenate([np.concatenate(balanced_y)] + [op_y])

    # Shuffle
    perm = np.random.permutation(len(all_X))
    all_X, all_y = all_X[perm], all_y[perm]
    print(f"Training data: {len(all_X)} samples, {NUM_CLASSES} classes")

    # Create DataLoader
    X_t = torch.FloatTensor(all_X).unsqueeze(1).to(device)
    y_t = torch.LongTensor(all_y).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    # Build and train model
    model = ExpressionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            correct += (out.argmax(1) == by).sum().item()
            total += bx.size(0)
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/total:.4f} - acc: {correct/total:.4f}")

    # Save model
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    path = os.path.join(SAVED_MODELS_DIR, "expression_cnn.pth")
    torch.save(model.state_dict(), path)
    print(f"Expression model saved to: {path}")

    return model


def load_expression_model():
    """Load the trained expression model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpressionCNN().to(device)
    path = os.path.join(SAVED_MODELS_DIR, "expression_cnn.pth")

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        return model
    return None


def classify_symbol(image, expression_model):
    """
    Classify a 28x28 image as a digit (0-9) or operator (+,-,*,/,(,)).

    Args:
        image: 28x28 numpy array, float32, values [0,1]
        expression_model: Trained ExpressionCNN model (16 classes)

    Returns:
        Tuple of (symbol_type, value)
        e.g., ('digit', 5) or ('operator', '+')
    """
    device = next(expression_model.parameters()).device

    X = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = expression_model(X)
        proba = torch.softmax(out, dim=1)
        pred = out.argmax(1).item()

    symbol = LABEL_MAP[pred]
    confidence = proba[0][pred].item()

    if pred <= 9:
        return ("digit", pred)
    else:
        return ("operator", symbol)
