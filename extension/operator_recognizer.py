"""
COS30018 - Extension Option 2: Operator Recognition
Recognizes mathematical operators (+, -, *, /) and parentheses in handwritten images.

Approach: Train a single CNN on 16 classes (digits 0-9 + 6 operators).
Uses synthetically generated operator images combined with MNIST digits.
Includes aspect-ratio heuristic to reduce confusion between +/1/7/4.
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


def generate_operator_images(num_per_class=5000):
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
    noise = np.random.randn(*img.shape) * noise_level
    img = np.clip(img + noise, 0, 1)

    angle = np.random.uniform(-10, 10)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img.astype(np.float32), M, (w, h))

    return img.astype(np.float32)


def _draw_plus():
    """Draw a + symbol - MUST have clear cross shape, both lines similar length."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx = 14 + np.random.randint(-2, 3)
    cy = 14 + np.random.randint(-2, 3)
    # Both arms similar length to make it clearly a cross
    length = np.random.randint(7, 12)
    thickness = np.random.randint(2, 4)

    # Horizontal line (must be prominent)
    h_len = length + np.random.randint(-1, 2)
    cv2.line(img, (cx - h_len, cy), (cx + h_len, cy), 1.0, thickness)
    # Vertical line (must be prominent)
    v_len = length + np.random.randint(-1, 2)
    cv2.line(img, (cx, cy - v_len), (cx, cy + v_len), 1.0, thickness)

    return _add_noise(img)


def _draw_minus():
    """Draw a - symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cy = 14 + np.random.randint(-3, 4)
    length = np.random.randint(7, 12)
    thickness = np.random.randint(2, 4)
    cx = 14 + np.random.randint(-2, 3)

    cv2.line(img, (cx - length, cy), (cx + length, cy), 1.0, thickness)
    return _add_noise(img)


def _draw_multiply():
    """Draw a * or x symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx = 14 + np.random.randint(-2, 3)
    cy = 14 + np.random.randint(-2, 3)
    length = np.random.randint(5, 9)
    thickness = np.random.randint(2, 4)

    cv2.line(img, (cx - length, cy - length), (cx + length, cy + length), 1.0, thickness)
    cv2.line(img, (cx + length, cy - length), (cx - length, cy + length), 1.0, thickness)
    return _add_noise(img)


def _draw_divide():
    """Draw a / symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    thickness = np.random.randint(2, 4)
    offset = np.random.randint(-2, 3)

    cv2.line(img, (8 + offset, 22 + offset), (20 + offset, 6 + offset), 1.0, thickness)
    return _add_noise(img)


def _draw_lparen():
    """Draw a ( symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx = 16 + np.random.randint(-2, 3)
    thickness = np.random.randint(2, 3)

    cv2.ellipse(img, (cx, 14), (6, 10), 0, 120, 240, 1.0, thickness)
    return _add_noise(img)


def _draw_rparen():
    """Draw a ) symbol with random variations."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    cx = 12 + np.random.randint(-2, 3)
    thickness = np.random.randint(2, 3)

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


def _compute_aspect_ratio(image):
    """
    Compute aspect ratio (width/height) of the actual content in a 28x28 image.
    Returns (aspect_ratio, fill_ratio, has_cross_pattern).
    """
    binary = (image > 0.15).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return 1.0, 0.0, False

    x, y, w, h = cv2.boundingRect(coords)
    aspect = w / max(h, 1)
    fill = np.sum(binary) / max(w * h, 1)

    # Detect cross pattern: check if there are pixels in all 4 quadrants
    # relative to center of bounding box
    cx, cy = x + w // 2, y + h // 2
    q_tl = np.sum(binary[:cy, :cx]) > 2
    q_tr = np.sum(binary[:cy, cx:]) > 2
    q_bl = np.sum(binary[cy:, :cx]) > 2
    q_br = np.sum(binary[cy:, cx:]) > 2
    has_cross = q_tl and q_tr and q_bl and q_br

    return aspect, fill, has_cross


def train_expression_model(epochs=12, batch_size=64):
    """
    Train the 16-class expression recognition model.
    Combines MNIST digits with synthetically generated operator images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MNIST digits...")
    from models.model_manager import load_mnist
    X_train, y_train, _, _ = load_mnist()

    # Subsample MNIST balanced (5000 per digit to match operator count)
    balanced_X, balanced_y = [], []
    for d in range(10):
        idx = np.where(y_train == d)[0][:5000]
        balanced_X.append(X_train[idx])
        balanced_y.append(y_train[idx])

    # Generate operator images (5000 per class)
    print("Generating operator training images...")
    op_X, op_y = generate_operator_images(num_per_class=5000)

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


def classify_symbol(image, expression_model, digit_model=None):
    """
    Classify a 28x28 image as a digit (0-9) or operator (+,-,*,/,(,)).

    Hybrid approach:
    1. ExpressionCNN determines if it's a digit or operator
    2. If digit → use the dedicated digit model (99.28% accuracy) for precise classification
    3. If operator → use ExpressionCNN result

    Args:
        image: 28x28 numpy array, float32, values [0,1]
        expression_model: Trained ExpressionCNN model (16 classes)
        digit_model: Optional dedicated digit model (10 classes, higher accuracy)

    Returns:
        Tuple of (symbol_type, value)
        e.g., ('digit', 5) or ('operator', '+')
    """
    device = next(expression_model.parameters()).device

    X = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = expression_model(X)
        proba = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = out.argmax(1).item()

    # Sum probabilities: digit classes (0-9) vs operator classes (10-15)
    digit_total_conf = float(np.sum(proba[:10]))
    operator_total_conf = float(np.sum(proba[10:]))

    # Aspect ratio heuristic
    aspect, fill, has_cross = _compute_aspect_ratio(image)

    # Determine if digit or operator
    is_operator = pred >= 10

    # Heuristic: if CNN says digit but image has clear cross pattern + square → "+"
    if not is_operator and has_cross and 0.6 < aspect < 1.6:
        if proba[10] > 0.05:  # class 10 = "+"
            is_operator = True
            pred = 10

    # Heuristic: if CNN says operator but image is very tall/narrow → digit
    if is_operator and pred == 10 and aspect < 0.35:
        is_operator = False
    if is_operator and pred == 11 and aspect < 0.4:
        is_operator = False

    if is_operator:
        # For operators, use ExpressionCNN result
        best_op = max(range(10, 16), key=lambda i: proba[i])
        symbol = LABEL_MAP[best_op]
        return ("operator", symbol)
    else:
        # For digits, use the dedicated digit model if available (much more accurate)
        if digit_model is not None:
            from models.model_manager import predict_digit
            label, confidence, digit_proba = predict_digit(digit_model, image)
            return ("digit", label)
        else:
            # Fallback to ExpressionCNN digit prediction
            best_digit = max(range(10), key=lambda i: proba[i])
            return ("digit", best_digit)
