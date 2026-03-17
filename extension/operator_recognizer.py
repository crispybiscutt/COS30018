"""
COS30018 - Extension Option 2: Operator Recognition
Recognizes mathematical operators (+, -, *, ÷) and parentheses in handwritten images.

Uses a CNN trained on a custom operator dataset derived from symbol patterns.
Falls back to template matching if no trained model is available.
"""
import os
import numpy as np
import cv2
from config import IMAGE_SIZE, SAVED_MODELS_DIR


# Operator label mapping
OPERATOR_LABELS = {
    0: "+",
    1: "-",
    2: "*",
    3: "/",
    4: "(",
    5: ")",
}

LABEL_TO_INDEX = {v: k for k, v in OPERATOR_LABELS.items()}


def classify_symbol(digit_image, digit_model):
    """
    Classify a segmented symbol as either a digit (0-9) or an operator (+,-,*,/,(,)).

    Strategy:
    1. Try digit model first - if confidence > threshold, it's a digit
    2. If low confidence, analyze shape features to detect operators
    3. Return (type, value) where type is 'digit' or 'operator'

    Args:
        digit_image: Preprocessed 28x28 image
        digit_model: Trained digit recognition model

    Returns:
        Tuple of (symbol_type, value)
        e.g., ('digit', 5) or ('operator', '+')
    """
    # Get digit prediction confidence
    proba = digit_model.predict_proba(digit_image)
    max_confidence = np.max(proba[0])
    digit_prediction = np.argmax(proba[0])

    # If high confidence for a digit, return digit
    if max_confidence > 0.8:
        return ("digit", int(digit_prediction))

    # Low confidence -> likely an operator, use shape analysis
    operator = _detect_operator_by_shape(digit_image)
    if operator:
        return ("operator", operator)

    # If still unsure, go with the digit prediction
    return ("digit", int(digit_prediction))


def _detect_operator_by_shape(image):
    """
    Detect operators using shape analysis (aspect ratio, pixel density, line detection).
    This is a heuristic approach that works for clearly written operators.

    Returns operator string or None if cannot determine.
    """
    # Ensure image is uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.copy()

    # Binarize
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Calculate features
    h, w = binary.shape
    pixel_density = np.sum(binary > 0) / (h * w)
    aspect_ratio = w / max(h, 1)

    # Find contours for shape analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Use Hough Line detection
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=10,
                            minLineLength=w * 0.3, maxLineGap=5)

    # --- Minus sign: single horizontal line, low pixel density ---
    if pixel_density < 0.15 and aspect_ratio > 1.5:
        return "-"

    # --- Plus sign: two perpendicular lines, cross shape ---
    if lines is not None and len(lines) >= 2:
        horizontal = 0
        vertical = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 30 or angle > 150:
                horizontal += 1
            elif 60 < angle < 120:
                vertical += 1

        if horizontal >= 1 and vertical >= 1 and pixel_density < 0.25:
            return "+"

    # --- Multiply (*): X shape or dot ---
    if pixel_density < 0.12:
        return "*"

    # --- Division (/): diagonal line ---
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 30 < angle < 70:
                return "/"

    # --- Parentheses: curved shapes ---
    if len(contours) == 1:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Parentheses are elongated curves, low circularity
            if circularity < 0.3 and aspect_ratio < 0.6:
                # Determine left or right parenthesis based on curve direction
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    if cx < w / 2:
                        return "("
                    else:
                        return ")"

    return None
