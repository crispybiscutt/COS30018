"""
COS30018 - Task 1: Image Preprocessing
Implements 3 different preprocessing techniques for handwritten digit images.

Techniques:
1. Basic: Grayscale -> Resize -> Normalize
2. Otsu Binarization: Grayscale -> Otsu Threshold -> Resize -> Normalize
3. Adaptive Threshold: Gaussian Blur -> Adaptive Threshold -> Morphological -> Resize -> Normalize
"""
import cv2
import numpy as np
from config import IMAGE_SIZE, PREPROCESS_BASIC, PREPROCESS_OTSU, PREPROCESS_ADAPTIVE


def preprocess_basic(image):
    """
    Technique 1 - Basic preprocessing.
    Steps: Convert to grayscale -> Resize to 28x28 -> Normalize pixel values to [0, 1].
    Simple and fast. Works well when input images are clean.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize to IMAGE_SIZE x IMAGE_SIZE
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def preprocess_otsu(image):
    """
    Technique 2 - Otsu Binarization.
    Steps: Grayscale -> Otsu automatic thresholding -> Resize -> Normalize.
    Otsu's method automatically determines the optimal threshold to separate
    foreground (digit) from background. Good for varying lighting conditions.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Otsu's thresholding (automatically finds optimal threshold)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize to IMAGE_SIZE x IMAGE_SIZE
    resized = cv2.resize(binary, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def preprocess_adaptive(image):
    """
    Technique 3 - Adaptive Threshold with Denoising.
    Steps: Grayscale -> Gaussian Blur (denoise) -> Adaptive Threshold -> Morphological
           closing (fill small holes) -> Resize -> Normalize.
    Best for noisy images or images with uneven lighting. The adaptive threshold
    computes threshold for each pixel based on its neighborhood.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold - threshold varies per-pixel based on local neighborhood
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to fill small holes in digits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Resize to IMAGE_SIZE x IMAGE_SIZE
    resized = cv2.resize(closed, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def preprocess(image, method=PREPROCESS_BASIC):
    """
    Main preprocessing function. Applies the selected technique.

    Args:
        image: Input image (numpy array, BGR or grayscale)
        method: One of 'basic', 'otsu', 'adaptive'

    Returns:
        Preprocessed image as numpy array of shape (28, 28), values in [0, 1]
    """
    methods = {
        PREPROCESS_BASIC: preprocess_basic,
        PREPROCESS_OTSU: preprocess_otsu,
        PREPROCESS_ADAPTIVE: preprocess_adaptive,
    }

    if method not in methods:
        raise ValueError(f"Unknown preprocessing method: {method}. "
                         f"Choose from: {list(methods.keys())}")

    return methods[method](image)


def preprocess_for_model(image, method=PREPROCESS_BASIC):
    """
    Preprocess an image and reshape it for model input.
    Returns shape (1, 28, 28, 1) suitable for CNN input.
    """
    processed = preprocess(image, method)
    return processed.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def normalize_segmented(image):
    """
    Normalize a segmented digit image that is ALREADY in MNIST format
    (white digit on black background, 28x28, uint8).

    This should be used AFTER segmentation, instead of preprocess().
    Segmentation output is already binarized and sized - just normalize to [0,1].
    """
    img = np.array(image, dtype=np.float32)

    # Ensure it's 28x28
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img_uint8 = img.astype(np.uint8) if img.max() <= 255 else img
        img = cv2.resize(img_uint8.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32)

    # Normalize to [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    return img


def invert_if_needed(image):
    """
    MNIST digits are white on black background.
    If input has dark digits on light background, invert it.
    """
    mean_val = np.mean(image)
    if mean_val > 127:
        return 255 - image
    return image
