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
    Normalize a segmented digit image to match MNIST format precisely.
    MNIST digits are:
    - White digit on black background
    - Fitted into a 20x20 pixel box (preserving aspect ratio)
    - Centered in 28x28 frame using center of mass
    - Anti-aliased grayscale values normalized to [0, 1]

    This should be used AFTER segmentation, instead of preprocess().
    """
    img = np.array(image, dtype=np.uint8)

    # Ensure we have white-on-black
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)

    # Find bounding box of the digit content
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Fit into 20x20 box preserving aspect ratio (like MNIST)
    target_size = 20
    if h > w:
        new_h = target_size
        new_w = max(1, int(w * (target_size / h)))
    else:
        new_w = target_size
        new_h = max(1, int(h * (target_size / w)))

    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place in 28x28 canvas, centered by center of mass
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Compute center of mass of the resized digit
    M = cv2.moments(resized)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = new_w / 2, new_h / 2

    # Place so that center of mass aligns with center of 28x28
    x_offset = int(round(14 - cx))
    y_offset = int(round(14 - cy))

    # Clamp offsets to keep digit within canvas
    x_offset = max(0, min(x_offset, IMAGE_SIZE - new_w))
    y_offset = max(0, min(y_offset, IMAGE_SIZE - new_h))

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Normalize to [0, 1]
    return canvas.astype(np.float32) / 255.0


def preprocess_photo(image):
    """
    Camera photo preprocessing pipeline.
    Optimized for photos taken with a phone/camera of handwritten expressions.

    Steps:
    1. Convert to grayscale
    2. Bilateral filter (edge-preserving denoising - better than Gaussian for photos)
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
    4. Adaptive threshold with larger block size for camera noise
    5. Morphological operations to clean up

    Returns: Binary image (uint8, white digits on black bg) at original resolution.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Bilateral filter: preserves edges while removing noise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE: normalize lighting across the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)

    # Adaptive threshold with larger block for camera photos
    binary = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Morphological: close small gaps then open to remove noise specks
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    return cleaned


def prepare_for_segmentation(image, method=PREPROCESS_BASIC):
    """
    Prepare an image for segmentation by converting to binary
    (white digits on black background) using the selected preprocessing method.

    Unlike preprocess() which outputs 28x28, this preserves the original resolution
    so segmentation can find bounding boxes accurately.

    Args:
        image: Input image (BGR or grayscale)
        method: 'basic', 'otsu', 'adaptive', or 'photo'

    Returns:
        Binary image (uint8, white on black) at original resolution
    """
    if method == "photo":
        return preprocess_photo(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert if needed (make digits white on black)
    if np.mean(gray) > 127:
        gray = 255 - gray

    if method == PREPROCESS_BASIC:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == PREPROCESS_OTSU:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == PREPROCESS_ADAPTIVE:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def invert_if_needed(image):
    """
    MNIST digits are white on black background.
    If input has dark digits on light background, invert it.
    """
    mean_val = np.mean(image)
    if mean_val > 127:
        return 255 - image
    return image
