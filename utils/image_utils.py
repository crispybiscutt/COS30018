"""
COS30018 - Image Utilities
Helper functions for loading, saving, and creating test images.
Includes creating multi-digit number images from folders of individual digit images.
"""
import os
import cv2
import numpy as np
from config import IMAGE_SIZE, TEST_IMAGES_DIR


def load_image(path):
    """
    Load an image from file.

    Args:
        path: Path to image file (png, jpg, bmp, etc.)

    Returns:
        Image as numpy array (BGR for color, grayscale for gray)
    """
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(image, path):
    """Save an image to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)


def create_number_from_digits(digit_image_paths, spacing=10, padding=20):
    """
    Create a multi-digit number image from a folder of individual digit images.
    This fulfills the requirement: "automatic creation of the image of a number
    from a folder of images of individual digits".

    Args:
        digit_image_paths: List of paths to individual digit images (sorted)
        spacing: Pixels between digits
        padding: Pixels of padding around the number

    Returns:
        Combined image as numpy array (grayscale)
    """
    digit_images = []
    max_height = 0

    for path in digit_image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        digit_images.append(img)
        max_height = max(max_height, img.shape[0])

    if not digit_images:
        raise ValueError("No valid digit images found")

    # Resize all digits to same height while keeping aspect ratio
    resized = []
    for img in digit_images:
        h, w = img.shape
        new_w = int(w * (max_height / h))
        resized.append(cv2.resize(img, (new_w, max_height)))

    # Calculate total width
    total_width = sum(img.shape[1] for img in resized) + spacing * (len(resized) - 1)

    # Create canvas (white background)
    canvas = np.ones((max_height + 2 * padding, total_width + 2 * padding), dtype=np.uint8) * 255

    # Place each digit
    x_offset = padding
    for img in resized:
        h, w = img.shape
        y_offset = padding + (max_height - h) // 2
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
        x_offset += w + spacing

    return canvas


def create_number_from_folder(folder_path, spacing=10, padding=20):
    """
    Create a multi-digit number image from all images in a folder.
    Images are sorted by filename to determine digit order.

    Args:
        folder_path: Path to folder containing digit images

    Returns:
        Combined number image
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ])

    if not files:
        raise ValueError(f"No image files found in: {folder_path}")

    return create_number_from_digits(files, spacing, padding)


def canvas_to_mnist_format(canvas_image):
    """
    Convert a drawing canvas image (white background, black strokes)
    to MNIST format (black background, white digit, 28x28).

    Args:
        canvas_image: Grayscale image from PyQt5 drawing canvas

    Returns:
        28x28 numpy array, float32, values [0, 1]
    """
    # Convert to grayscale if needed
    if len(canvas_image.shape) == 3:
        gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas_image.copy()

    # Invert (MNIST: white digit on black background)
    inverted = 255 - gray

    # Find bounding box of the digit
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # Extract digit and add padding
    digit = inverted[y:y+h, x:x+w]
    max_dim = max(w, h)
    pad = int(max_dim * 0.2)

    square = np.zeros((max_dim + 2 * pad, max_dim + 2 * pad), dtype=np.uint8)
    x_off = (max_dim + 2 * pad - w) // 2
    y_off = (max_dim + 2 * pad - h) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    # Resize to 28x28
    resized = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    return resized.astype(np.float32) / 255.0
