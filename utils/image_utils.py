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


def compose_mnist_number(digit_images, spacing=10, padding=20):
    """
    Compose multiple MNIST digit arrays into a single multi-digit number image.

    Args:
        digit_images: List of 28x28 numpy arrays (float32, 0-1, white-on-black)
        spacing: Pixels between digits
        padding: Pixels of padding around the number

    Returns:
        Grayscale uint8 image with black digits on white background
    """
    size = IMAGE_SIZE  # 28
    n = len(digit_images)
    canvas_w = n * size + (n - 1) * spacing + 2 * padding
    canvas_h = size + 2 * padding

    # White background
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    for i, digit in enumerate(digit_images):
        # Convert from float [0,1] white-on-black to uint8 black-on-white
        digit_uint8 = (255 - (digit * 255).astype(np.uint8))
        x = padding + i * (size + spacing)
        y = padding
        canvas[y:y + size, x:x + size] = digit_uint8

    return canvas


def canvas_to_mnist_format(canvas_image):
    """
    Convert a drawing canvas image (white background, black strokes)
    to MNIST format (black background, white digit, 28x28).

    Uses MNIST-style preprocessing:
    - Fit digit into 20x20 box preserving aspect ratio
    - Center in 28x28 frame using center of mass

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
    digit = inverted[y:y+h, x:x+w]

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

    M = cv2.moments(resized)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = new_w / 2, new_h / 2

    x_offset = int(round(14 - cx))
    y_offset = int(round(14 - cy))
    x_offset = max(0, min(x_offset, IMAGE_SIZE - new_w))
    y_offset = max(0, min(y_offset, IMAGE_SIZE - new_h))

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Normalize to [0, 1]
    return canvas.astype(np.float32) / 255.0
