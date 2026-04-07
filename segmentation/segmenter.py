"""
COS30018 - Task 2: Image Segmentation
Implements 3 different segmentation techniques to split a multi-digit number image
into individual digit images.

Techniques:
1. Contour-based: Uses OpenCV findContours to detect digit boundaries
2. Connected Components: Uses connectedComponentsWithStats to label regions
3. Vertical Projection: Analyzes vertical pixel distribution to find gaps between digits
"""
import cv2
import numpy as np
from config import IMAGE_SIZE, SEGMENT_CONTOUR, SEGMENT_CONNECTED, SEGMENT_PROJECTION, PREPROCESS_BASIC


def _prepare_binary(image, preprocess_method=None):
    """Convert input image to binary (white digits on black background).

    Args:
        image: Input image (BGR or grayscale)
        preprocess_method: If provided, use the corresponding preprocessing
            technique from preprocessor.py for better results on camera photos.
    """
    if preprocess_method is not None:
        from preprocessing.preprocessor import prepare_for_segmentation
        return prepare_for_segmentation(image, method=preprocess_method)

    # Default: simple Otsu (backwards compatible)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if np.mean(gray) > 127:
        gray = 255 - gray

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _extract_and_pad_digit(binary, x, y, w, h, padding=4):
    """
    Extract a digit region and pad it to make it square,
    then resize to IMAGE_SIZE x IMAGE_SIZE.
    """
    # Extract the digit region
    digit = binary[y:y+h, x:x+w]

    # Make the digit square by adding padding
    max_dim = max(w, h)
    square = np.zeros((max_dim + 2 * padding, max_dim + 2 * padding), dtype=np.uint8)

    # Center the digit in the square
    x_offset = (max_dim + 2 * padding - w) // 2
    y_offset = (max_dim + 2 * padding - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # Resize to IMAGE_SIZE x IMAGE_SIZE
    resized = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized


def segment_contour(image, preprocess_method=None):
    """
    Technique 1 - Contour-based segmentation.
    Steps: Binarize -> Find external contours -> Get bounding boxes ->
           Filter small contours (noise) -> Sort left-to-right -> Extract digits.

    Uses OpenCV's findContours which traces the boundaries of white regions.
    Good for well-separated digits with clear boundaries.
    """
    binary = _prepare_binary(image, preprocess_method)

    # Find external contours only (outermost boundaries)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes and filter out noise (too small regions)
    img_h, img_w = binary.shape
    min_area = (img_h * img_w) * 0.01  # At least 1% of image area

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        # Accept if tall enough OR wide enough (catches minus sign, etc.)
        if area > min_area and (h > img_h * 0.15 or w > img_w * 0.03):
            bounding_boxes.append((x, y, w, h))

    # Sort left-to-right by x coordinate
    bounding_boxes.sort(key=lambda b: b[0])

    # Extract each digit
    digits = []
    for (x, y, w, h) in bounding_boxes:
        digit = _extract_and_pad_digit(binary, x, y, w, h)
        digits.append(digit)

    return digits, bounding_boxes


def segment_connected_components(image, preprocess_method=None):
    """
    Technique 2 - Connected Components segmentation.
    Steps: Binarize -> Find connected components with stats ->
           Filter by area -> Sort left-to-right -> Extract digits.

    Uses OpenCV's connectedComponentsWithStats which labels each connected
    region of white pixels. More robust than contours for overlapping strokes.
    """
    binary = _prepare_binary(image, preprocess_method)

    # Find connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    img_h, img_w = binary.shape
    min_area = (img_h * img_w) * 0.005

    bounding_boxes = []
    for i in range(1, num_labels):  # Skip label 0 (background)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area and (h > img_h * 0.15 or w > img_w * 0.03):
            bounding_boxes.append((x, y, w, h))

    # Sort left-to-right
    bounding_boxes.sort(key=lambda b: b[0])

    # Merge overlapping boxes (digits with disconnected strokes)
    merged = _merge_overlapping_boxes(bounding_boxes)

    digits = []
    for (x, y, w, h) in merged:
        digit = _extract_and_pad_digit(binary, x, y, w, h)
        digits.append(digit)

    return digits, merged


def segment_projection(image, preprocess_method=None):
    """
    Technique 3 - Vertical Projection segmentation.
    Steps: Binarize -> Compute vertical projection (sum of white pixels per column) ->
           Find gaps (columns with zero/low projection) -> Split at gaps -> Extract digits.

    Analyzes the distribution of ink along the horizontal axis.
    Works best for digits that are clearly separated with gaps between them.
    """
    binary = _prepare_binary(image, preprocess_method)

    # Compute vertical projection (sum of white pixels for each column)
    projection = np.sum(binary > 0, axis=0)

    # Find digit regions (where projection > threshold)
    threshold = max(2, np.max(projection) * 0.08)
    is_digit = projection > threshold

    # Find start and end of each digit region
    regions = []
    in_region = False
    start = 0

    for i in range(len(is_digit)):
        if is_digit[i] and not in_region:
            start = i
            in_region = True
        elif not is_digit[i] and in_region:
            regions.append((start, i))
            in_region = False

    if in_region:
        regions.append((start, len(is_digit)))

    # Extract digits from each region
    digits = []
    bounding_boxes = []
    img_h = binary.shape[0]

    for (x_start, x_end) in regions:
        # Find vertical extent of the digit in this column range
        column_region = binary[:, x_start:x_end]
        rows_with_pixels = np.where(np.sum(column_region > 0, axis=1) > 0)[0]

        if len(rows_with_pixels) == 0:
            continue

        y = rows_with_pixels[0]
        h = rows_with_pixels[-1] - y + 1
        x = x_start
        w = x_end - x_start

        if w > 2 and h > img_h * 0.1:  # Filter very narrow/short segments
            bounding_boxes.append((x, y, w, h))
            digit = _extract_and_pad_digit(binary, x, y, w, h)
            digits.append(digit)

    return digits, bounding_boxes


def _merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """Merge bounding boxes that overlap significantly (for disconnected digit strokes)."""
    if len(boxes) <= 1:
        return boxes

    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue

            x1, y1, w1, h1 = merged[i]
            current = [x1, y1, x1 + w1, y1 + h1]

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue

                x2, y2, w2, h2 = merged[j]

                # Check horizontal overlap
                overlap_x = max(0, min(current[2], x2 + w2) - max(current[0], x2))
                min_w = min(current[2] - current[0], w2)

                if min_w > 0 and overlap_x / min_w > overlap_threshold:
                    # Merge the boxes
                    current[0] = min(current[0], x2)
                    current[1] = min(current[1], y2)
                    current[2] = max(current[2], x2 + w2)
                    current[3] = max(current[3], y2 + h2)
                    used[j] = True
                    changed = True

            new_merged.append((
                current[0], current[1],
                current[2] - current[0], current[3] - current[1]
            ))
            used[i] = True

        merged = new_merged

    merged.sort(key=lambda b: b[0])
    return merged


def segment(image, method=SEGMENT_CONTOUR, preprocess_method=None):
    """
    Main segmentation function. Applies the selected technique.

    Args:
        image: Input image containing multi-digit number
        method: One of 'contour', 'connected_components', 'projection'
        preprocess_method: Preprocessing to apply before segmentation
            (None=default Otsu, 'basic', 'otsu', 'adaptive', 'photo')

    Returns:
        Tuple of (list of digit images, list of bounding boxes)
        Each digit image is 28x28, uint8, white digits on black background
    """
    methods = {
        SEGMENT_CONTOUR: segment_contour,
        SEGMENT_CONNECTED: segment_connected_components,
        SEGMENT_PROJECTION: segment_projection,
    }

    if method not in methods:
        raise ValueError(f"Unknown segmentation method: {method}. "
                         f"Choose from: {list(methods.keys())}")

    return methods[method](image, preprocess_method=preprocess_method)
