"""
COS30018 - Extension Option 2: Expression Evaluator
Recognizes and evaluates handwritten arithmetic expressions.

Pipeline:
1. Segment the expression image into individual symbols
2. Classify each symbol as digit or operator
3. Build expression string
4. Safely evaluate the expression and return the result
"""
import numpy as np
import cv2
from preprocessing.preprocessor import preprocess
from segmentation.segmenter import segment
from extension.operator_recognizer import classify_symbol


def recognize_expression(image, digit_model, segment_method="contour"):
    """
    Recognize a handwritten arithmetic expression from an image.

    Args:
        image: Input image containing the expression
        digit_model: Trained digit recognition model
        segment_method: Segmentation method to use

    Returns:
        Dict with:
            - expression: The recognized expression string (e.g., "2+3*4")
            - result: The computed result
            - symbols: List of (type, value) for each symbol
            - error: Error message if evaluation fails
    """
    # Step 1: Segment the image into individual symbols
    digit_images, bounding_boxes = segment(image, method=segment_method)

    if not digit_images:
        return {
            "expression": "",
            "result": None,
            "symbols": [],
            "error": "No symbols found in image",
        }

    # Step 2: Classify each symbol
    symbols = []
    expression_parts = []

    for digit_img in digit_images:
        # Preprocess for model
        processed = preprocess(digit_img)
        symbol_type, value = classify_symbol(processed, digit_model)
        symbols.append((symbol_type, value))
        expression_parts.append(str(value))

    # Step 3: Build expression string
    # Handle multi-digit numbers (consecutive digits should be concatenated)
    expression = _build_expression(symbols)

    # Step 4: Safely evaluate
    result, error = _safe_eval(expression)

    return {
        "expression": expression,
        "result": result,
        "symbols": symbols,
        "error": error,
    }


def _build_expression(symbols):
    """
    Build expression string from classified symbols.
    Handles multi-digit numbers by concatenating consecutive digits.

    Args:
        symbols: List of (type, value) tuples

    Returns:
        Expression string like "23+45*6"
    """
    parts = []
    current_number = ""

    for symbol_type, value in symbols:
        if symbol_type == "digit":
            current_number += str(value)
        else:
            # Operator encountered
            if current_number:
                parts.append(current_number)
                current_number = ""
            parts.append(str(value))

    # Don't forget the last number
    if current_number:
        parts.append(current_number)

    return "".join(parts)


def _safe_eval(expression):
    """
    Safely evaluate a mathematical expression.
    Only allows digits and basic operators to prevent code injection.

    Args:
        expression: String like "2+3*4"

    Returns:
        Tuple of (result, error_message)
    """
    if not expression:
        return None, "Empty expression"

    # Validate: only allow digits, operators, parentheses, and spaces
    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return None, f"Invalid characters in expression: {expression}"

    # Replace ÷ with / if present
    expression = expression.replace("÷", "/")

    try:
        # Use Python's eval with restricted globals (no builtins)
        result = eval(expression, {"__builtins__": {}}, {})
        # Round to avoid floating point issues
        if isinstance(result, float):
            result = round(result, 6)
        return result, None
    except ZeroDivisionError:
        return None, "Division by zero"
    except SyntaxError:
        return None, f"Invalid expression syntax: {expression}"
    except Exception as e:
        return None, f"Evaluation error: {str(e)}"
