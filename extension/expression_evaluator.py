"""
COS30018 - Extension Option 2: Expression Evaluator
Recognizes and evaluates handwritten arithmetic expressions.

Pipeline:
1. Segment the expression image into individual symbols
2. Classify each symbol as digit or operator using the 16-class expression model
3. Build expression string (concatenating consecutive digits into numbers)
4. Safely evaluate the expression and return the result
"""
import numpy as np
import cv2
from preprocessing.preprocessor import normalize_segmented
from segmentation.segmenter import segment
from extension.operator_recognizer import classify_symbol, load_expression_model


def recognize_expression(image, expression_model=None, segment_method="contour"):
    """
    Recognize a handwritten arithmetic expression from an image.

    Args:
        image: Input image containing the expression
        expression_model: Trained ExpressionCNN (16 classes). Auto-loaded if None.
        segment_method: Segmentation method to use

    Returns:
        Dict with:
            - expression: The recognized expression string (e.g., "2+3*4")
            - result: The computed result
            - symbols: List of (type, value) for each symbol
            - error: Error message if evaluation fails
    """
    # Auto-load expression model if not provided
    if expression_model is None:
        expression_model = load_expression_model()
        if expression_model is None:
            return {
                "expression": "",
                "result": None,
                "symbols": [],
                "error": "Expression model not trained. Run train_expression_model() first.",
            }

    # Step 1: Segment the image into individual symbols
    digit_images, bounding_boxes = segment(image, method=segment_method)

    if not digit_images:
        return {
            "expression": "",
            "result": None,
            "symbols": [],
            "error": "No symbols found in image",
        }

    # Step 2: Classify each symbol using the 16-class model
    symbols = []
    for digit_img in digit_images:
        processed = normalize_segmented(digit_img)
        symbol_type, value = classify_symbol(processed, expression_model)
        symbols.append((symbol_type, value))

    # Step 3: Build expression string
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
    """
    parts = []
    current_number = ""

    for symbol_type, value in symbols:
        if symbol_type == "digit":
            current_number += str(value)
        else:
            if current_number:
                parts.append(current_number)
                current_number = ""
            parts.append(str(value))

    if current_number:
        parts.append(current_number)

    return "".join(parts)


def _safe_eval(expression):
    """
    Safely evaluate a mathematical expression.
    Only allows digits and basic operators to prevent code injection.
    """
    if not expression:
        return None, "Empty expression"

    allowed_chars = set("0123456789+-*/()., ")
    if not all(c in allowed_chars for c in expression):
        return None, f"Invalid characters in expression: {expression}"

    expression = expression.replace("÷", "/")

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        if isinstance(result, float):
            result = round(result, 6)
        return result, None
    except ZeroDivisionError:
        return None, "Division by zero"
    except SyntaxError:
        return None, f"Invalid expression syntax: {expression}"
    except Exception as e:
        return None, f"Evaluation error: {str(e)}"
