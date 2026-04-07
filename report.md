# COS30018 - Intelligent Systems
# Assignment Option B: Handwritten Number Recognition System (HNRS)

**Swinburne University of Technology**
**Semester: Summer 2025**

**Student:** Vu Minh - 104852111

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall System Architecture](#2-overall-system-architecture)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Image Segmentation](#4-image-segmentation)
5. [Handwritten Digit Recognition](#5-handwritten-digit-recognition)
6. [Overall System Functions and Performance](#6-overall-system-functions-and-performance)
7. [GUI Implementation](#7-gui-implementation)
8. [Extension: Arithmetic Expression Recognition](#8-extension-arithmetic-expression-recognition)
9. [Critical Analysis](#9-critical-analysis)
10. [Summary and Conclusion](#10-summary-and-conclusion)

---

## 1. Introduction

This report presents the design, implementation, and evaluation of a **Handwritten Number Recognition System (HNRS)** developed for the COS30018 Intelligent Systems unit. The system addresses the challenge of recognizing handwritten multi-digit numbers from images, combining image preprocessing, segmentation, and machine learning classification into an integrated pipeline.

The HNRS is trained on the MNIST dataset (60,000 training and 10,000 test images of handwritten digits 0-9) and supports three distinct ML models for performance comparison. A PyQt5-based graphical user interface allows users to draw digits, upload images, select models and preprocessing techniques, and view detailed evaluation metrics. As an extension, the system also recognizes simple arithmetic expressions containing operators (+, -, *, /, parentheses) and computes their results.

**Technology Stack:**
- Python 3.13, PyTorch 2.7 (deep learning), scikit-learn (classical ML, evaluation)
- OpenCV (image processing), PyQt5 (GUI), matplotlib/seaborn (visualization)

---

## 2. Overall System Architecture

The system follows a modular pipeline architecture with clear separation between preprocessing, segmentation, classification, and evaluation:

```
Input Image (Canvas / File / Folder)
    |
    v
[Preprocessing] --> Basic / Otsu Binarization / Adaptive Threshold
    |
    v
[Segmentation]  --> Contour / Connected Components / Vertical Projection
    |
    v
[Normalization] --> MNIST-style: 20x20 fit + center-of-mass centering in 28x28
    |
    v
[Classification] --> CNN (PyTorch) / SVM / KNN
    |
    v
[Output] --> Predicted digit(s) / expression with confidence scores
```

All models inherit from an abstract base class (`BaseModel`) that enforces a consistent interface: `build()`, `train()`, `predict()`, `predict_proba()`, `save()`, `load()`. A central `ModelManager` handles the model lifecycle (factory creation, training pipeline, serialization). This design allows adding new model types with minimal changes.

### Project Structure

```
COS30018/
├── main.py                     # Entry point
├── config.py                   # Global configuration
├── preprocessing/preprocessor.py  # 3 preprocessing techniques
├── segmentation/segmenter.py      # 3 segmentation methods
├── models/
│   ├── base_model.py           # Abstract base class
│   ├── cnn_pytorch.py          # CNN (PyTorch) with data augmentation
│   ├── svm_model.py            # SVM classifier (scikit-learn)
│   ├── knn_model.py            # KNN classifier (scikit-learn)
│   └── model_manager.py        # Model lifecycle management
├── evaluation/evaluator.py     # Metrics & visualization
├── gui/                        # PyQt5 GUI (3 tabs)
├── extension/                  # Arithmetic expression recognition
└── utils/image_utils.py        # Image I/O utilities
```

---

## 3. Data Preprocessing

Three preprocessing techniques were investigated to handle varying input quality. All techniques convert the input to a 28x28 grayscale image normalized to [0, 1].

### 3.1 Technique 1: Basic Preprocessing

The simplest pipeline: grayscale conversion, resize to 28x28 (using INTER_AREA interpolation for downsampling), and normalization. Suitable for clean input images where noise is minimal.

### 3.2 Technique 2: Otsu Binarization

Adds automatic threshold selection using Otsu's method, which minimizes intra-class variance between foreground (digit) and background pixels. The algorithm analyzes the image histogram and finds the optimal threshold that separates the two classes. This is effective for images with bimodal intensity distributions (clear separation between ink and paper) and varying lighting conditions.

### 3.3 Technique 3: Adaptive Threshold with Denoising

The most robust technique, designed for noisy or unevenly lit images:
1. Gaussian blur (5x5 kernel) to suppress high-frequency noise
2. Adaptive threshold (Gaussian-weighted, block size 11) where the threshold varies per-pixel based on the local neighborhood mean, handling uneven illumination
3. Morphological closing (3x3 kernel) to fill small gaps in digit strokes

### 3.4 Technique 4: Camera Photo Preprocessing

A specialized pipeline optimized for photos taken with a phone or camera of handwritten expressions on paper:

1. **Bilateral filter** (9px diameter, sigma 75): Edge-preserving denoising that removes camera sensor noise while keeping digit edges sharp - superior to Gaussian blur for real photos
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization, clip limit 2.0, 8x8 grid): Normalizes lighting across the image by equalizing contrast in local regions, handling shadows and uneven illumination common in camera photos
3. **Adaptive threshold** (Gaussian-weighted, block size 21, constant 10): Larger block size and higher constant than Technique 3 to handle the higher noise level in camera photos
4. **Morphological closing** (3x3 kernel) followed by **morphological opening** (2x2 kernel): First closes small gaps in digit strokes, then removes small noise specks

When a user uploads a photo, the system auto-detects it (by image resolution > 300px) and switches to this preprocessing mode.

### 3.5 MNIST-Style Normalization

A critical preprocessing step is the **MNIST-style normalization** applied to segmented digits before classification. The MNIST dataset digits are centered in a 28x28 frame using center of mass, with the digit content fitted into a 20x20 pixel bounding box. Our `normalize_segmented()` function replicates this:

1. Find the bounding box of the digit content
2. Resize to fit within a 20x20 pixel box (preserving aspect ratio)
3. Compute the center of mass of the resized digit
4. Place the digit in a 28x28 frame so the center of mass aligns with the frame center (14, 14)

This step significantly improves recognition accuracy for hand-drawn input, as it ensures the digit representation matches the format the models were trained on.

### 3.6 Comparison and Selection

| Technique | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| Basic | Fast, preserves grayscale detail | No noise handling | Clean digital images |
| Otsu | Automatic, handles lighting | Assumes bimodal distribution | Scanned documents |
| Adaptive | Handles uneven lighting, denoises | Slightly slower | Hand-drawn canvas input |
| **Photo** | **CLAHE + bilateral filter, best noise handling** | **Slowest** | **Camera photos of handwriting** |

**Selected technique:** The system auto-selects the appropriate technique based on input source. Canvas drawings use Adaptive Threshold, while uploaded camera photos use the Photo preprocessing pipeline. Users can manually override via the GUI dropdown.

---

## 4. Image Segmentation

Three segmentation techniques were implemented to extract individual digits from multi-digit number images.

### 4.1 Technique 1: Contour-Based Segmentation

Uses OpenCV's `findContours` with `RETR_EXTERNAL` mode to trace the outermost boundaries of white regions in a binarized image. Bounding boxes are computed for each contour, filtered by minimum area (>1% of image area) and minimum height (>15% of image height), then sorted left-to-right. Each extracted region is padded to square and resized to 28x28.

**Strengths:** Fast, works well for well-separated digits with clear boundaries.
**Weaknesses:** May fail when digit strokes touch or overlap.

### 4.2 Technique 2: Connected Components

Uses `cv2.connectedComponentsWithStats` with 8-connectivity to label each connected region of white pixels. Includes a merging step for overlapping bounding boxes (overlap threshold 0.3) to handle digits with disconnected strokes (e.g., broken "8", dots in "i").

**Strengths:** Handles disconnected strokes within a single digit.
**Weaknesses:** May over-segment digits with multiple components at wider separations.

### 4.3 Technique 3: Vertical Projection

Analyzes the vertical projection profile (sum of white pixels per column) to identify digit regions and gaps. Columns with projection above a threshold (8% of maximum) are classified as digit regions; gaps between regions indicate digit boundaries.

**Strengths:** Conceptually simple, works well for uniformly spaced digits.
**Weaknesses:** Fails when digits touch or have uneven vertical extent.

### 4.4 Comparison and Selection

| Technique | Speed | Touching Digits | Disconnected Strokes | Uneven Spacing |
|-----------|-------|-----------------|---------------------|----------------|
| **Contour** | **Fast** | Poor | Moderate | **Good** |
| Connected Components | Moderate | Moderate | **Good** | Good |
| Projection | Fast | Poor | Poor | Poor |

**Selected technique:** Contour-based segmentation is the default for its speed and reliability with typical hand-drawn input. Connected Components is recommended when digits have broken strokes.

---

## 5. Handwritten Digit Recognition

### 5.1 Model 1: CNN (PyTorch) - Best Model

**Architecture:**

| Layer | Type | Details | Output Shape |
|-------|------|---------|-------------|
| 1 | Conv2d + ReLU | 32 filters, 3x3 | (32, 26, 26) |
| 2 | MaxPool2d | 2x2 stride 2 | (32, 13, 13) |
| 3 | Conv2d + ReLU | 64 filters, 3x3 | (64, 11, 11) |
| 4 | MaxPool2d | 2x2 stride 2 | (64, 5, 5) |
| 5 | Conv2d + ReLU | 64 filters, 3x3 | (64, 3, 3) |
| 6 | Flatten + Dense + ReLU | 576 -> 128 | (128,) |
| 7 | Dropout | p=0.5 | (128,) |
| 8 | Dense (output) | 128 -> 10 | (10,) |

**Training Configuration:**
- Optimizer: Adam (lr=0.001) with StepLR scheduler (decay 0.5 every 5 epochs)
- Loss: CrossEntropyLoss
- Data augmentation: RandomAffine (rotation +/-10deg, translate 10%, scale 90-110%)
- Batch size: 64, Epochs: 15, Validation split: 10%

**Data augmentation** was added to improve robustness against hand-drawn input that differs from clean MNIST images. Random rotation, translation, and scaling simulate natural handwriting variation.

**Result: 99.45% accuracy on MNIST test set** (improved from 99.28% after adding augmentation)

### 5.2 Model 2: SVM (Support Vector Machine)

SVM finds the optimal hyperplane that maximizes the margin between classes in a high-dimensional feature space.

**Configuration:**
- Kernel: RBF (Radial Basis Function) - maps 784-dimensional input to infinite-dimensional space
- C = 10 (regularization: higher values = more complex decision boundary)
- gamma = "scale" (1 / (n_features * variance(X)))
- Input: Flattened 28x28 = 784 features, standardized via StandardScaler
- Training subset: 20,000 samples (SVM training is O(n^2) to O(n^3))

**Result: 96.31% accuracy on MNIST test set**

### 5.3 Model 3: KNN (K-Nearest Neighbors)

KNN is an instance-based learning algorithm that classifies by finding the K most similar training samples and voting.

**Configuration:**
- K = 5 neighbors with distance-weighted voting (closer neighbors have more influence)
- Algorithm: auto-selection between ball_tree, kd_tree, and brute-force
- Parallel processing: all CPU cores (n_jobs=-1)
- Input: Flattened 784 features, standardized via StandardScaler
- Training subset: 20,000 samples

**Result: 93.14% accuracy on MNIST test set**

### 5.4 Model Comparison and Selection

| Metric | CNN (PyTorch) | SVM | KNN |
|--------|--------------|-----|-----|
| **Accuracy** | **99.45%** | 96.31% | 93.14% |
| Inference/sample | 0.374ms | 5.189ms | 0.326ms |
| Training time | 413.2s | 343.7s | 7.7s |
| Total parameters | ~100K | N/A (support vectors) | N/A (stored samples) |

**Selected model:** CNN (PyTorch) is the best model for production use due to its superior accuracy and fast inference. The 99.45% accuracy represents near-human performance on MNIST.

---

## 6. Overall System Functions and Performance

### 6.1 Evaluation Methodology

All models were evaluated on the full MNIST test set (10,000 images) with the following metrics:
- **Accuracy**: Overall percentage of correct predictions
- **Precision**: TP / (TP + FP) per class - how many predicted positives are correct
- **Recall**: TP / (TP + FN) per class - how many actual positives are detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: 10x10 matrix showing classification patterns
- **Inference Time**: Average time per prediction and total for 10K samples

### 6.2 Per-Class Performance - CNN (Best Model)

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.9949 | 0.9969 | 0.9959 |
| 1 | 0.9982 | 0.9956 | 0.9969 |
| 2 | 0.9913 | 0.9952 | 0.9932 |
| 3 | 0.9902 | 0.9970 | 0.9936 |
| 4 | 0.9949 | 0.9969 | 0.9959 |
| 5 | 0.9944 | 0.9899 | 0.9921 |
| 6 | 0.9979 | 0.9916 | 0.9948 |
| 7 | 0.9942 | 0.9942 | 0.9942 |
| 8 | 0.9928 | 0.9959 | 0.9943 |
| 9 | 0.9920 | 0.9901 | 0.9911 |

All digits achieve F1-scores above 0.99, with digit "1" having the highest precision (0.9982) and digit "9" having the lowest recall (0.9901).

### 6.3 Multi-Digit Number Recognition

The system supports multi-digit number recognition through the segmentation pipeline:
1. Input image is segmented into individual digit regions
2. Each region is normalized to MNIST format (center-of-mass centering)
3. Each digit is classified independently
4. Consecutive digit predictions are concatenated to form the number

Testing with hand-drawn numbers on the canvas shows reliable recognition for clearly separated digits. The MNIST-style normalization (fitting to 20x20 box and centering by center of mass) is critical for bridging the gap between clean MNIST data and real handwritten input.

### 6.4 Image Acquisition

Two input methods are supported as required:
1. **Drawing Canvas**: 560x200 pixel canvas with dark ink on white background
2. **Image File Upload**: Supports PNG, JPG, BMP, TIFF formats
3. **Folder-based**: Automatically assembles a number image from a folder of individual digit images, sorted by filename

---

## 7. GUI Implementation

The GUI is built with PyQt5 using a modern Fusion style with a clean light theme.

### Tab 1: Recognition
- Interactive drawing canvas (560x200px)
- Dropdown selectors for model, preprocessing, and segmentation methods
- Expression Mode toggle for arithmetic recognition
- Result display with confidence score and probability bars for all 10 classes
- Scrollable segmented symbols display showing each detected digit

### Tab 2: Evaluation
- Summary cards per model showing accuracy and inference time
- Accuracy comparison bar chart (matplotlib embedded)
- Confusion matrix heatmap (seaborn) with previous/next navigation
- Per-class metrics table (precision, recall, F1-score)

### Tab 3: Models
- Model architecture cards with accuracy, training time, and status
- Re-training interface with configurable epochs and batch size
- Background training thread (QThread) to prevent GUI freezing
- Live training progress bar and log output

---

## 8. Extension: Arithmetic Expression Recognition

### 8.1 Approach

A 16-class CNN (same architecture as the digit CNN, with 16 output neurons) recognizes both digits (0-9) and operators (+, -, *, /, (, )). A hybrid classification strategy uses the 16-class model to determine if a symbol is a digit or operator, then delegates digit classification to the dedicated 99.45% digit model for higher accuracy.

### 8.2 Synthetic Operator Training Data

Since no standard dataset exists for handwritten operators, 5,000 synthetic images per operator class are generated using OpenCV drawing functions with random variations in position, thickness, and size. Data augmentation (Gaussian noise + random rotation +/-10deg) increases diversity. Combined with 5,000 MNIST samples per digit, the total training set is ~80,000 balanced samples.

### 8.3 Aspect Ratio Heuristics

To reduce confusion between visually similar symbols (e.g., "+" vs "1", "-" vs "1"):
- A cross pattern detector identifies "+" symbols based on pixel presence in all 4 quadrants
- Aspect ratio thresholds distinguish tall narrow symbols (likely digits) from wide symbols (likely operators)

### 8.4 Expression Evaluation

Recognized symbols are assembled into an expression string (consecutive digits concatenated into numbers), then safely evaluated using Python's `eval()` with restricted builtins and a character whitelist.

### 8.5 Camera Photo Upload

The system supports uploading camera photos of handwritten expressions. When a photo is uploaded:
1. The system auto-detects it as a camera image (resolution > 300px) and switches to Photo preprocessing
2. The Photo pipeline (bilateral filter + CLAHE + adaptive threshold) handles camera noise, shadows, and uneven lighting
3. The preprocessed binary image is segmented and each symbol is classified
4. Expression mode recognizes operators and computes the result

This enables a practical workflow: write an expression on paper, take a photo, upload it, and get the computed result.

---

## 9. Critical Analysis

### 9.1 Strengths

1. **Modular architecture**: The abstract base class pattern enables easy model swapping and extension. Adding a new model requires only implementing 6 methods.
2. **Comprehensive evaluation**: Per-class metrics, confusion matrices, and inference timing provide thorough model comparison.
3. **MNIST-style normalization**: Center-of-mass centering significantly bridges the gap between training data (clean MNIST) and real hand-drawn input.
4. **Data augmentation**: Random affine transforms during CNN training improve robustness to natural handwriting variation.

### 9.2 Limitations

1. **Domain gap**: Despite MNIST-style normalization, hand-drawn digits on a canvas still differ from actual handwritten digits on paper. The stroke width, ink distribution, and writing dynamics are different.
2. **SVM/KNN training subset**: Both use only 20,000 of 60,000 available training samples due to computational constraints. SVM in particular could benefit from the full dataset but training time would be prohibitive.
3. **Segmentation brittleness**: All three segmentation methods assume digits are separated by gaps. Touching or overlapping digits cause segmentation failures.
4. **Synthetic operator data**: The generated operator images are simplistic compared to real handwriting, which may reduce operator recognition accuracy on diverse handwriting styles.

### 9.3 Potential Improvements

1. **Data augmentation for SVM/KNN**: Feature extraction (HOG, PCA) could improve classical model accuracy without increasing dimensionality.
2. **Elastic deformation** augmentation (as used in the original MNIST paper) could further improve CNN robustness.
3. **Connected component analysis with stroke merging** could handle touching digits.
4. **Transfer learning** from pre-trained models (e.g., ResNet) could improve accuracy with fewer training epochs.
5. **Real handwriting data collection** for operators would improve expression recognition significantly.

---

## 10. Summary and Conclusion

This project successfully implements a complete Handwritten Number Recognition System that meets all assignment requirements:

- **Task 1 (Preprocessing)**: Four techniques implemented and compared (Basic, Otsu, Adaptive, Photo) - auto-selected based on input source
- **Task 2 (Segmentation)**: Three techniques implemented and compared - Contour-based selected for speed and reliability
- **Task 3 (ML Models)**: Three models trained and evaluated - CNN (PyTorch) achieves 99.45% accuracy, significantly outperforming SVM (96.31%) and KNN (93.14%)
- **Task 4 (Evaluation)**: Comprehensive evaluation with accuracy, precision, recall, F1-score, confusion matrices, and inference timing
- **GUI**: Complete PyQt5 interface with drawing canvas, image upload, model selection, and evaluation display
- **Extension**: Arithmetic expression recognition with 16-class CNN, synthetic operator data, safe expression evaluation, and camera photo upload support

The key insight from this project is that deep learning (CNN) significantly outperforms traditional ML methods for image classification, achieving near-human performance (99.45%) while also being faster at inference. The combination of data augmentation and MNIST-style normalization is critical for bridging the gap between training data and real-world handwritten input.

---

*COS30018 - Intelligent Systems, Swinburne University of Technology*
