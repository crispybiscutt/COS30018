# HNRS - Handwritten Number Recognition System

**COS30018 - Intelligent Systems | Swinburne University of Technology**

**Student:** Vu Minh - 104852111

---

## Overview

A complete system for recognizing handwritten digits and multi-digit numbers from images using machine learning. The system supports three ML models (CNN, SVM, KNN), four preprocessing techniques, three segmentation methods, and includes an extension for arithmetic expression recognition.

---

## Features

### Core Features
- **Handwritten digit recognition** (0-9) trained on MNIST dataset (60,000 images)
- **Multi-digit number recognition** via image segmentation
- **3 ML models** with comprehensive performance comparison:
  - CNN (PyTorch) - 99.45% accuracy
  - SVM (scikit-learn) - 96.31% accuracy
  - KNN (scikit-learn) - 93.14% accuracy
- **4 preprocessing techniques**: Basic, Otsu Binarization, Adaptive Threshold, Camera Photo
- **3 segmentation techniques**: Contour-based, Connected Components, Vertical Projection
- **Interactive GUI** with drawing canvas, image upload, and evaluation dashboard

### Extension: Arithmetic Expression Recognition
- Recognizes handwritten expressions containing digits (0-9) and operators (+, -, *, /, parentheses)
- 16-class CNN trained on MNIST digits + synthetically generated operator images
- Automatically computes and displays the expression result
- Supports camera photo upload with auto-preprocessing

### Data & Training
- **Dataset**: MNIST (60,000 training / 10,000 test images)
- **Data augmentation** (CNN): Random rotation (+-10 deg), translation (+-10%), scaling (90-110%)
- **MNIST-style normalization**: Digits fitted to 20x20 box, centered by center of mass in 28x28 frame
- **Training pipeline**: Automated via `train_and_evaluate.py` or GUI re-training

---

## Project Structure

```
COS30018/
├── main.py                          # Application entry point
├── config.py                        # Global configuration & hyperparameters
├── train_and_evaluate.py            # Batch training & evaluation script
├── requirements.txt                 # Python dependencies
├── report.md                        # Project report
│
├── preprocessing/                   # Task 1: Image Preprocessing
│   ├── __init__.py
│   └── preprocessor.py              # 4 techniques: Basic, Otsu, Adaptive, Photo
│
├── segmentation/                    # Task 2: Image Segmentation
│   ├── __init__.py
│   └── segmenter.py                 # 3 techniques: Contour, Connected Components, Projection
│
├── models/                          # Task 3: ML Models
│   ├── __init__.py
│   ├── base_model.py                # Abstract base class (BaseModel)
│   ├── cnn_pytorch.py               # CNN with data augmentation (PyTorch)
│   ├── cnn_keras.py                 # CNN (TensorFlow/Keras)
│   ├── svm_model.py                 # Support Vector Machine (scikit-learn)
│   ├── knn_model.py                 # K-Nearest Neighbors (scikit-learn)
│   ├── model_manager.py             # Model lifecycle: train, save, load, predict
│   └── saved_models/                # Trained model weights
│       ├── cnn_pytorch.pth
│       ├── expression_cnn.pth
│       ├── svm.pkl
│       └── knn.pkl
│
├── evaluation/                      # Task 4: Evaluation & Testing
│   ├── __init__.py
│   └── evaluator.py                 # Metrics, confusion matrix, model comparison
│
├── gui/                             # GUI Application (PyQt5)
│   ├── __init__.py
│   ├── main_window.py               # Main window with 3 tabs
│   ├── recognition_tab.py           # Draw/upload digits, get predictions
│   ├── evaluation_tab.py            # Charts, confusion matrix, per-class metrics
│   ├── training_tab.py              # Model info, re-training interface
│   ├── drawing_canvas.py            # Interactive drawing widget
│   └── theme.py                     # UI styling and color scheme
│
├── extension/                       # Extension: Arithmetic Expression Recognition
│   ├── __init__.py
│   ├── operator_recognizer.py       # 16-class CNN for digits + operators
│   └── expression_evaluator.py      # Expression parsing & safe evaluation
│
├── utils/                           # Utilities
│   ├── __init__.py
│   └── image_utils.py               # Image I/O, folder-to-number, canvas conversion
│
└── data/
    ├── mnist/                        # MNIST dataset (auto-downloaded)
    ├── test_images/                  # Sample test images
    └── evaluation_results/           # Evaluation results & charts
        ├── results.json
        └── charts/
```

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- PyQt5

### Setup

```bash
# Clone the repository
git clone https://github.com/minhvu278/COS30018-HNRS.git
cd COS30018-HNRS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Launch GUI
```bash
python main.py
```

### Train All Models (Command Line)
```bash
python train_and_evaluate.py
```

### GUI Tabs

**Tab 1 - Recognition:**
- Draw digits/numbers on the canvas and click "Recognize"
- Upload a photo of handwritten numbers/expressions
- Select model, preprocessing, and segmentation methods
- Toggle Expression Mode for arithmetic expression recognition

**Tab 2 - Evaluation:**
- View accuracy comparison across all models
- Navigate confusion matrices for each model
- Inspect per-class precision, recall, and F1-score

**Tab 3 - Models:**
- View pre-trained model details and architecture
- Re-train models with custom epochs and batch size

---

## Model Performance

| Model | Accuracy | Inference (10K) | Training Time |
|-------|----------|-----------------|---------------|
| **CNN (PyTorch)** | **99.45%** | 3.74s | 413.2s |
| SVM (scikit-learn) | 96.31% | 51.89s | 343.7s |
| KNN (scikit-learn) | 93.14% | 3.26s | 7.7s |

### CNN Architecture
```
Input(1x28x28) -> Conv2d(32,3x3) -> ReLU -> MaxPool(2x2)
              -> Conv2d(64,3x3) -> ReLU -> MaxPool(2x2)
              -> Conv2d(64,3x3) -> ReLU
              -> Flatten -> Dense(128) -> ReLU -> Dropout(0.5) -> Dense(10)
```

---

## Technical Details

### Preprocessing Pipeline
1. **Basic**: Grayscale -> Resize -> Normalize
2. **Otsu**: Grayscale -> Otsu threshold -> Resize -> Normalize
3. **Adaptive**: Gaussian blur -> Adaptive threshold -> Morphological closing -> Resize
4. **Photo**: Bilateral filter -> CLAHE -> Adaptive threshold -> Morphological cleanup

### Segmentation Methods
1. **Contour-based**: OpenCV findContours, bounding box filtering, left-to-right sorting
2. **Connected Components**: Pixel connectivity analysis with overlapping box merging
3. **Vertical Projection**: Column-wise pixel density analysis

### Expression Recognition
- 16-class CNN: digits (0-9) + operators (+, -, *, /, parentheses)
- Hybrid classification: ExpressionCNN for operator detection, dedicated digit model for digit classification
- Safe evaluation with restricted Python eval (whitelist-based)

---

## Dependencies

```
torch>=2.0
torchvision
numpy
opencv-python
Pillow
scikit-learn
matplotlib
seaborn
PyQt5
joblib
```

---

## License

This project is developed for educational purposes as part of COS30018 - Intelligent Systems at Swinburne University of Technology.
