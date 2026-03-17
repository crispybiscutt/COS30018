"""
COS30018 - Handwritten Number Recognition System
Global configuration and hyperparameters.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MNIST_DIR = os.path.join(DATA_DIR, "mnist")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# --- Image Settings ---
IMAGE_SIZE = 28          # MNIST standard: 28x28 pixels
IMAGE_CHANNELS = 1       # Grayscale

# --- Training Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1

# --- Model Names ---
MODEL_CNN_KERAS = "cnn_keras"
MODEL_CNN_PYTORCH = "cnn_pytorch"
MODEL_SVM = "svm"
MODEL_KNN = "knn"

AVAILABLE_MODELS = [MODEL_CNN_KERAS, MODEL_CNN_PYTORCH, MODEL_SVM, MODEL_KNN]

# --- Preprocessing Methods ---
PREPROCESS_BASIC = "basic"
PREPROCESS_OTSU = "otsu"
PREPROCESS_ADAPTIVE = "adaptive"

AVAILABLE_PREPROCESS = [PREPROCESS_BASIC, PREPROCESS_OTSU, PREPROCESS_ADAPTIVE]

# --- Segmentation Methods ---
SEGMENT_CONTOUR = "contour"
SEGMENT_CONNECTED = "connected_components"
SEGMENT_PROJECTION = "projection"

AVAILABLE_SEGMENTS = [SEGMENT_CONTOUR, SEGMENT_CONNECTED, SEGMENT_PROJECTION]

# --- KNN Hyperparameters ---
KNN_N_NEIGHBORS = 5

# --- SVM Hyperparameters ---
SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = "scale"

# --- Extension: Arithmetic Operators ---
OPERATORS = ["+", "-", "*", "/", "(", ")"]
