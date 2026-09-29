"""
Configuration Module for Cotton Leaf Disease Detection and Advisory System.

This module centralizes all project configurations, paths, dataset split parameters,
model hyperparameters, and class definitions to ensure reproducibility and clean
modularity across all scripts.
"""

from pathlib import Path

# ==============================================================================
# PROJECT DIRECTORIES (Dynamic Pathlib Resolution)
# ==============================================================================
# Resolves project root dynamically so code works on macOS, Linux, and Raspberry Pi
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Models and results paths
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Model Checkpoints & Artifacts
STAGE1_MODEL_PATH = MODELS_DIR / "mobilenetv2_stage1.keras"
FINAL_MODEL_PATH = MODELS_DIR / "mobilenetv2_cotton_final.keras"
CLASS_NAMES_JSON_PATH = MODELS_DIR / "class_names.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Phase 3 TFLite Model Paths
TFLITE_MODEL_PATH = MODELS_DIR / "cotton_leaf_mobilenetv2.tflite"
TFLITE_FLOAT16_PATH = MODELS_DIR / "cotton_leaf_mobilenetv2_float16.tflite"
TFLITE_INT8_PATH = MODELS_DIR / "cotton_leaf_mobilenetv2_int8.tflite"

# Plot & Metric Output Paths
TRAINING_ACCURACY_PLOT = RESULTS_DIR / "training_accuracy.png"
TRAINING_LOSS_PLOT = RESULTS_DIR / "training_loss.png"
CONFUSION_MATRIX_PLOT = RESULTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_JSON = RESULTS_DIR / "training_history.json"
MODEL_COMPARISON_TXT = RESULTS_DIR / "model_comparison.txt"

# Deployment Directory
DEPLOYMENT_DIR = BASE_DIR / "deployment"

# ==============================================================================
# DATASET REPRODUCIBILITY & SPLIT RATIOS
# ==============================================================================
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Assert that split ratios sum to 1.0 within floating point tolerance
assert abs((TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO) - 1.0) < 1e-6, (
    "Split ratios must sum to 1.0"
)

# ==============================================================================
# IMAGE & MODEL SPECIFICATIONS (MobileNetV2 Target)
# ==============================================================================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
NUM_CLASSES = 5

# Supported image file extensions (case-insensitive)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Ignored non-image and system artifact filenames/patterns
IGNORED_FILES = {
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    ".gitkeep",
}

# ==============================================================================
# TRAINING HYPERPARAMETERS (Two-Stage Transfer Learning)
# ==============================================================================
BATCH_SIZE = 32

# Stage 1: Feature Extraction (Frozen MobileNetV2 Backbone)
STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3

# Stage 2: Fine-Tuning (Unfreeze later layers of MobileNetV2)
STAGE2_EPOCHS = 20
STAGE2_LR = 1e-5
FINE_TUNE_AT_LAYER = 100  # MobileNetV2 has 154 layers; unfreeze top ~54 layers

# Regularization
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
L2_REGULARIZATION = 1e-4

# Early Stopping & LR Reduction
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ==============================================================================
# CLASS NAMES & CANONICAL MAPPINGS
# ==============================================================================
CLASS_NAMES = [
    "Alternaria_Leaf_Spot",
    "Bacterial_Blight",
    "Fusarium_Wilt",
    "Healthy",
    "Verticillium_Wilt",
]

# Mapping various raw folder naming conventions to canonical target class names
RAW_CLASS_MAPPING = {
    "alternaria leaf spot": "Alternaria_Leaf_Spot",
    "alternaria_leaf_spot": "Alternaria_Leaf_Spot",
    "alternaria": "Alternaria_Leaf_Spot",
    "bacterial blight": "Bacterial_Blight",
    "bacterial_blight": "Bacterial_Blight",
    "bacterial": "Bacterial_Blight",
    "fusarium wilt": "Fusarium_Wilt",
    "fusarium_wilt": "Fusarium_Wilt",
    "fusarium": "Fusarium_Wilt",
    "healthy leaf": "Healthy",
    "healthy_leaf": "Healthy",
    "healthy": "Healthy",
    "verticillium wilt": "Verticillium_Wilt",
    "verticillium_wilt": "Verticillium_Wilt",
    "verticillium": "Verticillium_Wilt",
}

# Human-readable labels for reports and UI
HUMAN_READABLE_LABELS = {
    "Alternaria_Leaf_Spot": "Alternaria Leaf Spot",
    "Bacterial_Blight": "Bacterial Blight",
    "Fusarium_Wilt": "Fusarium Wilt",
    "Healthy": "Healthy Leaf",
    "Verticillium_Wilt": "Verticillium Wilt",
}
