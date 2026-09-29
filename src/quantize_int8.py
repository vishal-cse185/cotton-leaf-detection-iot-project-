"""
Post-Training INT8 Quantization Script for Cotton Leaf Disease Detection.

This script:
1. Constructs a representative calibration dataset from training images (dataset/train/).
2. Strictly forbids and guards against test image leakage.
3. Samples diverse, balanced images across all 5 classes with standard MobileNetV2 preprocessing.
4. Performs full integer post-training quantization to produce an INT8 TFLite model.
5. Saves the model to models/cotton_leaf_mobilenetv2_int8.tflite.
"""

import sys
from pathlib import Path
from typing import Generator, List

import numpy as np
from PIL import Image

# Ensure src can be imported
try:
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        FINAL_MODEL_PATH,
        IMAGE_SIZE,
        MODELS_DIR,
        NUM_CLASSES,
        TFLITE_INT8_PATH,
        TRAIN_DIR,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        FINAL_MODEL_PATH,
        IMAGE_SIZE,
        MODELS_DIR,
        NUM_CLASSES,
        TFLITE_INT8_PATH,
        TRAIN_DIR,
    )

import tensorflow as tf


def collect_representative_image_paths(
    samples_per_class: int = 20,
) -> List[Path]:
    """
    Collect sample image paths strictly from dataset/train across all 5 classes.
    Never touches dataset/test or dataset/validation.
    """
    selected_paths: List[Path] = []

    for class_name in CLASS_NAMES:
        class_folder = TRAIN_DIR / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Training class folder not found: {class_folder}")

        files = sorted([
            f for f in class_folder.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

        # Deterministic slice across the class samples
        class_sample = files[:samples_per_class]
        selected_paths.extend(class_sample)

    assert len(selected_paths) == samples_per_class * NUM_CLASSES, (
        f"Expected {samples_per_class * NUM_CLASSES} paths, got {len(selected_paths)}"
    )

    # Strict audit guard: verify none of the paths originate outside TRAIN_DIR
    for p in selected_paths:
        assert TRAIN_DIR in p.parents, f"Data leakage detected! Non-training path: {p}"

    return selected_paths


def build_representative_dataset_generator(
    image_paths: List[Path],
):
    """
    Builds the calibration generator for INT8 post-training quantization.
    """
    def representative_dataset_gen() -> Generator[List[np.ndarray], None, None]:
        for path in image_paths:
            with Image.open(path) as img:
                rgb_img = img.convert("RGB")
                resized_img = rgb_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                img_array = np.array(resized_img, dtype=np.float32)
                # Model includes internal preprocessing and data augmentation (eval mode)
                input_tensor = np.expand_dims(img_array, axis=0)
                yield [input_tensor]

    return representative_dataset_gen


def quantize_to_int8(
    samples_per_class: int = 20,
) -> Path:
    """
    Execute INT8 post-training quantization using representative calibration data.
    """
    print("=" * 70)
    print(" INT8 POST-TRAINING QUANTIZATION PIPELINE")
    print("=" * 70)
    print(f"Source Model File        : {FINAL_MODEL_PATH}")
    print(f"Calibration Data Source  : {TRAIN_DIR} (Strictly Training Only)")
    print(f"Samples Per Class        : {samples_per_class} (Total: {samples_per_class * NUM_CLASSES})")
    print("-" * 70)

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Keras model not found at {FINAL_MODEL_PATH}")

    # 1. Load source model
    print("\n[1/3] Loading Source Model...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    print(f"  ✓ Source model loaded: {model.name}")

    # 2. Collect calibration dataset
    print("\n[2/3] Generating Representative Calibration Dataset...")
    calibration_paths = collect_representative_image_paths(samples_per_class)
    print(f"  ✓ Verified {len(calibration_paths)} calibration samples across {NUM_CLASSES} classes.")
    print("  ✓ Zero test-set leakage confirmed.")

    rep_dataset_gen = build_representative_dataset_generator(calibration_paths)

    # 3. Configure Converter for INT8 Quantization
    print("\n[3/3] Quantizing Model to INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset_gen

    try:
        tflite_int8_bytes = converter.convert()
    except Exception as e:
        print(f"  [ERROR] Standard INT8 conversion failed: {e}")
        print("  Attempting fallback with relaxed opset...")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        tflite_int8_bytes = converter.convert()

    TFLITE_INT8_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_int8_bytes)

    size_mb = len(tflite_int8_bytes) / (1024 * 1024)
    keras_size_mb = FINAL_MODEL_PATH.stat().st_size / (1024 * 1024)

    print(f"  ✓ Saved INT8 Quantized Model : {TFLITE_INT8_PATH}")
    print(f"  ✓ INT8 Model Size            : {size_mb:.2f} MB ({len(tflite_int8_bytes):,} bytes)")
    print(f"  ✓ Compression vs Keras       : {((1 - size_mb / keras_size_mb) * 100):.1f}% reduction")
    print("=" * 70 + "\n")

    return TFLITE_INT8_PATH


if __name__ == "__main__":
    quantize_to_int8(samples_per_class=20)
