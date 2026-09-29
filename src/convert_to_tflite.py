"""
TensorFlow Lite Model Conversion Script for Cotton Leaf Disease Detection.

This script:
1. Verifies the trained MobileNetV2 Keras model, input shape, output classes, and class mapping.
2. Converts the model to standard full-precision TensorFlow Lite format (.tflite).
3. Converts the model with Float16 post-training quantization for Raspberry Pi 5.
4. Validates TFLite interpreters, tensor dimensions, and data types.
5. Reports model sizes, compression ratios, and parameter diagnostics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Ensure src can be imported
try:
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        FINAL_MODEL_PATH,
        IMAGE_SIZE,
        INPUT_SHAPE,
        MODELS_DIR,
        NUM_CLASSES,
        TFLITE_FLOAT16_PATH,
        TFLITE_MODEL_PATH,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        FINAL_MODEL_PATH,
        IMAGE_SIZE,
        INPUT_SHAPE,
        MODELS_DIR,
        NUM_CLASSES,
        TFLITE_FLOAT16_PATH,
        TFLITE_MODEL_PATH,
    )

import tensorflow as tf


def verify_source_model() -> Tuple[tf.keras.Model, Dict[str, str]]:
    """
    Verify the existing Phase 2 trained model and class names JSON.

    Returns:
        Tuple of (loaded_keras_model, class_mapping_dict).
    """
    print("=" * 70)
    print(" [1/4] VERIFYING EXISTING MODEL & CLASS MAPPINGS")
    print("=" * 70)

    # 1. Locate trained model
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model file not found at: {FINAL_MODEL_PATH}\n"
            "Please train the model first via: python3 src/train_model.py"
        )
    print(f"  ✓ Located Keras Model: {FINAL_MODEL_PATH}")

    # 2. Verify model loads
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    print(f"  ✓ Model Loaded Successfully: {model.name}")

    # 3. Verify input shape
    input_shape = model.input_shape
    print(f"  ✓ Model Input Shape   : {input_shape} (Expected: (None, 224, 224, 3))")
    assert input_shape[1:] == (224, 224, 3), f"Unexpected input shape: {input_shape}"

    # 4. Verify output shape
    output_shape = model.output_shape
    print(f"  ✓ Model Output Shape  : {output_shape} (Expected: (None, 5))")
    assert output_shape[-1] == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, found {output_shape[-1]}"
    )

    # 5. Load and confirm class mapping
    if not CLASS_NAMES_JSON_PATH.exists():
        raise FileNotFoundError(f"Class mapping file not found at: {CLASS_NAMES_JSON_PATH}")

    with open(CLASS_NAMES_JSON_PATH, "r") as f:
        class_mapping = json.load(f)

    print("\n  Confirmed Class Order Mapping:")
    for idx_str, class_label in sorted(class_mapping.items(), key=lambda x: int(x[0])):
        print(f"    Index {idx_str} -> {class_label}")

    assert len(class_mapping) == NUM_CLASSES, (
        f"Class mapping contains {len(class_mapping)} classes, expected {NUM_CLASSES}"
    )

    return model, class_mapping


def inspect_tflite_model(tflite_path: Path) -> Tuple[list, list]:
    """
    Inspect a TFLite file using Interpreter to extract input/output tensor metadata.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return input_details, output_details


def convert_to_standard_tflite(model: tf.keras.Model) -> Path:
    """
    Convert Keras model to full-precision 32-bit float TensorFlow Lite format.
    """
    print("\n" + "=" * 70)
    print(" [2/4] CONVERTING TO STANDARD TFLITE (FULL PRECISION)")
    print("=" * 70)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    TFLITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_bytes)

    size_mb = len(tflite_bytes) / (1024 * 1024)
    print(f"  ✓ Saved Standard TFLite Model: {TFLITE_MODEL_PATH}")
    print(f"  ✓ Model Size                 : {size_mb:.2f} MB ({len(tflite_bytes):,} bytes)")

    # Validate interpreter
    input_details, output_details = inspect_tflite_model(TFLITE_MODEL_PATH)
    print(f"  ✓ Input Tensor Shape         : {input_details[0]['shape'].tolist()} ({input_details[0]['dtype'].__name__})")
    print(f"  ✓ Output Tensor Shape        : {output_details[0]['shape'].tolist()} ({output_details[0]['dtype'].__name__})")

    return TFLITE_MODEL_PATH


def convert_to_float16_tflite(model: tf.keras.Model) -> Path:
    """
    Convert Keras model with Float16 post-training quantization.
    Optimized for ARM Cortex CPU and NEON execution on Raspberry Pi 5.
    """
    print("\n" + "=" * 70)
    print(" [3/4] CONVERTING TO FLOAT16 QUANTIZED TFLITE (RECOMMENDED EDGE)")
    print("=" * 70)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_f16_bytes = converter.convert()

    TFLITE_FLOAT16_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_FLOAT16_PATH, "wb") as f:
        f.write(tflite_f16_bytes)

    size_mb = len(tflite_f16_bytes) / (1024 * 1024)
    print(f"  ✓ Saved Float16 TFLite Model : {TFLITE_FLOAT16_PATH}")
    print(f"  ✓ Model Size                 : {size_mb:.2f} MB ({len(tflite_f16_bytes):,} bytes)")

    # Validate interpreter
    input_details, output_details = inspect_tflite_model(TFLITE_FLOAT16_PATH)
    print(f"  ✓ Input Tensor Shape         : {input_details[0]['shape'].tolist()} ({input_details[0]['dtype'].__name__})")
    print(f"  ✓ Output Tensor Shape        : {output_details[0]['shape'].tolist()} ({output_details[0]['dtype'].__name__})")

    return TFLITE_FLOAT16_PATH


def run_conversion_pipeline() -> None:
    """Main execution pipeline for model verification and TFLite conversions."""
    model, class_mapping = verify_source_model()

    std_tflite = convert_to_standard_tflite(model)
    f16_tflite = convert_to_float16_tflite(model)

    # Calculate model size compression ratios
    keras_size_mb = FINAL_MODEL_PATH.stat().st_size / (1024 * 1024)
    std_size_mb = std_tflite.stat().st_size / (1024 * 1024)
    f16_size_mb = f16_tflite.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" [4/4] MODEL COMPRESSION & SIZE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Model Format':<26} | {'Size (MB)':>10} | {'Reduction':>12} | {'Deployment Fit':>18}")
    print("  " + "-" * 72)
    print(f"  {'Original Keras (.keras)':<26} | {keras_size_mb:>10.2f} | {'Baseline':>12} | {'Training / Server':>18}")
    print(f"  {'Standard TFLite (.tflite)':<26} | {std_size_mb:>10.2f} | {((1 - std_size_mb/keras_size_mb)*100):>11.1f}% | {'Edge Baseline':>18}")
    print(f"  {'Float16 TFLite (.tflite)':<26} | {f16_size_mb:>10.2f} | {((1 - f16_size_mb/keras_size_mb)*100):>11.1f}% | {'Raspberry Pi 5 (Best)':>18}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_conversion_pipeline()
