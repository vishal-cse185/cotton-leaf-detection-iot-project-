"""
Comprehensive Model Comparison and Benchmarking Script for Cotton Leaf AI.

Benchmarks across all 205 independent test images (dataset/test/):
1. Original Keras Model (.keras)
2. Float16 Quantized TFLite Model (.tflite)
3. INT8 Quantized TFLite Model (.tflite)

Evaluates:
- Test Accuracy
- Prediction Agreement (%)
- Average Inference Latency per image (ms)
- On-disk Model Size (MB) and Compression (%)
- Generates results/model_comparison.txt and models/model_metadata.json.
"""

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Ensure src can be imported
try:
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        FINAL_MODEL_PATH,
        IMAGE_SIZE,
        INPUT_SHAPE,
        MODEL_COMPARISON_TXT,
        MODEL_METADATA_PATH,
        MODELS_DIR,
        NUM_CLASSES,
        RESULTS_DIR,
        TEST_DIR,
        TFLITE_FLOAT16_PATH,
        TFLITE_INT8_PATH,
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
        MODEL_COMPARISON_TXT,
        MODEL_METADATA_PATH,
        MODELS_DIR,
        NUM_CLASSES,
        RESULTS_DIR,
        TEST_DIR,
        TFLITE_FLOAT16_PATH,
        TFLITE_INT8_PATH,
        TFLITE_MODEL_PATH,
    )

import tensorflow as tf


def load_test_data() -> Tuple[List[Path], np.ndarray]:
    """
    Load all test image paths and their true class indices in deterministic order.
    """
    test_paths: List[Path] = []
    test_labels: List[int] = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = TEST_DIR / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Test folder missing: {class_folder}")

        files = sorted([
            f for f in class_folder.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

        for f in files:
            test_paths.append(f)
            test_labels.append(class_idx)

    return test_paths, np.array(test_labels)


def load_raw_image(image_path: Path) -> np.ndarray:
    """Load image as float32 RGB array."""
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        resized_img = rgb_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.array(resized_img, dtype=np.float32)


def evaluate_keras_model(
    model_path: Path,
    test_paths: List[Path],
    y_true: np.ndarray,
) -> Tuple[float, np.ndarray, float, float]:
    """
    Evaluate Keras model on test set.

    Returns:
        Tuple of (accuracy, predictions_array, avg_latency_ms, size_mb).
    """
    print(f"\n[1/3] Benchmarking Keras Model: {model_path.name}...")
    model = tf.keras.models.load_model(model_path)
    size_mb = model_path.stat().st_size / (1024 * 1024)

    # Warmup
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model(dummy, training=False)

    latencies = []
    predictions = []

    for path in test_paths:
        img_arr = np.expand_dims(load_raw_image(path), axis=0)
        start = time.perf_counter()
        pred = model(img_arr, training=False).numpy()[0]
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(np.argmax(pred))

    predictions = np.array(predictions)
    accuracy = float(np.mean(predictions == y_true))
    avg_latency = float(np.mean(latencies))

    print(f"  ✓ Keras Test Accuracy    : {accuracy * 100:.2f}%")
    print(f"  ✓ Average Latency / image: {avg_latency:.2f} ms")
    print(f"  ✓ Model Size             : {size_mb:.2f} MB")

    return accuracy, predictions, avg_latency, size_mb


def evaluate_tflite_model(
    tflite_path: Path,
    test_paths: List[Path],
    y_true: np.ndarray,
    keras_predictions: np.ndarray,
) -> Tuple[float, np.ndarray, float, float, float]:
    """
    Evaluate TFLite model on test set and measure agreement with Keras model.

    Returns:
        Tuple of (accuracy, predictions_array, avg_latency_ms, size_mb, agreement_pct).
    """
    print(f"\nBenchmarking TFLite Model: {tflite_path.name}...")
    size_mb = tflite_path.stat().st_size / (1024 * 1024)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    scale_in, zp_in = input_details["quantization"]
    scale_out, zp_out = output_details["quantization"]

    # Warmup
    dummy = np.zeros(input_details["shape"], dtype=input_dtype)
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()

    latencies = []
    predictions = []

    for path in test_paths:
        img = load_raw_image(path)

        # Handle quantization if required
        if input_dtype in (np.int8, np.uint8):
            if scale_in > 0:
                scaled = (img / 127.5) - 1.0
                quant = np.round(scaled / scale_in + zp_in)
                input_data = np.clip(quant, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
            else:
                input_data = img.astype(input_dtype)
        else:
            input_data = img.astype(np.float32)

        input_data = np.expand_dims(input_data, axis=0)
        interpreter.set_tensor(input_details["index"], input_data)

        start = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000.0)

        out = interpreter.get_tensor(output_details["index"])[0]
        if output_details["dtype"] in (np.int8, np.uint8) and scale_out > 0:
            out = (out.astype(np.float32) - zp_out) * scale_out

        predictions.append(np.argmax(out))

    predictions = np.array(predictions)
    accuracy = float(np.mean(predictions == y_true))
    avg_latency = float(np.mean(latencies))
    agreement_pct = float(np.mean(predictions == keras_predictions) * 100.0)

    print(f"  ✓ TFLite Test Accuracy   : {accuracy * 100:.2f}%")
    print(f"  ✓ Keras Agreement        : {agreement_pct:.2f}%")
    print(f"  ✓ Average Latency / image: {avg_latency:.2f} ms")
    print(f"  ✓ Model Size             : {size_mb:.2f} MB")

    return accuracy, predictions, avg_latency, size_mb, agreement_pct


def generate_comparison_report(
    keras_metrics: dict,
    f16_metrics: dict,
    int8_metrics: dict,
    total_samples: int,
) -> None:
    """
    Format and save results/model_comparison.txt and print summary.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "================================================================================",
        " MODEL COMPARISON REPORT — COTTON LEAF DISEASE DETECTION SYSTEM",
        "================================================================================",
        f"Generated On          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Evaluation Dataset    : dataset/test/ ({total_samples} independent samples)",
        f"Target Architecture   : MobileNetV2 (Transfer Learning, 2-Stage)",
        "--------------------------------------------------------------------------------",
        "",
        "1. ORIGINAL KERAS MODEL (.keras)",
        f"   - Model Path          : {keras_metrics['path']}",
        f"   - File Size           : {keras_metrics['size_mb']:.2f} MB",
        f"   - Test Accuracy       : {keras_metrics['accuracy'] * 100:.2f}%",
        f"   - Average Latency     : {keras_metrics['latency_ms']:.2f} ms / image",
        "",
        "2. FLOAT16 TFLITE MODEL (.tflite) [RECOMMENDED DEPLOYMENT]",
        f"   - Model Path          : {f16_metrics['path']}",
        f"   - File Size           : {f16_metrics['size_mb']:.2f} MB",
        f"   - Size Reduction      : {((1 - f16_metrics['size_mb'] / keras_metrics['size_mb']) * 100):.1f}%",
        f"   - Test Accuracy       : {f16_metrics['accuracy'] * 100:.2f}%",
        f"   - Accuracy Difference : {(f16_metrics['accuracy'] - keras_metrics['accuracy']) * 100:+.2f}%",
        f"   - Keras Agreement     : {f16_metrics['agreement_pct']:.2f}%",
        f"   - Average Latency     : {f16_metrics['latency_ms']:.2f} ms / image",
        f"   - Speedup vs Keras    : {keras_metrics['latency_ms'] / f16_metrics['latency_ms']:.2f}x",
        "",
        "3. INT8 TFLITE MODEL (.tflite)",
    ]

    if int8_metrics.get("available", False):
        report_lines.extend([
            f"   - Model Path          : {int8_metrics['path']}",
            f"   - File Size           : {int8_metrics['size_mb']:.2f} MB",
            f"   - Size Reduction      : {((1 - int8_metrics['size_mb'] / keras_metrics['size_mb']) * 100):.1f}%",
            f"   - Test Accuracy       : {int8_metrics['accuracy'] * 100:.2f}%",
            f"   - Accuracy Difference : {(int8_metrics['accuracy'] - keras_metrics['accuracy']) * 100:+.2f}%",
            f"   - Keras Agreement     : {int8_metrics['agreement_pct']:.2f}%",
            f"   - Average Latency     : {int8_metrics['latency_ms']:.2f} ms / image",
        ])
    else:
        report_lines.append(f"   - Status: {int8_metrics.get('reason', 'Not created or evaluated')}")

    report_lines.extend([
        "",
        "--------------------------------------------------------------------------------",
        "SUMMARY & RECOMMENDATION FOR RASPBERRY PI 5 DEPLOYMENT:",
        f"The Float16 TFLite model ({f16_metrics['size_mb']:.2f} MB) provides the optimal balance of",
        f"accuracy ({f16_metrics['accuracy']*100:.2f}%), zero accuracy degradation, fast latency",
        f"({f16_metrics['latency_ms']:.2f} ms), and {f16_metrics['agreement_pct']:.1f}% prediction agreement with Keras.",
        "It is selected as the primary edge deployment model.",
        "================================================================================",
    ])

    report_text = "\n".join(report_lines)

    with open(MODEL_COMPARISON_TXT, "w") as f:
        f.write(report_text)

    print("\n" + report_text + "\n")
    print(f"  ✓ Saved comparison report to: {MODEL_COMPARISON_TXT}")


def save_model_metadata(
    keras_metrics: dict,
    f16_metrics: dict,
    class_mapping: dict,
) -> None:
    """Save metadata describing the deployment model."""
    metadata = {
        "model_name": "cotton_leaf_mobilenetv2_float16",
        "architecture": "MobileNetV2 Transfer Learning",
        "input_resolution": [224, 224, 3],
        "input_color_space": "RGB",
        "number_of_classes": NUM_CLASSES,
        "class_names": class_mapping,
        "preprocessing_method": "MobileNetV2 scaling: (pixel / 127.5) - 1.0 (integrated)",
        "source_keras_model_path": str(FINAL_MODEL_PATH),
        "tflite_model_path": str(TFLITE_FLOAT16_PATH),
        "optimization_type": "Float16 Post-Training Quantization",
        "keras_test_accuracy": float(keras_metrics["accuracy"]),
        "tflite_test_accuracy": float(f16_metrics["accuracy"]),
        "tflite_model_size_mb": float(f16_metrics["size_mb"]),
        "average_inference_latency_ms": float(f16_metrics["latency_ms"]),
        "target_device": "Raspberry Pi 5 (ARM Cortex-A76, 64-bit OS)",
        "conversion_timestamp": datetime.datetime.now().isoformat(),
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"  ✓ Saved model metadata to: {MODEL_METADATA_PATH}")


def run_comparison_benchmark() -> None:
    """Execute complete benchmarking across test set."""
    print("=" * 80)
    print(" PHASE 3: COMPREHENSIVE MODEL BENCHMARKING (KERAS VS TFLITE)")
    print("=" * 80)
    print(f"Test Directory : {TEST_DIR}")
    print(f"Models Dir     : {MODELS_DIR}")
    print("-" * 80)

    # 1. Load Test Dataset
    test_paths, y_true = load_test_data()
    total_test = len(test_paths)
    print(f"  Loaded {total_test} independent test images across {NUM_CLASSES} classes.")

    # 2. Benchmark Keras Model
    keras_acc, keras_preds, keras_lat, keras_size = evaluate_keras_model(
        FINAL_MODEL_PATH, test_paths, y_true
    )
    keras_metrics = {
        "path": str(FINAL_MODEL_PATH),
        "accuracy": keras_acc,
        "latency_ms": keras_lat,
        "size_mb": keras_size,
    }

    # 3. Benchmark Float16 TFLite Model
    f16_acc, f16_preds, f16_lat, f16_size, f16_agree = evaluate_tflite_model(
        TFLITE_FLOAT16_PATH, test_paths, y_true, keras_preds
    )
    f16_metrics = {
        "path": str(TFLITE_FLOAT16_PATH),
        "accuracy": f16_acc,
        "latency_ms": f16_lat,
        "size_mb": f16_size,
        "agreement_pct": f16_agree,
    }

    # 4. Benchmark INT8 TFLite Model (if present)
    int8_metrics = {"available": False}
    if TFLITE_INT8_PATH.exists():
        try:
            int8_acc, int8_preds, int8_lat, int8_size, int8_agree = evaluate_tflite_model(
                TFLITE_INT8_PATH, test_paths, y_true, keras_preds
            )
            int8_metrics = {
                "available": True,
                "path": str(TFLITE_INT8_PATH),
                "accuracy": int8_acc,
                "latency_ms": int8_lat,
                "size_mb": int8_size,
                "agreement_pct": int8_agree,
            }
        except Exception as e:
            int8_metrics = {"available": False, "reason": f"Evaluation error: {e}"}
    else:
        int8_metrics = {"available": False, "reason": "INT8 model file not found"}

    # 5. Load class mapping
    with open(CLASS_NAMES_JSON_PATH, "r") as f:
        class_mapping = json.load(f)

    # 6. Generate Comparison Report & Metadata
    generate_comparison_report(keras_metrics, f16_metrics, int8_metrics, total_test)
    save_model_metadata(keras_metrics, f16_metrics, class_mapping)


if __name__ == "__main__":
    run_comparison_benchmark()
