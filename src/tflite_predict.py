"""
TensorFlow Lite Single-Image Inference Utility for Cotton Leaf Disease Detection.

Supports both full TensorFlow (tf.lite.Interpreter) and lightweight
edge runtime (tflite_runtime.interpreter.Interpreter) on Raspberry Pi 5.

Usage:
    python3 src/tflite_predict.py --model models/cotton_leaf_mobilenetv2_float16.tflite --image path/to/image.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Import TFLite Interpreter (gracefully supporting tflite_runtime on Raspberry Pi)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

# Ensure src can be imported
try:
    from src.config import (
        CLASS_NAMES_JSON_PATH,
        IMAGE_SIZE,
        TFLITE_FLOAT16_PATH,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        CLASS_NAMES_JSON_PATH,
        IMAGE_SIZE,
        TFLITE_FLOAT16_PATH,
    )

DEFAULT_MODEL_PATH = TFLITE_FLOAT16_PATH


def load_class_names(json_path: Path) -> Dict[int, str]:
    """Load class mapping dictionary from JSON."""
    if not json_path.exists():
        # Look in same directory as model or fallback
        local_candidate = json_path.parent / "class_names.json"
        if local_candidate.exists():
            json_path = local_candidate
        else:
            raise FileNotFoundError(f"Class names file not found at: {json_path}")

    with open(json_path, "r") as f:
        mapping = json.load(f)
        return {int(k): v for k, v in mapping.items()}


def preprocess_image_for_tflite(
    image_path: Path,
    input_details: dict,
) -> np.ndarray:
    """
    Load, resize, format and scale image according to TFLite input tensor requirements.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        resized_img = rgb_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(resized_img, dtype=np.float32)

    input_dtype = input_details["dtype"]
    input_shape = input_details["shape"]

    # If the model requires quantized INT8/UINT8 input
    if input_dtype in (np.int8, np.uint8):
        scale, zero_point = input_details["quantization"]
        if scale > 0:
            # Model has internal preprocess_input or requires [-1, 1] scaled
            img_array = (img_array / 127.5) - 1.0
            quantized = np.round(img_array / scale + zero_point)
            img_array = np.clip(quantized, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        else:
            img_array = img_array.astype(input_dtype)
    else:
        # Float32 / Float16 model: model contains internal preprocessing layer
        img_array = img_array.astype(np.float32)

    # Ensure shape matches (1, 224, 224, 3)
    if len(input_shape) == 4 and img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def run_tflite_inference(
    model_path_str: str,
    image_path_str: str,
    top_k: int = 3,
) -> Tuple[str, float, List[Tuple[str, float]], float]:
    """
    Execute TFLite inference and return predicted class, confidence, top-k predictions, and latency.
    """
    model_path = Path(model_path_str)
    image_path = Path(image_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model file not found: {model_path}")

    # 1. Load Class Names
    class_names = load_class_names(CLASS_NAMES_JSON_PATH)

    # 2. Initialize TFLite Interpreter
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 3. Preprocess Input
    input_tensor = preprocess_image_for_tflite(image_path, input_details)
    interpreter.set_tensor(input_details["index"], input_tensor)

    # 4. Measure Inference Latency
    start_time = time.perf_counter()
    interpreter.invoke()
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    # 5. Extract Output and Dequantize if necessary
    output_data = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_details["quantization"]
        if scale > 0:
            output_data = (output_data.astype(np.float32) - zero_point) * scale

    # Ensure probabilities sum to 1
    exp_scores = np.exp(output_data - np.max(output_data))
    probabilities = exp_scores / np.sum(exp_scores) if np.max(output_data) > 1.0 else output_data

    # Normalize if slightly off
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum

    # 6. Rank Predictions
    top_indices = np.argsort(probabilities)[::-1]
    top_class_idx = top_indices[0]
    predicted_class = class_names.get(top_class_idx, f"Class {top_class_idx}")
    confidence_pct = float(probabilities[top_class_idx]) * 100.0

    ranked_predictions = []
    for idx in top_indices[:top_k]:
        name = class_names.get(idx, f"Class {idx}")
        pct = float(probabilities[idx]) * 100.0
        ranked_predictions.append((name, pct))

    # 7. Display Output
    print("\n" + "=" * 40)
    print("COTTON LEAF DISEASE PREDICTION")
    print("=" * 40)
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence_pct:.2f}%\n")
    print("Top Predictions:")
    for rank, (name, pct) in enumerate(ranked_predictions, 1):
        print(f"{rank}. {name:<22} {pct:>6.2f}%")
    print(f"\nInference Time : {inference_time_ms:.2f} ms")
    print("=" * 40 + "\n")

    return predicted_class, confidence_pct, ranked_predictions, inference_time_ms


def predict_leaf(image_path: Path, model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Structured prediction helper returning a clean dictionary for Web APIs and CLI.
    """
    target_model = model_path or TFLITE_FLOAT16_PATH
    pred_label, conf, top_preds, latency = run_tflite_inference(
        model_path_str=str(target_model),
        image_path_str=str(image_path),
        top_k=5,
    )
    return {
        "predicted_label": pred_label,
        "confidence": round(conf, 2),
        "top_predictions": [{"label": name, "confidence": round(pct, 2)} for name, pct in top_preds],
        "inference_time_ms": round(latency, 2),
        "model_used": str(target_model.name),
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a cotton leaf image using optimized TFLite model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(TFLITE_FLOAT16_PATH),
        help="Path to the TFLite model (.tflite). Default: models/cotton_leaf_mobilenetv2_float16.tflite",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the test cotton leaf image.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3).",
    )
    args = parser.parse_args()

    run_tflite_inference(args.model, args.image, top_k=args.top_k)
