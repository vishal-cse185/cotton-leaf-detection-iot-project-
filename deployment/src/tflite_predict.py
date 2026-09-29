"""
Self-Contained TFLite Inference Script for Raspberry Pi 5 Deployment.

Designed to run independently inside the deployment/ folder without external project dependencies.

Usage:
    python3 src/tflite_predict.py --image path/to/leaf.jpg
    python3 src/tflite_predict.py --model models/cotton_leaf_mobilenetv2_float16.tflite --image path/to/leaf.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Import TFLite Interpreter (supports LiteRT, tflite-runtime, and tensorflow)
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            try:
                import tensorflow as tf
                tflite = tf.lite
            except ImportError:
                tflite = None

# Dynamic local paths within deployment package
DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = DEPLOYMENT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "cotton_leaf_mobilenetv2_float16.tflite"
DEFAULT_CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)


def load_class_names(json_path: Path) -> Dict[int, str]:
    """Load class mapping dictionary from JSON."""
    candidate_paths = [
        json_path,
        DEFAULT_CLASS_NAMES_PATH,
        MODELS_DIR / "class_names.json",
        DEPLOYMENT_ROOT / "models" / "class_names.json",
    ]

    for p in candidate_paths:
        if p.exists():
            with open(p, "r") as f:
                mapping = json.load(f)
                return {int(k): v for k, v in mapping.items()}

    # Fallback to standard canonical class names
    return {
        0: "Alternaria Leaf Spot",
        1: "Bacterial Blight",
        2: "Fusarium Wilt",
        3: "Healthy Leaf",
        4: "Verticillium Wilt",
    }


def preprocess_image_for_tflite(
    image_path: Path,
    input_details: dict,
) -> np.ndarray:
    """Load, format, and resize image for TFLite edge input."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        resized_img = rgb_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        img_array = np.array(resized_img, dtype=np.float32)

    input_dtype = input_details["dtype"]
    input_shape = input_details["shape"]

    if input_dtype in (np.int8, np.uint8):
        scale, zero_point = input_details["quantization"]
        if scale > 0:
            img_array = (img_array / 127.5) - 1.0
            quantized = np.round(img_array / scale + zero_point)
            img_array = np.clip(quantized, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max).astype(input_dtype)
        else:
            img_array = img_array.astype(input_dtype)
    else:
        img_array = img_array.astype(np.float32)

    if len(input_shape) == 4 and img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    return img_array


def run_inference(
    model_path_str: str,
    image_path_str: str,
    top_k: int = 3,
) -> Tuple[str, float, List[Tuple[str, float]], float]:
    """Execute inference and display formatted output."""
    model_path = Path(model_path_str)
    image_path = Path(image_path_str)

    if not model_path.exists():
        # Fallback to any model in models dir
        if DEFAULT_MODEL_PATH.exists():
            model_path = DEFAULT_MODEL_PATH
        else:
            tflite_files = list(MODELS_DIR.glob("*.tflite"))
            if tflite_files:
                model_path = tflite_files[0]
            else:
                raise FileNotFoundError(f"TFLite model not found at {model_path}")

    class_names = load_class_names(DEFAULT_CLASS_NAMES_PATH)

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_tensor = preprocess_image_for_tflite(image_path, input_details)
    interpreter.set_tensor(input_details["index"], input_tensor)

    start_time = time.perf_counter()
    interpreter.invoke()
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    output_data = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] in (np.int8, np.uint8):
        scale, zero_point = output_details["quantization"]
        if scale > 0:
            output_data = (output_data.astype(np.float32) - zero_point) * scale

    exp_scores = np.exp(output_data - np.max(output_data))
    probabilities = exp_scores / np.sum(exp_scores) if np.max(output_data) > 1.0 else output_data

    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum

    top_indices = np.argsort(probabilities)[::-1]
    top_class_idx = top_indices[0]
    predicted_class = class_names.get(top_class_idx, f"Class {top_class_idx}")
    confidence_pct = float(probabilities[top_class_idx]) * 100.0

    ranked_predictions = []
    for idx in top_indices[:top_k]:
        name = class_names.get(idx, f"Class {idx}")
        pct = float(probabilities[idx]) * 100.0
        ranked_predictions.append((name, pct))

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
    target_model = model_path or DEFAULT_MODEL_PATH
    pred_label, conf, top_preds, latency = run_inference(
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
        description="Edge inference for Cotton Leaf Disease Detection on Raspberry Pi 5."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the TFLite model file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the cotton leaf image file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ranked predictions to show (default: 3).",
    )
    args = parser.parse_args()

    run_inference(args.model, args.image, top_k=args.top_k)
