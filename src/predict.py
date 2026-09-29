"""
Single-Image Prediction Utility for Cotton Leaf Disease Detection System.

Usage:
    python3 src/predict.py --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Ensure src can be imported
try:
    from src.config import (
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        FINAL_MODEL_PATH,
        HUMAN_READABLE_LABELS,
        IMAGE_SIZE,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        FINAL_MODEL_PATH,
        HUMAN_READABLE_LABELS,
        IMAGE_SIZE,
    )

import tensorflow as tf


def load_class_mapping() -> Dict[int, str]:
    """Load class mapping from JSON file or fall back to defaults."""
    if CLASS_NAMES_JSON_PATH.exists():
        with open(CLASS_NAMES_JSON_PATH, "r") as f:
            raw_map = json.load(f)
            return {int(k): v for k, v in raw_map.items()}
    return {i: HUMAN_READABLE_LABELS.get(c, c) for i, c in enumerate(CLASS_NAMES)}


def preprocess_image(image_path: Path) -> np.ndarray:
    """
    Load and format an image for model input:
    - Verifies readable image
    - Converts to RGB
    - Resizes to (224, 224)
    - Expands dimensions to (1, 224, 224, 3)

    Returns:
        NumPy array with shape (1, 224, 224, 3).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            resized_img = rgb_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            img_array = np.array(resized_img, dtype=np.float32)
            img_tensor = np.expand_dims(img_array, axis=0)
            return img_tensor
    except Exception as e:
        raise ValueError(f"Could not load or process image {image_path}: {e}")


def predict_single_image(image_path_str: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Run inference on a single cotton leaf image.

    Args:
        image_path_str: Path to the input image.
        top_k: Number of ranked predictions to display.

    Returns:
        Tuple of (predicted_class, confidence_pct, list_of_top_predictions).
    """
    image_path = Path(image_path_str)

    if not FINAL_MODEL_PATH.exists():
        print(f"\n[ERROR] Trained model file not found: {FINAL_MODEL_PATH}")
        print("Please train the model first by running: python3 src/train_model.py\n")
        sys.exit(1)

    # 1. Preprocess Image
    input_tensor = preprocess_image(image_path)

    # 2. Load Model and Class Mapping
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    class_map = load_class_mapping()

    # 3. Run Inference
    predictions = model(input_tensor, training=False).numpy()[0]

    # 4. Rank Predictions
    top_indices = np.argsort(predictions)[::-1]
    top_class_idx = top_indices[0]
    predicted_disease = class_map.get(top_class_idx, f"Class {top_class_idx}")
    top_confidence = float(predictions[top_class_idx]) * 100.0

    ranked_predictions = []
    for idx in top_indices[:top_k]:
        disease_name = class_map.get(idx, f"Class {idx}")
        conf = float(predictions[idx]) * 100.0
        ranked_predictions.append((disease_name, conf))

    # 5. Format and Print Results
    print("\n" + "=" * 50)
    print(" COTTON LEAF DISEASE INFERENCE RESULT")
    print("=" * 50)
    print(f"Image File       : {image_path.name}")
    print(f"Predicted disease: {predicted_disease}")
    print(f"Confidence       : {top_confidence:.2f}%\n")
    print("Top predictions:")
    for rank, (name, conf) in enumerate(ranked_predictions, 1):
        print(f"  {rank}. {name} — {conf:.2f}%")
    print("=" * 50 + "\n")

    return predicted_disease, top_confidence, ranked_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run disease inference on a single cotton leaf image using MobileNetV2."
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
        default=5,
        help="Number of top predictions to display (default: 5).",
    )
    args = parser.parse_args()

    predict_single_image(args.image, top_k=args.top_k)
