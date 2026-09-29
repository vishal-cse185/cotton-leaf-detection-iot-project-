"""
Evaluation Script for Cotton Leaf Disease Detection System.

This script performs independent evaluation on dataset/test/:
- Computes overall test accuracy and loss.
- Computes per-class Precision, Recall, F1-Score, and Support.
- Generates and saves a high-resolution Confusion Matrix heatmap.
- Analyzes per-class performance and potential confusion patterns.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Ensure src can be imported
try:
    from src.config import (
        BASE_DIR,
        BATCH_SIZE,
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        CONFUSION_MATRIX_PLOT,
        FINAL_MODEL_PATH,
        HUMAN_READABLE_LABELS,
        IMAGE_SIZE,
        RESULTS_DIR,
        TEST_DIR,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        BASE_DIR,
        BATCH_SIZE,
        CLASS_NAMES,
        CLASS_NAMES_JSON_PATH,
        CONFUSION_MATRIX_PLOT,
        FINAL_MODEL_PATH,
        HUMAN_READABLE_LABELS,
        IMAGE_SIZE,
        RESULTS_DIR,
        TEST_DIR,
    )

import tensorflow as tf


def load_class_mapping() -> Dict[int, str]:
    """Load class mapping from JSON file or fall back to config."""
    if CLASS_NAMES_JSON_PATH.exists():
        with open(CLASS_NAMES_JSON_PATH, "r") as f:
            raw_map = json.load(f)
            return {int(k): v for k, v in raw_map.items()}
    return {i: HUMAN_READABLE_LABELS.get(c, c) for i, c in enumerate(CLASS_NAMES)}


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: List[str],
    save_path: Path,
) -> None:
    """
    Plot and save a dual-annotated Confusion Matrix heatmap (Counts + Percentages).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute row-normalized percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    # Format cell annotations: "Count\n(Percentage%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"

    plt.figure(figsize=(9, 7.5), dpi=300)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#cccccc",
    )

    plt.title("Cotton Leaf Disease Detection — Test Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved confusion matrix to: {save_path}")


def analyze_per_class_performance(
    cm: np.ndarray,
    target_names: List[str],
    report_dict: dict,
) -> None:
    """Provide written analysis of strong and challenging disease classes."""
    print("\n" + "=" * 70)
    print(" PER-CLASS PERFORMANCE & CONFUSION ANALYSIS")
    print("=" * 70)

    f1_scores = []
    for idx, name in enumerate(target_names):
        metrics = report_dict[name]
        f1 = metrics["f1-score"]
        recall = metrics["recall"]
        precision = metrics["precision"]
        support = metrics["support"]
        f1_scores.append((name, f1, precision, recall, support, idx))

    # Sort by F1-score
    f1_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n1. Disease Classes Ranked by Diagnostic Performance (F1-Score):")
    for rank, (name, f1, prec, rec, supp, _) in enumerate(f1_scores, 1):
        status = "★ Excellent" if f1 >= 0.95 else ("✓ Good" if f1 >= 0.85 else "! Needs Improvement")
        print(f"   {rank}. {name:<22}: F1={f1*100:5.1f}% | Precision={prec*100:5.1f}% | Recall={rec*100:5.1f}% | Support={supp} [{status}]")

    print("\n2. Misclassification & Confusion Inspection:")
    misclassifications_found = False
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i, j] > 0:
                misclassifications_found = True
                print(f"   - {cm[i, j]} sample(s) of '{target_names[i]}' mistakenly predicted as '{target_names[j]}'")

    if not misclassifications_found:
        print("   - Perfect classification! Zero misclassifications found in the test dataset.")

    print("=" * 70 + "\n")


def evaluate_pipeline() -> Tuple[float, dict, np.ndarray]:
    """Execute evaluation pipeline on test dataset."""
    print("=" * 70)
    print(" INDEPENDENT TEST DATASET EVALUATION")
    print("=" * 70)
    print(f"Model File     : {FINAL_MODEL_PATH}")
    print(f"Test Directory : {TEST_DIR}")
    print("-" * 70)

    if not FINAL_MODEL_PATH.exists():
        print(f"  [ERROR] Trained model file not found: {FINAL_MODEL_PATH}")
        print("  Please run training first: python3 src/train_model.py")
        sys.exit(1)

    # 1. Load Model and Class Mapping
    print("\n[1/4] Loading Trained Model & Class Metadata...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    class_map = load_class_mapping()
    ordered_labels = [class_map[i] for i in range(len(class_map))]
    print(f"  Model Loaded Successfully. Target Classes: {ordered_labels}")

    # 2. Load Test Dataset (shuffle=False for exact alignment)
    print("\n[2/4] Loading Test Dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 3. Compute Predictions & Ground Truth
    print("\n[3/4] Running Inference on Independent Test Images...")
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model(images, training=False)
        y_pred_probs_list.append(preds.numpy())
        y_true_list.append(labels.numpy())

    y_true_cat = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

    y_true = np.argmax(y_true_cat, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate overall metrics
    test_accuracy = np.mean(y_true == y_pred)
    total_test = len(y_true)
    correct_count = np.sum(y_true == y_pred)

    print(f"\n  Total Test Samples   : {total_test}")
    print(f"  Correctly Classified : {correct_count} / {total_test}")
    print(f"  Final Test Accuracy  : {test_accuracy * 100:.2f}%\n")

    # 4. Classification Report
    print("=" * 70)
    print(" CLASSIFICATION REPORT")
    print("=" * 70)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=ordered_labels,
        digits=4,
    )
    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=ordered_labels,
        output_dict=True,
    )

    # 5. Confusion Matrix
    print("\n[4/4] Generating Confusion Matrix Plot...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ordered_labels, CONFUSION_MATRIX_PLOT)

    # 6. Detailed Performance Analysis
    analyze_per_class_performance(cm, ordered_labels, report_dict)

    return test_accuracy, report_dict, cm


if __name__ == "__main__":
    evaluate_pipeline()
