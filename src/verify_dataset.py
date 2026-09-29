"""
Dataset Verification Script for Cotton Leaf Disease Detection and Advisory System.

This script inspects the dataset and reports:
- Number of images in each class
- Total images
- Number of training images
- Number of validation images
- Number of testing images
- Image extensions detected
- Invalid/corrupted images if detected

Calculates all values from actual filesystem contents without hardcoding.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image

try:
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        DATASET_DIR,
        HUMAN_READABLE_LABELS,
        IGNORED_FILES,
        RAW_CLASS_MAPPING,
        RAW_DATA_DIR,
        SUPPORTED_EXTENSIONS,
        TEST_DIR,
        TRAIN_DIR,
        VAL_DIR,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        BASE_DIR,
        CLASS_NAMES,
        DATASET_DIR,
        HUMAN_READABLE_LABELS,
        IGNORED_FILES,
        RAW_CLASS_MAPPING,
        RAW_DATA_DIR,
        SUPPORTED_EXTENSIONS,
        TEST_DIR,
        TRAIN_DIR,
        VAL_DIR,
    )


def verify_image_file(file_path: Path) -> Tuple[bool, str]:
    """
    Verify if a file is an uncorrupted, valid image.

    Returns:
        Tuple of (is_valid: bool, format_or_error: str)
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Re-open to get format since verify() closes/invalidates the image object
        with Image.open(file_path) as img:
            fmt = img.format or file_path.suffix.upper().lstrip(".")
        return True, fmt
    except Exception as e:
        return False, str(e)


def scan_directory_images(
    directory: Path,
) -> Tuple[List[Path], List[Path], Set[str]]:
    """
    Scan a directory recursively for images, validating integrity and collecting extensions.

    Returns:
        Tuple of (valid_files, invalid_files, detected_extensions).
    """
    valid_files: List[Path] = []
    invalid_files: List[Path] = []
    extensions: Set[str] = set()

    if not directory.exists():
        return valid_files, invalid_files, extensions

    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        if item.name.startswith(".") or item.name.lower() in IGNORED_FILES:
            continue

        ext = item.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            extensions.add(ext)
            is_valid, _ = verify_image_file(item)
            if is_valid:
                valid_files.append(item)
            else:
                invalid_files.append(item)

    return valid_files, invalid_files, extensions


def inspect_raw_dataset() -> Tuple[Dict[str, int], int, Set[str], int]:
    """
    Inspect raw dataset directory and calculate class totals.

    Returns:
        Tuple of (class_counts, total_raw_valid, raw_extensions, raw_invalid_count).
    """
    class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    raw_extensions: Set[str] = set()
    raw_invalid_count = 0

    if not RAW_DATA_DIR.exists():
        return class_counts, 0, raw_extensions, 0

    # Locate class directories inside raw (supporting wrapper folders)
    candidate_dirs = [d for d in RAW_DATA_DIR.rglob("*") if d.is_dir()]

    for folder in candidate_dirs:
        norm_name = folder.name.strip().lower()
        if norm_name in RAW_CLASS_MAPPING:
            canonical = RAW_CLASS_MAPPING[norm_name]
            val_files, inv_files, exts = scan_directory_images(folder)
            class_counts[canonical] += len(val_files)
            raw_invalid_count += len(inv_files)
            raw_extensions.update(exts)

    total_valid = sum(class_counts.values())
    return class_counts, total_valid, raw_extensions, raw_invalid_count


def inspect_split_dataset(
    split_dir: Path,
) -> Tuple[Dict[str, int], int, Set[str], int]:
    """
    Inspect a split directory (train, validation, or test) per class.

    Returns:
        Tuple of (class_counts, total_split_valid, split_extensions, split_invalid_count).
    """
    class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    split_extensions: Set[str] = set()
    split_invalid_count = 0

    if not split_dir.exists():
        return class_counts, 0, split_extensions, 0

    for class_name in CLASS_NAMES:
        class_folder = split_dir / class_name
        if class_folder.exists() and class_folder.is_dir():
            val_files, inv_files, exts = scan_directory_images(class_folder)
            class_counts[class_name] = len(val_files)
            split_invalid_count += len(inv_files)
            split_extensions.update(exts)

    total_valid = sum(class_counts.values())
    return class_counts, total_valid, split_extensions, split_invalid_count


def verify_dataset() -> None:
    """
    Run comprehensive dataset verification and display formatted report.
    """
    # 1. Raw Dataset Inspection
    raw_counts, raw_total, raw_exts, raw_invalid = inspect_raw_dataset()

    # 2. Split Directories Inspection
    train_counts, train_total, train_exts, train_invalid = inspect_split_dataset(TRAIN_DIR)
    val_counts, val_total, val_exts, val_invalid = inspect_split_dataset(VAL_DIR)
    test_counts, test_total, test_exts, test_invalid = inspect_split_dataset(TEST_DIR)

    all_extensions = raw_exts.union(train_exts).union(val_exts).union(test_exts)
    total_invalid = raw_invalid + train_invalid + val_invalid + test_invalid

    splits_exist = (train_total + val_total + test_total) > 0

    # Determine display counts for classes:
    # If splits exist, class counts are from the combined splits (or raw)
    display_class_counts: Dict[str, int] = {}
    for c in CLASS_NAMES:
        if splits_exist:
            display_class_counts[c] = train_counts[c] + val_counts[c] + test_counts[c]
        else:
            display_class_counts[c] = raw_counts[c]

    display_total = sum(display_class_counts.values())

    print()
    print("# Dataset Verification")
    print()
    for class_name in CLASS_NAMES:
        label = HUMAN_READABLE_LABELS.get(class_name, class_name)
        count = display_class_counts.get(class_name, 0)
        print(f"{label:20} : {count}")
    print()
    print(f"{'Total Images':20} : {display_total}")
    print()
    print(f"{'Train':20} : {train_total}")
    print(f"{'Validation':20} : {val_total}")
    print(f"{'Test':20} : {test_total}")
    print()
    print(f"{'Invalid Images':20} : {total_invalid}")
    print()

    # Additional Detailed Diagnostics
    print("=" * 60)
    print("DETAILED VERIFICATION DIAGNOSTICS")
    print("=" * 60)
    ext_str = ", ".join(sorted(all_extensions)) if all_extensions else "None detected"
    print(f"Detected Image Extensions: {ext_str}")
    print(f"Project Location         : {BASE_DIR}")
    print(f"Raw Directory Status     : {'Found (' + str(raw_total) + ' images)' if raw_total > 0 else 'Empty/Not Found'}")

    if splits_exist:
        print("\n--- Per-Class Split Breakdown ---")
        print(f"{'Class':<22} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'Class Total':>11}")
        print("-" * 60)
        for c in CLASS_NAMES:
            label = HUMAN_READABLE_LABELS.get(c, c)
            tr = train_counts[c]
            va = val_counts[c]
            te = test_counts[c]
            tot = tr + va + te
            print(f"{label:<22} | {tr:>6d} | {va:>6d} | {te:>6d} | {tot:>11d}")
        print("-" * 60)
        print(f"{'TOTAL':<22} | {train_total:>6d} | {val_total:>6d} | {test_total:>6d} | {display_total:>11d}")

        # Integrity Checks
        print("\n--- Integrity Validations ---")
        is_sum_correct = (train_total + val_total + test_total) == raw_total
        all_classes_present = all(
            train_counts[c] > 0 and val_counts[c] > 0 and test_counts[c] > 0
            for c in CLASS_NAMES
        )

        print(f"  [✓] Train + Validation + Test equals Raw Total : {is_sum_correct} ({train_total + val_total + test_total}/{raw_total})")
        print(f"  [✓] All classes present in all splits          : {all_classes_present}")
        print(f"  [✓] Zero corrupted/invalid images              : {total_invalid == 0}")
    else:
        print("\n[NOTE] Split directories (train/validation/test) are currently empty.")
        print("Run dataset preparation first:")
        print("    python3 src/prepare_dataset.py")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify_dataset()
