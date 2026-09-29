"""
Dataset Preparation Script for Cotton Leaf Disease Detection and Advisory System.

This script:
1. Locates the raw cotton leaf disease class directories (handling wrapper folders dynamically).
2. Validates image integrity and filters out non-image or corrupt files.
3. Performs a reproducible, stratified 70% / 15% / 15% train/val/test split (Random Seed: 42).
4. Copies images into dataset/train, dataset/validation, and dataset/test using pathlib.
5. Protects existing split data from accidental overwriting unless explicit flags are given.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

# Import configuration constants
try:
    from src.config import (
        CLASS_NAMES,
        HUMAN_READABLE_LABELS,
        IGNORED_FILES,
        RANDOM_SEED,
        RAW_CLASS_MAPPING,
        RAW_DATA_DIR,
        SUPPORTED_EXTENSIONS,
        TEST_DIR,
        TEST_RATIO,
        TRAIN_DIR,
        TRAIN_RATIO,
        VAL_DIR,
        VALIDATION_RATIO,
    )
except ImportError:
    # Allows script to be executed directly from inside src/
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        CLASS_NAMES,
        HUMAN_READABLE_LABELS,
        IGNORED_FILES,
        RANDOM_SEED,
        RAW_CLASS_MAPPING,
        RAW_DATA_DIR,
        SUPPORTED_EXTENSIONS,
        TEST_DIR,
        TEST_RATIO,
        TRAIN_DIR,
        TRAIN_RATIO,
        VAL_DIR,
        VALIDATION_RATIO,
    )


def is_valid_image(file_path: Path) -> bool:
    """
    Verify if a file exists, has a supported extension, and can be read as a valid image.

    Args:
        file_path: Path to the image file.

    Returns:
        True if the image is valid and readable, False otherwise.
    """
    if not file_path.is_file():
        return False

    if file_path.name.startswith(".") or file_path.name.lower() in IGNORED_FILES:
        return False

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()  # Fast check for file corruption / integrity
        return True
    except Exception:
        return False


def discover_raw_class_directories(raw_root: Path) -> Dict[str, Path]:
    """
    Discover class folders inside raw_root, handling possible wrapper folders
    (e.g., 'Cotton_Original_Dataset' or 'Original Dataset').

    Args:
        raw_root: Path to the raw dataset directory.

    Returns:
        Dictionary mapping canonical class names to their source folder paths.
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")

    found_classes: Dict[str, Path] = {}

    # Gather candidate directories directly in raw_root or 1-2 levels nested
    candidate_dirs = [d for d in raw_root.rglob("*") if d.is_dir()]

    for folder in candidate_dirs:
        folder_norm = folder.name.strip().lower()
        if folder_norm in RAW_CLASS_MAPPING:
            canonical_name = RAW_CLASS_MAPPING[folder_norm]
            if canonical_name not in found_classes:
                found_classes[canonical_name] = folder

    # Verify all expected classes are detected
    missing = [c for c in CLASS_NAMES if c not in found_classes]
    if missing:
        readable_missing = [HUMAN_READABLE_LABELS.get(m, m) for m in missing]
        raise ValueError(
            f"Could not locate raw directories for expected classes: {readable_missing}\n"
            f"Searched inside: {raw_root}"
        )

    return found_classes


def collect_valid_images(class_dirs: Dict[str, Path]) -> Tuple[Dict[str, List[Path]], int]:
    """
    Collect and validate image file paths for each class folder.

    Args:
        class_dirs: Dictionary of canonical class names mapped to folder paths.

    Returns:
        Tuple of (dict mapping canonical class names to sorted list of valid paths, invalid_count).
    """
    dataset_paths: Dict[str, List[Path]] = {}
    invalid_count = 0

    for class_name, folder_path in class_dirs.items():
        valid_paths: List[Path] = []
        # Filter files in class directory
        for item in folder_path.iterdir():
            if item.name.startswith(".") or item.name.lower() in IGNORED_FILES:
                continue
            if item.is_file():
                if is_valid_image(item):
                    valid_paths.append(item)
                else:
                    invalid_count += 1

        # Remove duplicate paths and sort for reproducibility
        valid_paths = sorted(list(set(valid_paths)))
        dataset_paths[class_name] = valid_paths

    return dataset_paths, invalid_count


def split_class_samples(
    samples: List[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Stratified split for a single class list of samples with a fixed random seed.

    Args:
        samples: List of file paths for a single class.
        train_ratio: Proportion of training data (default: 0.70).
        val_ratio: Proportion of validation data (default: 0.15).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    total = len(samples)
    if total == 0:
        return [], [], []

    # Deterministic shuffle using dedicated random instance
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n_train = round(total * train_ratio)
    n_val = round(total * val_ratio)
    # Remaining goes to test to guarantee exact total count preservation
    n_test = total - (n_train + n_val)

    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train : n_train + n_val]
    test_samples = shuffled[n_train + n_val :]

    assert len(train_samples) + len(val_samples) + len(test_samples) == total

    return train_samples, val_samples, test_samples


def check_existing_splits(split_dirs: List[Path]) -> bool:
    """
    Check if any target split directories already contain files.
    """
    for d in split_dirs:
        if d.exists() and any(d.rglob("*")):
            # Check if there are actual files inside
            files = [f for f in d.rglob("*") if f.is_file()]
            if files:
                return True
    return False


def clean_split_directories(split_dirs: List[Path]) -> None:
    """
    Safely delete existing split directories to allow a clean re-split.
    """
    for d in split_dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def copy_files(file_list: List[Path], destination_dir: Path) -> None:
    """
    Copy a list of files to the destination directory preserving file metadata.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in file_list:
        shutil.copy2(file_path, destination_dir / file_path.name)


def prepare_dataset(force: bool = False, dry_run: bool = False) -> None:
    """
    Main orchestration routine for dataset discovery, validation, splitting, and copying.

    Args:
        force: Overwrite existing split directories if present.
        dry_run: Run discovery and split calculations without writing files.
    """
    print("=" * 70)
    print(" COTTON LEAF AI — DATASET PREPARATION PIPELINE")
    print("=" * 70)
    print(f"Project Root : {RAW_DATA_DIR.parent.parent}")
    print(f"Raw Directory: {RAW_DATA_DIR}")
    print(f"Random Seed  : {RANDOM_SEED}")
    print(f"Split Ratios : Train {TRAIN_RATIO*100:.0f}% | "
          f"Val {VALIDATION_RATIO*100:.0f}% | "
          f"Test {TEST_RATIO*100:.0f}%")
    print("-" * 70)

    # 1. Discover raw class folders
    print("[1/4] Discovering raw class folders...")
    try:
        class_dirs = discover_raw_class_directories(RAW_DATA_DIR)
        for class_name in CLASS_NAMES:
            folder = class_dirs[class_name]
            label = HUMAN_READABLE_LABELS.get(class_name, class_name)
            print(f"  ✓ {label:22}: {folder.relative_to(RAW_DATA_DIR)}")
    except Exception as e:
        print(f"  [ERROR] Class folder discovery failed: {e}")
        sys.exit(1)

    # 2. Collect and validate images
    print("\n[2/4] Validating images and filtering non-image files...")
    dataset_paths, invalid_count = collect_valid_images(class_dirs)

    total_valid = sum(len(paths) for paths in dataset_paths.values())
    for class_name in CLASS_NAMES:
        label = HUMAN_READABLE_LABELS.get(class_name, class_name)
        count = len(dataset_paths[class_name])
        print(f"  - {label:22}: {count:4d} valid images")

    print(f"  Total Valid Images   : {total_valid}")
    print(f"  Invalid/Corrupt Files: {invalid_count}")

    if total_valid == 0:
        print("  [ERROR] No valid images found! Please inspect dataset/raw/.")
        sys.exit(1)

    # 3. Check safety of target directories
    split_dirs = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    has_existing = check_existing_splits(split_dirs)

    if has_existing and not force and not dry_run:
        print("\n" + "!" * 70)
        print(" [WARNING] Existing split data detected in dataset/train, validation, or test.")
        print(" To prevent accidental data loss, existing files were NOT overwritten.")
        print(" Use '--force' flag to recreate the splits:")
        print("     python3 src/prepare_dataset.py --force")
        print("!" * 70)
        sys.exit(0)

    # 4. Perform Stratified Splitting
    print("\n[3/4] Performing reproducible stratified split (Seed = 42)...")
    split_results: Dict[str, Dict[str, List[Path]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }

    print(f"  {'Class':<22} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'Total':>6}")
    print("  " + "-" * 56)

    for class_name in CLASS_NAMES:
        label = HUMAN_READABLE_LABELS.get(class_name, class_name)
        paths = dataset_paths[class_name]
        train_p, val_p, test_p = split_class_samples(
            paths,
            train_ratio=TRAIN_RATIO,
            val_ratio=VALIDATION_RATIO,
            seed=RANDOM_SEED,
        )

        split_results["train"][class_name] = train_p
        split_results["validation"][class_name] = val_p
        split_results["test"][class_name] = test_p

        print(f"  {label:<22} | {len(train_p):>6d} | {len(val_p):>6d} | {len(test_p):>6d} | {len(paths):>6d}")

    total_train = sum(len(p) for p in split_results["train"].values())
    total_val = sum(len(p) for p in split_results["validation"].values())
    total_test = sum(len(p) for p in split_results["test"].values())

    print("  " + "-" * 56)
    print(f"  {'TOTALS':<22} | {total_train:>6d} | {total_val:>6d} | {total_test:>6d} | {total_valid:>6d}")
    print(f"  {'PERCENTAGES':<22} | {total_train/total_valid*100:>5.1f}% | {total_val/total_valid*100:>5.1f}% | {total_test/total_valid*100:>5.1f}% | 100.0%")

    if dry_run:
        print("\n[DRY RUN] Finished calculation. No files were written.")
        return

    # 5. Copy images into target directories
    print("\n[4/4] Writing split datasets to disk...")
    clean_split_directories(split_dirs)

    for split_name, split_dir in [("train", TRAIN_DIR), ("validation", VAL_DIR), ("test", TEST_DIR)]:
        print(f"  Writing {split_name} images to: {split_dir.relative_to(RAW_DATA_DIR.parent.parent)}/")
        for class_name in CLASS_NAMES:
            dest_folder = split_dir / class_name
            files = split_results[split_name][class_name]
            copy_files(files, dest_folder)

    print("\n" + "=" * 70)
    print(" ✓ DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Next step: Verify dataset using:")
    print("    python3 src/verify_dataset.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and split Cotton Leaf Disease dataset reproducibly."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split directories (train, validation, test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate split without copying files to disk.",
    )
    args = parser.parse_args()

    prepare_dataset(force=args.force, dry_run=args.dry_run)
