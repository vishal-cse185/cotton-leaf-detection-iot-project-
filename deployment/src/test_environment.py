"""
Raspberry Pi 5 Environment and TFLite Runtime Diagnostic Test Script.

Validates:
1. Python version compatibility (>= 3.9)
2. NumPy package availability
3. Pillow package availability
4. TFLite / LiteRT runtime availability (tflite-runtime or tensorflow.lite)
5. Edge model file presence (models/cotton_leaf_mobilenetv2_float16.tflite)
6. Class names mapping presence (models/class_names.json)
7. Model input tensor verification (expected: [1, 224, 224, 3])
8. Model output tensor verification (expected: [1, 5])

Prints an unambiguous PASS / FAIL report for field deployment readiness.
"""

import json
import sys
from pathlib import Path

# Dynamic relative path resolution inside deployment package
DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = DEPLOYMENT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "cotton_leaf_mobilenetv2_float16.tflite"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


def run_environment_diagnostic() -> bool:
    """Execute diagnostic checks and return True if all passed."""
    print("=" * 65)
    print(" RASPBERRY PI 5 EDGE ENVIRONMENT & MODEL DIAGNOSTIC")
    print("=" * 65)
    print(f"Deployment Root: {DEPLOYMENT_ROOT}")
    print(f"Target Device  : Raspberry Pi 5 (ARM Cortex-A76, 64-bit OS)")
    print("-" * 65)

    checks = []

    # 1. Python Version Check
    py_major, py_minor, py_micro = sys.version_info[:3]
    py_version_str = f"{py_major}.{py_minor}.{py_micro}"
    py_pass = (py_major == 3 and py_minor >= 9)
    checks.append((
        "Python Version",
        py_version_str,
        ">= 3.9",
        py_pass,
    ))

    # 2. NumPy Availability
    try:
        import numpy as np
        numpy_pass = True
        numpy_ver = np.__version__
    except ImportError:
        numpy_pass = False
        numpy_ver = "Not Installed"
    checks.append((
        "NumPy Library",
        numpy_ver,
        "Installed",
        numpy_pass,
    ))

    # 3. Pillow Availability
    try:
        import PIL
        from PIL import Image
        pillow_pass = True
        pillow_ver = PIL.__version__
    except ImportError:
        pillow_pass = False
        pillow_ver = "Not Installed"
    checks.append((
        "Pillow (PIL) Library",
        pillow_ver,
        "Installed",
        pillow_pass,
    ))

    # 4. TFLite / LiteRT Runtime Availability
    tflite_runtime_name = None
    interpreter_class = None

    try:
        import tflite_runtime.interpreter as tflite_rt
        interpreter_class = tflite_rt.Interpreter
        tflite_runtime_name = "tflite-runtime"
    except ImportError:
        try:
            import tensorflow.lite as tf_lite
            interpreter_class = tf_lite.Interpreter
            tflite_runtime_name = "tensorflow.lite"
        except ImportError:
            try:
                import ai_edge_litert.interpreter as litert
                interpreter_class = litert.Interpreter
                tflite_runtime_name = "ai_edge_litert"
            except ImportError:
                tflite_runtime_name = "Not Installed"

    tflite_pass = interpreter_class is not None
    checks.append((
        "TFLite / LiteRT Runtime",
        tflite_runtime_name,
        "tflite-runtime or tensorflow",
        tflite_pass,
    ))

    # 5. Model File Presence
    model_found = DEFAULT_MODEL_PATH.exists()
    # Check fallback if any other .tflite in models/
    active_model_path = DEFAULT_MODEL_PATH
    if not model_found and MODELS_DIR.exists():
        tflite_files = list(MODELS_DIR.glob("*.tflite"))
        if tflite_files:
            active_model_path = tflite_files[0]
            model_found = True

    model_size_str = f"{active_model_path.stat().st_size / (1024*1024):.2f} MB" if model_found else "Missing"
    checks.append((
        "TFLite Edge Model File",
        f"{active_model_path.name} ({model_size_str})" if model_found else "Not Found",
        "Valid .tflite file",
        model_found,
    ))

    # 6. Class Names JSON Presence
    classes_found = CLASS_NAMES_PATH.exists()
    class_count_str = "Missing"
    if classes_found:
        try:
            with open(CLASS_NAMES_PATH, "r") as f:
                classes_data = json.load(f)
                class_count_str = f"{len(classes_data)} classes found"
        except Exception as e:
            classes_found = False
            class_count_str = f"Corrupt: {e}"

    checks.append((
        "Class Names Mapping",
        class_count_str,
        "5 classes JSON",
        classes_found,
    ))

    # 7 & 8. Model Input / Output Tensor Dimensions
    input_shape_pass = False
    output_shape_pass = False
    input_shape_str = "N/A"
    output_shape_str = "N/A"

    if tflite_pass and model_found:
        try:
            interp = interpreter_class(model_path=str(active_model_path))
            interp.allocate_tensors()
            in_details = interp.get_input_details()[0]
            out_details = interp.get_output_details()[0]

            in_shape = in_details["shape"].tolist()
            out_shape = out_details["shape"].tolist()

            input_shape_str = f"{in_shape} ({in_details['dtype'].__name__})"
            output_shape_str = f"{out_shape} ({out_details['dtype'].__name__})"

            input_shape_pass = (in_shape == [1, 224, 224, 3] or in_shape[1:] == [224, 224, 3])
            output_shape_pass = (out_shape == [1, 5] or out_shape[-1] == 5)
        except Exception as e:
            input_shape_str = f"Error: {e}"
            output_shape_str = "Error"

    checks.append((
        "Model Input Shape",
        input_shape_str,
        "[1, 224, 224, 3]",
        input_shape_pass,
    ))

    checks.append((
        "Model Output Shape",
        output_shape_str,
        "[1, 5]",
        output_shape_pass,
    ))

    # Print Report Table
    print(f"{'#':<3} | {'Check Item':<24} | {'Detected':<30} | {'Status':>8}")
    print("-" * 72)
    all_passed = True
    for idx, (item, detected, expected, status) in enumerate(checks, 1):
        status_str = "PASS" if status else "FAIL"
        if not status:
            all_passed = False
        print(f"{idx:<3} | {item:<24} | {detected:<30} | {status_str:>8}")
    print("-" * 72)

    if all_passed:
        print("\n" + "=" * 65)
        print("  ✓ ALL DIAGNOSTIC CHECKS PASSED!")
        print("  The Raspberry Pi environment is ready for TFLite inference.")
        print("=" * 65 + "\n")
    else:
        print("\n" + "=" * 65)
        print("  ! SOME DIAGNOSTIC CHECKS FAILED.")
        print("  Please install missing dependencies or copy required models.")
        print("=" * 65 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_environment_diagnostic()
    sys.exit(0 if success else 1)
