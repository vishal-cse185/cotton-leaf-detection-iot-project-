"""
Raspberry Pi 5 Camera Module 3 Integration for Edge Deployment.

Part of: IoT, DNN and Generative AI Based Cotton Leaf Disease Detection and Advisory System.
Self-contained camera module for deployment on Raspberry Pi.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PiCameraEdge")

# Root of deployment package
DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

CAPTURES_DIR = DEPLOYMENT_ROOT / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = DEPLOYMENT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "cotton_leaf_mobilenetv2_float16.tflite"

PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
except (ImportError, Exception):
    PICAMERA2_AVAILABLE = False

CV2_AVAILABLE = False
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except (ImportError, Exception):
    CV2_AVAILABLE = False


class CameraManager:
    """Unified Camera Manager supporting Raspberry Pi Camera Module 3 and fallbacks."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        autofocus: bool = True,
        save_dir: Optional[Path] = None,
    ):
        self.resolution = resolution
        self.autofocus = autofocus
        self.save_dir = Path(save_dir) if save_dir else CAPTURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.backend = self._detect_backend()
        logger.info(f"Edge CameraManager active: '{self.backend}' | Resolution: {self.resolution}")

    def _detect_backend(self) -> str:
        if PICAMERA2_AVAILABLE:
            try:
                p2 = Picamera2()
                camera_info = p2.camera_info()
                p2.close()
                if camera_info:
                    return "picamera2"
            except Exception:
                pass

        if shutil.which("rpicam-still") is not None:
            return "rpicam-still"

        if shutil.which("libcamera-still") is not None:
            return "libcamera-still"

        if CV2_AVAILABLE:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        return "opencv"
            except Exception:
                pass

        return "simulation"

    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "target_device": "Raspberry Pi 5",
            "supported_sensors": ["Camera Module 3 (IMX708 12MP)", "Camera Module 2 (IMX219)", "USB Webcam"],
            "autofocus_supported": self.backend in ("picamera2", "rpicam-still"),
            "autofocus_enabled": self.autofocus,
            "configured_resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "captures_directory": str(self.save_dir),
            "is_hardware_camera": self.backend in ("picamera2", "rpicam-still", "libcamera-still", "opencv"),
        }

    def capture_image(
        self,
        output_filename: Optional[str] = None,
        timeout_ms: int = 1500,
    ) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"cotton_leaf_{timestamp_str}.jpg"

        output_path = self.save_dir / output_filename
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "backend_used": self.backend,
            "output_path": str(output_path),
            "resolution": self.resolution,
        }

        start_time = time.perf_counter()

        if self.backend == "picamera2":
            img_arr = self._capture_picamera2(output_path, timeout_ms)
        elif self.backend == "rpicam-still":
            img_arr = self._capture_rpicam_still(output_path, timeout_ms)
        elif self.backend == "libcamera-still":
            img_arr = self._capture_libcamera_still(output_path, timeout_ms)
        elif self.backend == "opencv":
            img_arr = self._capture_opencv(output_path)
        else:
            img_arr = self._capture_simulation(output_path)

        capture_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metadata["capture_latency_ms"] = capture_duration_ms
        return output_path, img_arr, metadata

    def _capture_picamera2(self, output_path: Path, timeout_ms: int) -> np.ndarray:
        picam2 = Picamera2()
        try:
            still_config = picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            picam2.configure(still_config)
            picam2.start()

            if self.autofocus:
                try:
                    picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
                except Exception:
                    pass

            time.sleep(timeout_ms / 1000.0)
            img_array = picam2.capture_array("main")
            pil_img = Image.fromarray(img_array)
            pil_img.save(output_path, quality=95)
            return img_array
        finally:
            picam2.stop()
            picam2.close()

    def _capture_rpicam_still(self, output_path: Path, timeout_ms: int) -> np.ndarray:
        cmd = [
            "rpicam-still",
            "-t", str(timeout_ms),
            "--width", str(self.resolution[0]),
            "--height", str(self.resolution[1]),
            "-o", str(output_path),
            "--nopreview",
        ]
        if self.autofocus:
            cmd.extend(["--autofocus-mode", "auto"])

        subprocess.run(cmd, check=True, capture_output=True)
        with Image.open(output_path) as img:
            rgb_img = img.convert("RGB")
            return np.array(rgb_img)

    def _capture_libcamera_still(self, output_path: Path, timeout_ms: int) -> np.ndarray:
        cmd = [
            "libcamera-still",
            "-t", str(timeout_ms),
            "--width", str(self.resolution[0]),
            "--height", str(self.resolution[1]),
            "-o", str(output_path),
            "--nopreview",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        with Image.open(output_path) as img:
            rgb_img = img.convert("RGB")
            return np.array(rgb_img)

    def _capture_opencv(self, output_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture from OpenCV device")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        pil_img.save(output_path, quality=95)
        return rgb_frame

    def _capture_simulation(self, output_path: Path) -> np.ndarray:
        samples_dir = DEPLOYMENT_ROOT / "static" / "samples"
        sample_img_path = None
        if samples_dir.exists():
            samples = list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg"))
            if samples:
                sample_img_path = samples[0]

        if sample_img_path and sample_img_path.exists():
            with Image.open(sample_img_path) as img:
                rgb_img = img.convert("RGB").resize(self.resolution, Image.Resampling.BILINEAR)
                rgb_img.save(output_path, quality=95)
                return np.array(rgb_img)

        w, h = self.resolution
        synthetic = np.full((h, w, 3), [34, 139, 34], dtype=np.uint8)
        for y in range(h):
            synthetic[y, :, 1] = np.clip(139 + int(30 * np.sin(y / 50.0)), 0, 255)
        pil_img = Image.fromarray(synthetic)
        pil_img.save(output_path, quality=95)
        return synthetic


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Camera Module 3 Edge Capture")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    parser.add_argument("--status", action="store_true", help="Print camera diagnostic status")
    parser.add_argument("--predict", action="store_true", help="Immediately run TFLite prediction")
    args = parser.parse_args()

    camera = CameraManager()
    if args.status:
        status = camera.get_status()
        print(json.dumps(status, indent=2))
        return

    output_path, _, meta = camera.capture_image(output_filename=args.output)
    print(f"Captured: {output_path} ({meta['capture_latency_ms']} ms via {meta['backend_used']})")

    if args.predict:
        from src.tflite_predict import predict_leaf
        res = predict_leaf(output_path)
        print(f"Prediction: {res['predicted_label']} ({res['confidence']}%) in {res['inference_time_ms']} ms")


if __name__ == "__main__":
    main()
