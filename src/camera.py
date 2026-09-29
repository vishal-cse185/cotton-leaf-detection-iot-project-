"""
Raspberry Pi 5 Camera Module 3 Integration & Real-Time Leaf Capture Module.

Part of: IoT, DNN and Generative AI Based Cotton Leaf Disease Detection and Advisory System.

Supported Backends (Automatic Fallback Priority):
1. Picamera2 (Official Python interface for Raspberry Pi OS Bookworm with autofocus support)
2. rpicam-still / libcamera-still (Raspberry Pi CLI commands via subprocess)
3. OpenCV VideoCapture (USB Webcams / V4L2 devices)
4. Synthetic / Test Sample Fallback (Offline development & simulated diagnostics)
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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CameraModule")

# Dynamic root path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CAPTURES_DIR = BASE_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Attempt imports for available camera backends
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
        self._picam2_instance = None
        logger.info(f"CameraManager initialized with backend: '{self.backend}' | Resolution: {self.resolution}")

    def _detect_backend(self) -> str:
        """Detect the optimal camera backend available on this system."""
        # Check Picamera2 (native Pi OS Bookworm / Raspberry Pi 5)
        if PICAMERA2_AVAILABLE:
            try:
                # Test initializing Picamera2
                p2 = Picamera2()
                camera_info = p2.camera_info()
                p2.close()
                if camera_info:
                    return "picamera2"
            except Exception as e:
                logger.debug(f"Picamera2 initialization check returned: {e}")

        # Check rpicam-still CLI (Raspberry Pi OS)
        if shutil.which("rpicam-still") is not None:
            return "rpicam-still"

        # Check libcamera-still CLI (Legacy Raspberry Pi OS)
        if shutil.which("libcamera-still") is not None:
            return "libcamera-still"

        # Check OpenCV Webcam / V4L2 device
        if CV2_AVAILABLE:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        return "opencv"
            except Exception as e:
                logger.debug(f"OpenCV video capture check returned: {e}")

        # Fallback to test/simulation mode
        return "simulation"

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and hardware status of the camera subsystem."""
        status: Dict[str, Any] = {
            "backend": self.backend,
            "target_device": "Raspberry Pi 5",
            "supported_sensors": ["Camera Module 3 (IMX708 12MP)", "Camera Module 2 (IMX219)", "USB Webcam"],
            "autofocus_supported": self.backend in ("picamera2", "rpicam-still"),
            "autofocus_enabled": self.autofocus,
            "configured_resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "captures_directory": str(self.save_dir),
            "is_hardware_camera": self.backend in ("picamera2", "rpicam-still", "libcamera-still", "opencv"),
        }

        # Attempt to gather hardware details from rpicam-hello / libcamera-hello if available
        if self.backend in ("rpicam-still", "picamera2", "libcamera-still"):
            cli_tool = "rpicam-hello" if shutil.which("rpicam-hello") else "libcamera-hello"
            if shutil.which(cli_tool):
                try:
                    res = subprocess.run([cli_tool, "--list-cameras"], capture_output=True, text=True, timeout=3)
                    status["camera_probe_output"] = res.stdout.strip() or res.stderr.strip()
                except Exception:
                    status["camera_probe_output"] = "Available on Raspberry Pi bus"
        return status

    def capture_image(
        self,
        output_filename: Optional[str] = None,
        timeout_ms: int = 1500,
    ) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
        """
        Capture a high-resolution image using the active backend.

        Returns:
            Tuple of (saved_file_path, rgb_numpy_array, metadata_dict)
        """
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
        logger.info(f"Captured leaf photo saved to: {output_path} ({capture_duration_ms} ms)")
        return output_path, img_arr, metadata

    def _capture_picamera2(self, output_path: Path, timeout_ms: int) -> np.ndarray:
        """Capture using official Picamera2 library."""
        picam2 = Picamera2()
        try:
            still_config = picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            picam2.configure(still_config)
            picam2.start()

            # Enable continuous autofocus if supported (Camera Module 3 IMX708)
            if self.autofocus:
                try:
                    picam2.set_controls({"AfMode": 2, "AfTrigger": 0})  # 2: Continuous Auto-Focus
                except Exception as af_err:
                    logger.debug(f"Autofocus control note: {af_err}")

            time.sleep(timeout_ms / 1000.0)  # Allow AGC/AEC convergence
            img_array = picam2.capture_array("main")
            pil_img = Image.fromarray(img_array)
            pil_img.save(output_path, quality=95)
            return img_array
        finally:
            picam2.stop()
            picam2.close()

    def _capture_rpicam_still(self, output_path: Path, timeout_ms: int) -> np.ndarray:
        """Capture using rpicam-still CLI tool."""
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
        """Capture using legacy libcamera-still CLI tool."""
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
        """Capture using OpenCV VideoCapture."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        # Warmup
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from OpenCV capture device")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        pil_img.save(output_path, quality=95)
        return rgb_frame

    def _capture_simulation(self, output_path: Path) -> np.ndarray:
        """
        Simulation fallback when no physical camera is attached.
        Selects a sample leaf from the test dataset if present, or creates a high-res leaf image.
        """
        logger.info("[SIMULATION MODE] No physical camera detected. Generating realistic test capture.")
        test_dir = BASE_DIR / "dataset" / "test"
        sample_img_path = None

        if test_dir.exists():
            for cls_folder in test_dir.iterdir():
                if cls_folder.is_dir():
                    images = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
                    if images:
                        sample_img_path = images[0]
                        break

        if sample_img_path and sample_img_path.exists():
            with Image.open(sample_img_path) as img:
                rgb_img = img.convert("RGB").resize(self.resolution, Image.Resampling.BILINEAR)
                rgb_img.save(output_path, quality=95)
                return np.array(rgb_img)

        # Generate a synthetic cotton leaf template with realistic chlorophyll color
        w, h = self.resolution
        synthetic = np.full((h, w, 3), [34, 139, 34], dtype=np.uint8)  # Forest Green
        # Add gradient/vein texture
        for y in range(h):
            synthetic[y, :, 1] = np.clip(139 + int(30 * np.sin(y / 50.0)), 0, 255)
        pil_img = Image.fromarray(synthetic)
        pil_img.save(output_path, quality=95)
        return synthetic


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Camera Module 3 Capture Utility")
    parser.add_argument("--output", type=str, default=None, help="Output image file path")
    parser.add_argument("--status", action="store_true", help="Print camera diagnostic status")
    parser.add_argument("--autofocus", action="store_true", default=True, help="Enable autofocus")
    parser.add_argument("--predict", action="store_true", help="Immediately run TFLite prediction on captured image")
    args = parser.parse_args()

    camera = CameraManager(autofocus=args.autofocus)

    if args.status:
        status = camera.get_status()
        print("\n" + "=" * 55)
        print("CAMERA SUBSYSTEM DIAGNOSTICS")
        print("=" * 55)
        for k, v in status.items():
            print(f"{k.replace('_', ' ').title():<28}: {v}")
        print("=" * 55)
        return

    output_path, _, meta = camera.capture_image(output_filename=args.output)
    print("\n" + "=" * 55)
    print("IMAGE CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"File Path        : {output_path}")
    print(f"Backend Used     : {meta['backend_used']}")
    print(f"Resolution       : {meta['resolution'][0]}x{meta['resolution'][1]}")
    print(f"Capture Duration : {meta['capture_latency_ms']} ms")
    print("=" * 55)

    if args.predict:
        print("\n[Executing Phase 3 TFLite Prediction on Captured Image...]")
        try:
            from src.tflite_predict import run_tflite_inference
            from src.config import TFLITE_FLOAT16_PATH
            pred_class, conf, top_preds, latency = run_tflite_inference(
                model_path_str=str(TFLITE_FLOAT16_PATH),
                image_path_str=str(output_path),
            )
            print(f"Prediction : {pred_class}")
            print(f"Confidence : {conf:.2f}%")
            print(f"Latency    : {latency:.2f} ms")
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")


if __name__ == "__main__":
    main()
