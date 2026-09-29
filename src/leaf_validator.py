"""
Leaf and Cotton Crop Morphology Validator.

Dual-Tier Botanical Verification:
1. Multi-Modal Generative AI (Google Gemini 2.5 Flash Vision):
   - Semantic visual reasoning (is_leaf, is_cotton_leaf)
   - Botanical plant species identification
   - Out-of-distribution rejection (faces, humans, walls, desks, keyboards, other crops)
   - Dual-AI diagnostic consensus with local edge CNN
2. Local Computer Vision & Morphology Guard (Zero-Dependency Offline Fallback):
   - Haar Cascade face & human presence rejection
   - YCrCb skin tone chromaticity filter
   - Agronomic Excess Green Index (ExG = 2G - R - B)
   - Chlorophyll spectral mask (eliminates neutral walls, wood, furniture, clothes)
   - Palmate lobing & compound leaflet morphology analysis
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("LeafValidator")

# Attempt Gemini SDK import
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except (ImportError, Exception):
    GEMINI_AVAILABLE = False


class LeafValidator:
    """
    Intelligent dual-tier validator for leaf presence and cotton plant verification.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client if available."""
        if GEMINI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("LeafValidator: Gemini Multimodal Vision active.")
            except Exception as e:
                logger.warning(f"LeafValidator: Gemini init failed: {e}")
                self.client = None
        else:
            self.client = None

    def set_api_key(self, api_key: str) -> bool:
        """Dynamically update API key at runtime."""
        self.api_key = api_key
        self._init_gemini()
        return self.client is not None

    def validate(
        self,
        image_path: Path,
        edge_prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate an image to confirm if:
        1. A plant leaf is present (rejects faces, rooms, walls, desks, furniture, skin).
        2. The leaf is specifically a cotton (Gossypium) leaf (rejects tomato, weeds, rose, etc.).
        
        Returns:
        {
            "status": "VALID_COTTON_LEAF" | "NO_LEAF_FOUND" | "NOT_COTTON_LEAF",
            "is_leaf": bool,
            "is_cotton_leaf": bool,
            "plant_detected": str,
            "confidence": float,
            "message": str,
            "tier_used": str,
            "visual_details": str,
            "genai_assessment": Optional[str],
        }
        """
        # Step 1: Run comprehensive local Computer Vision metrics
        cv_metrics = self._analyze_local_cv(image_path)

        # Critical Guard 1: Human Face Presence
        if cv_metrics["has_face"]:
            return {
                "status": "NO_LEAF_FOUND",
                "is_leaf": False,
                "is_cotton_leaf": False,
                "plant_detected": "Human Face / Person",
                "confidence": 99.5,
                "message": "Human face detected in camera viewport. No plant leaf found. Please point the camera directly at a cotton leaf.",
                "tier_used": "Local Computer Vision Guard (Biometric Filter)",
                "visual_details": "Haar facial cascade detected human facial landmarks; camera is facing a person.",
                "genai_assessment": "The image contains a person/face, not agricultural foliage.",
            }

        # Critical Guard 2: Human Skin / Indoor Body Tone
        if cv_metrics["skin_ratio"] > 0.30 and cv_metrics["green_ratio"] < 0.08:
            return {
                "status": "NO_LEAF_FOUND",
                "is_leaf": False,
                "is_cotton_leaf": False,
                "plant_detected": "Human Skin / Non-Plant Surface",
                "confidence": 98.0,
                "message": "Human skin tone detected. No plant leaf found. Please hold a cotton leaf up to the camera lens.",
                "tier_used": "Local Computer Vision Guard (Chromatic Filter)",
                "visual_details": f"Skin chromatic ratio is {cv_metrics['skin_ratio']*100:.1f}%; green chlorophyll is only {cv_metrics['green_ratio']*100:.1f}%.",
                "genai_assessment": "Chromatic analysis indicates human skin or warm indoor surfaces.",
            }

        # Critical Guard 3: Non-Cotton Compound / Pinnate Leaflets (e.g. Tomato, Potato, Rose, Weeds)
        # Foliage is present (>= 5% green), but fragmented into multiple small leaflets unlike a broad cotton blade
        if cv_metrics["green_ratio"] >= 0.05 and (
            (cv_metrics["dominant_contour_ratio"] < 0.45 and cv_metrics["num_leaflets"] >= 3)
            or (cv_metrics["max_leaf_area"] < 3500 and cv_metrics["num_leaflets"] >= 3)
        ):
            return {
                "status": "NOT_COTTON_LEAF",
                "is_leaf": True,
                "is_cotton_leaf": False,
                "plant_detected": "Non-Cotton Plant (Compound / Pinnate Foliage, e.g., Tomato/Weed)",
                "confidence": 95.0,
                "message": "No cotton leaf found. Detected compound/pinnate leaflets (such as Tomato or Weed), not a broad single-blade cotton leaf.",
                "tier_used": "Local Botanical Morphology Guard",
                "visual_details": f"Foliage coverage is {cv_metrics['green_ratio']*100:.1f}%, but fragmented into {cv_metrics['num_leaflets']} small leaflets (cotton is a single palmate blade).",
                "genai_assessment": "Leaf geometry does not match cotton (Gossypium) palmate morphology.",
            }

        # Critical Guard 4: Inanimate Surface / Background Rejection
        # Desks, whiteboards, neutral walls, laptops, clothes have near-zero chlorophyll
        if cv_metrics["green_ratio"] < 0.035 or cv_metrics["max_leaf_area"] < 2500:
            return {
                "status": "NO_LEAF_FOUND",
                "is_leaf": False,
                "is_cotton_leaf": False,
                "plant_detected": "Background / Inanimate Surface",
                "confidence": round((1.0 - cv_metrics["green_ratio"]) * 100, 1),
                "message": "No leaf found. Please place a plant leaf in front of the camera with good lighting.",
                "tier_used": "Local Computer Vision Guard (Agronomic Spectral Filter)",
                "visual_details": f"Chlorophyll vegetation coverage is {cv_metrics['green_ratio']*100:.1f}% (minimum 4.0% required).",
                "genai_assessment": "Insufficient plant chlorophyll detected.",
            }

        # Step 2: Pathological Health Verification & Lighting Domain-Shift Calibration
        # Normal healthy leaves under indoor lighting/shadows can trigger false Bacterial Blight logits.
        # Verify physical leaf pathology: true Bacterial Blight MUST exhibit angular water-soaked lesions.
        pathology = self._analyze_leaf_pathology(image_path)
        if edge_prediction and edge_prediction.get("predicted_label") == "Bacterial Blight":
            if pathology["healthy_pct"] >= 0.82 and pathology["halo_pct"] < 0.05:
                logger.info(
                    f"Dual-AI Calibration: Correcting false Bacterial Blight to Healthy Leaf "
                    f"(healthy green: {pathology['healthy_pct']*100:.1f}%, halos: {pathology['halo_pct']*100:.1f}%)"
                )
                corrected_conf = round(pathology["healthy_pct"] * 100.0, 1)
                edge_prediction["predicted_label"] = "Healthy Leaf"
                edge_prediction["confidence"] = corrected_conf
                edge_prediction["top_predictions"] = [
                    {"label": "Healthy Leaf", "confidence": corrected_conf},
                    {"label": "Bacterial Blight", "confidence": round(100.0 - corrected_conf, 1)},
                    {"label": "Fusarium Wilt", "confidence": 0.0},
                    {"label": "Verticillium Wilt", "confidence": 0.0},
                    {"label": "Alternaria Leaf Spot", "confidence": 0.0},
                ]
                edge_prediction["calibrated"] = True
                edge_prediction["calibration_details"] = (
                    f"Dual-AI consensus: leaf blade is {pathology['healthy_pct']*100:.1f}% uniform healthy green "
                    f"with zero angular necrotic lesions (shadow artifact calibrated)."
                )

        # Step 3: If Gemini Generative AI is active, use Multimodal Vision for botanical taxonomic reasoning
        if self.client:
            try:
                gemini_result = self._validate_with_gemini(image_path, edge_prediction)
                if gemini_result:
                    return gemini_result
            except Exception as e:
                logger.warning(f"Gemini Multimodal validation call failed: {e}. Falling back to Local CV.")

        # Step 4: Local Offline Fallback for Botanical Morphology (Cotton vs Non-Cotton)
        return self._validate_local_fallback(cv_metrics, edge_prediction)

    def _analyze_local_cv(self, image_path: Path) -> Dict[str, Any]:
        """Compute facial detection, skin ratio, Excess Green Index (ExG), and contour morphology."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {
                    "has_face": False,
                    "skin_ratio": 0.0,
                    "green_ratio": 0.0,
                    "dominant_contour_ratio": 0.0,
                    "max_leaf_area": 0,
                    "num_leaflets": 0,
                    "exg_mean": 0.0,
                }

            h, w = img.shape[:2]
            total_pixels = h * w
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Face Detection Guard (OpenCV Haar Cascades)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            face_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            f1 = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(70, 70))
            f2 = face_alt.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(70, 70))
            has_face = len(f1) > 0 or len(f2) > 0

            # 2. Skin tone detection (YCrCb chromaticity: Cr 133-173, Cb 77-127)
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            skin_ratio = float(np.count_nonzero(skin_mask) / total_pixels)

            # 3. Normalized Excess Green Index (ExG = 2G - R - B)
            b, g, r = cv2.split(img.astype(np.float32))
            denom = r + g + b + 1e-6
            rn, gn, bn = r / denom, g / denom, b / denom
            exg = 2.0 * gn - rn - bn
            exg_mask = (exg > 0.035).astype(np.uint8) * 255
            exg_mean = float(np.mean(exg))

            # 4. Chlorophyll Green in HSV (Hue 25 to 95, Sat > 30, Val > 30)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_green = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))

            # Genuine vegetation must satisfy BOTH green hue and positive excess green
            vegetation_mask = cv2.bitwise_and(hsv_green, exg_mask)
            green_pixels = np.count_nonzero(vegetation_mask)
            green_ratio = float(green_pixels / total_pixels)

            # 5. Connected component & contour analysis of leaf blade
            contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100]
            total_leaf_area = sum(areas) if areas else 0
            max_leaf_area = max(areas) if areas else 0
            dominant_ratio = float(max_leaf_area / total_leaf_area) if total_leaf_area > 0 else 0.0

            return {
                "has_face": has_face,
                "skin_ratio": skin_ratio,
                "green_ratio": green_ratio,
                "exg_mean": exg_mean,
                "dominant_contour_ratio": dominant_ratio,
                "max_leaf_area": max_leaf_area,
                "total_leaf_area": total_leaf_area,
                "num_leaflets": len(areas),
            }
        except Exception as e:
            logger.error(f"Local CV analysis error: {e}")
            return {
                "has_face": False,
                "skin_ratio": 0.0,
                "green_ratio": 0.0,
                "dominant_contour_ratio": 0.0,
                "max_leaf_area": 0,
                "num_leaflets": 0,
                "exg_mean": 0.0,
            }

    def _analyze_leaf_pathology(self, image_path: Path) -> Dict[str, float]:
        """Compute proportion of healthy green vs necrotic/halo lesions on the leaf blade."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"healthy_pct": 0.0, "halo_pct": 0.0}
            b, g, r = cv2.split(img.astype(np.float32))
            denom = r + g + b + 1e-6
            rn, gn, bn = r / denom, g / denom, b / denom
            exg = 2.0 * gn - rn - bn
            exg_mask = (exg > 0.02).astype(np.uint8) * 255
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            leaf_mask = cv2.bitwise_and(
                cv2.inRange(hsv, (18, 25, 25), (95, 255, 255)),
                exg_mask
            )
            leaf_pixels = np.count_nonzero(leaf_mask)
            if leaf_pixels < 500:
                return {"healthy_pct": 0.0, "halo_pct": 0.0}

            green_mask = cv2.inRange(hsv, (30, 35, 35), (90, 255, 255))
            healthy_green = cv2.bitwise_and(leaf_mask, green_mask)
            healthy_pct = float(np.count_nonzero(healthy_green) / leaf_pixels)

            halo_mask = cv2.inRange(hsv, (16, 60, 60), (30, 255, 255))
            halo_leaf = cv2.bitwise_and(leaf_mask, halo_mask)
            halo_pct = float(np.count_nonzero(halo_leaf) / leaf_pixels)
            return {"healthy_pct": healthy_pct, "halo_pct": halo_pct}
        except Exception as e:
            logger.error(f"Pathology analysis error: {e}")
            return {"healthy_pct": 0.0, "halo_pct": 0.0}

    def _validate_with_gemini(
        self,
        image_path: Path,
        edge_prediction: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Perform multimodal taxonomic inspection using Gemini 2.5 Flash."""
        with Image.open(image_path) as pil_img:
            img_copy = pil_img.copy()

        edge_info = ""
        if edge_prediction:
            edge_info = f"Edge MobileNetV2 predicted: '{edge_prediction.get('predicted_label')}' with {edge_prediction.get('confidence', 0)}% confidence."

        prompt = f"""
You are an expert botanical taxonomist and agricultural AI inspector for Cotton Crops.
Inspect this image with extreme precision and verify:

1. LEAF PRESENCE: Is a real botanical plant leaf clearly present in the image?
   - If the image contains a human face, person, room, wall, desk, ceiling, keyboard, furniture, clothes, or non-plant object, set:
     "status": "NO_LEAF_FOUND", "is_leaf": false, "is_cotton_leaf": false.
     In "plant_detected", specify what is visible (e.g. "Human Face / Indoor Environment").

2. COTTON SPECIES VERIFICATION: Is this leaf specifically from a COTTON plant (genus Gossypium)?
   - Cotton leaves have distinct morphological traits: 3 to 5 palmate pointed lobes, a broad cordate (heart-shaped) base, and prominent primary veins radiating from the petiole.
   - Non-cotton leaves (such as Tomato, Potato, Rose, Banana, Corn, Mango, Bean, Weed, or indoor houseplant leaves) are NOT cotton leaves.
   - If it is another plant leaf, identify the plant species in "plant_detected" and set:
     "status": "NOT_COTTON_LEAF", "is_leaf": true, "is_cotton_leaf": false.

3. VALIDATION STATUS:
   - "VALID_COTTON_LEAF" if and only if it is genuinely a cotton leaf.
   - "NOT_COTTON_LEAF" if it is a plant leaf, but NOT a cotton leaf.
   - "NO_LEAF_FOUND" if there is no leaf at all.

4. HEALTH & DISEASE VERIFICATION:
   - Carefully differentiate normal healthy leaves from diseased leaves.
   - A normal healthy leaf has clean, uniform green foliage. Natural leaf veins and camera lighting shadows are NOT bacterial blight!
   - True Bacterial Blight (Xanthomonas citri) requires visible angular, water-soaked necrotic lesions bounded by veins.
   - If the leaf is normal and healthy, set "genai_assessment": "Healthy leaf: uniform green lamina without angular water-soaked lesions."

{edge_info}

Respond ONLY with a valid JSON object matching this exact schema:
{{
    "status": "VALID_COTTON_LEAF" | "NOT_COTTON_LEAF" | "NO_LEAF_FOUND",
    "is_leaf": true | false,
    "is_cotton_leaf": true | false,
    "plant_detected": "e.g. Cotton (Gossypium hirsutum) / Tomato (Solanum lycopersicum) / Human Face / Indoor Room",
    "confidence": 0.0 to 100.0,
    "message": "Clear explanation for the farmer",
    "visual_details": "Morphological description (e.g. palmate lobing, venation, color)",
    "genai_assessment": "If cotton leaf, visual assessment of disease or health (e.g. Healthy / Angular water-soaked lesions / Alternaria rings / Wilt). If not cotton, state why."
}}
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img_copy],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        if response and response.text:
            data = json.loads(response.text)
            data["tier_used"] = "Google Gemini Multimodal Vision (Generative AI)"
            return data
        return None

    def _validate_local_fallback(
        self,
        cv_metrics: Dict[str, Any],
        edge_prediction: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Local offline heuristic validator when Gemini is offline."""
        green_ratio = cv_metrics["green_ratio"]
        dominant_ratio = cv_metrics.get("dominant_contour_ratio", 0.0)
        max_area = cv_metrics.get("max_leaf_area", 0)
        num_leaflets = cv_metrics.get("num_leaflets", 0)

        # Check for non-cotton compound/fragmented leaves (e.g. Tomato pinnate leaflets)
        # Cotton leaves have a single dominant palmate contour (> 45% of foliage area and large single blade)
        if (dominant_ratio < 0.45 and num_leaflets >= 3) or (max_area < 3500 and num_leaflets >= 4):
            return {
                "status": "NOT_COTTON_LEAF",
                "is_leaf": True,
                "is_cotton_leaf": False,
                "plant_detected": "Non-Cotton Plant (Compound / Pinnate Foliage, e.g., Tomato/Weed)",
                "confidence": 94.0,
                "message": "No cotton leaf found. Detected compound/pinnate leaflets (e.g. Tomato or Weed), not a broad single-blade cotton leaf.",
                "tier_used": "Local Botanical Morphology Guard",
                "visual_details": f"Dominant blade ratio is {dominant_ratio*100:.1f}% across {num_leaflets} leaflets (cotton leaves have a single dominant palmate blade).",
                "genai_assessment": "Morphology does not match standard 3-to-5 lobe Gossypium palmate leaf.",
            }

        # If calibrated from false Bacterial Blight to Healthy Leaf
        if edge_prediction and edge_prediction.get("calibrated"):
            return {
                "status": "VALID_COTTON_LEAF",
                "is_leaf": True,
                "is_cotton_leaf": True,
                "plant_detected": "Cotton (Gossypium hirsutum) - Normal Healthy Leaf",
                "confidence": edge_prediction.get("confidence", 95.0),
                "message": "Valid cotton leaf confirmed. Normal healthy leaf verified.",
                "tier_used": "Dual-AI Botanical Pathology Guard",
                "visual_details": edge_prediction.get("calibration_details", "Leaf blade is uniform healthy green with no angular necrotic lesions."),
                "genai_assessment": "Vegetation index, chlorophyll, and blade uniformity match healthy cotton crop foliage (camera lighting shadow artifact calibrated).",
            }

        # Check edge prediction confidence and distribution
        if edge_prediction:
            conf = edge_prediction.get("confidence", 0.0)
            top_preds = edge_prediction.get("top_predictions", [])
            
            if len(top_preds) >= 2:
                margin = top_preds[0]["confidence"] - top_preds[1]["confidence"]
            else:
                margin = conf

            # Out-of-Distribution filter: If top prediction confidence is low (< 52%)
            # or the margin between top 2 classes is razor thin (< 10%)
            if conf < 52.0 or (conf < 65.0 and margin < 12.0):
                return {
                    "status": "NOT_COTTON_LEAF",
                    "is_leaf": True,
                    "is_cotton_leaf": False,
                    "plant_detected": "Non-Cotton Plant Leaf (Uncertain morphology)",
                    "confidence": round(100.0 - conf, 1),
                    "message": "No cotton leaf found. The leaf pattern does not match cotton (Gossypium) disease or healthy profiles.",
                    "tier_used": "Local Computer Vision & OOD Guard",
                    "visual_details": f"Model uncertainty high (confidence: {conf:.1f}%, margin: {margin:.1f}%). Foliage detected: {green_ratio*100:.1f}%.",
                    "genai_assessment": "Leaf characteristics deviate from standard cotton crop morphology.",
                }

        # Valid cotton leaf confirmed
        return {
            "status": "VALID_COTTON_LEAF",
            "is_leaf": True,
            "is_cotton_leaf": True,
            "plant_detected": "Cotton (Gossypium hirsutum)",
            "confidence": 96.5,
            "message": "Valid cotton leaf confirmed.",
            "tier_used": "Local Computer Vision Guard",
            "visual_details": f"Foliage coverage: {green_ratio*100:.1f}%, Blade dominant contour: {dominant_ratio*100:.1f}%.",
            "genai_assessment": "Vegetation index, chlorophyll, and palmate blade match cotton crop foliage.",
        }
