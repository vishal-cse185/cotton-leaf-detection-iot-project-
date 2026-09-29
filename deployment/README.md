# Raspberry Pi 5 Edge Deployment Package

**Project Title**: *IoT, DNN and Generative AI Based Cotton Leaf Disease Detection and Advisory System*  
**Target Hardware**: Raspberry Pi 5 (Quad-core ARM Cortex-A76 @ 2.4 GHz, 4GB/8GB RAM)  
**Edge Framework**: TensorFlow Lite / LiteRT Runtime (Float16 Quantized MobileNetV2)  
**Camera Hardware**: Raspberry Pi Camera Module 3 (IMX708 12 MP Autofocus)  
**Advisory Layer**: Google Gemini Generative AI + Built-in Multilingual Offline Knowledge Base  

---

## 1. Edge Architecture Overview

This standalone deployment package contains only the lightweight artifacts necessary for on-device edge inference, real-time camera capture, generative agronomic advisory, and the interactive web dashboard. It excludes heavy training scripts, training datasets, and development dependencies.

### Edge Inference & Advisory Pipeline

```text
Raspberry Pi Camera Module 3 (12 MP, Autofocus)
                │
                ▼
Image Capture (picamera2 / rpicam-still / OpenCV fallback)  [Phase 4]
                │
                ▼
Image Preprocessing (224 × 224 × 3 RGB, MobileNetV2 normalization)
                │
                ▼
Edge DNN Inference (Float16 TFLite MobileNetV2 ~4.62 MB, ~8 ms latency)  [Phase 3]
                │
                ▼
Disease Classification (5 Classes: Alternaria, Blight, Fusarium, Healthy, Verticillium)
                │
                ▼
Generative AI & Agronomic Advisory Engine (Gemini / Offline Multilingual)  [Phase 5]
                │
                ▼
CottonGuard AI Farmer Web Dashboard (Real-time UI, Voice Audio TTS, Printable Reports)
```

---

## 2. Package Contents

```text
deployment/
│
├── models/
│   ├── cotton_leaf_mobilenetv2_float16.tflite   # Float16 optimized edge model (4.62 MB)
│   ├── class_names.json                         # Serialized class index-to-name mapping
│   └── model_metadata.json                      # Complete model specifications & metrics
│
├── src/
│   ├── camera.py                                # Raspberry Pi Camera Module 3 capture & diagnostics
│   ├── advisory_engine.py                       # Generative AI & multilingual agronomic advisory
│   ├── tflite_predict.py                        # Standalone TFLite inference CLI & API helper
│   └── test_environment.py                      # Raspberry Pi diagnostic environment validation
│
├── templates/
│   └── index.html                               # Modern glassmorphism farmer web dashboard
│
├── static/
│   ├── style.css                                # Responsive dark-mode emerald styling
│   ├── app.js                                   # Web client logic, voice speech, and history
│   └── samples/                                 # 5 pre-loaded demo leaf sample images
│
├── app.py                                       # Flask web application server & REST API
├── start_dashboard.sh                           # 1-click startup script for web dashboard
├── setup_pi.sh                                  # 1-click automated setup script for Raspberry Pi OS
├── requirements.txt                             # Ultra-lightweight edge dependencies
└── README.md                                    # This setup and deployment guide
```

---

## 3. Quick Setup on Raspberry Pi 5 (SD Card Transfer)

### Method A: Automated 1-Click Setup (Recommended)

1. **Extract this deployment package** on your Raspberry Pi (e.g. `/home/pi/cotton_deployment`):
   ```bash
   unzip Cotton_Leaf_RaspberryPi_Deployment.zip -d ~/cotton_deployment
   cd ~/cotton_deployment
   ```

2. **Run the automated setup script**:
   ```bash
   bash setup_pi.sh
   ```
   *This automatically updates apt, installs required camera drivers (`python3-picamera2`, `rpicam-apps`), creates a virtual environment, installs lightweight Python dependencies, and runs environment verification.*

3. **Start the Farmer Web Dashboard**:
   ```bash
   bash start_dashboard.sh
   ```
   *The server starts on port `5000`. You can open `http://localhost:5000` directly on the Raspberry Pi, or navigate to `http://<raspberry-pi-ip>:5000` from any smartphone, tablet, or laptop connected to the same Wi-Fi network!*

---

### Method B: Manual Setup

1. **Install System Prerequisites**:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv python3-numpy python3-pil python3-picamera2 rpicam-apps
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv --system-site-packages env
   source env/bin/activate
   ```

3. **Install Requirements**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Environment**:
   ```bash
   python3 src/test_environment.py
   ```

---

## 4. CLI Execution & Diagnostics

### 1. Test Camera Module 3 Sensor
```bash
python3 src/camera.py --status
```

### 2. Capture a Leaf Photo & Predict Immediately
```bash
python3 src/camera.py --predict
```

### 3. Run Inference on Any Local Image
```bash
python3 src/tflite_predict.py --image static/samples/bacterial_sample.png
```

### 4. Query Generative AI / Offline Advisory (5 Languages Supported)
Supported languages: `en` (English), `hi` (Hindi), `mr` (Marathi), `te` (Telugu), `gu` (Gujarati).

```bash
# In Hindi
python3 src/advisory_engine.py --disease "Bacterial_Blight" --lang hi

# In Marathi
python3 src/advisory_engine.py --disease "Fusarium_Wilt" --lang mr

# In Telugu
python3 src/advisory_engine.py --disease "Alternaria_Leaf_Spot" --lang te
```

*(Optional: Set `export GEMINI_API_KEY="your-api-key"` to enable dynamic Google Gemini Generative AI synthesis. If unset or offline in the field, the system automatically uses the extensive built-in expert agronomic knowledge base).*

---

## 5. Web Dashboard Features

- **Live Camera Capture**: Real-time snapshot from Camera Module 3 with autofocus.
- **Photo Upload**: Drag-and-drop or select any leaf photo.
- **Demo Gallery**: Instant 1-click test cards for all 5 conditions.
- **Real-Time Edge Inference**: Sub-10ms classification with confidence meter & 5-class distribution bar chart.
- **Multilingual Advisory**: English, Hindi, Marathi, Telugu, and Gujarati with instant switching.
- **Voice Read-Aloud**: Audio narration of treatment advice for farmers.
- **Printable Certificate**: Clean printable diagnosis report with timestamp and prescriptions.
- **Field Scan History**: Persistent local log of all field scans.
