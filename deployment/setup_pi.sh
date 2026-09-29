#!/usr/bin/env bash
# ==============================================================================
# Automated Setup Script for Raspberry Pi 5 Edge Deployment
# Project: IoT, DNN and Generative AI Cotton Leaf Disease Detection & Advisory
# ==============================================================================

set -e

echo "====================================================================="
echo " Setting up Cotton Leaf AI System on Raspberry Pi 5..."
echo "====================================================================="

# 1. Update package list and install system libraries for camera & python
echo "[Step 1/5] Checking and installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv python3-numpy python3-pil python3-picamera2
    # Ensure libcamera / rpicam tools are present
    if ! command -v rpicam-still &> /dev/null && ! command -v libcamera-still &> /dev/null; then
        sudo apt-get install -y rpicam-apps || sudo apt-get install -y libcamera-apps || true
    fi
fi

# 2. Setup Python Virtual Environment (handling PEP 668 externally-managed environments)
echo "[Step 2/5] Creating Python virtual environment..."
VENV_DIR="env"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
fi

# 3. Activate Virtual Environment
source "$VENV_DIR/bin/activate"

# 4. Install Python Dependencies
echo "[Step 3/5] Installing lightweight edge dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run Environment Diagnostic
echo "[Step 4/5] Running Raspberry Pi 5 Diagnostic Test..."
python3 src/test_environment.py

# 6. Check Camera Connection
echo "[Step 5/5] Checking Camera Sensor Status..."
python3 src/camera.py --status

echo "====================================================================="
echo " SETUP COMPLETED SUCCESSFULLY!"
echo " To start the Farmer Web Dashboard, run:"
echo "     bash start_dashboard.sh"
echo "====================================================================="
