#!/usr/bin/env bash
# ==============================================================================
# Launch Script for CottonGuard AI Farmer Web Dashboard
# Raspberry Pi 5 Edge Deployment
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Activate virtual environment if present
if [ -d "env" ]; then
    source "env/bin/activate"
fi

# Determine IP address of Raspberry Pi
IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$IP_ADDR" ]; then
    IP_ADDR="localhost"
fi

if [ -z "$PORT" ]; then
    if lsof -i :5000 >/dev/null 2>&1; then
        PORT=5001
    else
        PORT=5000
    fi
fi

echo "====================================================================="
echo " STARTING COTTONGUARD AI EDGE DASHBOARD"
echo "====================================================================="
echo " Target Hardware : Raspberry Pi 5 (Quad-core ARM Cortex-A76)"
echo " Edge Model      : Float16 TFLite MobileNetV2 (4.62 MB)"
echo " Camera Module   : Raspberry Pi Camera Module 3 (IMX708)"
echo " Advisory Engine : Generative AI & Multilingual Offline Knowledge"
echo "---------------------------------------------------------------------"
echo " Access locally on Raspberry Pi: http://localhost:${PORT}"
echo " Access from any phone/PC on Wi-Fi: http://${IP_ADDR}:${PORT}"
echo "====================================================================="

export HOST="0.0.0.0"
export PORT="${PORT}"
python3 app.py
