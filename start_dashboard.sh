#!/usr/bin/env bash
# ==============================================================================
# Launch Script for CottonGuard AI Farmer Web Dashboard (Root Workspace)
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -z "$PORT" ]; then
    if lsof -i :5000 >/dev/null 2>&1; then
        PORT=5001
    else
        PORT=5000
    fi
fi

echo "====================================================================="
echo " STARTING COTTONGUARD AI FARMER WEB DASHBOARD"
echo "====================================================================="
echo " Open in your browser: http://localhost:${PORT}"
echo "====================================================================="

export HOST="0.0.0.0"
export PORT="${PORT}"
python3 app.py

