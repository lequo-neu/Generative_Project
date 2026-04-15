#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# IE7615 Group 8 — Demo launcher
# Activates the project venv and starts the Flask caption server.
# Usage: bash demo/run.sh   (from project root)
#        or double-click in Finder after: chmod +x demo/run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# venv may be at project root or one level up (Prj/)
if   [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    VENV="$PROJECT_ROOT/.venv"
elif [ -f "$(dirname "$PROJECT_ROOT")/.venv/bin/activate" ]; then
    VENV="$(dirname "$PROJECT_ROOT")/.venv"
else
    VENV=""
fi

echo "========================================"
echo " IE7615 Group 8 — Caption Demo"
echo " Project: $PROJECT_ROOT"
echo "========================================"

# Activate venv
if [ -f "$VENV/bin/activate" ]; then
    echo "[INFO] Activating venv: $VENV"
    source "$VENV/bin/activate"
else
    echo "[WARN] No .venv found at $VENV — using system Python"
fi

PYTHON=$(command -v python3)
echo "[INFO] Python: $PYTHON ($(python3 --version 2>&1))"

# Install flask into venv if missing
# python3 -c "import flask" 2>/dev/null || {
#     echo "[INFO] Installing flask into venv..."
#     pip install flask --quiet
# }

# Fix OMP/dill tmp file issue on macOS (Warning #179)
export TMPDIR="$HOME/.cache/tmp"
mkdir -p "$TMPDIR"

# Start server
echo "[INFO] Starting server at http://127.0.0.1:5000"
echo "[INFO] Press Ctrl+C to stop"
echo ""
cd "$PROJECT_ROOT"
python3 demo/app.py
