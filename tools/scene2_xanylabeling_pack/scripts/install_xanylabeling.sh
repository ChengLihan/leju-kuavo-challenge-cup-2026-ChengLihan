#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PACK_DIR"

echo "[INFO] Current directory: $PACK_DIR"

if [ ! -d ".venv_label" ]; then
    echo "[INFO] Creating Python virtual environment: .venv_label"
    python3 -m venv .venv_label
fi

source .venv_label/bin/activate

echo "[INFO] Upgrading pip..."
pip install -U pip

echo "[INFO] Installing X-AnyLabeling..."
pip install anylabeling

echo "[INFO] Installation finished."
echo "[INFO] Run the following command to start:"
echo "      ./scripts/start_scene2_labeling.sh"
