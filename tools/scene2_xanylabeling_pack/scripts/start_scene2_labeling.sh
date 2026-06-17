#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PACK_DIR"

if [ ! -d ".venv_label" ]; then
    echo "[ERROR] .venv_label not found."
    echo "[ERROR] Please run: ./scripts/install_xanylabeling.sh"
    exit 1
fi

source .venv_label/bin/activate

echo "[INFO] Starting X-AnyLabeling..."
echo "[INFO] Label config:"
echo "       $PACK_DIR/configs/scene2_labels.txt"

anylabeling \
  --labels "$PACK_DIR/configs/scene2_labels.txt" \
  --validatelabel exact
