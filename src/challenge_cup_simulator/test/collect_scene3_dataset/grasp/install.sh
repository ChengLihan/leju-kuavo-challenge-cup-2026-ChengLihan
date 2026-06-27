#!/bin/bash
# ==============================================================================
# install.sh — cache-only cleanup for Scene3 grasp module.
#
# Run inside container:  bash grasp/install.sh
# This script ONLY clears __pycache__ across the relevant source tree
# to guarantee that code changes are picked up.
# ==============================================================================
set -e
cd "$(dirname "$0")"

echo "=== Scene3 grasp cache cleanup ==="

rm -rf __pycache__ 2>/dev/null || true
rm -rf configs/__pycache__ 2>/dev/null || true

# Also clean scene2 (referenced via sys.path) and scene3 parent caches
rm -rf ../../__pycache__ 2>/dev/null || true
rm -rf ../../collect_scene2_dataset/__pycache__ 2>/dev/null || true

echo "[done] all __pycache__ cleaned"
