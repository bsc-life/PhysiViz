#!/usr/bin/env bash
# Install everything the viewer and the exporter need.
#
# Usage:
#   ./scripts/install.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== 1/2  Node.js dependencies (viewer app) =="
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js (>= 20) first: https://nodejs.org/" >&2
  exit 1
fi
npm install

echo ""
echo "== 2/2  Python dependencies (exporter) =="
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3 first." >&2
  exit 1
fi

VENV_DIR="$ROOT/exporter/.venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet numpy pandas scipy

echo ""
echo "== (optional) ffmpeg — only needed for video export (WebM/MP4) =="
if command -v ffmpeg >/dev/null 2>&1; then
  echo "Found: $(command -v ffmpeg)"
else
  echo "Not found. Video export will fail until it's installed:"
  echo "  Ubuntu/Debian:  sudo apt install ffmpeg"
  echo "  macOS:          brew install ffmpeg"
  echo "  Windows:        https://ffmpeg.org/download.html"
  echo "(Everything else — the 3D viewer, PNG/SVG export, dataset loading — works without it.)"
fi

echo ""
echo "Done."
echo "  - Export a PhysiCell run:  ./scripts/prepare_dataset.sh <physicell_output_dir> <dataset_name>"
echo "  - Start the viewer:        ./scripts/start.sh"
