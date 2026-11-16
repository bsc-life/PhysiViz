#!/usr/bin/env bash
# Builds and runs the official, unmodified PhysiCell "template" sample
# project (MathCancer/PhysiCell), then exports it through this tool's own
# pipeline -- a complete, run-it-yourself demo that needs nothing
# BSC-specific: just a C++ toolchain, on top of what install.sh already
# sets up (Node.js + the Python exporter venv).
#
# Usage:
#   ./scripts/demo_from_physicell_sample.sh [dataset_name]
#
# Requires a C++17 toolchain (g++, make) and git:
#   Ubuntu/Debian:  sudo apt install build-essential git
#   macOS:          xcode-select --install
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-output_physicell_template_demo}"
SRC="$ROOT/exporter/.physicell_sample_src"
MAX_TIME_MIN="${PHYSICELL_DEMO_MAX_TIME:-300}"  # keep the run short by default

for tool in git g++ make; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: '$tool' not found. Install a C++ build toolchain first:" >&2
    echo "  Ubuntu/Debian:  sudo apt install build-essential git" >&2
    echo "  macOS:          xcode-select --install" >&2
    exit 1
  fi
done

if [[ -x "$SRC/project" ]]; then
  echo "== 1/4  Reusing existing build at $SRC/project =="
else
  echo "== 1/4  Cloning PhysiCell (MathCancer/PhysiCell, shallow) =="
  rm -rf "$SRC"
  git clone --quiet --depth 1 https://github.com/MathCancer/PhysiCell.git "$SRC"

  echo ""
  echo "== 2/4  Building the official 'template' sample project =="
  (cd "$SRC" && make template >/dev/null && make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)")
fi

cd "$SRC"
echo ""
echo "== 3/4  Running a short simulation (max_time=${MAX_TIME_MIN} min) =="
# Shorten max_time so this finishes in seconds regardless of the sample
# project's own default; portable sed-via-temp-file (avoids the GNU vs BSD
# `sed -i` flag incompatibility).
sed "s#<max_time units=\"min\">[0-9.]*</max_time>#<max_time units=\"min\">${MAX_TIME_MIN}</max_time>#" \
  config/PhysiCell_settings.xml > config/PhysiCell_settings.xml.tmp
mv config/PhysiCell_settings.xml.tmp config/PhysiCell_settings.xml
rm -rf output
mkdir -p output
./project

echo ""
echo "== 4/4  Exporting to the viewer's dataset format =="
PYTHON="$ROOT/exporter/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="python3"
DEST="$ROOT/public/data/$NAME"
"$PYTHON" "$ROOT/exporter/export_crossfeeding_viz.py" \
  --output-dir "$SRC/output" \
  --out "$DEST" \
  --cell-type-xml "$SRC/output/PhysiCell_settings.xml" \
  --description "Official PhysiCell 'template' sample project (MathCancer/PhysiCell), built and run by this script -- no custom model, no PhysiBoSS, nothing BSC-specific."

echo ""
echo "Done. Start the viewer with ./scripts/start.sh, then open:"
echo "  http://localhost:3000/?dataset=$NAME"
