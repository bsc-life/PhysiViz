#!/usr/bin/env bash
# Export a PhysiCell output folder into the compact bundle the viewer reads.
#
# Usage:
#   ./scripts/prepare_dataset.sh <physicell_output_dir> <dataset_name> [extra exporter flags...]
#
# Examples:
#   ./scripts/prepare_dataset.sh ~/PhysiCell/output my_run
#   ./scripts/prepare_dataset.sh ~/PhysiCell/output my_run --description "AMLO treatment, 2D"
#   ./scripts/prepare_dataset.sh ~/PhysiCell/output my_run \
#       --bnd-file ~/PhysiCell/config/model.bnd --cell-type-xml ~/PhysiCell/config/PhysiCell_settings.xml
#
# Run `./scripts/prepare_dataset.sh --help` to see every exporter flag
# (--frame-step, --downsample, --substrates, --cell-columns, --max-frames, ...).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  PYTHON="$ROOT/exporter/.venv/bin/python3"
  [[ -x "$PYTHON" ]] || PYTHON="python3"
  exec "$PYTHON" "$ROOT/exporter/export_crossfeeding_viz.py" --help
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <physicell_output_dir> <dataset_name> [extra exporter flags...]" >&2
  echo "       $0 --help    (full list of exporter flags)" >&2
  exit 1
fi

OUTPUT_DIR="$1"
NAME="$2"
shift 2

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: '$OUTPUT_DIR' is not a directory." >&2
  exit 1
fi

PYTHON="$ROOT/exporter/.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  echo "No exporter/.venv found — run ./scripts/install.sh first, or falling back to system python3." >&2
  PYTHON="python3"
fi

DEST="$ROOT/public/data/$NAME"
echo "Exporting '$OUTPUT_DIR' -> '$DEST' ..."
"$PYTHON" "$ROOT/exporter/export_crossfeeding_viz.py" \
  --output-dir "$OUTPUT_DIR" \
  --out "$DEST" \
  "$@"

echo ""
echo "Done. Start the viewer with ./scripts/start.sh, then open:"
echo "  http://localhost:3000/?dataset=$NAME"
