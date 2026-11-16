#!/usr/bin/env bash
# Start the local viewer server.
#
# Usage:
#   ./scripts/start.sh
#   PHYSICELL_ROOT=/path/to/simulations ./scripts/start.sh   # also list un-exported
#                                                             # PhysiCell output* folders
#                                                             # in the dataset dropdown
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d node_modules ]]; then
  echo "node_modules/ not found — run ./scripts/install.sh first." >&2
  exit 1
fi

URL="http://localhost:3000"
echo "Starting PhysiCell Web Viewer at $URL ..."
echo "(Ctrl+C to stop)"

# Best-effort auto-open once the server is likely up; never fails the script.
(
  sleep 3
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 || true
  fi
) &

exec npm run dev
