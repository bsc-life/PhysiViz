#!/usr/bin/env python3
"""
Backfill population_summary.json for crossfeeding-viz datasets that were
exported before the population-sparkline feature existed, by reading their
already-exported manifest.json + per-frame cells.json (no raw MultiCellDS
output needed).

Usage:
    python3 backfill_population_summary.py crossfeeding-viz/public/data/output_pasmc_no_amlo [more dirs...]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_crossfeeding_viz import compute_frame_population_summary  # noqa: E402


def backfill(dataset_dir: str) -> None:
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    boolean_node_columns = manifest.get("boolean_node_columns") or []
    frames_out = []
    for fmeta in manifest["frames"]:
        cells_path = os.path.join(dataset_dir, fmeta["dir"], "cells.json")
        with open(cells_path, encoding="utf-8") as f:
            cells_payload = json.load(f)["cells"]
        frames_out.append(
            {
                "index": fmeta["index"],
                "time_min": fmeta["time_min"],
                **compute_frame_population_summary(cells_payload, boolean_node_columns),
            }
        )

    population_summary = {
        "version": 1,
        "cell_type_names": manifest.get("cell_type_names"),
        "cell_type_colors": manifest.get("cell_type_colors"),
        "boolean_node_columns": boolean_node_columns or None,
        "frames": frames_out,
    }
    out_path = os.path.join(dataset_dir, "population_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(population_summary, f)
    print(f"Wrote {out_path} ({len(frames_out)} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: backfill_population_summary.py <dataset_dir> [more dirs...]")
    for d in sys.argv[1:]:
        backfill(d)
