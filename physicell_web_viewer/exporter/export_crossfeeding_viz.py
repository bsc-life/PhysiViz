#!/usr/bin/env python3
"""
Export PhysiCell MultiCellDS output for the PhysiViz web viewer.

Uses a vendored minimal copy of pctk's MultiCellDS reader (pctk_min/).
Writes manifest.json + per-frame substrates.f32 and cells.f32 (both float32 LE
binary; cells.f32 is a flat row-major [n_cells x len(cell_columns)] matrix,
NaN for any column not present on a given cell -- see manifest.json's
cell_columns for column order).

Usage:
    python export_crossfeeding_viz.py --output-dir /path/to/PhysiCell/output \\
        --out ../public/data/my_run
    python export_crossfeeding_viz.py -o /path/to/PhysiCell/output \\
        --out ../public/data/my_run --frame-step 2 --downsample 2 --skip-t0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# This script ships with a minimal, vendored copy of the two pctk modules it
# actually needs (pctk_min/multicellds.py + config.py) so the exporter has no
# dependency on the full pctk package or any particular repo layout.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pctk_min import multicellds  # noqa: E402


# Tab20-style palette (20 visually distinct colors); for models with more
# than 20 cell types we fall back to an HSV sweep so colors keep spreading
# out instead of repeating (see _color_for_index).
_TYPE_PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]


def _color_for_index(i: int, n_total: int) -> str:
    """Pick a color for the i-th (of n_total) cell type.

    Uses the fixed 20-color palette directly when it covers n_total; beyond
    that, cycles an HSV sweep (golden-angle hue steps) so colors for >20
    cell types still spread out instead of silently repeating in pairs.
    """
    if n_total <= len(_TYPE_PALETTE):
        return _TYPE_PALETTE[i % len(_TYPE_PALETTE)]
    import colorsys

    hue = (i * 0.6180339887498949) % 1.0  # golden ratio conjugate
    sat = 0.55 + 0.25 * ((i // 1) % 2)
    val = 0.85 if (i % 2 == 0) else 0.65
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

# Prefix used for one-hot PhysiBoSS/MaBoSS node columns merged into cells.json,
# so they can never collide with an existing cell_columns / custom_data name
# (e.g. a model can have both a continuous "CREC_1" custom_data level AND a
# boolean node named "CREC_1" -- these must stay distinguishable).
BOOLEAN_NODE_COLUMN_PREFIX = "bnode_"


def parse_bnd_node_names(bnd_path: str) -> list[str]:
    """Extract MaBoSS/PhysiBoSS node names from a .bnd file (lines 'node NAME {').

    Order follows first appearance in the file.
    """
    text = open(bnd_path, encoding="utf-8").read()
    # MaBoSS .bnd node declarations are lowercase "node" followed by the name.
    return re.findall(r"^\s*node\s+([A-Za-z0-9_]+)", text, flags=re.M)


def parse_cell_type_names(settings_xml_path: str) -> dict[int, str]:
    """Extract {cell_definition ID: name} from a PhysiCell_settings.xml.

    Uses a regex over the raw text rather than ElementTree: hand-edited
    PhysiCell_settings.xml files commonly contain "--" inside XML comments
    (e.g. flags like "--smc-diameter"), which is invalid XML and makes
    strict parsers choke even though the document is otherwise fine.
    """
    text = open(settings_xml_path, encoding="utf-8").read()
    mapping: dict[int, str] = {}
    for m in re.finditer(
        r'<cell_definition\s+name="([^"]+)"\s+ID="(\d+)"', text
    ):
        name, cid = m.group(1), m.group(2)
        mapping[int(cid)] = name
    return mapping


def compute_frame_population_summary(
    cells_payload: list[dict],
    boolean_node_columns: list[str] | None = None,
) -> dict:
    """Tally one frame's cells.json into compact per-frame population stats.

    Returns {n_cells, dead_count, counts_by_type: {"<cell_type>": n, ...},
    node_active_fraction: {"<bnode_NAME>": frac_active_among_modeled_cells, ...}}.
    A node's fraction is computed only over cells that actually carry that
    boolean-node key (i.e. cells with an intracellular model) so cell types
    without one don't dilute the fraction; it is omitted (not zero) if no
    cell in the frame carries that key.
    """
    counts_by_type: dict[str, int] = {}
    dead_count = 0
    node_names = boolean_node_columns or []
    node_totals = {n: 0 for n in node_names}
    node_active = {n: 0 for n in node_names}

    for row in cells_payload:
        ct = row.get("cell_type")
        if ct is not None:
            key = str(int(ct))
            counts_by_type[key] = counts_by_type.get(key, 0) + 1
        if float(row.get("dead", 0.0) or 0.0) > 0.5:
            dead_count += 1
        for n in node_names:
            if n in row:
                node_totals[n] += 1
                if row[n] >= 0.5:
                    node_active[n] += 1

    node_active_fraction = {
        n: (node_active[n] / node_totals[n]) for n in node_names if node_totals[n] > 0
    }

    return {
        "n_cells": len(cells_payload),
        "dead_count": dead_count,
        "counts_by_type": counts_by_type,
        "node_active_fraction": node_active_fraction,
    }


def load_boolean_states(csv_path: str, node_names: list[str]) -> dict[int, dict[str, float]]:
    """Parse a PhysiBoSS 'output*_boolean_intracellular.csv' (columns ID, state),
    where `state` is a ' -- '-joined list of the currently-ACTIVE node names.

    Returns {cell_ID: {node_name: 0.0/1.0 for every node in node_names}}.
    Cells not present in the CSV (e.g. a cell type without an intracellular
    model) are simply absent from the returned dict -- callers should leave
    their boolean-node columns unset rather than defaulting them to 0, so the
    viewer renders them as "not applicable" (gray) instead of "node off".
    """
    if not node_names or not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns or "state" not in df.columns:
        return {}
    out: dict[int, dict[str, float]] = {}
    for _, row in df.iterrows():
        cid = int(row["ID"])
        state = row["state"]
        active = set()
        if isinstance(state, str) and state.strip():
            active = {tok.strip() for tok in state.split("--")}
        out[cid] = {n: (1.0 if n in active else 0.0) for n in node_names}
    return out


def _reshape_to_grid(x: np.ndarray, y: np.ndarray, v: np.ndarray):
    """Reshape scattered (x,y,v) to regular Z[ny, nx] (same as plot_substrate_concentration_gif)."""
    xu = np.unique(x)
    yu = np.unique(y)
    nx, ny = xu.size, yu.size
    if nx * ny != v.size:
        raise ValueError(f"Cannot reshape: expected {nx * ny} points, got {v.size}")
    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)
    z = np.full((ny, nx), np.nan, dtype=np.float64)
    z[iy, ix] = v
    return xu, yu, z.astype(np.float32)


def _block_downsample(z: np.ndarray, k: int) -> np.ndarray:
    """Mean-pool k×k blocks; crop to multiple of k."""
    if k <= 1:
        return z
    ny, nx = z.shape
    ny2, nx2 = (ny // k) * k, (nx // k) * k
    z = z[:ny2, :nx2]
    return z.reshape(ny2 // k, k, nx2 // k, k).mean(axis=(1, 3)).astype(np.float32)


def _block_downsample_3d(vol: np.ndarray, k: int) -> np.ndarray:
    """Mean-pool k×k×k blocks; crop to multiples of k. vol shape (nz, ny, nx)."""
    if k <= 1:
        return vol.astype(np.float32)
    nz, ny, nx = vol.shape
    nz2, ny2, nx2 = (nz // k) * k, (ny // k) * k, (nx // k) * k
    v = vol[:nz2, :ny2, :nx2]
    return v.reshape(
        nz2 // k, k, ny2 // k, k, nx2 // k, k
    ).mean(axis=(1, 3, 5)).astype(np.float32)


def _substrate_layers_2d_from_matrix(
    A: np.ndarray,
    substrate_indices: list[int],
    row_offset: int,
    downsample: int,
) -> tuple[list[np.ndarray], dict]:
    """
    Build substrate layers for the web viewer.

    Supports 2D microenvironment (nx·ny voxels) and 3D Cartesian (nx·ny·nz) as a
    full volume (BioFVM column order: x fastest, then y, then z).
    """
    x_all = A[0, :].astype(np.float64)
    y_all = A[1, :].astype(np.float64)
    z_all = A[2, :].astype(np.float64)
    xu = np.unique(x_all)
    yu = np.unique(y_all)
    zu = np.unique(z_all)
    nx, ny, nz = int(xu.size), int(yu.size), int(zu.size)
    n = int(A.shape[1])

    meta: dict = {"microenvironment_grid": "2d"}

    if nx * ny == n:

        def slice_and_downsample(v: np.ndarray) -> np.ndarray:
            _xu, _yu, z = _reshape_to_grid(x_all, y_all, v)
            return _block_downsample(z, downsample)

    elif nx * ny * nz == n:
        meta = {
            "microenvironment_grid": "3d_volume",
            "z_min": float(zu.min()),
            "z_max": float(zu.max()),
        }

        def slice_and_downsample(v: np.ndarray) -> np.ndarray:
            vol = v.reshape((nz, ny, nx))
            return _block_downsample_3d(vol, downsample)

    else:
        raise ValueError(
            f"Cannot map microenvironment: nx={nx}, ny={ny}, nz={nz}, n_voxels={n} "
            f"(expected {nx * ny} for 2D or {nx * ny * nz} for 3D)"
        )

    layers: list[np.ndarray] = []
    for si in substrate_indices:
        row_idx = si + row_offset
        if row_idx >= A.shape[0]:
            continue
        v = A[row_idx, :].astype(np.float64)
        layers.append(slice_and_downsample(v))

    return layers, meta


DEFAULT_CELL_KEYS = [
    "ID",
    "x_position",
    "y_position",
    "z_position",
    "total_volume",
    "radius",
    "cell_type",
    "apoptotic",
    "necrotic",
    "dead",
    "current_death_model",
    "x_death_rates",
    "y_death_rates",
    "death_rate",
    "CB_death_rate",
    "MB_death_rate",
    "x_velocity",
    "y_velocity",
    "z_velocity",
    "migration_speed",
    "pressure",
    "CB_growth_rate",
    "MB_growth_rate",
    "CB_fba_infeasible",
    "MB_fba_infeasible",
    "glucose_flux",
    "CB_h2_flux",
    "CB_co2_flux",
    "CB_acetate_flux",
    "MB_h2_flux",
    "MB_co2_flux",
    "MB_acetate_flux",
    "methane_flux",
]


def export_viz(
    output_dir: str,
    out_dir: str,
    frame_step: int = 1,
    downsample: int = 1,
    skip_t0: bool = False,
    substrates: list[str] | None = None,
    cell_keys: list[str] | None = None,
    max_frames: int | None = None,
    bnd_file: str | None = None,
    boolean_nodes: list[str] | None = None,
    boolean_csv_suffix: str = "_boolean_intracellular.csv",
    cell_type_xml: str | None = None,
    description: str | None = None,
) -> None:
    if not os.path.isdir(output_dir):
        raise SystemExit(f"Output directory not found: {output_dir}")

    # --- Optional PhysiBoSS/MaBoSS node list (for merging boolean_intracellular.csv) ---
    node_names: list[str] = []
    if boolean_nodes:
        node_names = list(boolean_nodes)
    elif bnd_file:
        if not os.path.isfile(bnd_file):
            raise SystemExit(f".bnd file not found: {bnd_file}")
        node_names = parse_bnd_node_names(bnd_file)
        print(f"Parsed {len(node_names)} PhysiBoSS/MaBoSS nodes from {bnd_file}")

    # --- Optional cell-type ID -> name mapping (for manifest.cell_type_names) ---
    cell_type_names: dict[int, str] = {}
    if cell_type_xml:
        if not os.path.isfile(cell_type_xml):
            raise SystemExit(f"Cell-type XML not found: {cell_type_xml}")
        cell_type_names = parse_cell_type_names(cell_type_xml)
        print(f"Parsed cell types from {cell_type_xml}: {cell_type_names}")

    reader = multicellds.MultiCellDS(output_folder=output_dir)
    substrate_info = reader.microenvironment_columns
    substrate_names_all = [name for (name, _units, _ID) in substrate_info]

    if substrates:
        resolved = []
        for s in substrates:
            for name in substrate_names_all:
                if name.lower() == s.lower():
                    resolved.append(name)
                    break
            else:
                print(f"Warning: substrate '{s}' not found; available: {substrate_names_all}")
        if not resolved:
            raise SystemExit("No valid substrates after filtering.")
        substrate_names = resolved
    else:
        substrate_names = list(substrate_names_all)

    try:
        substrate_indices = [substrate_names_all.index(n) for n in substrate_names]
    except ValueError as e:
        raise SystemExit(f"Substrate name mismatch: {e}") from e
    row_offset = 4

    cell_cols = list(reader.cell_columns)
    keys = cell_keys if cell_keys else DEFAULT_CELL_KEYS
    cell_column_keys = [k for k in keys if k in cell_cols]
    missing = [k for k in keys if k not in cell_cols]
    if missing:
        print(f"Note: skipping cell columns not in output schema: {missing}")
    cell_indices = [cell_cols.index(k) for k in cell_column_keys]

    # Need the raw cell ID per row to merge in boolean node states, even if the
    # caller didn't request "ID" as one of the exported cell_keys.
    id_col_idx = cell_cols.index("ID") if "ID" in cell_cols else None
    if node_names and id_col_idx is None:
        print("Warning: no 'ID' column in cell schema; cannot merge boolean node states.")
        node_names = []
    boolean_node_columns = [f"{BOOLEAN_NODE_COLUMN_PREFIX}{n}" for n in node_names]
    all_cell_columns = cell_column_keys + boolean_node_columns
    cell_col_index = {k: i for i, k in enumerate(all_cell_columns)}

    # xml_list drives the boolean-CSV filenames; it uses the exact same glob +
    # sort order that MultiCellDS.cells_as_matrix_iterator() uses internally,
    # so index idx below lines up with cell_iter[idx] / micro_iter[idx].
    xml_list = sorted(glob.glob(os.path.join(output_dir, "output*.xml")))

    micro_iter = list(reader.microenvironment_as_matrix_iterator())
    cell_iter = list(reader.cells_as_matrix_iterator())
    if len(micro_iter) != len(cell_iter):
        print(
            f"Warning: microenvironment frames ({len(micro_iter)}) != cell frames ({len(cell_iter)}); "
            "pairing by min length."
        )
    n_total = min(len(micro_iter), len(cell_iter))

    frames_meta = []
    population_summary_frames: list[dict] = []
    os.makedirs(out_dir, exist_ok=True)
    frames_root = os.path.join(out_dir, "frames")
    os.makedirs(frames_root, exist_ok=True)

    nx = ny = None
    x_min = x_max = y_min = y_max = None
    exported = 0
    grid_meta_out: dict | None = None
    substrates_order_final = "substrate_major_then_y_then_x"
    nz_out = 1
    substrate_alpha_max: list[float] | None = None

    for idx in range(n_total):
        if max_frames is not None and exported >= max_frames:
            break
        if frame_step > 1 and idx % frame_step != 0:
            continue

        t_micro, A = micro_iter[idx]
        t_cell, cells_mat = cell_iter[idx]
        if skip_t0 and t_micro <= 0:
            continue

        if A is None:
            print(f"Skipping frame {idx}: no microenvironment matrix")
            continue

        x_all = A[0, :].astype(np.float64)
        y_all = A[1, :].astype(np.float64)

        try:
            layers, grid_meta = _substrate_layers_2d_from_matrix(
                A, substrate_indices, row_offset, downsample
            )
        except ValueError as e:
            raise SystemExit(f"Frame {idx}: {e}") from e

        if not layers:
            raise SystemExit("No substrate layers exported (check matrix shape).")

        if grid_meta_out is None:
            grid_meta_out = grid_meta

        stack = np.stack(layers, axis=0)
        if stack.ndim == 4:
            _S, nz_out, ny, nx = stack.shape
            flat = np.ascontiguousarray(stack.reshape(-1), dtype=np.float32)
            substrates_order_final = "substrate_major_then_z_then_y_then_x"
        elif stack.ndim == 3:
            _S, ny, nx = stack.shape
            nz_out = 1
            flat = np.ascontiguousarray(stack.reshape(-1), dtype=np.float32)
            substrates_order_final = "substrate_major_then_y_then_x"
        else:
            raise SystemExit(f"Unexpected stack shape {stack.shape}")

        n_sub = int(stack.shape[0])
        if substrate_alpha_max is None:
            substrate_alpha_max = [0.0] * n_sub
        elif len(substrate_alpha_max) != n_sub:
            raise SystemExit(
                "Number of substrates changed between time steps; cannot build substrate_alpha_ref_max."
            )
        for si in range(n_sub):
            mx = float(np.max(stack[si]))
            if mx > substrate_alpha_max[si]:
                substrate_alpha_max[si] = mx

        if x_min is None:
            x_min = float(np.min(x_all))
            x_max = float(np.max(x_all))
            y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))

        frame_name = f"{exported:05d}"
        frame_dir = os.path.join(frames_root, frame_name)
        os.makedirs(frame_dir, exist_ok=True)
        substrate_path = os.path.join(frame_dir, "substrates.f32")
        flat.tofile(substrate_path)

        bool_states: dict[int, dict[str, float]] = {}
        if node_names and idx < len(xml_list):
            boolean_csv_path = xml_list[idx][: -len(".xml")] + boolean_csv_suffix
            bool_states = load_boolean_states(boolean_csv_path, node_names)

        cells_payload = []
        if (
            cells_mat.size > 0
            and cell_indices
            and cells_mat.shape[1] > max(cell_indices)
        ):
            for r in range(cells_mat.shape[0]):
                row = {}
                for k, j in zip(cell_column_keys, cell_indices):
                    val = cells_mat[r, j]
                    row[k] = float(np.asarray(val).item())
                if bool_states and id_col_idx is not None:
                    cid = int(np.asarray(cells_mat[r, id_col_idx]).item())
                    states_for_cell = bool_states.get(cid)
                    if states_for_cell is not None:
                        for n in node_names:
                            row[f"{BOOLEAN_NODE_COLUMN_PREFIX}{n}"] = states_for_cell[n]
                    # else: leave boolean-node keys unset for this cell (e.g. a
                    # cell type with no intracellular model) -> renders as N/A.
                cells_payload.append(row)

        # Binary cell frame format: a flat row-major float32 matrix, one row
        # per cell, columns in the fixed `all_cell_columns` order (same order
        # written to manifest.json's cell_columns). This replaces the old
        # array-of-keyed-objects cells.json, which for large agent-based
        # models (hundreds of thousands of cells) cost both a much bigger
        # transfer (text repeats every key name per cell) and a much slower
        # client-side JSON.parse -- measured ~6x smaller and faster to decode
        # on a 261k-cell frame. A column absent for a given cell (e.g. a
        # boolean-node value when that cell has no intracellular model) is
        # written as NaN, which the viewer already treats as "missing" via
        # Number.isFinite() guards everywhere a cell field is read.
        n_cells = len(cells_payload)
        cell_mat = np.full((n_cells, len(all_cell_columns)), np.nan, dtype=np.float32)
        for r, row in enumerate(cells_payload):
            for k, v in row.items():
                ci = cell_col_index.get(k)
                if ci is not None:
                    cell_mat[r, ci] = v
        cells_path = os.path.join(frame_dir, "cells.f32")
        cell_mat.tofile(cells_path)

        frames_meta.append(
            {
                "index": exported,
                "dir": f"frames/{frame_name}",
                "time_min": float(t_micro),
                "time_cell_min": float(t_cell),
            }
        )
        population_summary_frames.append(
            {
                "index": exported,
                "time_min": float(t_micro),
                **compute_frame_population_summary(cells_payload, boolean_node_columns),
            }
        )
        exported += 1

    manifest = {
        "version": 1,
        "times_min": [f["time_min"] for f in frames_meta],
        "nx": int(nx) if nx is not None else 0,
        "ny": int(ny) if ny is not None else 0,
        **(grid_meta_out or {}),
        "n_substrates": len(substrate_names),
        "substrate_names": substrate_names,
        "downsample": int(downsample),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "spatial_units": "micron",
        "cell_columns": all_cell_columns,
        "frames": frames_meta,
        "substrates_file": "substrates.f32",
        "cells_file": "cells.f32",
        "cells_format": "binary_f32",
        "substrates_dtype": "float32",
        "substrates_order": substrates_order_final,
    }
    if substrate_alpha_max is not None:
        manifest["substrate_alpha_ref_max"] = substrate_alpha_max
    if (grid_meta_out or {}).get("microenvironment_grid") == "3d_volume":
        manifest["nz"] = int(nz_out)
    if boolean_node_columns:
        manifest["boolean_node_columns"] = boolean_node_columns
        manifest["boolean_node_column_prefix"] = BOOLEAN_NODE_COLUMN_PREFIX
    if cell_type_names:
        # JSON object keys must be strings; the viewer looks these up as
        # cell_type_names[String(cell_type)].
        manifest["cell_type_names"] = {str(k): v for k, v in sorted(cell_type_names.items())}
        _sorted_type_keys = sorted(cell_type_names.keys())
        manifest["cell_type_colors"] = {
            str(k): _color_for_index(i, len(_sorted_type_keys))
            for i, k in enumerate(_sorted_type_keys)
        }
    if description:
        manifest["description"] = description

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    population_summary = {
        "version": 1,
        "cell_type_names": manifest.get("cell_type_names"),
        "cell_type_colors": manifest.get("cell_type_colors"),
        "boolean_node_columns": boolean_node_columns or None,
        "frames": population_summary_frames,
    }
    with open(os.path.join(out_dir, "population_summary.json"), "w", encoding="utf-8") as f:
        json.dump(population_summary, f)

    print(f"Wrote {exported} frames to {out_dir}")
    print(f"Manifest: {os.path.join(out_dir, 'manifest.json')}")
    print(f"Population summary: {os.path.join(out_dir, 'population_summary.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MultiCellDS for crossfeeding-viz web app")
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="PhysiCell output folder (contains output*.xml and .mat files)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination directory (e.g. crossfeeding-viz/public/data/exp2)",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Export every N-th snapshot (default 1)")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Mean-pool spatial grid by this factor after reshape (default 1)",
    )
    parser.add_argument("--skip-t0", action="store_true", help="Skip snapshot at t<=0")
    parser.add_argument("--substrates", nargs="*", help="Subset of substrate names (default: all)")
    parser.add_argument(
        "--cell-columns",
        nargs="*",
        help="Cell matrix columns to include in cells.json (default: built-in list)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many exported frames")
    parser.add_argument(
        "--bnd-file",
        help=(
            "Path to a MaBoSS/PhysiBoSS .bnd file; node names are auto-parsed and merged "
            "from each frame's 'output*_boolean_intracellular.csv' (ID,state columns, "
            "state = ' -- '-joined list of ACTIVE node names) into cells.json as "
            "0/1 'bnode_<NAME>' columns, selectable in the viewer as custom:bnode_<NAME>."
        ),
    )
    parser.add_argument(
        "--boolean-nodes",
        nargs="*",
        help="Explicit list of PhysiBoSS node names (overrides --bnd-file parsing)",
    )
    parser.add_argument(
        "--boolean-csv-suffix",
        default="_boolean_intracellular.csv",
        help="Suffix appended to each 'outputNNNNNNNN' stem to find its boolean-state CSV",
    )
    parser.add_argument(
        "--cell-type-xml",
        help=(
            "Path to a PhysiCell_settings.xml; <cell_definitions> name/ID pairs are "
            "written to manifest.json as cell_type_names/cell_type_colors so the "
            "viewer's cell_type legend shows real names for any number of types."
        ),
    )
    parser.add_argument(
        "--description",
        help=(
            "Short human-readable description of this run, written to manifest.json "
            "and shown in the dataset dropdown / control panel (e.g. 'No-AMLO control, "
            "180 days, 2D monolayer')."
        ),
    )
    args = parser.parse_args()

    export_viz(
        output_dir=os.path.abspath(args.output_dir),
        out_dir=os.path.abspath(args.out),
        frame_step=args.frame_step,
        downsample=args.downsample,
        skip_t0=args.skip_t0,
        substrates=args.substrates if args.substrates else None,
        cell_keys=args.cell_columns if args.cell_columns else None,
        max_frames=args.max_frames,
        bnd_file=os.path.abspath(args.bnd_file) if args.bnd_file else None,
        boolean_nodes=args.boolean_nodes if args.boolean_nodes else None,
        boolean_csv_suffix=args.boolean_csv_suffix,
        cell_type_xml=os.path.abspath(args.cell_type_xml) if args.cell_type_xml else None,
        description=args.description,
    )


if __name__ == "__main__":
    main()
