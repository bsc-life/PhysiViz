# PhysiCell Web Viewer

(part of the [bsc-life/PhysiViz](https://github.com/bsc-life/PhysiViz) tool collection)

A Next.js + TypeScript 3D viewer for exploring PhysiCell-family (PhysiCell,
PhysiBoSS, PhysiCelldFBA) simulation output — cells and microenvironment over
time, in an interactive 3D scene (orbit / rotate / pan / zoom) that runs
entirely on your own machine. The viewer is model-agnostic: everything it
knows about a dataset comes from `manifest.json` + `cells.f32`, so it works
for 2D or 3D runs, any number of cell types, with or without a
PhysiBoSS/MaBoSS intracellular model.

## Quick start

Requires [Node.js](https://nodejs.org/) ≥ 20 and Python 3 (for the exporter).

```bash
git clone https://github.com/bsc-life/PhysiViz.git
cd PhysiViz/physicell_web_viewer

./scripts/install.sh                                          # 1. one-time setup
./scripts/prepare_dataset.sh /path/to/PhysiCell/output my_run # 2. export a run you have
./scripts/start.sh                                             # 3. open http://localhost:3000
```

That's it — the viewer opens with a dataset dropdown; pick `my_run`, or open
it directly at `http://localhost:3000/?dataset=my_run`.

No PhysiCell output of your own yet? `public/data/` is intentionally empty
on a fresh clone (see `public/data/README.md`) — run
`./scripts/demo_from_physicell_sample.sh` instead of step 2 above for a
complete, run-it-yourself demo: it clones, builds and runs the official,
unmodified PhysiCell `template` sample project (no custom model, nothing
BSC-specific) and exports it, so there's something to look at using only
the real pipeline. That one extra script needs a C++ toolchain (`g++`,
`make`) on top of Node/Python; the viewer and `prepare_dataset.sh` itself
never do.

Each script is standalone and explains itself with `--help` /
by reading it; `./scripts/prepare_dataset.sh --help` prints the full list of
exporter flags (PhysiBoSS node coloring, cell-type names, frame
stepping/downsampling, substrate filtering, ...).

## Features

- Interactive 3D scene: orbit/pan/zoom, playback with loop/speed control, a
  time-step slider (arrow keys / Home / End supported).
- Cell coloring by type, growth rate, death state, or any exported numeric
  column — including one-hot PhysiBoSS/MaBoSS node columns.
- **Population sparkline**: a stacked-area chart of cell-type counts (or a
  selected boolean node's active-fraction) over the whole run, with a
  draggable playhead synced to the 3D view.
- **Click-to-isolate**: click a legend swatch (cell type or alive/pre-dead/
  apoptotic/necrotic state) to hide that population in the 3D view and the
  sparkline; click again to restore, or use "show all".
- 3D slicers to peel away outer voxel/cell layers along any axis.
- PNG (with DPI scaling, rendered natively at the target resolution — not an
  upscaled screenshot) and WebM/MP4 video export. Video export
  captures one still per frame deterministically (never drops/duplicates a
  frame, however heavy the scene) and encodes them server-side in a single
  `ffmpeg` pass — **requires `ffmpeg` on `PATH`** where the dev server runs,
  for both formats.
- A dataset dropdown with per-dataset descriptions, backed by a live scan of
  your simulations folder.

## Repository layout

```
physicell_web_viewer/
├── scripts/
│   ├── install.sh                      # npm install + a Python venv with numpy/pandas/scipy
│   ├── prepare_dataset.sh              # wraps the exporter: <physicell_output_dir> <name> [flags...]
│   ├── demo_from_physicell_sample.sh   # builds+runs the official PhysiCell 'template' sample, exports it
│   └── start.sh                        # npm run dev, opens http://localhost:3000
├── app/
│   ├── page.tsx                    # entry page; sets the default dataset (dataPrefix)
│   ├── api/output-folders/route.ts # lists output* folders for the dataset dropdown
│   └── api/render-video/route.ts   # ffmpeg: encodes a captured frame sequence into MP4/WebM
├── components/
│   ├── SimulationViewer.tsx        # the whole viewer (3D scene + control panel)
│   └── PopulationSparkline.tsx     # population/node-activity chart strip
├── lib/                             # viewer types, color maps
├── public/data/                     # exported bundles -- empty on a fresh clone, see its own README.md
└── exporter/
    ├── export_crossfeeding_viz.py  # the exporter (standalone; vendors a minimal pctk reader)
    ├── backfill_population_summary.py
    └── pctk_min/                   # vendored MultiCellDS reader (BSD-3-Clause, see NOTICE.md)
```

## Manual setup (without the scripts)

```bash
npm install
python3 -m venv exporter/.venv && exporter/.venv/bin/pip install numpy pandas scipy

python3 exporter/export_crossfeeding_viz.py \
  --output-dir /path/to/PhysiCell/output \
  --out public/data/my_run

npm run dev   # http://localhost:3000
```

Exporter flags:

| Flag | Meaning |
| --- | --- |
| `--output-dir, -o` | PhysiCell output folder (contains `output*.xml` + `.mat` files) |
| `--out` | Destination directory, typically `public/data/<name>` |
| `--frame-step N` | Export every N-th snapshot (smaller bundle) |
| `--downsample K` | Mean-pool the microenvironment grid by factor K (2D: K×K; 3D: K×K×K) |
| `--skip-t0` | Omit the initial `t ≤ 0` snapshot |
| `--substrates NAME...` | Only these substrates (names must match the XML) |
| `--cell-columns KEY...` | Cell matrix columns to include in the export (default: a built-in list) |
| `--max-frames M` | Cap the number of exported frames |
| `--bnd-file PATH` | MaBoSS/PhysiBoSS `.bnd`; merges each frame's `output*_boolean_intracellular.csv` as one-hot `bnode_<NAME>` columns, selectable as `custom:bnode_<NAME>` |
| `--boolean-nodes NAME...` | Explicit node list, overrides `--bnd-file` parsing |
| `--boolean-csv-suffix SUFFIX` | Per-frame boolean-state CSV suffix (default `_boolean_intracellular.csv`) |
| `--cell-type-xml PATH` | `PhysiCell_settings.xml`; writes `cell_type_names`/`cell_type_colors` to the manifest for an N-type legend |
| `--description TEXT` | Short description shown in the dataset dropdown and control panel |

The exporter also writes `population_summary.json` (per-frame cell-type
counts + boolean-node active fractions) alongside `manifest.json` — this
powers the sparkline. For datasets exported before this feature existed, or
if you only have the already-exported `public/data/<name>/` bundle (not the
raw PhysiCell output), regenerate it cheaply with:

```bash
python3 exporter/backfill_population_summary.py public/data/<name>
```

### Data format (what the viewer reads)

```
public/data/<dataset>/
├── manifest.json
├── population_summary.json
└── frames/NNNNN/
    ├── substrates.f32   # float32, length n_substrates * ny * nx [* nz]
    └── cells.f32        # float32, flat row-major [n_cells x len(cell_columns)]
```

- `substrates.f32` is **substrate-major**, then row-major `y`, then `x` (z
  outermost for 3D volume grids).
- `cells.f32` is one row per cell, columns in the exact order of
  `manifest.json`'s `cell_columns`; a column with no value for a given cell
  (e.g. a boolean-node value on a cell type with no intracellular model) is
  `NaN`. This binary format replaced an earlier `cells.json` (array of
  `{column: value}` objects) — for large models (hundreds of thousands of
  cells) the JSON form cost a much bigger transfer (every column name
  repeated per cell) and a much slower browser-side parse; datasets
  exported before this change still load fine (`manifest.json`'s
  `cells_format` field, absent or `"json"`, tells the viewer which to
  expect).
- `manifest.json` key fields: `times_min[]`, `nx`/`ny`/`nz`,
  `microenvironment_grid`, `substrate_names[]`, `cell_columns[]`,
  `cells_format`, `frames[]`, and optionally `description`,
  `cell_type_names`/`cell_type_colors`, `boolean_node_columns`.
- Any column in `cell_columns` is selectable in the viewer as a
  `custom:<column>` color mode.

## Using the viewer

- **`PHYSICELL_ROOT`** env var (default: the parent of this folder): the
  folder the dataset dropdown scans for `output*` directories. Folders
  without an export in `public/data/` show as "(not exported)":

  ```bash
  PHYSICELL_ROOT=/path/to/simulations ./scripts/start.sh
  ```
- **Dataset dropdown + `Open`**: selecting a dataset loads it automatically;
  the `Open` button re-triggers the current selection.
- **Time step**: slider (arrow keys / Home / End), Play/Pause, Loop, 1×/2×/4×
  speed.
- **Cell color by**: `cell_type`, `growth_rate`, `state`, `dead`,
  `current_death_model`, `custom:<any exported column>`.
- **Population sparkline**: shows cell-type counts (or a selected boolean
  node's active fraction) across the whole run; drag or click to scrub.
- **Legend click-to-isolate**: click a swatch to hide that population;
  "show all" resets.
- **3D slicers**: clamp voxel/cell rendering to an index box per axis.
- **Exports**: PNG (native-resolution re-render, not an upscale), WebM/MP4
  (both need `ffmpeg` on `PATH` where the dev server runs —
  `./scripts/install.sh` checks for it).

### URL parameters

| Param | Values | Effect |
| --- | --- | --- |
| `dataset` | folder name under `public/data/` | load that dataset (overrides `dataPrefix`) |
| `frame` | index | jump to a time step |
| `substrate` | name or index | preselect the substrate |
| `cellColorBy` | mode name or any exported column | preselect cell coloring (`custom:` prefix optional) |
| `showField` / `showCells` | `0`/`1` | toggle layers |
| `minimal` | `0`/`1` | hide the control panel (for screenshots) |

## Extending to other simulations

1. **Export its output** with the same script. Substrates come from the XML.
   For PhysiBoSS boolean node values, use `--bnd-file` or `--boolean-nodes`;
   for cell-type names, use `--cell-type-xml`. Any other numeric cell column
   can be requested via `--cell-columns` and becomes a `custom:<column>`
   color mode automatically — no code changes needed.
2. **Cell type legend** is manifest-driven (`cell_type_names`/
   `cell_type_colors`); without `--cell-type-xml` it falls back to a generic
   2-color split.
3. **Naming**: the dropdown only lists folders starting with `output`, and
   datasets load from `public/data/<name>` (or `?dataset=<name>` directly).
4. **2D simulations**: the default camera is now elevated automatically for
   non-3D-volume datasets (previously required manual orbiting).

## Credits

- Built on [PhysiCell](https://physicell.org/) / [PhysiBoSS](https://sysbio-curie.github.io/physiboss/) MultiCellDS output.
- The exporter vendors a minimal MultiCellDS reader from
  [pctk](https://github.com/PhysiBoSS/pctk) (Miguel Ponce de Leon), BSD-3-Clause
  — see `exporter/pctk_min/NOTICE.md` and `exporter/pctk_min/LICENSE`.

## Known issues

- The `Open` button is now mostly redundant (dropdown auto-loads) but kept
  for re-triggering the current selection.
- Video export (both WebM and MP4) requires `ffmpeg` on `PATH` where the
  dev server runs; `./scripts/install.sh` checks for it and tells you how
  to install it if missing.
