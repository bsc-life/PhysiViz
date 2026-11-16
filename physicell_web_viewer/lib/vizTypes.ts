/** manifest.json written by analysis/export_crossfeeding_viz.py */

export type VizFrameRef = {
  index: number;
  dir: string;
  time_min: number;
  time_cell_min: number;
};

export type VizManifest = {
  version: number;
  times_min: number[];
  nx: number;
  ny: number;
  n_substrates: number;
  substrate_names: string[];
  /** Per-substrate max concentration over all exported frames; voxel alpha = value / this. */
  substrate_alpha_ref_max?: number[];
  downsample: number;
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  spatial_units: string;
  cell_columns: string[];
  frames: VizFrameRef[];
  substrates_file: string;
  cells_file: string;
  substrates_dtype: string;
  substrates_order: string;
  /** Legacy: one xy slice from a 3D grid. */
  microenvironment_grid?: "2d" | "3d_xy_slice" | "3d_volume";
  /** Source nz (legacy slice exports only). */
  microenvironment_nz?: number;
  z_slice_index?: number;
  z_slice_micron?: number;
  /** Exported z-depth for full 3D voxel dumps. */
  nz?: number;
  z_min?: number;
  z_max?: number;
  /** cell_type (numeric, as string key) -> display name, from --cell-type-xml. */
  cell_type_names?: Record<string, string>;
  /** cell_type (numeric, as string key) -> hex color, from --cell-type-xml. */
  cell_type_colors?: Record<string, string>;
  /** Names of one-hot PhysiBoSS/MaBoSS node columns merged by --bnd-file, e.g. "bnode_Nfat". */
  boolean_node_columns?: string[];
  boolean_node_column_prefix?: string;
  /** Short human-readable run description, from --description. */
  description?: string;
  /**
   * Per-frame cell data wire format. "binary_f32" (current exporter default)
   * is a flat row-major float32 matrix [n_cells x cell_columns.length],
   * column order given by cell_columns, NaN where a column is absent for a
   * given cell; "json" (legacy datasets exported before this field existed)
   * is the old {cells: [{col: value, ...}, ...]} array-of-objects shape.
   * Absent means "json" for backward compatibility with older exports.
   */
  cells_format?: "json" | "binary_f32";
};

export type CellsFile = {
  cells: Record<string, number>[];
};

/** population_summary.json written by analysis/export_crossfeeding_viz.py */
export type PopulationSummaryFrame = {
  index: number;
  time_min: number;
  n_cells: number;
  dead_count: number;
  /** cell_type (numeric, as string key) -> count. */
  counts_by_type: Record<string, number>;
  /** bnode_<NAME> -> fraction of cells carrying that node that are active (1.0). */
  node_active_fraction: Record<string, number>;
};

export type PopulationSummary = {
  version: number;
  cell_type_names?: Record<string, string> | null;
  cell_type_colors?: Record<string, string> | null;
  boolean_node_columns?: string[] | null;
  frames: PopulationSummaryFrame[];
};
