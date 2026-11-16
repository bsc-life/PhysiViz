"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { CellsFile, VizManifest, PopulationSummary } from "@/lib/vizTypes";
import PopulationSparkline from "@/components/PopulationSparkline";
import {
  heatToRgb,
  microenvironmentColormapCssGradientVertical,
  microenvironmentHeatToRgb,
} from "@/lib/heatColor";
import {
  microenvironmentFrameVoxelAlpha,
  VOXEL_INSTANCE_ALPHA_CACHE_KEY,
  voxelInstanceAlphaOnBeforeCompile,
} from "@/lib/voxelInstanceAlphaShader";
import type { CellColorMode } from "@/lib/vectorCellExport";

type Props = {
  /** Public URL prefix, e.g. /data/output_marco (no trailing slash) */
  dataPrefix: string;
};

type FramePayload = {
  substrates: Float32Array;
  cells: CellsFile["cells"];
};

const FRAME_CACHE_MAX = 4;

function parseBoolParam(v: string | null | undefined, fallback: boolean): boolean {
  if (v == null) return fallback;
  const s = v.trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(s)) return true;
  if (["0", "false", "no", "off"].includes(s)) return false;
  return fallback;
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.floor(v)));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Decodes the binary_f32 cell-frame format: a flat row-major float32 matrix
 * [n_cells x columns.length] into the same {cells: [{col: value, ...}]}
 * shape every downstream consumer (CellSpheres, tooltips, isolate filters,
 * growth-rate getters) already expects, so none of that code needed to
 * change. NaN entries (a column absent for a given cell, e.g. a boolean
 * node on a cell type with no intracellular model) are skipped -- the key
 * is simply left unset, matching the old JSON format's behavior exactly,
 * and every reader already treats a missing/undefined field the same way
 * it treats NaN (Number.isFinite guards throughout).
 */
function decodeCellsBinary(buf: ArrayBuffer, columns: string[]): Record<string, number>[] {
  const flat = new Float32Array(buf);
  const nCols = columns.length;
  const n = nCols > 0 ? Math.floor(flat.length / nCols) : 0;
  const cells: Record<string, number>[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const base = i * nCols;
    const row: Record<string, number> = {};
    for (let j = 0; j < nCols; j++) {
      const v = flat[base + j];
      if (!Number.isNaN(v)) row[columns[j]] = v;
    }
    cells[i] = row;
  }
  return cells;
}

// Reusable THREE.Color instances for CellSpheres' hot per-cell render loop.
// `InstancedMesh.setColorAt()` copies r/g/b into the instanceColor buffer the
// moment it's called, so sharing these mutable instances across iterations is
// safe -- it just avoids allocating + hex-string-parsing (THREE.Color's
// setStyle) up to hundreds of thousands of times per frame on large models
// (measured ~100ms/261k cells for the string-parse path alone vs ~0ms reused).
const COLOR_TYPE_DEFAULT = new THREE.Color("#38bdf8");
const COLOR_NECROTIC = new THREE.Color("#7f1d1d");
const COLOR_APOPTOTIC = new THREE.Color("#ef4444");
const COLOR_PRE_DEAD = new THREE.Color("#f59e0b");
const COLOR_DEAD_GENERIC = new THREE.Color("#64748b");
const COLOR_ALIVE_STATE = new THREE.Color("#22c55e");
const COLOR_NA_FALLBACK = new THREE.Color("#64748b");
const COLOR_TYPE_FALLBACK_POS = new THREE.Color("#fb923c");
const COLOR_TYPE_FALLBACK_NEG = new THREE.Color("#38bdf8");

/**
 * Builds a (rounded-sm cell-type code -> reusable THREE.Color) lookup once per
 * manifest, so the per-cell render loop never re-parses a hex string.
 */
function buildCellTypeColorMap(manifest: VizManifest): Map<string, THREE.Color> {
  const map = new Map<string, THREE.Color>();
  const colors = manifest.cell_type_colors;
  if (colors) {
    for (const [code, hex] of Object.entries(colors)) {
      map.set(code, new THREE.Color(hex));
    }
  }
  return map;
}

function numericCellColor(value: number, mn: number, mx: number): THREE.Color {
  if (!(mx > mn)) return new THREE.Color("#cbd5e1");
  const t = Math.max(0, Math.min(1, (value - mn) / (mx - mn)));
  const [r, g, b] = heatToRgb(t);
  return new THREE.Color(r / 255, g / 255, b / 255);
}

function getCellRadius(cell: Record<string, number>, fallbackRadius: number): number {
  const explicitR = Number(cell.radius);
  if (Number.isFinite(explicitR) && explicitR > 0) return explicitR;
  const totalVol = Number(cell.total_volume);
  if (Number.isFinite(totalVol) && totalVol > 0) {
    return Math.cbrt((3 * totalVol) / (4 * Math.PI));
  }
  return fallbackRadius;
}

function getCellGrowthRate(cell: Record<string, number>): number | null {
  const candidates = [
    Number(cell.CB_growth_rate),
    Number(cell.MB_growth_rate),
    Number(cell.growth_rate),
  ];
  for (const v of candidates) {
    if (Number.isFinite(v)) return v;
  }
  return null;
}

function getCellDeathRate(cell: Record<string, number>): number | null {
  const vectorCandidates = [Number(cell.x_death_rates), Number(cell.y_death_rates)].filter(
    (v) => Number.isFinite(v)
  );
  if (vectorCandidates.length > 0) {
    return vectorCandidates.reduce((acc, v) => acc + v, 0);
  }
  const candidates = [
    Number(cell.CB_death_rate),
    Number(cell.MB_death_rate),
    Number(cell.death_rate),
  ];
  for (const v of candidates) {
    if (Number.isFinite(v)) return v;
  }
  return null;
}

/**
 * Resolve a cell_type numeric code to a display color using manifest.cell_type_colors
 * (written by --cell-type-xml, supports any number of types). Falls back to the
 * original 2-species binary split (type 0 = blue, type >=1 = orange) when the
 * manifest has no cell_type_colors, so pre-existing exports keep rendering the
 * same way they always did.
 */
function getCellTypeColor(
  typeColorMap: Map<string, THREE.Color>,
  ct: number,
  dead: boolean
): THREE.Color {
  if (dead) return COLOR_DEAD_GENERIC;
  const hit = typeColorMap.get(String(Math.round(ct)));
  if (hit) return hit;
  return ct >= 0.5 ? COLOR_TYPE_FALLBACK_POS : COLOR_TYPE_FALLBACK_NEG;
}

/** Ordered [code, name, color] legend entries from manifest.cell_type_names/_colors. */
function getCellTypeLegendEntries(
  manifest: VizManifest
): { code: string; name: string; color: string }[] {
  const names = manifest.cell_type_names;
  if (!names) return [];
  const colors = manifest.cell_type_colors ?? {};
  return Object.keys(names)
    .sort((a, b) => Number(a) - Number(b))
    .map((code) => ({
      code,
      name: names[code],
      color: colors[code] ?? "#38bdf8",
    }));
}

/** Skip raycast hits on this object so pointer events reach meshes behind (cells vs substrate voxels). */
const skipPointerRaycast: THREE.Object3D["raycast"] = (_raycaster, _intersects) => {};

function getCellId(cell: Record<string, number>, fallbackIndex: number): number {
  const keys = ["ID", "id", "Id", "global_ID", "cell_ID", "parent_ID"] as const;
  for (const k of keys) {
    const raw = cell[k] as number | undefined;
    if (raw == null) continue;
    const n = Number(raw);
    if (Number.isFinite(n)) return Math.trunc(n);
  }
  return fallbackIndex;
}

function resolveInstanceId(
  e: ThreeEvent<PointerEvent | MouseEvent>
): number | null {
  const id = e.instanceId;
  if (typeof id === "number" && id >= 0) return id;
  const hit = e.intersections?.find((h) => h.eventObject === e.eventObject);
  if (typeof hit?.instanceId === "number" && hit.instanceId >= 0) return hit.instanceId;
  return null;
}

function ensureInstanceAlphaArray(
  geometry: THREE.BufferGeometry,
  count: number
): Float32Array {
  let attr = geometry.getAttribute("instanceAlpha") as
    | THREE.InstancedBufferAttribute
    | undefined;
  if (!attr || attr.count !== count) {
    if (attr) geometry.deleteAttribute("instanceAlpha");
    const arr = new Float32Array(count);
    arr.fill(1);
    attr = new THREE.InstancedBufferAttribute(arr, 1);
    attr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("instanceAlpha", attr);
  }
  return (geometry.getAttribute("instanceAlpha") as THREE.InstancedBufferAttribute)
    .array as Float32Array;
}

/**
 * Allocate {@link THREE.InstancedMesh.instanceColor} before first draw so shader
 * programs include `USE_INSTANCING_COLOR`/`USE_COLOR` from frame 0 (otherwise the
 * cache can compile without per-instance tints and voxels stay white).
 */
function syncInstancedMeshColorBuffer(mesh: THREE.InstancedMesh): void {
  const n = mesh.instanceMatrix.count;
  if (n <= 0) return;
  const existing = mesh.instanceColor;
  if (existing == null || existing.count !== n) {
    const colors = new Float32Array(n * 3);
    colors.fill(1);
    mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
    mesh.instanceColor.needsUpdate = true;
  }
}

const VOXEL_MAT_PROPS = {
  color: "#ffffff" as const,
  transparent: true,
  opacity: 1,
  depthWrite: false,
  onBeforeCompile: voxelInstanceAlphaOnBeforeCompile,
};

/** Cube edge length as a fraction of grid spacing (per axis). Slightly <1 avoids coplanar z-fighting; smaller values (e.g. 0.9) leave obvious gaps. */
const VOXEL_MESH_FILL = 0.998;

const CELL_MEMBRANE_SHADER_KEY = "cell_membrane_variation_v1";

function cellMembraneOnBeforeCompile(shader: THREE.WebGLProgramParametersWithUniforms) {
  shader.vertexShader = shader.vertexShader.replace(
    "#include <begin_vertex>",
    `
      #include <begin_vertex>
      // Add subtle membrane irregularity so cells are less perfectly spherical.
      float wobble =
        sin(position.x * 7.1) *
        sin(position.y * 6.3) *
        sin(position.z * 5.7);
      transformed += normal * (0.08 * wobble);
    `
  );
}

function formatConc(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(3);
  return v.toPrecision(4);
}

function voxelsPerSubstrate(manifest: VizManifest): number {
  if (manifest.microenvironment_grid === "3d_volume") {
    const nz = manifest.nz ?? 1;
    return manifest.nx * manifest.ny * nz;
  }
  return manifest.nx * manifest.ny;
}

function computeSubstrateRange(
  payload: FramePayload | null,
  manifest: VizManifest,
  substrateIndex: number
): { min: number; max: number } | null {
  if (!payload) return null;
  const count = voxelsPerSubstrate(manifest);
  const base = substrateIndex * count;
  const { substrates } = payload;
  if (substrates.length < base + count) return null;
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < count; i++) {
    const v = substrates[base + i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!(mx > mn)) {
    mn = 0;
    mx = 1;
  }
  return { min: mn, max: mx };
}

/** Edge length along one domain axis between voxel centers (nx cells → nx−1 gaps). */
function voxelStep1D(lo: number, hi: number, nCells: number): number {
  const span = Number(hi) - Number(lo);
  if (!Number.isFinite(span)) return 1;
  if (nCells <= 1) return Math.max(Math.abs(span), 1);
  return Math.max(Math.abs(span) / Math.max(1, nCells - 1), 1e-6);
}

function VoxelField2D({
  payload,
  manifest,
  substrateIndex,
  substrateRange,
}: {
  payload: FramePayload | null;
  manifest: VizManifest;
  substrateIndex: number;
  /** Min/max used for color scale (same as colorbar). Falls back to per-frame scan if null. */
  substrateRange: { min: number; max: number } | null;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { nx, ny, x_min, x_max, y_min, y_max } = manifest;

  const dx = voxelStep1D(x_min, x_max, nx);
  const dy = voxelStep1D(y_min, y_max, ny);
  // Horizontal plane: Three X = PhysiCell x, Three Z = PhysiCell y (see instanced positions).
  const boxX = dx * VOXEL_MESH_FILL;
  const boxZ = dy * VOXEL_MESH_FILL;
  const boxY = Math.max(Math.min(dx, dy) * 0.25, 1e-4);
  const yBase = -boxY * 0.55;
  const count = nx * ny;

  const transforms = useMemo(() => {
    const m = new THREE.Matrix4();
    const v = new THREE.Vector3();
    const list: THREE.Matrix4[] = [];
    list.length = count;
    for (let iy = 0; iy < ny; iy++) {
      const y = y_min + iy * dy;
      for (let ix = 0; ix < nx; ix++) {
        const x = x_min + ix * dx;
        // Scene uses y-up; map PhysiCell (x,y,z) -> (x,z,y).
        v.set(x, yBase, y);
        m.compose(v, new THREE.Quaternion(), new THREE.Vector3(1, 1, 1));
        list[iy * nx + ix] = m.clone();
      }
    }
    return list;
  }, [count, dx, dy, nx, ny, x_min, y_min, yBase]);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh?.geometry) return;
    ensureInstanceAlphaArray(mesh.geometry, count);
    syncInstancedMeshColorBuffer(mesh);
  }, [count]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh?.geometry) return;
    syncInstancedMeshColorBuffer(mesh);
    const dummy = new THREE.Object3D();
    const alphaArr = ensureInstanceAlphaArray(mesh.geometry, count);
    const bounds =
      payload &&
      (substrateRange ?? computeSubstrateRange(payload, manifest, substrateIndex));
    if (!payload || !bounds) {
      for (let i = 0; i < count; i++) {
        alphaArr[i] = 1;
        dummy.position.setFromMatrixPosition(transforms[i]);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(1, 1, 1);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }
      mesh.geometry.getAttribute("instanceAlpha").needsUpdate = true;
      mesh.instanceMatrix.needsUpdate = true;
      return;
    }
    const { substrates } = payload;
    const base = substrateIndex * ny * nx;
    let { min: mn, max: mx } = bounds;
    if (!(mx > mn)) {
      mn = 0;
      mx = 1;
    }
    const alphaAttr = mesh.geometry.getAttribute("instanceAlpha");
    for (let i = 0; i < count; i++) {
      const v = substrates[base + i];
      dummy.position.setFromMatrixPosition(transforms[i]);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      alphaArr[i] = microenvironmentFrameVoxelAlpha(v, mn, mx);
      const t = (v - mn) / (mx - mn);
      const [r, g, b] = microenvironmentHeatToRgb(t);
      mesh.setColorAt(i, new THREE.Color(r / 255, g / 255, b / 255));
    }
    alphaAttr.needsUpdate = true;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor!.needsUpdate = true;
  }, [
    count,
    manifest,
    nx,
    ny,
    payload,
    substrateIndex,
    substrateRange,
    transforms,
  ]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      raycast={skipPointerRaycast}
      // InstancedMesh.boundingSphere starts null and, once any render pass
      // lazily computes it (Frustum.intersectsObject), is NEVER recomputed
      // on later setMatrixAt() updates -- so with `count` fixed per dataset
      // (it never changes between frames here, unlike CellSpheres), a bad
      // first computation (e.g. before this component's effect has written
      // real instance transforms) stays wrong for the component's entire
      // lifetime, silently culling voxels that are actually in view. Opt
      // out of frustum culling entirely rather than chase that race.
      frustumCulled={false}
    >
      <boxGeometry args={[boxX, boxY, boxZ]} />
      <meshStandardMaterial
        ref={(m) => {
          if (m) m.customProgramCacheKey = () => VOXEL_INSTANCE_ALPHA_CACHE_KEY;
        }}
        {...VOXEL_MAT_PROPS}
      />
    </instancedMesh>
  );
}

function VoxelField3D({
  payload,
  manifest,
  substrateIndex,
  substrateRange,
  clipIx,
  clipIy,
  clipIz,
}: {
  payload: FramePayload | null;
  manifest: VizManifest;
  substrateIndex: number;
  substrateRange: { min: number; max: number } | null;
  /** Inclusive grid index range [lo, hi] along X — voxels outside are hidden. */
  clipIx: readonly [number, number];
  clipIy: readonly [number, number];
  clipIz: readonly [number, number];
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { nx, ny, x_min, x_max, y_min, y_max, z_min = 0, z_max = 0 } = manifest;
  const nz = manifest.nz ?? 1;

  const dx = voxelStep1D(x_min, x_max, nx);
  const dy = voxelStep1D(y_min, y_max, ny);
  const zSpan = z_max - z_min;
  const dz = nz <= 1 ? 0 : zSpan / Math.max(1, nz - 1);
  const planMin = Math.min(dx, dy);
  const tol =
    1e-9 *
    Math.max(Math.abs(x_max - x_min), Math.abs(y_max - y_min), Math.abs(zSpan), 1);
  const dzBox =
    nz <= 1 || zSpan < tol || dz < tol ? planMin : dz;
  // Instance position is v.set(x, z_phys, y_phys): box edges along Three.X,Y,Z = dx, dz, dy.
  const boxGeoX = Math.max(dx * VOXEL_MESH_FILL, 1e-4);
  const boxGeoY = Math.max(dzBox * VOXEL_MESH_FILL, 1e-4);
  const boxGeoZ = Math.max(dy * VOXEL_MESH_FILL, 1e-4);
  const count = nx * ny * nz;

  const [ixLo, ixHi] = clipIx;
  const [iyLo, iyHi] = clipIy;
  const [izLo, izHi] = clipIz;

  const transforms = useMemo(() => {
    const m = new THREE.Matrix4();
    const v = new THREE.Vector3();
    const list: THREE.Matrix4[] = [];
    list.length = count;
    let i = 0;
    for (let iz = 0; iz < nz; iz++) {
      const z = nz <= 1 ? z_min : z_min + iz * dz;
      for (let iy = 0; iy < ny; iy++) {
        const y = y_min + iy * dy;
        for (let ix = 0; ix < nx; ix++) {
          const x = x_min + ix * dx;
          // Scene uses y-up; map PhysiCell (x,y,z) -> (x,z,y).
          v.set(x, z, y);
          m.compose(v, new THREE.Quaternion(), new THREE.Vector3(1, 1, 1));
          list[i++] = m.clone();
        }
      }
    }
    return list;
  }, [count, dx, dy, dz, nx, ny, nz, x_min, y_min, z_min]);

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh?.geometry) return;
    ensureInstanceAlphaArray(mesh.geometry, count);
    syncInstancedMeshColorBuffer(mesh);
  }, [count]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh?.geometry) return;
    syncInstancedMeshColorBuffer(mesh);
    const dummy = new THREE.Object3D();
    const alphaArr = ensureInstanceAlphaArray(mesh.geometry, count);
    const bounds =
      payload &&
      (substrateRange ?? computeSubstrateRange(payload, manifest, substrateIndex));

    let mn = 0;
    let mx = 1;
    let base = 0;
    const substrates = payload?.substrates;
    if (payload && bounds) {
      mn = bounds.min;
      mx = bounds.max;
      if (!(mx > mn)) {
        mn = 0;
        mx = 1;
      }
      base = substrateIndex * count;
    }

    const alphaAttr = mesh.geometry.getAttribute("instanceAlpha");

    for (let i = 0; i < count; i++) {
      const ix = i % nx;
      const iy = Math.floor(i / nx) % ny;
      const iz = Math.floor(i / (nx * ny));
      const inClip =
        ix >= ixLo &&
        ix <= ixHi &&
        iy >= iyLo &&
        iy <= iyHi &&
        iz >= izLo &&
        iz <= izHi;

      let show = false;
      if (inClip) {
        if (payload && bounds && substrates) {
          const val = substrates[base + i];
          show = true;
          alphaArr[i] = microenvironmentFrameVoxelAlpha(val, mn, mx);
          const t = (val - mn) / (mx - mn);
          const [r, g, b] = microenvironmentHeatToRgb(t);
          mesh.setColorAt(i, new THREE.Color(r / 255, g / 255, b / 255));
        } else {
          show = true;
          alphaArr[i] = 1;
        }
      } else {
        alphaArr[i] = 0;
      }

      dummy.position.setFromMatrixPosition(transforms[i]);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(show ? 1 : 0, show ? 1 : 0, show ? 1 : 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    alphaAttr.needsUpdate = true;
    mesh.instanceMatrix.needsUpdate = true;
    if (payload && bounds) {
      mesh.instanceColor!.needsUpdate = true;
    }
  }, [
    count,
    transforms,
    ixLo,
    ixHi,
    iyLo,
    iyHi,
    izLo,
    izHi,
    nx,
    ny,
    manifest,
    payload,
    substrateIndex,
    substrateRange,
  ]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      raycast={skipPointerRaycast}
      // See the matching comment in VoxelField2D: a fixed `count` means
      // this mesh never remounts across frames, so a bad lazily-cached
      // boundingSphere (computed before setMatrixAt wrote real transforms)
      // would stay wrong forever. Skip frustum culling instead.
      frustumCulled={false}
    >
      <boxGeometry args={[boxGeoX, boxGeoY, boxGeoZ]} />
      <meshStandardMaterial
        ref={(m) => {
          if (m) m.customProgramCacheKey = () => VOXEL_INSTANCE_ALPHA_CACHE_KEY;
        }}
        {...VOXEL_MAT_PROPS}
      />
    </instancedMesh>
  );
}

function DomainBoundingGrid({
  manifest,
}: {
  manifest: VizManifest;
}) {
  const {
    x_min,
    x_max,
    y_min,
    y_max,
    z_min = 0,
    z_max = 0,
    nx,
    ny,
    nz = 1,
  } = manifest;

  const geometry = useMemo(() => {
    const lines: number[] = [];
    const xVals = Array.from({ length: Math.max(2, nx) }, (_, i) =>
      x_min + (i * (x_max - x_min)) / Math.max(1, nx - 1)
    );
    const yVals = Array.from({ length: Math.max(2, ny) }, (_, i) =>
      y_min + (i * (y_max - y_min)) / Math.max(1, ny - 1)
    );
    const zVals = Array.from({ length: Math.max(2, nz) }, (_, i) =>
      z_min + (i * (z_max - z_min)) / Math.max(1, nz - 1)
    );

    const add = (a: [number, number, number], b: [number, number, number]) => {
      // Scene uses y-up; map PhysiCell (x,y,z) -> (x,z,y).
      lines.push(a[0], a[2], a[1], b[0], b[2], b[1]);
    };

    for (const y of yVals) {
      for (const z of zVals) {
        if (y === y_min || y === y_max || z === z_min || z === z_max) {
          add([x_min, y, z], [x_max, y, z]);
        }
      }
    }
    for (const x of xVals) {
      for (const z of zVals) {
        if (x === x_min || x === x_max || z === z_min || z === z_max) {
          add([x, y_min, z], [x, y_max, z]);
        }
      }
    }
    for (const x of xVals) {
      for (const y of yVals) {
        if (x === x_min || x === x_max || y === y_min || y === y_max) {
          add([x, y, z_min], [x, y, z_max]);
        }
      }
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
    return g;
  }, [nx, ny, nz, x_max, x_min, y_max, y_min, z_max, z_min]);

  return (
    <lineSegments geometry={geometry} raycast={skipPointerRaycast}>
      <lineBasicMaterial color="#94a3b8" transparent opacity={0.5} />
    </lineSegments>
  );
}

function CellSpheres({
  payload,
  manifest,
  colorMode,
  clipIx,
  clipIy,
  clipIz,
  onHoverCellId,
  onSelectCellId,
  hiddenCellTypes,
  hiddenStates,
}: {
  payload: FramePayload | null;
  manifest: VizManifest;
  colorMode: CellColorMode;
  clipIx?: readonly [number, number];
  clipIy?: readonly [number, number];
  clipIz?: readonly [number, number];
  onHoverCellId?: (id: number | null) => void;
  onSelectCellId?: (id: number | null) => void;
  /** Cell-type codes (string keys) hidden via legend click-to-isolate; applies regardless of colorMode. */
  hiddenCellTypes?: Set<string>;
  /** "alive" | "pre_dead" | "apoptotic" | "necrotic" | "dead" states hidden via legend click-to-isolate. */
  hiddenStates?: Set<string>;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const cells = useMemo(() => payload?.cells ?? [], [payload]);
  const { nx, ny, x_min, x_max } = manifest;
  const fallbackRadius = Math.max(0.25, (x_max - x_min) / Math.max(1, nx) * 0.35);
  const count = cells.length;
  const cellIds = useMemo(() => cells.map((c, i) => getCellId(c, i)), [cells]);
  const typeColorMap = useMemo(
    () => buildCellTypeColorMap(manifest),
    [manifest.cell_type_colors]
  );
  // Icosahedron subdivision ("detail") level for the shared cell-sphere
  // geometry. Triangle count is 20 * 4^detail, so a fixed detail=4 (5120
  // tri/sphere) is fine for a few thousand cells but explodes past any GPU's
  // budget for large agent-based models (e.g. 261k cells x 5120 tri = ~1.3B
  // triangles for a single instanced draw call). Scale detail down as the
  // instance count grows so total triangle count stays roughly bounded.
  const sphereDetail = useMemo(() => {
    if (count > 80000) return 0; // 20 tri/sphere
    if (count > 20000) return 1; // 80 tri/sphere
    if (count > 3000) return 3; // 1280 tri/sphere
    return 4; // 5120 tri/sphere -- original fidelity for small/medium models
  }, [count]);
  const useRawZ =
    manifest.microenvironment_grid === "3d_volume" &&
    (manifest.nz ?? 1) > 1;
  const nz = manifest.nz ?? 1;
  const dx = (manifest.x_max - manifest.x_min) / Math.max(1, nx - 1);
  const dy = (manifest.y_max - manifest.y_min) / Math.max(1, ny - 1);
  const dz = ((manifest.z_max ?? 0) - (manifest.z_min ?? 0)) / Math.max(1, nz - 1);
  const [ixLo, ixHi] = clipIx ?? [0, Math.max(0, nx - 1)];
  const [iyLo, iyHi] = clipIy ?? [0, Math.max(0, ny - 1)];
  const [izLo, izHi] = clipIz ?? [0, Math.max(0, nz - 1)];

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const customKey = colorMode.startsWith("custom:")
      ? colorMode.slice("custom:".length)
      : null;
    const numericKey = colorMode === "growth_rate" ? "__growth_rate__" : customKey;
    let mn = Infinity;
    let mx = -Infinity;
    if (numericKey) {
      for (let i = 0; i < count; i++) {
        const val: number =
          (numericKey === "__growth_rate__"
            ? getCellGrowthRate(cells[i] ?? {})
            : Number(cells[i]?.[numericKey])) ?? NaN;
        if (Number.isFinite(val)) {
          if (val < mn) mn = val;
          if (val > mx) mx = val;
        }
      }
      if (!(mx > mn)) {
        mn = 0;
        mx = 1;
      }
    }
    for (let i = 0; i < count; i++) {
      const c = cells[i];
      const x = c.x_position ?? 0;
      const y = c.y_position ?? 0;
      const z = c.z_position ?? 0;
      const cellRadius = getCellRadius(c, fallbackRadius);
      // Scene uses y-up; map PhysiCell (x,y,z) -> (x,z,y).
      dummy.position.set(x, useRawZ ? z : z + cellRadius * 0.65, y);

      const dead = (c.dead ?? 0) > 0.5;
      const apoptoticFlag = Number(c.apoptotic ?? 0) > 0.5;
      const necroticFlag = Number(c.necrotic ?? 0) > 0.5;
      const deathModel = Number(c.current_death_model ?? -1);
      const growthRate: number = getCellGrowthRate(c) ?? NaN;
      const deathRate: number = getCellDeathRate(c) ?? NaN;
      const apoptosisRate = Number(c.x_death_rates ?? 0);
      const necrosisRate = Number(c.y_death_rates ?? 0);
      const hasApoptosisRate = Number.isFinite(apoptosisRate) && apoptosisRate > 1e-12;
      const hasNecrosisRate = Number.isFinite(necrosisRate) && necrosisRate > 1e-12;
      const hasNearZeroGrowth = Number.isFinite(growthRate) && Math.abs(growthRate) <= 1e-12;
      const hasNonzeroDeathRate = Number.isFinite(deathRate) && deathRate > 1e-12;
      // Primary mapping for vessel datasets: x_death_rates -> apoptosis, y_death_rates -> necrosis.
      // Fallback to explicit flags/model IDs when per-pathway rates are unavailable.
      const apoptoticModel = dead && (deathModel === 100 || deathModel === 0);
      const necroticModel = dead && (deathModel === 101 || deathModel === 1);
      const isApoptotic =
        hasApoptosisRate
          ? apoptosisRate >= necrosisRate
          : apoptoticFlag || apoptoticModel;
      const isNecrotic =
        hasNecrosisRate
          ? necrosisRate > apoptosisRate
          : necroticFlag || necroticModel;
      const isPreDead =
        !dead && !isApoptotic && !isNecrotic && hasNearZeroGrowth && hasNonzeroDeathRate;
      const ct = c.cell_type ?? 0;
      const stateKey = isNecrotic
        ? "necrotic"
        : isApoptotic
          ? "apoptotic"
          : isPreDead
            ? "pre_dead"
            : dead
              ? "dead"
              : "alive";

      let show = true;
      if (useRawZ && dx > 0 && dy > 0 && dz > 0) {
        const ix = clampInt(Math.round((x - manifest.x_min) / dx), 0, nx - 1);
        const iy = clampInt(Math.round((y - manifest.y_min) / dy), 0, ny - 1);
        const iz = clampInt(
          Math.round((z - (manifest.z_min ?? 0)) / dz),
          0,
          Math.max(0, nz - 1)
        );
        show =
          ix >= ixLo &&
          ix <= ixHi &&
          iy >= iyLo &&
          iy <= iyHi &&
          iz >= izLo &&
          iz <= izHi;
      }
      if (hiddenCellTypes?.has(String(Math.round(ct)))) show = false;
      if (hiddenStates?.has(stateKey)) show = false;
      dummy.scale.set(
        show ? cellRadius : 0,
        show ? cellRadius : 0,
        show ? cellRadius : 0
      );
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      let color: THREE.Color = COLOR_TYPE_DEFAULT;
      if (colorMode === "state") {
        if (isNecrotic) {
          color = COLOR_NECROTIC;
        } else if (isApoptotic) {
          color = COLOR_APOPTOTIC;
        } else if (isPreDead) {
          color = COLOR_PRE_DEAD;
        } else if (dead) {
          color = COLOR_DEAD_GENERIC;
        } else {
          color = COLOR_ALIVE_STATE;
        }
      } else if (colorMode === "dead") {
        color = dead ? COLOR_APOPTOTIC : COLOR_ALIVE_STATE;
      } else if (colorMode === "current_death_model") {
        color = deathModel >= 0 ? COLOR_PRE_DEAD : COLOR_ALIVE_STATE;
      } else if (colorMode === "growth_rate") {
        const val: number = getCellGrowthRate(c) ?? NaN;
        color = Number.isFinite(val) ? numericCellColor(val, mn, mx) : COLOR_NA_FALLBACK;
      } else if (customKey) {
        const val =
          customKey === "derived_radius"
            ? getCellRadius(c, fallbackRadius)
            : Number(c[customKey]);
        color = Number.isFinite(val) ? numericCellColor(val, mn, mx) : COLOR_NA_FALLBACK;
      } else {
        color = getCellTypeColor(typeColorMap, ct, dead);
      }
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [
    cells,
    clipIx,
    clipIy,
    clipIz,
    colorMode,
    count,
    dx,
    dy,
    dz,
    ixHi,
    ixLo,
    iyHi,
    iyLo,
    izHi,
    izLo,
    manifest.x_min,
    manifest.y_min,
    manifest.z_min,
    nx,
    ny,
    nz,
    fallbackRadius,
    useRawZ,
    hiddenCellTypes,
    hiddenStates,
    typeColorMap,
  ]);

  if (count === 0) return null;
  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      // InstancedMesh.boundingSphere starts null and is only ever computed
      // lazily, once, the first time a render pass needs it -- later
      // setMatrixAt() updates (every frame change, every clip/legend-isolate
      // filter toggle) never invalidate that cached sphere. When `count`
      // happens not to change between updates (e.g. scrubbing a clip range
      // while paused on one frame, or toggling a legend filter) this mesh
      // isn't remounted either, so a bad first computation stays wrong
      // indefinitely -- cells that are genuinely in view get silently
      // frustum-culled because the renderer is testing a stale sphere.
      // Disabling culling trades a (negligible, for one bounded dataset)
      // optimization for always-correct rendering.
      frustumCulled={false}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        const ii = resolveInstanceId(e);
        onHoverCellId?.(ii == null ? null : (cellIds[ii] ?? ii));
      }}
      onPointerOut={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        onHoverCellId?.(null);
      }}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation();
        const ii = resolveInstanceId(e);
        onSelectCellId?.(ii == null ? null : (cellIds[ii] ?? ii));
      }}
    >
      <icosahedronGeometry args={[1, sphereDetail]} />
      <meshPhysicalMaterial
        ref={(m) => {
          if (m) m.customProgramCacheKey = () => CELL_MEMBRANE_SHADER_KEY;
        }}
        color="#ffffff"
        roughness={0.5}
        metalness={0.02}
        clearcoat={0.35}
        clearcoatRoughness={0.7}
        reflectivity={0.35}
        sheen={0.35}
        sheenRoughness={0.7}
        onBeforeCompile={cellMembraneOnBeforeCompile}
      />
    </instancedMesh>
  );
}

/** Handle to the live renderer/scene/camera, kept fresh so export functions
 * can render an extra off-screen frame at an arbitrary resolution on demand
 * (see exportSnapshot) instead of upscaling the already-rendered canvas. */
type RenderExportHandle = {
  gl: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
};

/** Syncs the live renderer/scene/camera so export functions can read them at any time. */
function RenderExportSync({
  exportRef,
}: {
  exportRef: React.MutableRefObject<RenderExportHandle | null>;
}) {
  const { camera, gl, scene } = useThree();
  useFrame(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      exportRef.current = { gl, scene, camera };
    }
  });
  return null;
}

export default function SimulationViewer({ dataPrefix }: Props) {
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const renderExportRef = useRef<RenderExportHandle | null>(null);
  const [effectiveDataPrefix, setEffectiveDataPrefix] = useState(dataPrefix);
  const [manifest, setManifest] = useState<VizManifest | null>(null);
  const [framePayload, setFramePayload] = useState<FramePayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loopPlayback, setLoopPlayback] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 4>(1);
  const [hoveredCellId, setHoveredCellId] = useState<number | null>(null);
  const [selectedCellId, setSelectedCellId] = useState<number | null>(null);
  const [substrateIndex, setSubstrateIndex] = useState(0);
  const [showField, setShowField] = useState(true);
  const [showCells, setShowCells] = useState(true);
  const [showXyGrid, setShowXyGrid] = useState(true);
  const [showDomainBoundingGrid, setShowDomainBoundingGrid] = useState(true);
  const [backgroundTheme, setBackgroundTheme] = useState<"black" | "white">("black");
  const [hideUi, setHideUi] = useState(false);
  /** Right-side "Viewer controls" panel visibility (independent of `hideUi`,
   * which also hides 3D scene helpers for clean exports). Has its own
   * always-visible re-open button since the panel's own close button
   * obviously can't be reached once the panel is hidden. */
  const [showControlsPanel, setShowControlsPanel] = useState(true);
  /** Population sparkline visibility, independently closeable. */
  const [showPopulationPlot, setShowPopulationPlot] = useState(true);
  const [cellColorMode, setCellColorMode] = useState<CellColorMode>("cell_type");
  const [exportingVideo, setExportingVideo] = useState(false);
  const [videoExportProgress, setVideoExportProgress] = useState<{
    phase: "capturing" | "encoding";
    done: number;
    total: number;
  } | null>(null);
  const [videoStartFrame, setVideoStartFrame] = useState(0);
  const [videoEndFrame, setVideoEndFrame] = useState(0);
  const [videoFps, setVideoFps] = useState(8);
  const [videoFormat, setVideoFormat] = useState<"mp4" | "webm">("mp4");
  const [videoQuality, setVideoQuality] = useState<"standard" | "high" | "ultra">(
    "high"
  );
  const [pngDpi, setPngDpi] = useState(300);
  const [outputFolders, setOutputFolders] = useState<
    { name: string; exported: boolean; description?: string }[]
  >([]);
  const [selectedOutputFolder, setSelectedOutputFolder] = useState("");

  /** Switch the live dataset to `name` (a folder under public/data/) and sync the URL. */
  const openDatasetByName = useCallback((name: string) => {
    if (!name) return;
    const next = `/data/${name}`;
    setEffectiveDataPrefix(next);
    if (typeof window !== "undefined") {
      const u = new URL(window.location.href);
      u.searchParams.set("dataset", name);
      window.history.replaceState({}, "", u.toString());
    }
  }, []);

  const [populationSummary, setPopulationSummary] = useState<PopulationSummary | null>(null);
  const [sparklineNode, setSparklineNode] = useState<string | null>(null);
  /** Cell-type codes (as string keys) toggled off via legend click-to-isolate. */
  const [hiddenCellTypes, setHiddenCellTypes] = useState<Set<string>>(new Set());
  /** "state" legend entries (alive/pre_dead/apoptotic/necrotic/dead) toggled off via click-to-isolate. */
  const [hiddenStates, setHiddenStates] = useState<Set<string>>(new Set());

  const toggleHidden = useCallback(
    (set: React.Dispatch<React.SetStateAction<Set<string>>>, key: string) => {
      set((prev) => {
        const next = new Set(prev);
        if (next.has(key)) next.delete(key);
        else next.add(key);
        return next;
      });
    },
    []
  );

  /** Inclusive grid index ranges for 3D microenvironment slab clipping [lo, hi] per axis. */
  const [volIx, setVolIx] = useState<[number, number]>([0, 0]);
  const [volIy, setVolIy] = useState<[number, number]>([0, 0]);
  const [volIz, setVolIz] = useState<[number, number]>([0, 0]);
  const [cellIx, setCellIx] = useState<[number, number]>([0, 0]);
  const [cellIy, setCellIy] = useState<[number, number]>([0, 0]);
  const [cellIz, setCellIz] = useState<[number, number]>([0, 0]);
  const cacheRef = useRef<Map<number, FramePayload>>(new Map());

  const volumeDatasetKey =
    manifest?.microenvironment_grid === "3d_volume" &&
    manifest.nz != null &&
    manifest.nz >= 1
      ? `${effectiveDataPrefix}:${manifest.nx}×${manifest.ny}×${manifest.nz}`
      : "";

  useEffect(() => {
    if (typeof window === "undefined") return;
    const p = new URLSearchParams(window.location.search);
    const dataset = p.get("dataset")?.trim();
    if (dataset) {
      setEffectiveDataPrefix(`/data/${dataset.replace(/^\/+/, "")}`);
    } else {
      setEffectiveDataPrefix(dataPrefix);
    }
    setShowField(parseBoolParam(p.get("showField"), true));
    setShowCells(parseBoolParam(p.get("showCells"), true));
    setHideUi(parseBoolParam(p.get("minimal"), false));
    const modeRaw = p.get("cellColorBy")?.trim();
    if (modeRaw) {
      if (modeRaw.startsWith("custom:")) {
        setCellColorMode(modeRaw as CellColorMode);
      } else if (
        modeRaw === "cell_type" ||
        modeRaw === "growth_rate" ||
        modeRaw === "state" ||
        modeRaw === "dead" ||
        modeRaw === "current_death_model"
      ) {
        setCellColorMode(modeRaw);
      } else {
        setCellColorMode(`custom:${modeRaw}`);
      }
    }
  }, [dataPrefix]);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/output-folders")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((d: { outputs?: { name: string; exported: boolean }[] }) => {
        if (cancelled) return;
        const list = d.outputs ?? [];
        setOutputFolders(list);
        const current = effectiveDataPrefix.replace(/^\/data\//, "");
        const found = list.find((x) => x.name === current && x.exported);
        const firstExported = list.find((x) => x.exported);
        setSelectedOutputFolder(found?.name ?? (firstExported?.name ?? list[0]?.name ?? ""));
        // The initial dataPrefix (from app/page.tsx or ?dataset=) may name a
        // dataset that doesn't exist on this machine -- nothing is bundled
        // in the repo anymore (see public/data/README.md). Redirect once to
        // whatever's actually been exported rather than 404ing on a fixed
        // name; if nothing has been exported yet, leave it as-is so the
        // empty-state message below can point at the two setup scripts.
        if (!found && firstExported) {
          setEffectiveDataPrefix(`/data/${firstExported.name}`);
        }
      })
      .catch(() => {
        if (!cancelled) setOutputFolders([]);
      });
    return () => {
      cancelled = true;
    };
    // Converges in at most one extra pass: redirecting sets effectiveDataPrefix
    // to firstExported.name, so the next run finds it (found truthy) and stops.
  }, [effectiveDataPrefix]);

  useLayoutEffect(() => {
    if (!volumeDatasetKey || !manifest) return;
    setVolIx([0, Math.max(0, manifest.nx - 1)]);
    setVolIy([0, Math.max(0, manifest.ny - 1)]);
    setVolIz([0, Math.max(0, manifest.nz! - 1)]);
    setCellIx([0, Math.max(0, manifest.nx - 1)]);
    setCellIy([0, Math.max(0, manifest.ny - 1)]);
    setCellIz([0, Math.max(0, manifest.nz! - 1)]);
  }, [volumeDatasetKey]); // eslint-disable-line react-hooks/exhaustive-deps -- reset slicers only when grid dims / dataPrefix change

  useEffect(() => {
    let cancelled = false;
    setErr(null);
    fetch(`${effectiveDataPrefix}/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} loading manifest`);
        return r.json();
      })
      .then((m: VizManifest) => {
        if (!cancelled) {
          setManifest(m);
          setFrameIndex(0);
          cacheRef.current.clear();
        }
      })
      .catch((e: Error) => {
        if (!cancelled) setErr(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [effectiveDataPrefix]);

  // Compact per-frame population/node-activity summary for the sparkline strip.
  // Optional file: older exports (or a dataset missing it for any reason) just
  // render without the sparkline rather than erroring.
  useEffect(() => {
    let cancelled = false;
    setPopulationSummary(null);
    setSparklineNode(null);
    setHiddenCellTypes(new Set());
    setHiddenStates(new Set());
    fetch(`${effectiveDataPrefix}/population_summary.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((s: PopulationSummary | null) => {
        if (!cancelled) setPopulationSummary(s);
      })
      .catch(() => {
        if (!cancelled) setPopulationSummary(null);
      });
    return () => {
      cancelled = true;
    };
  }, [effectiveDataPrefix]);

  const loadFrame = useCallback(
    async (idx: number): Promise<FramePayload | null> => {
      if (!manifest || idx < 0 || idx >= manifest.frames.length) return null;
      const cached = cacheRef.current.get(idx);
      if (cached) return cached;

      const ref = manifest.frames[idx];
      const baseUrl = `${effectiveDataPrefix}/${ref.dir}`;
      const [subRes, cellRes] = await Promise.all([
        fetch(`${baseUrl}/${manifest.substrates_file}`),
        fetch(`${baseUrl}/${manifest.cells_file}`),
      ]);
      if (!subRes.ok) throw new Error(`substrates ${subRes.status}`);
      if (!cellRes.ok) throw new Error(`cells ${cellRes.status}`);
      const buf = await subRes.arrayBuffer();
      const substrates = new Float32Array(buf);
      let cells: Record<string, number>[];
      if (manifest.cells_format === "binary_f32") {
        const cellBuf = await cellRes.arrayBuffer();
        cells = decodeCellsBinary(cellBuf, manifest.cell_columns);
      } else {
        const cellsJson = (await cellRes.json()) as CellsFile;
        cells = cellsJson.cells ?? [];
      }
      const payload: FramePayload = {
        substrates,
        cells,
      };

      const cache = cacheRef.current;
      cache.set(idx, payload);
      while (cache.size > FRAME_CACHE_MAX) {
        const first = cache.keys().next().value;
        if (first === undefined) break;
        cache.delete(first);
      }
      return payload;
    },
    [effectiveDataPrefix, manifest]
  );

  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    loadFrame(frameIndex)
      .then((p) => {
        if (cancelled || !p) return;
        setFramePayload(p);
      })
      .catch((e: Error) => setErr(e.message));
    return () => {
      cancelled = true;
    };
  }, [manifest, frameIndex, loadFrame]);

  useEffect(() => {
    if (!isPlaying || !manifest) return;
    if (frameIndex >= manifest.frames.length - 1 && !loopPlayback) {
      setIsPlaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      setFrameIndex((prev) => {
        const next = prev + 1;
        if (next >= manifest.frames.length) {
          window.clearInterval(timer);
          if (loopPlayback) return 0;
          setIsPlaying(false);
          return Math.max(0, manifest.frames.length - 1);
        }
        return next;
      });
    }, Math.max(30, Math.floor(150 / playbackSpeed)));
    return () => window.clearInterval(timer);
  }, [frameIndex, isPlaying, loopPlayback, manifest, playbackSpeed]);

  const maxFrame = manifest ? manifest.frames.length - 1 : 0;
  const timeLabel = useMemo(() => {
    if (!manifest || frameIndex >= manifest.frames.length) return "";
    const t = manifest.frames[frameIndex].time_min;
    return `${t.toFixed(0)} min (${(t / 60).toFixed(2)} h)`;
  }, [manifest, frameIndex]);

  const substrateRange = useMemo(
    () =>
      manifest
        ? computeSubstrateRange(framePayload, manifest, substrateIndex)
        : null,
    [framePayload, manifest, substrateIndex]
  );

  const gridSteps = useMemo(() => {
    if (!manifest || manifest.microenvironment_grid !== "3d_volume")
      return null;
    const { nx, ny, nz = 1, x_min, x_max, y_min, y_max, z_min = 0, z_max = 0 } =
      manifest;
    const dx = (x_max - x_min) / Math.max(1, nx - 1);
    const dy = (y_max - y_min) / Math.max(1, ny - 1);
    const dz = (z_max - z_min) / Math.max(1, nz - 1);
    return { dx, dy, dz, x_min, y_min, z_min };
  }, [manifest]);

  const sceneSpan = useMemo(() => {
    if (!manifest) return 1;
    const xy = Math.max(
      manifest.x_max - manifest.x_min,
      manifest.y_max - manifest.y_min
    );
    if (
      manifest.microenvironment_grid === "3d_volume" &&
      manifest.z_min != null &&
      manifest.z_max != null
    ) {
      const zz = Math.max(manifest.z_max - manifest.z_min, 1e-6);
      return Math.max(xy, zz);
    }
    return xy;
  }, [manifest]);

  // Default camera position. World layout is (THREE x, y, z) = (data x, data z, data y),
  // so a flat 2D dataset (data z ~ 0) lies in the THREE XZ ("floor") plane. The naive
  // camera at (0, 0, +span) sits IN that plane looking along -Z, i.e. edge-on to a 2D
  // dataset -- it collapses to a near-invisible line. Elevate it for anything that isn't
  // a true 3D volume so a freshly loaded 2D dataset is visible without the user orbiting.
  const initialCameraPosition = useMemo((): [number, number, number] => {
    if (manifest?.microenvironment_grid === "3d_volume") {
      return [0, 0, sceneSpan * 1.2];
    }
    return [sceneSpan * 0.55, sceneSpan * 0.95, sceneSpan * 0.95];
  }, [manifest, sceneSpan]);

  const domainMetrics = useMemo(() => {
    if (!manifest) return null;
    const sx = manifest.x_max - manifest.x_min;
    const sy = manifest.y_max - manifest.y_min;
    const nx = Math.max(1, manifest.nx - 1);
    const ny = Math.max(1, manifest.ny - 1);
    const dx = sx / nx;
    const dy = sy / ny;
    if (manifest.microenvironment_grid === "3d_volume" && manifest.nz != null) {
      const z0 = manifest.z_min ?? 0;
      const z1 = manifest.z_max ?? 0;
      const sz = z1 - z0;
      const dz = sz / Math.max(1, manifest.nz - 1);
      return { sx, sy, sz, dx, dy, dz };
    }
    return { sx, sy, sz: 0, dx, dy, dz: 0 };
  }, [manifest]);

  const heatGradient = useMemo(
    () => microenvironmentColormapCssGradientVertical(),
    []
  );
  const substrateLabel =
    manifest && substrateIndex < manifest.substrate_names.length
      ? manifest.substrate_names[substrateIndex]
      : "Substrate";
  const numericCellColumns = useMemo(() => {
    if (!manifest) return [] as string[];
    const blocked = new Set(["x_position", "y_position", "z_position"]);
    const cols = manifest.cell_columns.filter((k) => !blocked.has(k));
    if (!cols.includes("derived_radius")) cols.unshift("derived_radius");
    return cols;
  }, [manifest]);

  const currentDatasetName = useMemo(
    () => effectiveDataPrefix.replace(/^\/data\//, ""),
    [effectiveDataPrefix]
  );

  useEffect(() => {
    if (typeof window === "undefined" || !manifest) return;
    const p = new URLSearchParams(window.location.search);
    const frameRaw = p.get("frame");
    if (frameRaw != null) {
      const idx = Number(frameRaw);
      if (Number.isFinite(idx)) {
        setFrameIndex(clampInt(idx, 0, Math.max(0, manifest.frames.length - 1)));
      }
    }
    const subRaw = p.get("substrate");
    if (subRaw != null) {
      const asNum = Number(subRaw);
      if (Number.isFinite(asNum)) {
        setSubstrateIndex(
          clampInt(asNum, 0, Math.max(0, manifest.substrate_names.length - 1))
        );
      } else {
        const byName = manifest.substrate_names.findIndex(
          (s) => s.toLowerCase() === subRaw.toLowerCase()
        );
        if (byName >= 0) setSubstrateIndex(byName);
      }
    }
  }, [manifest]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    (window as Window & { __vizReady?: boolean }).__vizReady = !!manifest;
  }, [frameIndex, framePayload, manifest]);

  useEffect(() => {
    if (!manifest) return;
    setVideoStartFrame(0);
    setVideoEndFrame(Math.max(0, manifest.frames.length - 1));
  }, [manifest]);

  const exportSnapshot = useCallback(async () => {
    const canvas = canvasHostRef.current?.querySelector("canvas");
    const handle = renderExportRef.current;
    if (!(canvas instanceof HTMLCanvasElement) || !manifest || !handle) return;
    const { gl, scene, camera } = handle;

    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    const scale = Math.max(0.1, pngDpi / 96);
    const outW = Math.max(1, Math.round(cssW * scale));
    const outH = Math.max(1, Math.round(cssH * scale));

    // Native re-render at the target resolution instead of upscaling the
    // already-rendered (on-screen-sized) canvas through a 2D context: the
    // old path just blew up a lower-resolution raster and blurred it. This
    // asks the GPU to draw the real scene at the requested pixel count, so
    // a higher "PNG DPI" setting buys genuinely more detail, not a softer
    // enlargement -- and it's cheaper too, since there's no extra full-size
    // 2D blit of the on-screen canvas on the common (opaque-bg) path.
    const prevSize = new THREE.Vector2();
    gl.getSize(prevSize);
    const prevPixelRatio = gl.getPixelRatio();
    const prevAspect = camera.aspect;

    gl.setPixelRatio(1);
    gl.setSize(outW, outH, false);
    camera.aspect = outW / outH;
    camera.updateProjectionMatrix();
    gl.render(scene, camera);

    try {
      // toBlob() snapshots the current drawing buffer synchronously at call
      // time (preserveDrawingBuffer: true keeps it readable); only the PNG
      // encoding itself happens async, so it's safe to restore the live
      // view's size right after this call without racing the capture.
      const blob: Blob | null = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/png")
      );
      if (!blob) return;

      // Composite onto an opaque backdrop at the SAME resolution (a 1:1
      // copy, not a resize) -- matches the on-screen look, where the
      // dark/white background is a CSS layer behind the alpha-enabled
      // WebGL canvas, not part of its own pixels.
      let outBlob: Blob = blob;
      const bgCanvas = document.createElement("canvas");
      bgCanvas.width = outW;
      bgCanvas.height = outH;
      const ctx = bgCanvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = backgroundTheme === "white" ? "#ffffff" : "#09090b";
        ctx.fillRect(0, 0, outW, outH);
        const bitmap = await createImageBitmap(blob);
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        const composited: Blob | null = await new Promise((resolve) =>
          bgCanvas.toBlob(resolve, "image/png")
        );
        if (composited) outBlob = composited;
      }

      const url = URL.createObjectURL(outBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `viz_${(manifest.frames[frameIndex]?.time_min ?? 0).toFixed(0)}min_f${frameIndex}_${outW}x${outH}.png`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      // Restore the live view's size so R3F's own render loop and
      // OrbitControls keep working normally on the next frame.
      gl.setPixelRatio(prevPixelRatio);
      gl.setSize(prevSize.x, prevSize.y, false);
      camera.aspect = prevAspect;
      camera.updateProjectionMatrix();
      gl.render(scene, camera);
    }
  }, [backgroundTheme, frameIndex, manifest, pngDpi]);

  const exportVideo = useCallback(async () => {
    if (!manifest || exportingVideo) return;
    const canvas = canvasHostRef.current?.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) return;

    const start = clampInt(
      Math.min(videoStartFrame, videoEndFrame),
      0,
      Math.max(0, manifest.frames.length - 1)
    );
    const end = clampInt(
      Math.max(videoStartFrame, videoEndFrame),
      0,
      Math.max(0, manifest.frames.length - 1)
    );
    const fps = clampInt(videoFps, 1, 30);

    const prevFrame = frameIndex;
    setExportingVideo(true);
    setErr(null);
    const total = end - start + 1;
    setVideoExportProgress({ phase: "capturing", done: 0, total });

    // Wait for the frameIndex state update to commit, then give the GPU a
    // couple of animation frames to actually draw the new scene, before
    // reading the canvas back. Unlike the old real-time MediaRecorder path,
    // this loop is bound only by how long a frame genuinely takes to
    // render -- not by the video's target playback fps -- so it can only be
    // as slow as the scene itself, never slower, and it never drops or
    // duplicates a frame under load.
    const waitForRender = () =>
      sleep(20).then(
        () =>
          new Promise<void>((resolve) => {
            requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
          })
      );

    const captureFrame = (): Promise<Blob> =>
      new Promise((resolve, reject) => {
        canvas.toBlob(
          (blob) => (blob ? resolve(blob) : reject(new Error("canvas.toBlob returned null"))),
          "image/jpeg",
          0.92
        );
      });

    try {
      // Capture every requested frame as a still JPEG first (deterministic,
      // one real render per frame) rather than recording in real time.
      // JPEG at q=0.92 is visually lossless for this purpose and much
      // cheaper to encode/upload than PNG; the final video codec is lossy
      // either way, so this costs nothing perceptible while cutting both
      // capture time and upload size noticeably for long exports.
      const frames: Blob[] = [];
      for (let i = start; i <= end; i++) {
        setFrameIndex(i);
        await waitForRender();
        frames.push(await captureFrame());
        setVideoExportProgress({ phase: "capturing", done: i - start + 1, total });
      }

      // Single server-side ffmpeg pass straight from the still sequence --
      // no intermediate real-time WebM and no second lossy re-encode.
      setVideoExportProgress({ phase: "encoding", done: 0, total: 1 });
      const fd = new FormData();
      fd.append("fps", String(fps));
      fd.append("quality", videoQuality);
      fd.append("format", videoFormat);
      fd.append("ext", "jpg");
      frames.forEach((blob, i) => {
        fd.append(`frame_${String(i).padStart(5, "0")}`, blob, `frame_${i}.jpg`);
      });

      const res = await fetch("/api/render-video", { method: "POST", body: fd });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Video render failed: ${txt}`);
      }
      const outBlob = await res.blob();
      const outName = `viz_f${start}-${end}_${fps}fps.${videoFormat}`;

      const url = URL.createObjectURL(outBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = outName;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to export video.");
    } finally {
      setFrameIndex(prevFrame);
      setExportingVideo(false);
      setVideoExportProgress(null);
    }
  }, [
    exportingVideo,
    frameIndex,
    manifest,
    videoEndFrame,
    videoFormat,
    videoFps,
    videoQuality,
    videoStartFrame,
  ]);


  return (
    <div className="relative h-screen w-screen overflow-hidden bg-zinc-950">
      {err && outputFolders.every((f) => !f.exported) && (
        <div className="absolute left-3 top-3 z-30 max-w-[min(75vw,560px)] rounded-sm border border-zinc-700 bg-zinc-950/90 px-3 py-2 text-sm text-zinc-200">
          <p className="mb-1.5 font-semibold text-zinc-100">No dataset exported yet</p>
          <p className="text-zinc-400">
            <code className="bg-zinc-800 px-1 rounded-sm">public/data/</code> is empty on a fresh
            clone. Export your own PhysiCell run:
          </p>
          <code className="mt-1 block bg-zinc-800 px-1.5 py-1 text-xs text-zinc-300">
            ./scripts/prepare_dataset.sh /path/to/PhysiCell/output my_run
          </code>
          <p className="mt-1.5 text-zinc-400">
            — or try the pipeline with no simulation of your own (needs a C++ toolchain):
          </p>
          <code className="mt-1 block bg-zinc-800 px-1.5 py-1 text-xs text-zinc-300">
            ./scripts/demo_from_physicell_sample.sh
          </code>
        </div>
      )}
      {err && outputFolders.some((f) => f.exported) && (
        <p className="absolute left-3 top-3 z-30 max-w-[min(75vw,720px)] rounded-sm border border-red-700/60 bg-zinc-950/90 px-3 py-2 text-red-300 text-sm">
          {err} loading <code className="bg-zinc-800 px-1 rounded-sm">{effectiveDataPrefix}</code> — pick a
          dataset from the dropdown instead, or check it still exists under{" "}
          <code className="bg-zinc-800 px-1 rounded-sm">public/data/</code>.
        </p>
      )}

      {manifest && (
        <>
          {!hideUi && !showControlsPanel && (
            <button
              type="button"
              onClick={() => setShowControlsPanel(true)}
              className="absolute right-4 top-4 z-20 rounded-none border border-zinc-700 bg-zinc-950/95 px-3 py-2 text-sm text-zinc-200 shadow-sm hover:bg-zinc-900"
              title="Show viewer controls"
            >
              ☰ Controls
            </button>
          )}

          {!hideUi && showControlsPanel && (
            <div className="absolute right-4 top-4 z-20 max-h-[calc(100vh-2rem)] w-[min(420px,92vw)] overflow-y-auto rounded-none border border-zinc-700 bg-zinc-950/95 p-3 text-sm shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-zinc-200 font-semibold">Viewer controls</div>
                <button
                  type="button"
                  onClick={() => setShowControlsPanel(false)}
                  className="shrink-0 rounded-sm border border-zinc-700 px-1.5 py-0.5 text-xs text-zinc-400 hover:bg-zinc-800"
                  title="Hide viewer controls"
                >
                  ✕
                </button>
              </div>

              <div className="mb-4 flex items-end gap-2">
                <select
                  value={selectedOutputFolder}
                  onChange={(e) => {
                    const name = e.target.value;
                    setSelectedOutputFolder(name);
                    openDatasetByName(name);
                  }}
                  title={manifest?.description ?? undefined}
                  className="min-w-0 flex-1 rounded-sm border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-xs"
                >
                  {outputFolders.length === 0 && (
                    <option value={currentDatasetName}>{currentDatasetName}</option>
                  )}
                  {outputFolders.map((f) => (
                    <option key={f.name} value={f.name} title={f.description ?? undefined}>
                      {f.name}
                      {f.exported ? "" : "  (not exported)"}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => openDatasetByName(selectedOutputFolder)}
                  className="shrink-0 rounded-sm border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 hover:bg-zinc-800"
                  title="Dataset selection now loads automatically; this re-opens the current selection."
                >
                  Open
                </button>
              </div>

              <div className="mb-4 flex flex-wrap gap-3 items-end text-sm">
            <label className="flex flex-col gap-1">
              <span className="text-zinc-400">Time step</span>
              <input
                type="range"
                min={0}
                max={maxFrame}
                value={frameIndex}
                onChange={(e) => {
                  setIsPlaying(false);
                  setFrameIndex(Number(e.target.value));
                }}
                onKeyDown={(e) => {
                  // Explicit handling so Left/Right/Home/End reliably step frames even
                  // if focus/other listeners (e.g. OrbitControls) interfere with the
                  // browser's native range-input key behavior.
                  let next: number | null = null;
                  if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
                    next = Math.max(0, frameIndex - 1);
                  } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
                    next = Math.min(maxFrame, frameIndex + 1);
                  } else if (e.key === "Home") {
                    next = 0;
                  } else if (e.key === "End") {
                    next = maxFrame;
                  }
                  if (next != null) {
                    e.preventDefault();
                    setIsPlaying(false);
                    setFrameIndex(next);
                  }
                }}
                className="w-56 accent-sky-500"
              />
              <span className="text-zinc-500 text-xs font-mono">{timeLabel}</span>
            </label>
            <button
              type="button"
              onClick={() => {
                if (!manifest) return;
                if (frameIndex >= manifest.frames.length - 1) {
                  setFrameIndex(0);
                }
                setIsPlaying((v) => !v);
              }}
              className="rounded-sm border border-zinc-700 bg-zinc-900 px-2 py-1 hover:bg-zinc-800"
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={loopPlayback}
                onChange={(e) => setLoopPlayback(e.target.checked)}
              />
              Loop
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-zinc-400">Playback speed</span>
              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value) as 1 | 2 | 4)}
                className="rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
              >
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={4}>4x</option>
              </select>
            </label>

            <label className="flex flex-col gap-1">
              <span className="text-zinc-400">Substrate</span>
              <select
                value={substrateIndex}
                onChange={(e) => setSubstrateIndex(Number(e.target.value))}
                className="bg-zinc-900 border border-zinc-700 rounded-sm px-2 py-1 min-w-[140px]"
              >
                {manifest.substrate_names.map((name, i) => (
                  <option key={name} value={i}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showField}
                onChange={(e) => setShowField(e.target.checked)}
              />
              Microenvironment
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showCells}
                onChange={(e) => setShowCells(e.target.checked)}
              />
              Cells
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showXyGrid}
                onChange={(e) => setShowXyGrid(e.target.checked)}
              />
              XY grid
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showDomainBoundingGrid}
                onChange={(e) => setShowDomainBoundingGrid(e.target.checked)}
              />
              Domain bounding grid
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showPopulationPlot}
                onChange={(e) => setShowPopulationPlot(e.target.checked)}
              />
              Population plot
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-zinc-400">Background</span>
              <select
                value={backgroundTheme}
                onChange={(e) =>
                  setBackgroundTheme(e.target.value as "black" | "white")
                }
                className="rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
              >
                <option value="black">black</option>
                <option value="white">white</option>
              </select>
            </label>
            <div className="flex flex-wrap items-center gap-1">
              <button
                type="button"
                onClick={exportSnapshot}
                className="rounded-sm border border-zinc-700 bg-zinc-900 px-2 py-1 hover:bg-zinc-800"
              >
                Export PNG
              </button>
            </div>
            <label className="flex flex-col gap-1">
              <span className="text-zinc-400">PNG DPI</span>
              <input
                type="number"
                min={72}
                max={1200}
                value={pngDpi}
                onChange={(e) =>
                  setPngDpi(clampInt(Number(e.target.value) || 300, 72, 1200))
                }
                className="w-24 rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-zinc-400">Cell color by</span>
              <select
                value={cellColorMode}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v.startsWith("custom:")) {
                    setCellColorMode(v as CellColorMode);
                    return;
                  }
                  if (
                    v === "cell_type" ||
                    v === "growth_rate" ||
                    v === "state" ||
                    v === "dead" ||
                    v === "current_death_model"
                  ) {
                    setCellColorMode(v);
                  }
                }}
                className="bg-zinc-900 border border-zinc-700 rounded-sm px-2 py-1 min-w-[180px]"
              >
                <option value="cell_type">cell_type</option>
                <option value="growth_rate">growth_rate</option>
                <option value="state">state (alive/pre-dead/apoptotic/necrotic)</option>
                <option value="dead">dead</option>
                <option value="current_death_model">current_death_model</option>
                {numericCellColumns.map((k) => (
                  <option key={k} value={`custom:${k}`}>
                    {k === "derived_radius" ? "custom:derived_radius (from radius/volume)" : `custom:${k}`}
                  </option>
                ))}
              </select>
            </label>
            {domainMetrics && (
              <div className="text-[11px] text-zinc-400 rounded-sm border border-zinc-700 bg-zinc-900/60 px-2 py-1.5">
                Domain: {domainMetrics.sx.toFixed(1)} x {domainMetrics.sy.toFixed(1)}
                {manifest.microenvironment_grid === "3d_volume"
                  ? ` x ${domainMetrics.sz.toFixed(1)}`
                  : ""}
                {" "}{manifest.spatial_units}
                <br />
                Step: dx={domainMetrics.dx.toFixed(2)}, dy={domainMetrics.dy.toFixed(2)}
                {manifest.microenvironment_grid === "3d_volume"
                  ? `, dz=${domainMetrics.dz.toFixed(2)}`
                  : ""}
                {" "}{manifest.spatial_units}
              </div>
            )}
              </div>

            <details className="rounded-none border border-zinc-700 bg-zinc-900/75 p-3 text-sm max-w-5xl" open>
              <summary className="cursor-pointer font-medium text-zinc-300">
                Export video
              </summary>
              <div className="mt-2 flex flex-wrap items-end gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-zinc-400">Start frame</span>
                  <input
                    type="number"
                    min={0}
                    max={maxFrame}
                    value={videoStartFrame}
                    onChange={(e) =>
                      setVideoStartFrame(clampInt(Number(e.target.value), 0, maxFrame))
                    }
                    className="w-24 rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-zinc-400">End frame</span>
                  <input
                    type="number"
                    min={0}
                    max={maxFrame}
                    value={videoEndFrame}
                    onChange={(e) =>
                      setVideoEndFrame(clampInt(Number(e.target.value), 0, maxFrame))
                    }
                    className="w-24 rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-zinc-400">FPS</span>
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={videoFps}
                    onChange={(e) => setVideoFps(clampInt(Number(e.target.value), 1, 30))}
                    className="w-20 rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-zinc-400">Format</span>
                  <select
                    value={videoFormat}
                    onChange={(e) => setVideoFormat(e.target.value as "mp4" | "webm")}
                    className="rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
                  >
                    <option value="mp4">mp4</option>
                    <option value="webm">webm</option>
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-zinc-400">Quality</span>
                  <select
                    value={videoQuality}
                    onChange={(e) =>
                      setVideoQuality(
                        e.target.value as "standard" | "high" | "ultra"
                      )
                    }
                    className="rounded-sm border border-zinc-700 bg-zinc-950 px-2 py-1"
                  >
                    <option value="standard">standard</option>
                    <option value="high">high</option>
                    <option value="ultra">ultra</option>
                  </select>
                </label>
                <button
                  type="button"
                  onClick={exportVideo}
                  disabled={exportingVideo}
                  className="rounded-sm border border-zinc-700 bg-zinc-900 px-3 py-1.5 hover:bg-zinc-800 disabled:opacity-50"
                >
                  {exportingVideo ? "Rendering..." : "Export video"}
                </button>
                {videoExportProgress && (
                  <span className="text-xs text-zinc-400">
                    {videoExportProgress.phase === "capturing"
                      ? `Capturing frame ${videoExportProgress.done}/${videoExportProgress.total}…`
                      : "Encoding on server…"}
                  </span>
                )}
              </div>
            </details>

          {!hideUi && showField && manifest.microenvironment_grid === "3d_volume" &&
            manifest.nz != null &&
            manifest.nz > 0 &&
            gridSteps && (
              <details className="rounded-none border border-zinc-700 bg-zinc-900/75 p-3 text-sm max-w-5xl" open>
                <summary className="cursor-pointer font-medium text-zinc-300">
                  3D microenvironment slicers
                </summary>
                <div className="mt-2">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <button
                    type="button"
                    className="text-xs text-sky-400 hover:underline"
                    onClick={() => {
                      setVolIx([0, Math.max(0, manifest.nx - 1)]);
                      setVolIy([0, Math.max(0, manifest.ny - 1)]);
                      setVolIz([0, Math.max(0, manifest.nz! - 1)]);
                    }}
                  >
                    Show full volume
                  </button>
                </div>
                <p className="text-zinc-500 text-xs mb-3">
                  Per axis: set min / max <strong>grid index</strong> (inclusive).
                  Only voxels inside the box are drawn so you can peel away outer
                  layers and see inside. Coordinates are voxel centers in{" "}
                  {manifest.spatial_units}.
                </p>
                <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-3">
                  {(
                    [
                      {
                        letter: "X",
                        pair: volIx,
                        setPair: setVolIx,
                        maxIdx: Math.max(0, manifest.nx - 1),
                        pos: (i: number) =>
                          gridSteps.x_min + i * gridSteps.dx,
                      },
                      {
                        letter: "Y",
                        pair: volIy,
                        setPair: setVolIy,
                        maxIdx: Math.max(0, manifest.ny - 1),
                        pos: (i: number) =>
                          gridSteps.y_min + i * gridSteps.dy,
                      },
                      {
                        letter: "Z",
                        pair: volIz,
                        setPair: setVolIz,
                        maxIdx: Math.max(0, manifest.nz - 1),
                        pos: (i: number) =>
                          gridSteps.z_min + i * gridSteps.dz,
                      },
                    ] as const
                  ).map(({ letter, pair, setPair, maxIdx, pos }) => (
                    <div key={letter} className="flex flex-col gap-2 min-w-0">
                      <span className="text-zinc-400 text-xs font-semibold">
                        {letter} axis (0…{maxIdx})
                      </span>
                      <label className="flex flex-col gap-0.5">
                        <span className="text-zinc-500 text-[11px]">
                          Min index {pair[0]} →≈ {pos(pair[0]).toFixed(1)}{" "}
                          {manifest.spatial_units}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={maxIdx}
                          value={pair[0]}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            setPair((prev) => {
                              const lo = Math.min(v, prev[1]);
                              const hi = Math.max(v, prev[1]);
                              return [lo, hi];
                            });
                          }}
                          className="w-full accent-sky-500"
                        />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <span className="text-zinc-500 text-[11px]">
                          Max index {pair[1]} →≈ {pos(pair[1]).toFixed(1)}{" "}
                          {manifest.spatial_units}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={maxIdx}
                          value={pair[1]}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            setPair((prev) => {
                              const lo = Math.min(prev[0], v);
                              const hi = Math.max(prev[0], v);
                              return [lo, hi];
                            });
                          }}
                          className="w-full accent-amber-500"
                        />
                      </label>
                    </div>
                  ))}
                </div>
                </div>
              </details>
            )}

          {!hideUi && showCells && manifest.microenvironment_grid === "3d_volume" &&
            manifest.nz != null &&
            manifest.nz > 0 &&
            gridSteps && (
              <details className="rounded-none border border-zinc-700 bg-zinc-900/75 p-3 text-sm max-w-5xl mt-3" open>
                <summary className="cursor-pointer font-medium text-zinc-300">
                  3D cell slicers
                </summary>
                <div className="mt-2">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <button
                    type="button"
                    className="text-xs text-sky-400 hover:underline"
                    onClick={() => {
                      setCellIx([0, Math.max(0, manifest.nx - 1)]);
                      setCellIy([0, Math.max(0, manifest.ny - 1)]);
                      setCellIz([0, Math.max(0, manifest.nz! - 1)]);
                    }}
                  >
                    Show all cells
                  </button>
                </div>
                <p className="text-zinc-500 text-xs mb-3">
                  Slice cell rendering independently from voxels to inspect tumor
                  interior.
                </p>
                <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-3">
                  {(
                    [
                      {
                        letter: "X",
                        pair: cellIx,
                        setPair: setCellIx,
                        maxIdx: Math.max(0, manifest.nx - 1),
                        pos: (i: number) => gridSteps.x_min + i * gridSteps.dx,
                      },
                      {
                        letter: "Y",
                        pair: cellIy,
                        setPair: setCellIy,
                        maxIdx: Math.max(0, manifest.ny - 1),
                        pos: (i: number) => gridSteps.y_min + i * gridSteps.dy,
                      },
                      {
                        letter: "Z",
                        pair: cellIz,
                        setPair: setCellIz,
                        maxIdx: Math.max(0, manifest.nz - 1),
                        pos: (i: number) => gridSteps.z_min + i * gridSteps.dz,
                      },
                    ] as const
                  ).map(({ letter, pair, setPair, maxIdx, pos }) => (
                    <div key={`cell-${letter}`} className="flex flex-col gap-2 min-w-0">
                      <span className="text-zinc-400 text-xs font-semibold">
                        {letter} axis (0…{maxIdx})
                      </span>
                      <label className="flex flex-col gap-0.5">
                        <span className="text-zinc-500 text-[11px]">
                          Min index {pair[0]} →≈ {pos(pair[0]).toFixed(1)}{" "}
                          {manifest.spatial_units}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={maxIdx}
                          value={pair[0]}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            setPair((prev) => [Math.min(v, prev[1]), Math.max(v, prev[1])]);
                          }}
                          className="w-full accent-sky-500"
                        />
                      </label>
                      <label className="flex flex-col gap-0.5">
                        <span className="text-zinc-500 text-[11px]">
                          Max index {pair[1]} →≈ {pos(pair[1]).toFixed(1)}{" "}
                          {manifest.spatial_units}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={maxIdx}
                          value={pair[1]}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            setPair((prev) => [Math.min(prev[0], v), Math.max(prev[0], v)]);
                          }}
                          className="w-full accent-amber-500"
                        />
                      </label>
                    </div>
                  ))}
                </div>
                </div>
              </details>
            )}


            </div>
          )}

          <div
            ref={canvasHostRef}
            className="absolute inset-0 overflow-hidden"
            style={{
              background: backgroundTheme === "white" ? "#ffffff" : "#09090b",
            }}
          >
            {!hideUi && showCells && (
              <div className="pointer-events-auto absolute left-4 top-4 z-20 max-w-[240px] rounded-none border border-zinc-600 bg-zinc-950/97 px-3 py-2 text-xs text-zinc-200 shadow-sm">
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="font-semibold text-zinc-300">
                    Legend — Cells ({cellColorMode})
                  </span>
                  <div className="flex shrink-0 gap-1">
                    {(cellColorMode === "cell_type" || cellColorMode === "state") && (
                      <button
                        type="button"
                        onClick={() => {
                          if (cellColorMode === "cell_type") {
                            const entries = manifest ? getCellTypeLegendEntries(manifest) : [];
                            setHiddenCellTypes(new Set(entries.map((e) => e.code)));
                            setHiddenStates(new Set(["dead"]));
                          } else {
                            setHiddenStates(
                              new Set(["alive", "pre_dead", "apoptotic", "necrotic"])
                            );
                          }
                        }}
                        className="rounded-sm border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800"
                        title="Hide every population in the legend"
                      >
                        hide all
                      </button>
                    )}
                    {(hiddenCellTypes.size > 0 || hiddenStates.size > 0) && (
                      <button
                        type="button"
                        onClick={() => {
                          setHiddenCellTypes(new Set());
                          setHiddenStates(new Set());
                        }}
                        className="rounded-sm border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800"
                      >
                        show all
                      </button>
                    )}
                  </div>
                </div>
                {(cellColorMode === "cell_type" || cellColorMode === "state") && (
                  <p className="mb-1 text-[10px] text-zinc-500">
                    Click a swatch to isolate / show that population.
                  </p>
                )}
                {cellColorMode === "cell_type" ? (
                  <ul className="space-y-1.5">
                    {(manifest && getCellTypeLegendEntries(manifest).length > 0
                      ? getCellTypeLegendEntries(manifest)
                      : [
                          { code: "0", name: "C. beijerinckii", color: "#38bdf8" },
                          { code: "1", name: "M. barkeri", color: "#fb923c" },
                        ]
                    ).map(({ code, name, color }) => {
                      const hidden = hiddenCellTypes.has(code);
                      return (
                        <li key={code}>
                          <button
                            type="button"
                            onClick={() => toggleHidden(setHiddenCellTypes, code)}
                            className="flex w-full items-center gap-2 rounded-sm px-0.5 py-0.5 text-left hover:bg-zinc-800/80"
                            style={{ opacity: hidden ? 0.4 : 1 }}
                            title={hidden ? "Hidden — click to show" : "Click to hide"}
                          >
                            <span
                              className="h-3 w-3 shrink-0 rounded-full border border-zinc-500"
                              style={{ background: color }}
                            />
                            <span>
                              <span className="text-zinc-300">{name}</span>
                              <span className="text-zinc-500"> (type {code})</span>
                            </span>
                          </button>
                        </li>
                      );
                    })}
                    <li>
                      <button
                        type="button"
                        onClick={() => toggleHidden(setHiddenStates, "dead")}
                        className="flex w-full items-center gap-2 rounded-sm px-0.5 py-0.5 text-left hover:bg-zinc-800/80"
                        style={{ opacity: hiddenStates.has("dead") ? 0.4 : 1 }}
                        title={hiddenStates.has("dead") ? "Hidden — click to show" : "Click to hide"}
                      >
                        <span
                          className="h-3 w-3 shrink-0 rounded-full border border-zinc-500"
                          style={{ background: "#64748b" }}
                        />
                        <span className="text-zinc-300">Dead</span>
                      </button>
                    </li>
                  </ul>
                ) : cellColorMode === "state" ? (
                  <ul className="space-y-1.5">
                    {(
                      [
                        ["alive", "Alive", "#22c55e"],
                        ["pre_dead", "Pre-dead (growth=0, death>0)", "#f59e0b"],
                        ["apoptotic", "Apoptotic", "#ef4444"],
                        ["necrotic", "Necrotic", "#7f1d1d"],
                      ] as const
                    ).map(([key, label, color]) => {
                      const hidden = hiddenStates.has(key);
                      return (
                        <li key={key}>
                          <button
                            type="button"
                            onClick={() => toggleHidden(setHiddenStates, key)}
                            className="flex w-full items-center gap-2 rounded-sm px-0.5 py-0.5 text-left hover:bg-zinc-800/80"
                            style={{ opacity: hidden ? 0.4 : 1 }}
                            title={hidden ? "Hidden — click to show" : "Click to hide"}
                          >
                            <span
                              className="h-3 w-3 shrink-0 rounded-full border border-zinc-500"
                              style={{ background: color }}
                            />
                            <span className="text-zinc-300">{label}</span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="text-zinc-400 text-[11px]">
                    Colormap applied to selected field values.
                  </p>
                )}
              </div>
            )}

            {!hideUi && showField && manifest && (
              <div className="pointer-events-none absolute bottom-3 left-3 z-10 flex min-h-[120px] max-h-[min(52vh,360px)] flex-row items-stretch gap-2 rounded-none border border-zinc-600 bg-zinc-950/97 px-2 py-2 text-xs text-zinc-200 shadow-sm">
                <div className="flex w-[5.5rem] shrink-0 flex-col justify-between py-0.5 text-right font-mono leading-tight tabular-nums">
                  <span title="Max (this frame)">
                    {substrateRange ? formatConc(substrateRange.max) : "—"}
                  </span>
                  <span className="text-center text-[10px] font-sans leading-snug text-zinc-400">
                    {substrateLabel}
                    <br />
                    <span className="text-zinc-500">per-frame scale</span>
                  </span>
                  <span title="Min (this frame)">
                    {substrateRange ? formatConc(substrateRange.min) : "—"}
                  </span>
                </div>
                <div
                  className="w-5 min-h-[100px] shrink-0 self-stretch rounded-sm border border-zinc-500/90"
                  style={{ background: heatGradient }}
                  title="This frame: min (bottom) → max (top); colormap tuned for dark background"
                />
              </div>
            )}

            {!hideUi && showCells && (
              <div className="pointer-events-none absolute bottom-3 right-3 z-10 rounded-none border border-zinc-600 bg-zinc-950/97 px-3 py-2 text-xs text-zinc-200 shadow-sm">
                <div className="mb-1 font-semibold text-zinc-300">Cell ID</div>
                <div className="font-mono text-zinc-400">Hover: {hoveredCellId ?? "—"}</div>
                <div className="font-mono text-zinc-300">Selected: {selectedCellId ?? "—"}</div>
              </div>
            )}

            {!hideUi && showCells && showPopulationPlot && populationSummary && (
              <div className="absolute bottom-3 left-1/2 z-10 w-[min(800px,94vw)] -translate-x-1/2">
                <button
                  type="button"
                  onClick={() => setShowPopulationPlot(false)}
                  className="absolute right-1 top-1 z-10 rounded-sm border border-zinc-700 bg-zinc-950/90 px-1.5 py-0.5 text-[10px] text-zinc-400 hover:bg-zinc-800"
                  title="Hide population plot"
                >
                  ✕
                </button>
                <PopulationSparkline
                  summary={populationSummary}
                  frameIndex={frameIndex}
                  onScrub={(idx) => {
                    setIsPlaying(false);
                    setFrameIndex(idx);
                  }}
                  selectedNode={sparklineNode}
                  onSelectNode={setSparklineNode}
                  hiddenTypes={hiddenCellTypes}
                />
              </div>
            )}

            <Canvas
              key={effectiveDataPrefix}
              gl={{ preserveDrawingBuffer: true, alpha: true }}
              camera={{
                position: initialCameraPosition,
                fov: 45,
                near: 0.1,
                far: Math.max(4000, sceneSpan * 10),
              }}
              onPointerMissed={() => {
                setHoveredCellId(null);
                setSelectedCellId(null);
              }}
            >
              <RenderExportSync exportRef={renderExportRef} />
              <ambientLight intensity={0.72} />
              <directionalLight position={[120, 100, 180]} intensity={0.75} />
              {!hideUi && showXyGrid && (
                <gridHelper
                  raycast={skipPointerRaycast}
                  args={[sceneSpan, 20, "#334155", "#1f2937"]}
                  position={[
                    0,
                    manifest.microenvironment_grid === "3d_volume" &&
                    manifest.z_min != null &&
                    manifest.z_max != null
                      ? (manifest.z_min + manifest.z_max) / 2
                      : -0.8,
                    manifest.microenvironment_grid === "3d_volume" &&
                    manifest.y_min != null &&
                    manifest.y_max != null
                      ? (manifest.y_min + manifest.y_max) / 2
                      : 0,
                  ]}
                />
              )}
              {showDomainBoundingGrid && (
                <DomainBoundingGrid manifest={manifest} />
              )}
              {showField &&
                (manifest.microenvironment_grid === "3d_volume" ? (
                  <VoxelField3D
                    payload={framePayload}
                    manifest={manifest}
                    substrateIndex={substrateIndex}
                    substrateRange={substrateRange}
                    clipIx={volIx}
                    clipIy={volIy}
                    clipIz={volIz}
                  />
                ) : (
                  <VoxelField2D
                    payload={framePayload}
                    manifest={manifest}
                    substrateIndex={substrateIndex}
                    substrateRange={substrateRange}
                  />
                ))}
              {showCells && (
                <CellSpheres
                  payload={framePayload}
                  manifest={manifest}
                  colorMode={cellColorMode}
                  clipIx={cellIx}
                  clipIy={cellIy}
                  clipIz={cellIz}
                  onHoverCellId={setHoveredCellId}
                  onSelectCellId={setSelectedCellId}
                  hiddenCellTypes={hiddenCellTypes}
                  hiddenStates={hiddenStates}
                />
              )}
              <OrbitControls
                makeDefault
                enablePan
                enableRotate
                enableZoom
                dampingFactor={0.08}
              />
            </Canvas>
          </div>
        </>
      )}
    </div>
  );
}
