/**
 * Vector exports aligned with the SimulationViewer camera.
 * - Cells: projected disks with approximate membrane shading (not GLSL-identical).
 * - Microenvironment: projected voxel boxes, same heat colormap and alpha as the 3D field.
 * - Combined: depth-sorted field + cells in one SVG.
 */

import * as THREE from "three";
import { heatToRgb, microenvironmentHeatToRgb } from "@/lib/heatColor";
import type { VizManifest } from "@/lib/vizTypes";
import { microenvironmentFrameVoxelAlpha } from "@/lib/voxelInstanceAlphaShader";

export type CellColorMode =
  | "cell_type"
  | "growth_rate"
  | "state"
  | "dead"
  | "current_death_model"
  | `custom:${string}`;

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.floor(v)));
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

function numericCellColorHex(value: number, mn: number, mx: number): string {
  if (!(mx > mn)) return "#cbd5e1";
  const t = Math.max(0, Math.min(1, (value - mn) / (mx - mn)));
  const [r, g, b] = heatToRgb(t);
  const h = (n: number) =>
    Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`;
}

function threeColorToHex(c: THREE.Color): string {
  const h = (n: number) =>
    Math.max(0, Math.min(255, Math.round(n * 255))).toString(16).padStart(2, "0");
  return `#${h(c.r)}${h(c.g)}${h(c.b)}`;
}

/** Match SimulationViewer CellSpheres coloring (flat fill for export). */
export function getCellColorHex(
  cell: Record<string, number>,
  colorMode: CellColorMode,
  allCells: Record<string, number>[],
  fallbackRadius: number
): string {
  const customKey = colorMode.startsWith("custom:")
    ? colorMode.slice("custom:".length)
    : null;
  const numericKey = colorMode === "growth_rate" ? "__growth_rate__" : customKey;
  let mn = Infinity;
  let mx = -Infinity;
  if (numericKey) {
    for (const c of allCells) {
      const val: number =
        (numericKey === "__growth_rate__"
          ? getCellGrowthRate(c)
          : Number(c[numericKey])) ?? NaN;
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

  const dead = (cell.dead ?? 0) > 0.5;
  const apoptoticFlag = Number(cell.apoptotic ?? 0) > 0.5;
  const necroticFlag = Number(cell.necrotic ?? 0) > 0.5;
  const deathModel = Number(cell.current_death_model ?? -1);
  const growthRate: number = getCellGrowthRate(cell) ?? NaN;
  const deathRate: number = getCellDeathRate(cell) ?? NaN;
  const apoptosisRate = Number(cell.x_death_rates ?? 0);
  const necrosisRate = Number(cell.y_death_rates ?? 0);
  const hasApoptosisRate = Number.isFinite(apoptosisRate) && apoptosisRate > 1e-12;
  const hasNecrosisRate = Number.isFinite(necrosisRate) && necrosisRate > 1e-12;
  const hasNearZeroGrowth = Number.isFinite(growthRate) && Math.abs(growthRate) <= 1e-12;
  const hasNonzeroDeathRate = Number.isFinite(deathRate) && deathRate > 1e-12;
  const apoptoticModel = dead && (deathModel === 100 || deathModel === 0);
  const necroticModel = dead && (deathModel === 101 || deathModel === 1);
  const isApoptotic =
    hasApoptosisRate ? apoptosisRate >= necrosisRate : apoptoticFlag || apoptoticModel;
  const isNecrotic =
    hasNecrosisRate ? necrosisRate > apoptosisRate : necroticFlag || necroticModel;
  const isPreDead =
    !dead && !isApoptotic && !isNecrotic && hasNearZeroGrowth && hasNonzeroDeathRate;
  const ct = cell.cell_type ?? 0;

  let color = new THREE.Color("#38bdf8");
  if (colorMode === "state") {
    if (isNecrotic) color = new THREE.Color("#7f1d1d");
    else if (isApoptotic) color = new THREE.Color("#ef4444");
    else if (isPreDead) color = new THREE.Color("#f59e0b");
    else if (dead) color = new THREE.Color("#64748b");
    else color = new THREE.Color("#22c55e");
  } else if (colorMode === "dead") {
    color = dead ? new THREE.Color("#ef4444") : new THREE.Color("#22c55e");
  } else if (colorMode === "current_death_model") {
    color = deathModel >= 0 ? new THREE.Color("#f59e0b") : new THREE.Color("#22c55e");
  } else if (colorMode === "growth_rate") {
    const val: number = getCellGrowthRate(cell) ?? NaN;
    return Number.isFinite(val) ? numericCellColorHex(val, mn, mx) : "#64748b";
  } else if (customKey) {
    const val =
      customKey === "derived_radius"
        ? getCellRadius(cell, fallbackRadius)
        : Number(cell[customKey]);
    return Number.isFinite(val) ? numericCellColorHex(val, mn, mx) : "#64748b";
  } else {
    if (ct >= 0.5) color = new THREE.Color("#fb923c");
    if (dead) color = new THREE.Color("#64748b");
  }
  return threeColorToHex(color);
}

export type VectorCellExportInput = {
  cells: Record<string, number>[];
  manifest: VizManifest;
  camera: THREE.PerspectiveCamera;
  /** Output dimensions in CSS pixels (match canvas client size). */
  width: number;
  height: number;
  colorMode: CellColorMode;
  clipIx: readonly [number, number];
  clipIy: readonly [number, number];
  clipIz: readonly [number, number];
  /** If set, SVG gets this background; omit rect when null/undefined for transparent. */
  backgroundHex?: string | null;
  /** Optional subtitle in SVG (e.g. time). */
  title?: string;
};

/** Substrate field only (uses {@link volIx}/volIy/volIz like VoxelField3D; 2D uses full XY slab). */
export type VectorMicroEnvExportInput = {
  manifest: VizManifest;
  camera: THREE.PerspectiveCamera;
  width: number;
  height: number;
  substrates: Float32Array;
  substrateIndex: number;
  substrateRange: { min: number; max: number } | null;
  volIx: readonly [number, number];
  volIy: readonly [number, number];
  volIz: readonly [number, number];
  backgroundHex?: string | null;
  title?: string;
};

/** Field + cells: cell clips from {@link VectorCellExportInput}; volume clip from volI*. */
export type VectorCombinedExportInput = VectorCellExportInput & {
  substrates: Float32Array;
  substrateIndex: number;
  substrateRange: { min: number; max: number } | null;
  volIx: readonly [number, number];
  volIy: readonly [number, number];
  volIz: readonly [number, number];
};

function escAttr(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const VOXEL_MESH_FILL = 0.998;

function voxelStep1D(lo: number, hi: number, nCells: number): number {
  const span = Number(hi) - Number(lo);
  if (!Number.isFinite(span)) return 1;
  if (nCells <= 1) return Math.max(Math.abs(span), 1);
  return Math.max(Math.abs(span) / Math.max(1, nCells - 1), 1e-6);
}

/** Screen AABB of a world-space axis-aligned box under perspective (8 corners). */
function projectBoxToScreenRect(
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number,
  center: THREE.Vector3,
  halfX: number,
  halfY: number,
  halfZ: number
): { sx: number; sy: number; sw: number; sh: number } | null {
  const corner = new THREE.Vector3();
  let minSx = Infinity;
  let maxSx = -Infinity;
  let minSy = Infinity;
  let maxSy = -Infinity;
  for (const a of [-1, 1] as const) {
    for (const b of [-1, 1] as const) {
      for (const c of [-1, 1] as const) {
        corner.copy(center).add(new THREE.Vector3(a * halfX, b * halfY, c * halfZ));
        corner.project(camera);
        const sx = (corner.x * 0.5 + 0.5) * width;
        const sy = (-corner.y * 0.5 + 0.5) * height;
        minSx = Math.min(minSx, sx);
        maxSx = Math.max(maxSx, sx);
        minSy = Math.min(minSy, sy);
        maxSy = Math.max(maxSy, sy);
      }
    }
  }
  if (!(maxSx > minSx && maxSy > minSy)) return null;
  return { sx: minSx, sy: minSy, sw: maxSx - minSx, sh: maxSy - minSy };
}

type VoxelSvgItem = {
  depth: number;
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  fillRgb: string;
  opacity: number;
};

function collectMicroenvironmentVoxels(input: VectorMicroEnvExportInput): VoxelSvgItem[] {
  const {
    manifest,
    camera,
    width,
    height,
    substrates,
    substrateIndex,
    substrateRange,
    volIx,
    volIy,
    volIz,
  } = input;
  const nx = manifest.nx;
  const ny = manifest.ny;
  const nz = manifest.nz ?? 1;
  const is3d = manifest.microenvironment_grid === "3d_volume" && nz > 1;

  let mn = substrateRange?.min ?? 0;
  let mx = substrateRange?.max ?? 1;
  if (!(mx > mn)) {
    mn = 0;
    mx = 1;
  }

  const count = is3d ? nx * ny * nz : nx * ny;
  const base = substrateIndex * count;
  if (substrates.length < base + count) return [];

  const { x_min, x_max, y_min, y_max, z_min = 0, z_max = 0 } = manifest;
  const dx = voxelStep1D(x_min, x_max, nx);
  const dy = voxelStep1D(y_min, y_max, ny);
  const zSpan = z_max - z_min;
  const dz = nz <= 1 ? 0 : zSpan / Math.max(1, nz - 1);
  const planMin = Math.min(dx, dy);
  const tol =
    1e-9 * Math.max(Math.abs(x_max - x_min), Math.abs(y_max - y_min), Math.abs(zSpan), 1);
  const dzBox = nz <= 1 || zSpan < tol || dz < tol ? planMin : dz;

  const [ixLo, ixHi] = volIx;
  const [iyLo, iyHi] = volIy;
  const [izLo, izHi] = volIz;

  const camPos = new THREE.Vector3();
  camera.getWorldPosition(camPos);

  const out: VoxelSvgItem[] = [];

  if (is3d) {
    const boxGeoX = Math.max(dx * VOXEL_MESH_FILL, 1e-4) / 2;
    const boxGeoY = Math.max(dzBox * VOXEL_MESH_FILL, 1e-4) / 2;
    const boxGeoZ = Math.max(dy * VOXEL_MESH_FILL, 1e-4) / 2;
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        for (let ix = 0; ix < nx; ix++) {
          if (ix < ixLo || ix > ixHi || iy < iyLo || iy > iyHi || iz < izLo || iz > izHi) {
            continue;
          }
          const idx = iz * nx * ny + iy * nx + ix;
          const val = substrates[base + idx]!;
          const x = x_min + ix * dx;
          const y = y_min + iy * dy;
          const z = nz <= 1 ? z_min : z_min + iz * dz;
          const center = new THREE.Vector3(x, z, y);
          const pr = projectBoxToScreenRect(
            camera,
            width,
            height,
            center,
            boxGeoX,
            boxGeoY,
            boxGeoZ
          );
          if (!pr || pr.sw < 0.2 || pr.sh < 0.2) continue;
          const t = (val - mn) / (mx - mn);
          const [r, g, b] = microenvironmentHeatToRgb(t);
          const opacity = microenvironmentFrameVoxelAlpha(val, mn, mx);
          out.push({
            depth: center.distanceTo(camPos),
            sx: pr.sx,
            sy: pr.sy,
            sw: pr.sw,
            sh: pr.sh,
            fillRgb: `rgb(${r},${g},${b})`,
            opacity,
          });
        }
      }
    }
  } else {
    const boxX = (dx * VOXEL_MESH_FILL) / 2;
    const boxZ = (dy * VOXEL_MESH_FILL) / 2;
    const boxYTh = Math.max(Math.min(dx, dy) * 0.25, 1e-4);
    const boxY = boxYTh / 2;
    const yBase = -boxYTh * 0.55;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const idx = iy * nx + ix;
        const val = substrates[base + idx]!;
        const x = x_min + ix * dx;
        const yPhys = y_min + iy * dy;
        const center = new THREE.Vector3(x, yBase, yPhys);
        const pr = projectBoxToScreenRect(camera, width, height, center, boxX, boxY, boxZ);
        if (!pr || pr.sw < 0.15 || pr.sh < 0.15) continue;
        const t = (val - mn) / (mx - mn);
        const [r, g, b] = microenvironmentHeatToRgb(t);
        const opacity = microenvironmentFrameVoxelAlpha(val, mn, mx);
        out.push({
          depth: center.distanceTo(camPos),
          sx: pr.sx,
          sy: pr.sy,
          sw: pr.sw,
          sh: pr.sh,
          fillRgb: `rgb(${r},${g},${b})`,
          opacity,
        });
      }
    }
  }
  return out;
}

export function buildMicroenvironmentSvgString(input: VectorMicroEnvExportInput): string {
  const { width, height, backgroundHex, title } = input;
  const voxels = collectMicroenvironmentVoxels(input);
  voxels.sort((a, b) => b.depth - a.depth);

  const esc = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  const parts: string[] = [];
  parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`
  );
  if (backgroundHex) {
    parts.push(`<rect width="100%" height="100%" fill="${esc(backgroundHex)}"/>`);
  }
  if (title) {
    const bg = backgroundHex ? new THREE.Color(backgroundHex) : null;
    const bgLum = bg ? bg.r * 0.2126 + bg.g * 0.7152 + bg.b * 0.0722 : 0;
    const titleFill = bgLum > 0.72 ? "#171717" : "#ffffff";
    parts.push(
      `<text x="8" y="19" fill="${escAttr(titleFill)}" font-family="Helvetica Neue, Arial, Noto Sans, sans-serif" font-size="13" font-weight="600" letter-spacing="0.01em">${esc(
        title
      )}</text>`
    );
  }
  for (const v of voxels) {
    parts.push(
      `<rect x="${v.sx.toFixed(2)}" y="${v.sy.toFixed(2)}" width="${v.sw.toFixed(2)}" height="${v.sh.toFixed(2)}" fill="${escAttr(v.fillRgb)}" fill-opacity="${v.opacity.toFixed(4)}" stroke="none"/>`
    );
  }
  parts.push(`</svg>`);
  return parts.join("\n");
}

/**
 * Same direction as SimulationViewer: <directionalLight position={[120, 100, 180]} /> —
 * parallel rays; L points from the cell toward the light source (world).
 */
const SCENE_LIGHT_TOWARD = new THREE.Vector3(120, 100, 180).normalize();

/** Ambient / directional intensities from the viewer scene (approximate). */
const SCENE_AMBIENT_I = 0.72;
const SCENE_DIR_I = 0.75;

/**
 * Viewer uses white meshPhysicalMaterial × per-instance color → albedo equals {@link getCellColorHex}.
 * SVG cannot run that shader; we preserve hue/chroma by **scaling** the same base RGB (diffuse)
 * and add only a **small additive** highlight (no heavy lerp to white/cool, which shifted colors).
 * Layout is kept calm for print-style figures (soft radial, thin neutral outline).
 */
function addSubtleSpecular(baseScaled: THREE.Color, spec: number): THREE.Color {
  const out = baseScaled.clone();
  const s = Math.min(0.2, spec);
  out.r = Math.min(1, out.r + s * 0.92);
  out.g = Math.min(1, out.g + s * 0.94);
  out.b = Math.min(1, out.b + s);
  return out;
}

function cellMembraneSvgShading(
  baseHex: string,
  wpos: THREE.Vector3,
  camera: THREE.PerspectiveCamera,
  sx: number,
  sy: number,
  rPx: number
): { gx: number; gy: number; gradR: number; bright: string; mid: string; edge: string } {
  const base = new THREE.Color(baseHex);
  const camPos = new THREE.Vector3();
  camera.getWorldPosition(camPos);
  const N = camPos.clone().sub(wpos).normalize();
  const L = SCENE_LIGHT_TOWARD.clone();
  const V = N.clone();
  const ndotl = Math.max(0, N.dot(L));
  const H = L.clone().add(V).normalize();
  const ndoth = Math.max(0, N.dot(H));
  const specAmt = Math.pow(ndoth, 24) * 0.16 + Math.pow(ndoth, 72) * 0.07;

  const amb = 0.48 * SCENE_AMBIENT_I;
  const dir = 0.52 * SCENE_DIR_I * ndotl;
  const lighting = amb + dir;
  const t = Math.min(1, lighting / 0.88);
  const diffuseScalar = 0.5 + 0.48 * t;

  const brightBase = base.clone().multiplyScalar(diffuseScalar * 1.03);
  const midBase = base.clone().multiplyScalar(diffuseScalar * 0.88);
  const edgeBase = base.clone().multiplyScalar(diffuseScalar * 0.63);

  const brightC = addSubtleSpecular(brightBase, specAmt);
  const midC = addSubtleSpecular(midBase, specAmt * 0.32);
  const edgeC = edgeBase;

  const lView = L.clone().transformDirection(camera.matrixWorldInverse);
  let rawDx = lView.x * rPx * 0.38;
  let rawDy = -lView.y * rPx * 0.38;
  const maxO = rPx * 0.34;
  const len = Math.hypot(rawDx, rawDy);
  if (len > maxO && len > 1e-6) {
    const s = maxO / len;
    rawDx *= s;
    rawDy *= s;
  }
  const gx = sx + rawDx;
  const gy = sy + rawDy;
  const distFocus = Math.hypot(rawDx, rawDy);
  const gradR = Math.max(rPx * 1.08, distFocus + rPx * 0.9);

  return {
    gx,
    gy,
    gradR,
    bright: threeColorToHex(brightC),
    mid: threeColorToHex(midC),
    edge: threeColorToHex(edgeC),
  };
}

export function buildCellsSvgString(input: VectorCellExportInput): string {
  const { width, height, backgroundHex, title, camera } = input;
  const prepared = prepareCellsRaw(input);
  prepared.sort((a, b) => b.depth - a.depth);

  const esc = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  const parts: string[] = [];
  parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`
  );
  const defs: string[] = ["<defs>"];
  prepared.forEach((p, i) => {
    const sh = cellMembraneSvgShading(p.fill, p.wpos, camera, p.sx, p.sy, p.rPx);
    const id = `cellMemb_${i}`;
    defs.push(
      `<radialGradient id="${id}" gradientUnits="userSpaceOnUse" cx="${sh.gx.toFixed(2)}" cy="${sh.gy.toFixed(2)}" r="${sh.gradR.toFixed(2)}">` +
        `<stop offset="0%" stop-color="${escAttr(sh.bright)}"/>` +
        `<stop offset="52%" stop-color="${escAttr(sh.mid)}"/>` +
        `<stop offset="100%" stop-color="${escAttr(sh.edge)}"/>` +
        `</radialGradient>`
    );
  });
  defs.push("</defs>");
  parts.push(defs.join(""));

  if (backgroundHex) {
    parts.push(`<rect width="100%" height="100%" fill="${esc(backgroundHex)}"/>`);
  }
  if (title) {
    const bg = backgroundHex ? new THREE.Color(backgroundHex) : null;
    const bgLum = bg
      ? bg.r * 0.2126 + bg.g * 0.7152 + bg.b * 0.0722
      : 0;
    const titleFill = bgLum > 0.72 ? "#171717" : "#ffffff";
    parts.push(
      `<text x="8" y="19" fill="${escAttr(titleFill)}" font-family="Helvetica Neue, Arial, Noto Sans, sans-serif" font-size="13" font-weight="600" letter-spacing="0.01em">${esc(
        title
      )}</text>`
    );
  }

  for (let i = 0; i < prepared.length; i++) {
    const p = prepared[i]!;
    const id = `cellMemb_${i}`;
    parts.push(
      `<circle cx="${p.sx.toFixed(2)}" cy="${p.sy.toFixed(2)}" r="${p.rPx.toFixed(2)}" fill="url(#${id})" stroke="#111827" stroke-opacity="0.10" stroke-width="${Math.max(0.25, p.rPx * 0.022).toFixed(2)}" opacity="1"/>`
    );
  }
  parts.push(`</svg>`);
  return parts.join("\n");
}

/** Internal: prepared cells with projected screen position and radius. */
type PreparedCellFull = {
  wpos: THREE.Vector3;
  radius: number;
  fill: string;
  depth: number;
  sx: number;
  sy: number;
  rPx: number;
};

function prepareCellsRaw(input: VectorCellExportInput): PreparedCellFull[] {
  const {
    cells,
    manifest,
    camera,
    colorMode,
    clipIx,
    clipIy,
    clipIz,
    width,
    height,
  } = input;
  const { nx, ny, x_min, x_max } = manifest;
  const fallbackRadius = Math.max(0.25, (x_max - x_min) / Math.max(1, nx) * 0.35);
  const useRawZ =
    manifest.microenvironment_grid === "3d_volume" && (manifest.nz ?? 1) > 1;
  const nz = manifest.nz ?? 1;
  const dx = (manifest.x_max - manifest.x_min) / Math.max(1, nx - 1);
  const dy = (manifest.y_max - manifest.y_min) / Math.max(1, ny - 1);
  const dz = ((manifest.z_max ?? 0) - (manifest.z_min ?? 0)) / Math.max(1, nz - 1);
  const [ixLo, ixHi] = clipIx;
  const [iyLo, iyHi] = clipIy;
  const [izLo, izHi] = clipIz;

  const camPos = new THREE.Vector3();
  camera.getWorldPosition(camPos);
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();

  const out: PreparedCellFull[] = [];

  for (const c of cells) {
    const x = c.x_position ?? 0;
    const y = c.y_position ?? 0;
    const z = c.z_position ?? 0;
    const cellRadius = getCellRadius(c, fallbackRadius);
    const wy = useRawZ ? z : z + cellRadius * 0.65;
    const wpos = new THREE.Vector3(x, wy, y);

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
    if (!show) continue;

    const pCenter = wpos.clone().project(camera);
    if (
      pCenter.z < -1 ||
      pCenter.z > 1 ||
      pCenter.x < -1.05 ||
      pCenter.x > 1.05 ||
      pCenter.y < -1.05 ||
      pCenter.y > 1.05
    ) {
      continue;
    }

    const pEdge = wpos.clone().addScaledVector(right, cellRadius).project(camera);
    const sx = (pCenter.x * 0.5 + 0.5) * width;
    const sy = (-pCenter.y * 0.5 + 0.5) * height;
    const sx1 = (pEdge.x * 0.5 + 0.5) * width;
    const sy1 = (-pEdge.y * 0.5 + 0.5) * height;
    const rPx = Math.max(0.5, Math.hypot(sx1 - sx, sy1 - sy));

    const fill = getCellColorHex(c, colorMode, cells, fallbackRadius);
    const depth = wpos.distanceTo(camPos);
    out.push({ wpos, radius: cellRadius, fill, depth, sx, sy, rPx });
  }
  return out;
}

export function buildCombinedFieldAndCellsSvg(input: VectorCombinedExportInput): string {
  const {
    width,
    height,
    backgroundHex,
    title,
    camera,
    manifest,
    substrates,
    substrateIndex,
    substrateRange,
    volIx,
    volIy,
    volIz,
  } = input;

  const voxels = collectMicroenvironmentVoxels({
    manifest,
    camera,
    width,
    height,
    substrates,
    substrateIndex,
    substrateRange,
    volIx,
    volIy,
    volIz,
  });
  const cellsPrepared = prepareCellsRaw(input);

  type Layer =
    | { k: "v"; depth: number; v: VoxelSvgItem }
    | { k: "c"; depth: number; i: number; p: PreparedCellFull };

  const layers: Layer[] = [
    ...voxels.map((v) => ({ k: "v" as const, depth: v.depth, v })),
    ...cellsPrepared.map((p, i) => ({ k: "c" as const, depth: p.depth, i, p })),
  ];
  layers.sort((a, b) => b.depth - a.depth);

  const esc = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  const parts: string[] = [];
  parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`
  );

  const defs: string[] = ["<defs>"];
  for (let i = 0; i < cellsPrepared.length; i++) {
    const p = cellsPrepared[i]!;
    const sh = cellMembraneSvgShading(p.fill, p.wpos, camera, p.sx, p.sy, p.rPx);
    const id = `cellMemb_${i}`;
    defs.push(
      `<radialGradient id="${id}" gradientUnits="userSpaceOnUse" cx="${sh.gx.toFixed(2)}" cy="${sh.gy.toFixed(2)}" r="${sh.gradR.toFixed(2)}">` +
        `<stop offset="0%" stop-color="${escAttr(sh.bright)}"/>` +
        `<stop offset="52%" stop-color="${escAttr(sh.mid)}"/>` +
        `<stop offset="100%" stop-color="${escAttr(sh.edge)}"/>` +
        `</radialGradient>`
    );
  }
  defs.push("</defs>");
  parts.push(defs.join(""));

  if (backgroundHex) {
    parts.push(`<rect width="100%" height="100%" fill="${esc(backgroundHex)}"/>`);
  }
  if (title) {
    const bg = backgroundHex ? new THREE.Color(backgroundHex) : null;
    const bgLum = bg ? bg.r * 0.2126 + bg.g * 0.7152 + bg.b * 0.0722 : 0;
    const titleFill = bgLum > 0.72 ? "#171717" : "#ffffff";
    parts.push(
      `<text x="8" y="19" fill="${escAttr(titleFill)}" font-family="Helvetica Neue, Arial, Noto Sans, sans-serif" font-size="13" font-weight="600" letter-spacing="0.01em">${esc(
        title
      )}</text>`
    );
  }

  for (const L of layers) {
    if (L.k === "v") {
      const v = L.v;
      parts.push(
        `<rect x="${v.sx.toFixed(2)}" y="${v.sy.toFixed(2)}" width="${v.sw.toFixed(2)}" height="${v.sh.toFixed(2)}" fill="${escAttr(v.fillRgb)}" fill-opacity="${v.opacity.toFixed(4)}" stroke="none"/>`
      );
    } else {
      const p = L.p;
      const id = `cellMemb_${L.i}`;
      parts.push(
        `<circle cx="${p.sx.toFixed(2)}" cy="${p.sy.toFixed(2)}" r="${p.rPx.toFixed(2)}" fill="url(#${id})" stroke="#111827" stroke-opacity="0.10" stroke-width="${Math.max(0.25, p.rPx * 0.022).toFixed(2)}" opacity="1"/>`
      );
    }
  }
  parts.push(`</svg>`);
  return parts.join("\n");
}

/**
 * Wrap a high-res PNG (native canvas pixels, transparent background) in an SVG that declares
 * CSS display dimensions. Tools like Figma import the SVG at the correct visual size while
 * the embedded PNG carries the full rendering resolution.
 *
 * @param displayWidth  CSS pixel width  (canvas.clientWidth)
 * @param displayHeight CSS pixel height (canvas.clientHeight)
 * @param nativeWidth   Actual PNG width  (canvas.width — boosted by DPR during export)
 * @param nativeHeight  Actual PNG height (canvas.height)
 */
export function buildRasterizedSvgFromPngDataUrl(input: {
  displayWidth: number;
  displayHeight: number;
  nativeWidth: number;
  nativeHeight: number;
  pngDataUrl: string;
  title?: string;
}): string {
  const { displayWidth, displayHeight, nativeWidth, nativeHeight, pngDataUrl, title } = input;
  const esc = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const parts: string[] = [];
  parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
  // viewBox matches display size; width/height are in px so Figma respects dimensions.
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"` +
      ` width="${displayWidth}" height="${displayHeight}"` +
      ` viewBox="0 0 ${displayWidth} ${displayHeight}">`
  );
  // The image element stretches the high-res PNG into the display viewport.
  // Figma imports this as an embedded image at full resolution.
  parts.push(
    `<image x="0" y="0" width="${displayWidth}" height="${displayHeight}"` +
      ` xlink:href="${esc(pngDataUrl)}"` +
      ` image-rendering="optimizeQuality" preserveAspectRatio="none"` +
      ` data-native-width="${nativeWidth}" data-native-height="${nativeHeight}"/>`
  );
  if (title) {
    parts.push(
      `<text x="10" y="22" fill="#ffffff" fill-opacity="0.75"` +
        ` font-family="Helvetica Neue, Arial, Noto Sans, sans-serif"` +
        ` font-size="13" font-weight="600" letter-spacing="0.01em">${esc(title)}</text>`
    );
  }
  parts.push(`</svg>`);
  return parts.join("\n");
}

export function downloadTextFile(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
