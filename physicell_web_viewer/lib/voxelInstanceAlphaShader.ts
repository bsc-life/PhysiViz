import type { WebGLProgramParametersWithUniforms } from "three";

/** Must match material `customProgramCacheKey` so Three.js rebuilds program when hook changes. */
export const VOXEL_INSTANCE_ALPHA_CACHE_KEY = "voxelInstanceAlpha_v1";

/**
 * Per-instance opacity via `instanceAlpha` InstancedBufferAttribute (float, 0–1).
 * Call from `meshStandardMaterial` `onBeforeCompile`; set `customProgramCacheKey` to
 * {@link VOXEL_INSTANCE_ALPHA_CACHE_KEY}.
 */
export function voxelInstanceAlphaOnBeforeCompile(
  shader: WebGLProgramParametersWithUniforms
): void {
  shader.vertexShader = shader.vertexShader.replace(
    "#include <common>",
    `
    #include <common>
    attribute float instanceAlpha;
    varying float vInstanceAlpha;
    `
  );
  shader.vertexShader = shader.vertexShader.replace(
    "#include <begin_vertex>",
    `
    #include <begin_vertex>
    vInstanceAlpha = instanceAlpha;
    `
  );
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <common>",
    `
    #include <common>
    varying float vInstanceAlpha;
    `
  );
  shader.fragmentShader = shader.fragmentShader.replace(
    "#include <opaque_fragment>",
    `
    diffuseColor.a *= vInstanceAlpha;
    #include <opaque_fragment>
    `
  );
}

/** Same normalization as heat color: 0 at min, 1 at max (clamped). */
export function substrateNormalizedAlpha(
  v: number,
  mn: number,
  mx: number
): number {
  if (!(mx > mn)) return 1;
  const t = (v - mn) / (mx - mn);
  return Math.max(0, Math.min(1, t));
}

/**
 * Per-frame min–max opacity for substrate voxels on dark backgrounds.
 * Avoids near-invisible lows when absolute concentration is small.
 */
export function microenvironmentFrameVoxelAlpha(
  concentration: number,
  frameMin: number,
  frameMax: number
): number {
  const t = substrateNormalizedAlpha(concentration, frameMin, frameMax);
  return Math.max(0.38, 0.28 + 0.72 * t);
}

/**
 * Voxel opacity: if `exportedSeriesMax > 0`, alpha = c / ref (absolute scale over the whole export).
 * Otherwise falls back to per-frame heat normalization (legacy bundles).
 */
export function substrateVoxelAlpha(
  concentration: number,
  frameMin: number,
  frameMax: number,
  exportedSeriesMax: number | null | undefined
): number {
  if (exportedSeriesMax != null && exportedSeriesMax > 0) {
    return Math.max(0, Math.min(1, concentration / exportedSeriesMax));
  }
  return substrateNormalizedAlpha(concentration, frameMin, frameMax);
}
