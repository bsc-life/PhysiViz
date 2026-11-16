/** Inferno-like colormap (approximate, compact). t in [0,1]. */

/** Multi-stop lerp for microenvironment legend and voxels (designed for dark backgrounds). */
function lerpRgb(
  a: [number, number, number],
  b: [number, number, number],
  u: number
): [number, number, number] {
  return [
    Math.round(a[0] + (b[0] - a[0]) * u),
    Math.round(a[1] + (b[1] - a[1]) * u),
    Math.round(a[2] + (b[2] - a[2]) * u),
  ];
}

const MICROENVIRONMENT_STOPS: Array<{ t: number; rgb: [number, number, number] }> = [
  { t: 0, rgb: [20, 45, 130] },
  { t: 0.18, rgb: [0, 140, 215] },
  { t: 0.4, rgb: [45, 210, 155] },
  { t: 0.62, rgb: [255, 215, 70] },
  { t: 1, rgb: [255, 255, 248] },
];

/**
 * Microenvironment heat colors: dark saturated blue → cyan → green → yellow → off-white.
 * Low end stays clearly visible on near-black (#09090b) backgrounds.
 */
export function microenvironmentHeatToRgb(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < MICROENVIRONMENT_STOPS.length - 1 && x > MICROENVIRONMENT_STOPS[i + 1].t) {
    i += 1;
  }
  const a = MICROENVIRONMENT_STOPS[i];
  const b = MICROENVIRONMENT_STOPS[i + 1] ?? a;
  const span = b.t - a.t || 1;
  const u = (x - a.t) / span;
  return lerpRgb(a.rgb, b.rgb, u);
}

/** CSS `linear-gradient` matching {@link microenvironmentHeatToRgb} (low → bottom, high → top). */
export function microenvironmentColormapCssGradientVertical(): string {
  const stops: string[] = [];
  const n = 40;
  for (let k = 0; k <= n; k++) {
    const tt = k / n;
    const [r, g, b] = microenvironmentHeatToRgb(tt);
    stops.push(`rgb(${r},${g},${b}) ${(tt * 100).toFixed(2)}%`);
  }
  return `linear-gradient(to top, ${stops.join(", ")})`;
}

/** CSS `linear-gradient` matching `heatToRgb` (low → bottom, high → top). */
export function heatColormapCssGradientVertical(): string {
  const stops: string[] = [];
  const n = 36;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const [r, g, b] = heatToRgb(t);
    stops.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(2)}%`);
  }
  return `linear-gradient(to top, ${stops.join(", ")})`;
}

export function heatToRgb(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  // Piecewise smooth purple→blue→green→yellow
  const r = Math.floor(
    255 * (0.06 + 2.2 * x * x * x + 1.2 * Math.max(0, x - 0.75) * 4)
  );
  const g = Math.floor(255 * (0.03 + 1.8 * Math.pow(x, 1.4)));
  const b = Math.floor(
    255 * (0.2 + 1.4 * (1 - x) * (1 - x) + 0.5 * Math.max(0, x - 0.5))
  );
  return [
    Math.min(255, Math.max(0, r)),
    Math.min(255, Math.max(0, g)),
    Math.min(255, Math.max(0, b)),
  ];
}
