"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import type { PopulationSummary } from "@/lib/vizTypes";

type Props = {
  summary: PopulationSummary | null;
  frameIndex: number;
  onScrub: (frameIndex: number) => void;
  /** null = stacked population-by-type view; otherwise a bnode_<NAME> active-fraction line. */
  selectedNode: string | null;
  onSelectNode: (node: string | null) => void;
  /** Hex colors to dim/highlight in lockstep with the 3D legend's click-to-isolate state. */
  hiddenTypes: Set<string>;
};

const CHART_W = 760;
const CHART_H = 110;
const PAD_L = 4;
const PAD_R = 4;
const PAD_T = 6;
const PAD_B = 4;

function nodeLabel(bnodeKey: string): string {
  return bnodeKey.startsWith("bnode_") ? bnodeKey.slice("bnode_".length) : bnodeKey;
}

export default function PopulationSparkline({
  summary,
  frameIndex,
  onScrub,
  selectedNode,
  onSelectNode,
  hiddenTypes,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const frames = summary?.frames ?? [];
  const n = frames.length;
  const typeNames = useMemo(() => summary?.cell_type_names ?? {}, [summary]);
  const typeColors = summary?.cell_type_colors ?? {};
  const typeCodes = useMemo(
    () => Object.keys(typeNames).sort((a, b) => Number(a) - Number(b)),
    [typeNames]
  );
  const visibleTypeCodes = useMemo(
    () => typeCodes.filter((c) => !hiddenTypes.has(c)),
    [typeCodes, hiddenTypes]
  );

  const xForFrame = useCallback(
    (i: number) => PAD_L + (n <= 1 ? 0 : (i / (n - 1)) * (CHART_W - PAD_L - PAD_R)),
    [n]
  );

  const frameForX = useCallback(
    (clientX: number) => {
      const svg = svgRef.current;
      if (!svg || n === 0) return 0;
      const rect = svg.getBoundingClientRect();
      const frac = (clientX - rect.left) / Math.max(1, rect.width);
      const x = frac * CHART_W;
      const t = (x - PAD_L) / Math.max(1e-6, CHART_W - PAD_L - PAD_R);
      return Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
    },
    [n]
  );

  const handlePointer = useCallback(
    (clientX: number) => {
      onScrub(frameForX(clientX));
    },
    [frameForX, onScrub]
  );

  if (!summary || n === 0) return null;

  let content: React.ReactNode;
  let yMax = 1;
  let yLabel = "";

  if (selectedNode) {
    const vals = frames.map((f) => f.node_active_fraction[selectedNode] ?? NaN);
    yMax = 1;
    yLabel = "active fraction";
    const innerH = CHART_H - PAD_T - PAD_B;
    const pts = vals
      .map((v, i) => {
        if (!Number.isFinite(v)) return null;
        const x = xForFrame(i);
        const y = PAD_T + innerH * (1 - v);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .filter((p): p is string => p != null)
      .join(" ");
    content = (
      <polyline
        points={pts}
        fill="none"
        stroke="#38bdf8"
        strokeWidth={1.75}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    );
  } else {
    // Stacked area over visible cell types, in ascending type-code order.
    const stacks = frames.map((f) => {
      let running = 0;
      return visibleTypeCodes.map((code) => {
        const v = f.counts_by_type[code] ?? 0;
        const y0 = running;
        running += v;
        return [y0, running] as const;
      });
    });
    yMax = Math.max(1, ...stacks.map((s) => (s.length ? s[s.length - 1][1] : 0)));
    yLabel = "cells";
    const innerH = CHART_H - PAD_T - PAD_B;
    const yOf = (v: number) => PAD_T + innerH * (1 - v / yMax);

    content = visibleTypeCodes.map((code, ti) => {
      const top = frames.map((_, i) => `${xForFrame(i).toFixed(1)},${yOf(stacks[i][ti][1]).toFixed(1)}`);
      const bottom = frames
        .map((_, i) => `${xForFrame(n - 1 - i).toFixed(1)},${yOf(stacks[n - 1 - i][ti][0]).toFixed(1)}`);
      const path = [...top, ...bottom].join(" ");
      return (
        <polygon
          key={code}
          points={path}
          fill={typeColors[code] ?? "#38bdf8"}
          fillOpacity={0.85}
          stroke="none"
        />
      );
    });
  }

  const playheadX = xForFrame(Math.min(frameIndex, n - 1));

  return (
    <div className="pointer-events-auto rounded-none border border-zinc-600 bg-zinc-950/97 px-2.5 py-2 text-[11px] text-zinc-300 shadow-sm">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="font-semibold text-zinc-300">
          {selectedNode ? `Node activity: ${nodeLabel(selectedNode)}` : "Population by cell type"}
        </span>
        <div className="flex items-center gap-1.5">
          <span className="text-zinc-500">{yLabel}</span>
          {summary.boolean_node_columns && summary.boolean_node_columns.length > 0 && (
            <select
              value={selectedNode ?? ""}
              onChange={(e) => onSelectNode(e.target.value || null)}
              className="rounded-sm border border-zinc-700 bg-zinc-900 px-1 py-0.5 text-[10px]"
              title="Plot a boolean node's active-fraction instead of population counts"
            >
              <option value="">(population)</option>
              {summary.boolean_node_columns.map((bn) => (
                <option key={bn} value={bn}>
                  {nodeLabel(bn)}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        width="100%"
        height={CHART_H}
        className="cursor-pointer select-none"
        onPointerDown={(e) => {
          setDragging(true);
          handlePointer(e.clientX);
          (e.target as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (dragging) handlePointer(e.clientX);
        }}
        onPointerUp={() => setDragging(false)}
      >
        {content}
        <line
          x1={playheadX}
          x2={playheadX}
          y1={PAD_T}
          y2={CHART_H - PAD_B}
          stroke="#f8fafc"
          strokeWidth={1.25}
          strokeOpacity={0.9}
        />
      </svg>
      {!selectedNode && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-zinc-400">
          {typeCodes.map((code) => (
            <span
              key={code}
              className="flex items-center gap-1"
              style={{ opacity: hiddenTypes.has(code) ? 0.35 : 1 }}
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ background: typeColors[code] ?? "#38bdf8" }}
              />
              {typeNames[code] ?? `type ${code}`}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
