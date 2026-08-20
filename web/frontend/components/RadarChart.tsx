'use client';

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { DIM_LABELS, DIM_ORDER } from '../lib/kvs';

interface RadarChartProps {
  /** Primary dataset — always shown */
  scores: Record<string, number>;
  /** Optional second dataset for compare view */
  scoresB?: Record<string, number>;
  labelA?: string;
  labelB?: string;
  /** Dimensions that were not tested — rendered as gaps, not zeros */
  skipped?: string[];
  skippedB?: string[];
}

export default function KaalRadarChart({
  scores, scoresB, labelA = 'Model', labelB = 'After', skipped, skippedB,
}: RadarChartProps) {
  // An untested dimension must not plot at 0. On a vulnerability scale 0 reads
  // as "perfectly robust", which is the opposite of "we have no evidence".
  // null makes recharts break the polygon at that axis instead.
  const valueFor = (
    src: Record<string, number> | undefined,
    key: string,
    skips: string[] | undefined,
  ): number | null => {
    if (!src) return null;
    if (skips?.includes(key)) return null;
    const v = src[key];
    return typeof v === 'number' ? v : null;
  };

  const data = DIM_ORDER.map((key) => ({
    dim: DIM_LABELS[key] ?? key,
    A: valueFor(scores, key, skipped),
    ...(scoresB ? { B: valueFor(scoresB, key, skippedB ?? skipped) } : {}),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="#333" />
        <PolarAngleAxis
          dataKey="dim"
          tick={{ fill: '#9CA3AF', fontSize: 11 }}
        />
        {/* KVS dimensions are always 0–10. Without an explicit domain recharts
            auto-scales the radius to a "nice" number above the data max, so
            the outer ring moves between audits and two fingerprints cannot be
            compared by shape or area. */}
        <PolarRadiusAxis
          domain={[0, 10]}
          tickCount={6}
          tick={false}
          axisLine={false}
        />
        <Radar
          name={labelA}
          dataKey="A"
          stroke="#CC0000"
          fill="#CC0000"
          fillOpacity={0.25}
          strokeWidth={2}
        />
        {scoresB && (
          <Radar
            name={labelB}
            dataKey="B"
            stroke="#3B82F6"
            fill="#3B82F6"
            fillOpacity={0.15}
            strokeWidth={2}
          />
        )}
        {scoresB && (
          <Legend
            wrapperStyle={{ fontSize: '12px', color: '#9CA3AF' }}
          />
        )}
      </RadarChart>
    </ResponsiveContainer>
  );
}
