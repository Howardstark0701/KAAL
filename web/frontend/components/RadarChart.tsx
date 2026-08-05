'use client';

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
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
}

export default function KaalRadarChart({ scores, scoresB, labelA = 'Model', labelB = 'After' }: RadarChartProps) {
  const data = DIM_ORDER.map((key) => ({
    dim: DIM_LABELS[key] ?? key,
    A: scores[key] ?? 0,
    ...(scoresB ? { B: scoresB[key] ?? 0 } : {}),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="#333" />
        <PolarAngleAxis
          dataKey="dim"
          tick={{ fill: '#9CA3AF', fontSize: 11 }}
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
