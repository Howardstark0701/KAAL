'use client';

import { useEffect, useRef } from 'react';
import { animate } from 'framer-motion';
import { getRiskTier, formatScore } from '../lib/kvs';

// Circle geometry
const RADIUS       = 86;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS; // ≈ 540.35

interface KVSGaugeProps {
  score: number;
  size?: number;
}

export default function KVSGauge({ score, size = 200 }: KVSGaugeProps) {
  const tier       = getRiskTier(score);
  const targetOffset = CIRCUMFERENCE - (score / 10) * CIRCUMFERENCE;

  // Ref to the arc element so we can animate its strokeDashoffset directly
  const arcRef = useRef<SVGCircleElement>(null);

  useEffect(() => {
    if (!arcRef.current) return;

    // Start from empty (full offset) and animate to the target fill
    const controls = animate(CIRCUMFERENCE, targetOffset, {
      duration: 0.8,
      ease: 'easeOut',
      onUpdate(value) {
        if (arcRef.current) {
          arcRef.current.style.strokeDashoffset = String(value);
        }
      },
    });

    return () => controls.stop();
  }, [score, targetOffset]);

  return (
    <svg
      viewBox="0 0 200 200"
      width={size}
      height={size}
      aria-label={`KVS score ${score.toFixed(1)} out of 10 — ${tier.label}`}
      role="meter"
      aria-valuenow={score}
      aria-valuemin={0}
      aria-valuemax={10}
    >
      {/* Background track */}
      <circle
        cx={100}
        cy={100}
        r={RADIUS}
        fill="none"
        stroke="#1F1F1F"
        strokeWidth={8}
      />

      {/* Animated score arc — rotated so start is at 12 o'clock */}
      <circle
        ref={arcRef}
        cx={100}
        cy={100}
        r={RADIUS}
        fill="none"
        stroke={tier.hex}
        strokeWidth={8}
        strokeLinecap="round"
        strokeDasharray={CIRCUMFERENCE}
        strokeDashoffset={CIRCUMFERENCE}   // starts empty; JS animates to targetOffset
        style={{ transform: 'rotate(-90deg)', transformOrigin: '100px 100px' }}
      />

      {/* Score text — centered */}
      <text
        x={100}
        y={96}
        textAnchor="middle"
        dominantBaseline="middle"
        fontFamily="'JetBrains Mono', 'Fira Code', monospace"
        fontSize={32}
        fontWeight={700}
        fill="#F2F2F2"
      >
        {score.toFixed(1)}
      </text>

      {/* Risk label — below score */}
      <text
        x={100}
        y={122}
        textAnchor="middle"
        dominantBaseline="middle"
        fontFamily="Inter, -apple-system, sans-serif"
        fontSize={12}
        fill="#888888"
      >
        {tier.label}
      </text>
    </svg>
  );
}
