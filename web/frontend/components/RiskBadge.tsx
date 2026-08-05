import { getRiskTier, formatScore } from '../lib/kvs';

interface RiskBadgeProps {
  score: number;
  showScore?: boolean;
}

export default function RiskBadge({ score, showScore = true }: RiskBadgeProps) {
  const tier = getRiskTier(score);
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border"
      style={{ color: tier.hex, borderColor: tier.hex, backgroundColor: `${tier.hex}18` }}
    >
      {showScore && <span className="font-mono">{formatScore(score)}</span>}
      <span>{tier.label}</span>
    </span>
  );
}
