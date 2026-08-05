// KVS score helpers — single source of truth for risk tiers, colours, labels.

export interface RiskTier {
  label: string;
  hex: string;
  tailwindText: string;
  tailwindBg: string;
  tailwindBorder: string;
}

const TIERS: Array<{ max: number } & RiskTier> = [
  { max: 2.0,  label: 'Robust',        hex: '#4ADE80', tailwindText: 'text-green-400',    tailwindBg: 'bg-green-400/10',    tailwindBorder: 'border-green-400' },
  { max: 4.0,  label: 'Low Risk',      hex: '#A3E635', tailwindText: 'text-lime-400',     tailwindBg: 'bg-lime-400/10',     tailwindBorder: 'border-lime-400' },
  { max: 6.0,  label: 'Medium Risk',   hex: '#FACC15', tailwindText: 'text-yellow-400',   tailwindBg: 'bg-yellow-400/10',   tailwindBorder: 'border-yellow-400' },
  { max: 8.0,  label: 'High Risk',     hex: '#FB923C', tailwindText: 'text-orange-400',   tailwindBg: 'bg-orange-400/10',   tailwindBorder: 'border-orange-400' },
  { max: 9.5,  label: 'Critical',      hex: '#CC0000', tailwindText: 'text-red-600',      tailwindBg: 'bg-red-600/10',      tailwindBorder: 'border-red-600' },
  { max: 10.0, label: 'Catastrophic',  hex: '#7F0000', tailwindText: 'text-red-900',      tailwindBg: 'bg-red-900/10',      tailwindBorder: 'border-red-900' },
];

export function getRiskTier(score: number): RiskTier {
  for (const tier of TIERS) {
    if (score <= tier.max) return tier;
  }
  return TIERS[TIERS.length - 1];
}

export function formatScore(score: number): string {
  return score.toFixed(2);
}

// Canonical dimension display names
export const DIM_LABELS: Record<string, string> = {
  fgsm_susceptibility:    'FGSM',
  pgd_susceptibility:     'PGD',
  perturbation_threshold: 'Perturbation Threshold',
  physical_survivability: 'Physical',
  blackbox_efficiency:    'Black-Box',
};

export const DIM_ORDER = [
  'fgsm_susceptibility',
  'pgd_susceptibility',
  'perturbation_threshold',
  'physical_survivability',
  'blackbox_efficiency',
];
