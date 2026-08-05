interface ProgressBarProps {
  pct: number;
  label?: string;
  className?: string;
}

export default function ProgressBar({ pct, label, className = '' }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className={`w-full ${className}`} role="progressbar" aria-valuenow={clamped} aria-valuemin={0} aria-valuemax={100} aria-label={label ?? 'Progress'}>
      {label && (
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span className="truncate pr-2">{label}</span>
          <span className="shrink-0">{clamped}%</span>
        </div>
      )}
      <div className="h-2 rounded-full bg-[#222] overflow-hidden">
        <div
          className="h-full rounded-full bg-[#CC0000] transition-all duration-300"
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
