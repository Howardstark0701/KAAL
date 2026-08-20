import Head from 'next/head';
import { useId, useState } from 'react';
import dynamic from 'next/dynamic';
import RiskBadge from '../components/RiskBadge';
import { useToast } from '../components/ErrorToast';
import { compareAudits, ApiError } from '../lib/api';
import { DIM_LABELS, DIM_ORDER, formatScore, getRiskTier } from '../lib/kvs';
import type { CompareResponse } from '../lib/types';

const KaalRadarChart = dynamic(() => import('../components/RadarChart'), { ssr: false });

export default function ComparePage() {
  const { showError } = useToast();
  const beforeId = useId();
  const afterId  = useId();

  const [beforeJobId, setBeforeJobId] = useState('');
  const [afterJobId,  setAfterJobId]  = useState('');
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState<CompareResponse | null>(null);

  const canCompare = beforeJobId.trim().length > 0 && afterJobId.trim().length > 0 && !loading;

  const handleCompare = async () => {
    if (!canCompare) return;
    setLoading(true); setResult(null);
    try {
      const res = await compareAudits(beforeJobId.trim(), afterJobId.trim());
      setResult(res);
    } catch (e) {
      const msg = e instanceof ApiError
        ? (e.status === 404 ? 'One or both job IDs were not found or not yet complete.' : e.message)
        : 'Compare failed';
      showError(msg);
    } finally {
      setLoading(false);
    }
  };

  const deltaColor = (d: number) =>
    d < 0 ? 'text-green-400' : d > 0 ? 'text-red-400' : 'text-gray-400';

  const deltaSign = (d: number) => (d > 0 ? '+' : '') + d.toFixed(2);

  return (
    <>
      <Head><title>Compare — KAAL</title></Head>
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-1">Compare Audits</h1>
        <p className="text-sm text-gray-500 mb-8">
          Enter two job IDs to compare KVS scores and vulnerability dimensions side by side.
        </p>

        {/* Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          <div>
            <label htmlFor={beforeId} className="block text-xs text-gray-400 mb-1">Before Audit — Job ID</label>
            <input
              id={beforeId}
              type="text"
              placeholder="43-character job ID"
              value={beforeJobId}
              onChange={(e) => setBeforeJobId(e.target.value)}
              className="w-full bg-[#111] border border-[#333] rounded px-3 py-2 text-sm text-gray-200 font-mono focus:outline-none focus:border-[#CC0000]"
            />
          </div>
          <div>
            <label htmlFor={afterId} className="block text-xs text-gray-400 mb-1">After Audit — Job ID</label>
            <input
              id={afterId}
              type="text"
              placeholder="43-character job ID"
              value={afterJobId}
              onChange={(e) => setAfterJobId(e.target.value)}
              className="w-full bg-[#111] border border-[#333] rounded px-3 py-2 text-sm text-gray-200 font-mono focus:outline-none focus:border-[#CC0000]"
            />
          </div>
        </div>

        <p className="text-xs text-gray-600 mb-4">
          Job IDs are shown in the URL after completing an audit:{' '}
          <span className="font-mono text-gray-500">/results?job_id=…</span>{' '}
          Job IDs expire when the server restarts.
        </p>

        <button
          onClick={handleCompare}
          disabled={!canCompare}
          className="w-full sm:w-auto px-8 py-2.5 rounded-lg font-semibold text-sm transition-colors mb-8
            disabled:opacity-40 disabled:cursor-not-allowed
            bg-[#CC0000] hover:bg-[#aa0000] text-white
            focus-visible:ring-2 focus-visible:ring-[#CC0000] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0A0A0A]"
        >
          {loading ? 'Comparing…' : 'Compare →'}
        </button>

        {/* Results */}
        {result && (
          <>
            {/* Score comparison */}
            <section className="grid grid-cols-2 gap-4 mb-6">
              {([['Before', result.before], ['After', result.after]] as const).map(([label, kvs]) => (
                <div key={label} className="bg-[#111] border border-[#222] rounded-lg p-5 text-center">
                  <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">{label}</div>
                  <div
                    className="text-4xl font-bold font-mono mb-2"
                    style={{ color: getRiskTier(kvs.score).hex }}
                  >
                    {formatScore(kvs.score)}
                  </div>
                  <RiskBadge score={kvs.score} showScore={false} />
                </div>
              ))}
            </section>

            {/* Overall delta */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-5 mb-6 flex items-center justify-between">
              <span className="text-sm text-gray-400">Overall KVS change</span>
              <span className={`text-2xl font-bold font-mono ${deltaColor(result.delta.overall)}`}>
                {deltaSign(result.delta.overall)}
              </span>
            </section>

            {/* Overlaid radar */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Vulnerability Fingerprint</h2>
              <KaalRadarChart
                scores={result.before.dimension_scores}
                scoresB={result.after.dimension_scores}
                skipped={result.before.dimensions_skipped ?? []}
                skippedB={result.after.dimensions_skipped ?? []}
                labelA="Before"
                labelB="After"
              />
            </section>

            {/* Per-dimension deltas */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-6">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Dimension Deltas</h2>
              <div className="space-y-3">
                {DIM_ORDER.map((key) => {
                  const delta = result.delta.dimensions[key] ?? 0;
                  const before = result.before.dimension_scores[key] ?? 0;
                  const after  = result.after.dimension_scores[key]  ?? 0;
                  return (
                    <div key={key} className="flex items-center gap-3 text-sm">
                      <span className="text-gray-400 w-44 shrink-0">{DIM_LABELS[key] ?? key}</span>
                      <span className="font-mono text-gray-500 w-10 text-right">{before.toFixed(1)}</span>
                      <span className="text-gray-600">→</span>
                      <span className="font-mono text-gray-300 w-10">{after.toFixed(1)}</span>
                      <span className={`font-mono font-semibold ml-auto ${deltaColor(delta)}`}>
                        {deltaSign(delta)}
                      </span>
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-gray-600 mt-4">
                Green = improvement (lower score). Red = regression (higher score).
              </p>
            </section>
          </>
        )}
      </div>
    </>
  );
}
