import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import ProgressBar from '../components/ProgressBar';
import KVSGauge from '../components/KVSGauge';
import RiskBadge from '../components/RiskBadge';
import { useToast } from '../components/ErrorToast';
import { connectProgressWS, getAuditResult, pdfReportUrl, patchPngUrl } from '../lib/api';
import { DIM_LABELS, DIM_ORDER } from '../lib/kvs';
import type { AuditResult, WSMessage } from '../lib/types';

// Recharts needs client-only render
const KaalRadarChart = dynamic(() => import('../components/RadarChart'), { ssr: false });

export default function ResultsPage() {
  const router = useRouter();
  const { showError } = useToast();
  const jobId = typeof router.query.job_id === 'string' ? router.query.job_id : null;

  const [progress, setProgress] = useState(0);
  const [step,     setStep]     = useState('Waiting…');
  const [done,     setDone]     = useState(false);
  const [failed,   setFailed]   = useState(false);
  const [errMsg,   setErrMsg]   = useState('');
  const [result,   setResult]   = useState<AuditResult | null>(null);

  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const cleanup = connectProgressWS(
      jobId,
      async (msg: WSMessage) => {
        if (msg.type === 'error') {
          setFailed(true);
          setErrMsg(msg.message);
          return;
        }
        setProgress(msg.progress_pct);
        setStep(msg.step_name || msg.message);

        if (msg.type === 'done') {
          setDone(true);
          try {
            const r = await getAuditResult(jobId);
            setResult(r);
          } catch (e) {
            showError(`Failed to load result: ${e}`);
          }
        }
      },
      (err) => showError(err),
    );
    cleanupRef.current = cleanup;
    return () => cleanup();
  }, [jobId]);

  // No job_id in URL
  if (!jobId) {
    return (
      <>
        <Head><title>Results — KAAL</title></Head>
        <div className="flex flex-col items-center justify-center min-h-[40vh] text-center gap-4">
          <p className="text-gray-400">No job ID provided.</p>
          <Link href="/audit" className="text-[#CC0000] hover:underline text-sm">
            ← Start an audit
          </Link>
        </div>
      </>
    );
  }

  return (
    <>
      <Head><title>Results — KAAL</title></Head>

      <div className="max-w-3xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-white">Audit Results</h1>
          <span className="text-xs font-mono text-gray-600 bg-[#111] border border-[#222] px-2 py-1 rounded">
            {jobId.slice(0, 8)}
          </span>
        </div>

        {/* Progress phase */}
        {!done && !failed && (
          <div className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6">
            <p className="text-sm text-gray-300 mb-3">Audit in progress…</p>
            <ProgressBar pct={progress} label={step} />
          </div>
        )}

        {/* Error state */}
        {failed && (
          <div className="bg-[#1a0000] border border-[#CC0000] rounded-lg p-6 mb-6 text-red-300">
            <p className="font-semibold mb-1">Audit failed</p>
            <p className="text-sm">{errMsg}</p>
            <Link href="/audit" className="text-[#CC0000] text-sm mt-3 inline-block hover:underline">
              ← Back to audit
            </Link>
          </div>
        )}

        {/* Results */}
        {done && result && (
          <>
            {/* KVS score */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6 text-center">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">KVS Score</h2>
              <KVSGauge score={result.kvs.score} size={200} />
              {result.kvs.plain_english && (
                <p className="text-xs text-gray-500 mt-4 max-w-sm mx-auto">{result.kvs.plain_english}</p>
              )}
            </section>

            {/* Radar chart */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Vulnerability Fingerprint</h2>
              <KaalRadarChart scores={result.kvs.dimension_scores} />
            </section>

            {/* Dimension scores */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Dimension Scores</h2>
              <div className="space-y-3">
                {DIM_ORDER.map((key) => {
                  const score = result.kvs.dimension_scores[key];
                  const skipped = result.kvs.dimensions_skipped.includes(key);
                  return (
                    <div key={key} className="flex items-center gap-3">
                      <span className="text-xs text-gray-400 w-44 shrink-0">{DIM_LABELS[key] ?? key}</span>
                      {skipped ? (
                        <span className="text-xs text-gray-600 italic">not tested</span>
                      ) : (
                        <>
                          <div className="flex-1 h-1.5 bg-[#222] rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-[#CC0000]"
                              style={{ width: `${(score / 10) * 100}%` }}
                            />
                          </div>
                          <RiskBadge score={score} showScore />
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            {/* Remediation */}
            {result.kvs.remediation.length > 0 && (
              <section className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6">
                <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Remediation</h2>
                <ul className="space-y-2">
                  {result.kvs.remediation.map((r, i) => (
                    <li key={i} className="flex gap-2 text-sm text-gray-300">
                      <span className="text-[#CC0000] shrink-0">→</span>
                      {r}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Model / dataset info */}
            <section className="bg-[#111] border border-[#222] rounded-lg p-6 mb-6">
              <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Audit Details</h2>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                {[
                  ['Model',       result.model?.name ?? '—'],
                  ['Framework',   result.model?.framework ?? '—'],
                  ['Input shape', result.model?.input_shape?.join('×') ?? '—'],
                  ['Classes',     result.model?.num_classes ?? '—'],
                  ['Images',      result.dataset?.total_images ?? '—'],
                  ['Duration',    result.audit_duration_seconds ? `${result.audit_duration_seconds.toFixed(1)}s` : '—'],
                ].map(([k, v]) => (
                  <>
                    <dt key={`k-${k}`} className="text-gray-500">{k}</dt>
                    <dd key={`v-${k}`} className="text-gray-200 font-mono">{String(v)}</dd>
                  </>
                ))}
              </dl>
            </section>

            {/* Downloads */}
            <section className="flex flex-wrap gap-3 mb-8">
              <a
                href={pdfReportUrl(jobId)}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 bg-[#CC0000] hover:bg-[#aa0000] text-white text-sm font-semibold rounded-lg transition-colors"
                aria-label="Download PDF report"
              >
                ↓ Download PDF Report
              </a>
              {Boolean((result.attacks as Record<string, unknown>)?.patch) && (
                <a
                  href={patchPngUrl(jobId)}
                  target="_blank"
                  rel="noreferrer"
                  className="px-4 py-2 border border-[#333] hover:border-[#555] text-gray-300 hover:text-white text-sm font-semibold rounded-lg transition-colors"
                  aria-label="Download adversarial patch PNG"
                >
                  ↓ Patch PNG
                </a>
              )}
            </section>
          </>
        )}
      </div>
    </>
  );
}
