import Head from 'next/head';
import { useId, useRef, useState } from 'react';
import UploadZone from '../components/UploadZone';
import ProgressBar from '../components/ProgressBar';
import { useToast } from '../components/ErrorToast';
import { uploadModel, uploadDataset, generatePatch, getAuditResult, connectProgressWS, patchPngUrl, patchPrintUrl, ApiError } from '../lib/api';
import type { ModelUploadResponse, DatasetUploadResponse, PatchResult, WSMessage } from '../lib/types';

export default function PatchPage() {
  const { showError } = useToast();
  const targetId    = useId();
  const fractionId  = useId();
  const iterationsId = useId();
  const printCmId   = useId();

  const [modelInfo,   setModelInfo]   = useState<ModelUploadResponse | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetUploadResponse | null>(null);
  const [modelStatus,   setModelStatus]   = useState<'idle'|'uploading'|'success'|'error'>('idle');
  const [datasetStatus, setDatasetStatus] = useState<'idle'|'uploading'|'success'|'error'>('idle');
  const [modelMsg,   setModelMsg]   = useState('');
  const [datasetMsg, setDatasetMsg] = useState('');

  const [targetClass,    setTargetClass]    = useState(0);
  const [patchFraction,  setPatchFraction]  = useState(0.05);
  const [iterations,     setIterations]     = useState(500);
  const [printCm,        setPrintCm]        = useState(15.0);

  const [running,   setRunning]   = useState(false);
  const [progress,  setProgress]  = useState(0);
  const [stepLabel, setStepLabel] = useState('');
  const [jobId,     setJobId]     = useState<string | null>(null);
  const [patchResult, setPatchResult] = useState<PatchResult | null>(null);
  const [failed,    setFailed]    = useState(false);
  const [errMsg,    setErrMsg]    = useState('');

  const cleanupRef = useRef<(() => void) | null>(null);

  const handleModelFiles = async (files: File[]) => {
    const file = files[0]; if (!file) return;
    setModelStatus('uploading'); setModelMsg('Uploading…');
    try {
      const info = await uploadModel(file);
      setModelInfo(info); setModelStatus('success');
      setModelMsg(`${info.framework} · ${info.input_shape.join('×')} · ${info.num_classes} classes`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Upload failed';
      setModelStatus('error'); setModelMsg(msg); setModelInfo(null); showError(msg);
    }
  };

  const handleDatasetFiles = async (files: File[]) => {
    if (!files.length) return;
    setDatasetStatus('uploading'); setDatasetMsg('Uploading…');
    try {
      const info = await uploadDataset(files);
      setDatasetInfo(info); setDatasetStatus('success');
      setDatasetMsg(`${info.count} images`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Upload failed';
      setDatasetStatus('error'); setDatasetMsg(msg); setDatasetInfo(null); showError(msg);
    }
  };

  const canGenerate = modelInfo !== null && datasetInfo !== null && !running;

  const handleGenerate = async () => {
    if (!canGenerate || !modelInfo || !datasetInfo) return;
    setRunning(true); setFailed(false); setErrMsg(''); setPatchResult(null);
    try {
      const { job_id } = await generatePatch({
        model_id:       modelInfo.model_id,
        dataset_id:     datasetInfo.dataset_id,
        target_class:   targetClass,
        patch_fraction: patchFraction,
        iterations,
        print_cm:       printCm,
      });
      setJobId(job_id);

      const cleanup = connectProgressWS(
        job_id,
        async (msg: WSMessage) => {
          setProgress(msg.progress_pct);
          setStepLabel(msg.step_name || msg.message);
          if (msg.type === 'error') {
            setFailed(true); setErrMsg(msg.message); setRunning(false);
          }
          if (msg.type === 'done') {
            // The WS 'done' frame only carries {kvs_score, job_id}. The patch
            // stats live in the job's result_data, fetched here — otherwise
            // the results would silently fall back to 0%.
            try {
              const res = await getAuditResult(job_id);
              const data = res as unknown as Record<string, unknown>;
              setPatchResult({
                target_class:            (data.target_class as number) ?? targetClass,
                attack_success_rate:     (data.attack_success_rate as number) ?? 0,
                avg_confidence_on_target:(data.avg_confidence_on_target as number) ?? 0,
                patch_fraction_used:     (data.patch_fraction_used as number) ?? patchFraction,
                iterations_used:         (data.iterations_used as number) ?? iterations,
                plain_english:           (data.plain_english as string) ?? '',
              });
            } catch (e) {
              showError(`Failed to load patch result: ${e instanceof ApiError ? e.message : e}`);
            }
            setRunning(false);
          }
        },
        (err) => showError(err),
      );
      cleanupRef.current = cleanup;
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Failed to start patch generation';
      showError(msg); setRunning(false);
    }
  };

  return (
    <>
      <Head><title>Patch — KAAL</title></Head>
      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-1">Adversarial Patch Generator</h1>
        <p className="text-sm text-gray-500 mb-8">
          Generate a printable patch that causes misclassification from any position.
        </p>

        <section className="mb-6">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">1 · Model</h2>
          <UploadZone
            label="Model file (.pt, .pth, .h5, .keras, .onnx, .tflite)"
            accept=".pt,.pth,.h5,.keras,.onnx,.tflite"
            onFiles={handleModelFiles}
            status={modelStatus}
            statusMessage={modelMsg}
            disabled={modelStatus === 'uploading' || running}
          />
        </section>

        <section className="mb-6">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">2 · Dataset</h2>
          <UploadZone
            label="Images (.jpg, .png, .jpeg, .bmp, .webp)"
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            multiple
            onFiles={handleDatasetFiles}
            status={datasetStatus}
            statusMessage={datasetMsg}
            disabled={datasetStatus === 'uploading' || running}
          />
        </section>

        <section className="mb-8 bg-[#111] border border-[#222] rounded-lg p-5 space-y-5">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">3 · Configuration</h2>

          <div>
            <label htmlFor={targetId} className="block text-xs text-gray-400 mb-1">Target Class Index</label>
            <input
              id={targetId}
              type="number"
              min={0}
              value={targetClass}
              onChange={(e) => setTargetClass(Math.max(0, parseInt(e.target.value) || 0))}
              className="w-28 bg-[#0A0A0A] border border-[#333] rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-[#CC0000]"
            />
          </div>

          <div>
            <label htmlFor={fractionId} className="block text-xs text-gray-400 mb-1">
              Patch Fraction
              <span className="ml-2 font-mono text-gray-300">{patchFraction.toFixed(3)}</span>
            </label>
            <input
              id={fractionId}
              type="range"
              min={0.001}
              max={0.5}
              step={0.001}
              value={patchFraction}
              onChange={(e) => setPatchFraction(parseFloat(e.target.value))}
              className="w-full accent-[#CC0000]"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-0.5"><span>0.001</span><span>0.5</span></div>
          </div>

          <div>
            <label htmlFor={iterationsId} className="block text-xs text-gray-400 mb-1">Iterations</label>
            <input
              id={iterationsId}
              type="number"
              min={1}
              value={iterations}
              onChange={(e) => setIterations(Math.max(1, parseInt(e.target.value) || 500))}
              className="w-28 bg-[#0A0A0A] border border-[#333] rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-[#CC0000]"
            />
          </div>

          <div>
            <label htmlFor={printCmId} className="block text-xs text-gray-400 mb-1">Print Size (cm)</label>
            <input
              id={printCmId}
              type="number"
              min={0.1}
              step={0.1}
              value={printCm}
              onChange={(e) => setPrintCm(Math.max(0.1, parseFloat(e.target.value) || 15))}
              className="w-28 bg-[#0A0A0A] border border-[#333] rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-[#CC0000]"
            />
          </div>
        </section>

        {running && (
          <ProgressBar pct={progress} label={stepLabel || 'Generating patch…'} className="mb-4" />
        )}

        {failed && (
          <div className="bg-[#1a0000] border border-[#CC0000] rounded-lg p-4 mb-4 text-red-300 text-sm">
            {errMsg}
          </div>
        )}

        <button
          onClick={handleGenerate}
          disabled={!canGenerate}
          className="w-full py-3 rounded-lg font-semibold text-sm transition-colors
            disabled:opacity-40 disabled:cursor-not-allowed
            bg-[#CC0000] hover:bg-[#aa0000] text-white
            focus-visible:ring-2 focus-visible:ring-[#CC0000] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0A0A0A]"
        >
          {running ? 'Generating…' : 'Generate Patch →'}
        </button>

        {/* Results */}
        {patchResult && jobId && (
          <section className="mt-8 bg-[#111] border border-[#222] rounded-lg p-6">
            <h2 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Patch Results</h2>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm mb-4">
              {[
                ['Target class',          patchResult.target_class],
                ['Attack success rate',   `${(patchResult.attack_success_rate * 100).toFixed(1)}%`],
                ['Avg confidence',        `${(patchResult.avg_confidence_on_target * 100).toFixed(1)}%`],
                ['Patch fraction used',   patchResult.patch_fraction_used.toFixed(3)],
                ['Iterations used',       patchResult.iterations_used],
              ].map(([k, v]) => (
                <>
                  <dt key={`k-${k}`} className="text-gray-500">{k}</dt>
                  <dd key={`v-${k}`} className="text-gray-200 font-mono">{String(v)}</dd>
                </>
              ))}
            </dl>
            {patchResult.plain_english && (
              <p className="text-xs text-gray-500 italic mb-5">{patchResult.plain_english}</p>
            )}
            {/* Inline patch preview */}
            <div className="my-4 border border-[#222] rounded-lg overflow-hidden bg-[#0A0A0A] flex items-center justify-center p-4">
              <img
                src={patchPngUrl(jobId)}
                alt="Generated adversarial patch"
                className="max-w-full max-h-48 object-contain"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
            </div>
            <div className="flex gap-3 flex-wrap">
              <a
                href={patchPngUrl(jobId)}
                target="_blank"
                rel="noreferrer"
                aria-label="Download patch PNG"
                className="px-4 py-2 bg-[#CC0000] hover:bg-[#aa0000] text-white text-sm font-semibold rounded-lg transition-colors"
              >
                ↓ Patch PNG
              </a>
              <a
                href={patchPrintUrl(jobId)}
                target="_blank"
                rel="noreferrer"
                aria-label="Download printable patch PDF"
                className="px-4 py-2 border border-[#333] hover:border-[#555] text-gray-300 hover:text-white text-sm font-semibold rounded-lg transition-colors"
              >
                ↓ Printable PDF
              </a>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
