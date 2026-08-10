import Head from 'next/head';
import { useRouter } from 'next/router';
import { useId, useState } from 'react';
import UploadZone from '../components/UploadZone';
import ProgressBar from '../components/ProgressBar';
import { useToast } from '../components/ErrorToast';
import { uploadModel, uploadDataset, startAudit, ApiError } from '../lib/api';
import type { ModelUploadResponse, DatasetUploadResponse } from '../lib/types';

const ATTACK_OPTIONS = ['fgsm', 'pgd', 'patch', 'blackbox', 'physical'];

export default function AuditPage() {
  const router  = useRouter();
  const { showError } = useToast();
  const epsilonId = useId();
  const stepsId   = useId();
  const noGcId    = useId();

  // Upload state
  const [modelInfo,   setModelInfo]   = useState<ModelUploadResponse | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetUploadResponse | null>(null);
  const [modelStatus,   setModelStatus]   = useState<'idle'|'uploading'|'success'|'error'>('idle');
  const [datasetStatus, setDatasetStatus] = useState<'idle'|'uploading'|'success'|'error'>('idle');
  const [modelMsg,   setModelMsg]   = useState('');
  const [datasetMsg, setDatasetMsg] = useState('');

  // Config state
  const [attacks,   setAttacks]   = useState<string[]>(['fgsm', 'pgd', 'patch', 'physical']);
  const [epsilon,   setEpsilon]   = useState(0.03);
  const [steps,     setSteps]     = useState(40);
  const [noGradcam, setNoGradcam] = useState(false);

  // Launch state
  const [launching, setLaunching] = useState(false);

  // ---------------------------------------------------------------------------
  // Model upload
  // ---------------------------------------------------------------------------
  const handleModelFiles = async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setModelStatus('uploading');
    setModelMsg('Uploading...');
    try {
      const info = await uploadModel(file);
      setModelInfo(info);
      setModelStatus('success');
      setModelMsg(`${info.framework} · ${info.input_shape.join('×')} · ${info.num_classes} classes`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Upload failed';
      setModelStatus('error');
      setModelMsg(msg);
      setModelInfo(null);
      showError(`Model upload: ${msg}`);
    }
  };

  // ---------------------------------------------------------------------------
  // Dataset upload
  // ---------------------------------------------------------------------------
  const handleDatasetFiles = async (files: File[]) => {
    if (files.length === 0) return;
    setDatasetStatus('uploading');
    setDatasetMsg('Uploading...');
    try {
      const info = await uploadDataset(files);
      setDatasetInfo(info);
      setDatasetStatus('success');
      const fmts = Object.entries(info.formats).map(([k, v]) => `${k}:${v}`).join(', ');
      setDatasetMsg(`${info.count} images · ${fmts}`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Upload failed';
      setDatasetStatus('error');
      setDatasetMsg(msg);
      setDatasetInfo(null);
      showError(`Dataset upload: ${msg}`);
    }
  };

  // ---------------------------------------------------------------------------
  // Toggle attacks
  // ---------------------------------------------------------------------------
  const toggleAttack = (a: string) => {
    setAttacks((prev) =>
      prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]
    );
  };

  // ---------------------------------------------------------------------------
  // Start audit
  // ---------------------------------------------------------------------------
  const canStart = modelInfo !== null && datasetInfo !== null && attacks.length > 0 && !launching;

  const handleStart = async () => {
    if (!canStart || !modelInfo || !datasetInfo) return;
    setLaunching(true);
    try {
      const { job_id } = await startAudit({
        model_id:       modelInfo.model_id,
        dataset_id:     datasetInfo.dataset_id,
        attacks,
        epsilon,
        steps,
        report_formats: ['pdf', 'json'],
        no_gradcam:     noGradcam,
      });
      router.push(`/results?job_id=${job_id}`);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : 'Failed to start audit';
      showError(msg);
      setLaunching(false);
    }
  };

  return (
    <>
      <Head><title>Audit — KAAL</title></Head>

      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-1">New Audit</h1>
        <p className="text-sm text-gray-500 mb-8">Upload your model and dataset, configure attacks, then start.</p>

        {/* Model upload */}
        <section className="mb-6">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">1 · Model</h2>
          <UploadZone
            label="Model file (.pt, .pth, .h5, .keras, .onnx, .tflite)"
            accept=".pt,.pth,.h5,.keras,.onnx,.tflite"
            onFiles={handleModelFiles}
            status={modelStatus}
            statusMessage={modelMsg}
            disabled={modelStatus === 'uploading'}
          />
        </section>

        {/* Dataset upload */}
        <section className="mb-6">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">2 · Dataset</h2>
          <UploadZone
            label="Images (.jpg, .jpeg, .png, .bmp, .webp)"
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            multiple
            onFiles={handleDatasetFiles}
            status={datasetStatus}
            statusMessage={datasetMsg}
            disabled={datasetStatus === 'uploading'}
          />
        </section>

        {/* Config */}
        <section className="mb-8 bg-[#111] border border-[#222] rounded-lg p-5 space-y-5">
          <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">3 · Configuration</h2>

          {/* Attack selection */}
          <div>
            <fieldset>
              <legend className="text-xs text-gray-400 mb-2">Attacks</legend>
              <div className="flex flex-wrap gap-2">
                {ATTACK_OPTIONS.map((a) => (
                  <label key={a} className="flex items-center gap-1.5 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={attacks.includes(a)}
                      onChange={() => toggleAttack(a)}
                      className="accent-[#CC0000] w-4 h-4"
                    />
                    <span className="text-sm text-gray-300 uppercase font-mono">{a}</span>
                  </label>
                ))}
              </div>
            </fieldset>
          </div>

          {/* Epsilon */}
          <div>
            <label htmlFor={epsilonId} className="block text-xs text-gray-400 mb-1">
              Epsilon (ε) — perturbation magnitude
              <span className="ml-2 font-mono text-gray-300">{epsilon.toFixed(3)}</span>
            </label>
            <input
              id={epsilonId}
              type="range"
              min={0.001}
              max={1.0}
              step={0.001}
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full accent-[#CC0000]"
            />
            <div className="flex justify-between text-xs text-gray-600 mt-0.5">
              <span>0.001</span><span>1.0</span>
            </div>
            {epsilon > 0.1 && (
              <p className="text-xs text-yellow-500 mt-1">
                &#9888; Epsilon &gt; 0.1 produces visually perceptible perturbations — use for stress testing only.
              </p>
            )}
          </div>

          {/* PGD steps */}
          <div>
            <label htmlFor={stepsId} className="block text-xs text-gray-400 mb-1">
              PGD Steps
            </label>
            <input
              id={stepsId}
              type="number"
              min={1}
              max={200}
              value={steps}
              onChange={(e) => setSteps(Math.max(1, Math.min(200, parseInt(e.target.value) || 40)))}
              className="w-28 bg-[#0A0A0A] border border-[#333] rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-[#CC0000]"
            />
          </div>

          {/* No GradCAM */}
          <div>
            <label htmlFor={noGcId} className="flex items-center gap-2 cursor-pointer select-none">
              <input
                id={noGcId}
                type="checkbox"
                checked={noGradcam}
                onChange={(e) => setNoGradcam(e.target.checked)}
                className="accent-[#CC0000] w-4 h-4"
              />
              <span className="text-sm text-gray-300">Skip GradCAM (faster)</span>
            </label>
          </div>
        </section>

        {/* Launch */}
        {launching && (
          <ProgressBar pct={5} label="Starting audit…" className="mb-4" />
        )}

        <button
          onClick={handleStart}
          disabled={!canStart}
          className="w-full py-3 rounded-lg font-semibold text-sm transition-colors focus-visible:ring-2 focus-visible:ring-[#CC0000] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0A0A0A]
            disabled:opacity-40 disabled:cursor-not-allowed
            bg-[#CC0000] hover:bg-[#aa0000] text-white"
        >
          {launching ? 'Starting…' : 'Start Audit →'}
        </button>

        {!modelInfo && (
          <p className="text-xs text-gray-600 text-center mt-2">Upload a model and dataset to begin</p>
        )}
      </div>
    </>
  );
}
