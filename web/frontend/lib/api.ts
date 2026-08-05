// KAAL API client — all HTTP and WebSocket calls to the FastAPI backend.
// Backend URL is proxied via next.config.js rewrites in dev,
// and pointed directly at localhost:8080 for WebSocket (not proxied by Next.js).

import type {
  ModelUploadResponse,
  DatasetUploadResponse,
  AuditStartRequest,
  AuditStartResponse,
  AuditStatusResponse,
  AuditResult,
  PatchGenerateRequest,
  PatchGenerateResponse,
  CompareResponse,
  WSMessage,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080';
const WS_BASE  = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8080')
  .replace(/^http/, 'ws');

// ---------------------------------------------------------------------------
// ApiError
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch { /* ignore parse error */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

export async function uploadModel(file: File): Promise<ModelUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload/model`, { method: 'POST', body: form });
  return handleResponse<ModelUploadResponse>(res);
}

export async function uploadDataset(files: File[]): Promise<DatasetUploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  const res = await fetch(`${API_BASE}/api/upload/dataset`, { method: 'POST', body: form });
  return handleResponse<DatasetUploadResponse>(res);
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

export async function startAudit(req: AuditStartRequest): Promise<AuditStartResponse> {
  const res = await fetch(`${API_BASE}/api/audit/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return handleResponse<AuditStartResponse>(res);
}

export async function getAuditStatus(jobId: string): Promise<AuditStatusResponse> {
  const res = await fetch(`${API_BASE}/api/audit/status/${jobId}`);
  return handleResponse<AuditStatusResponse>(res);
}

export async function getAuditResult(jobId: string): Promise<AuditResult> {
  const res = await fetch(`${API_BASE}/api/audit/result/${jobId}`);
  return handleResponse<AuditResult>(res);
}

// ---------------------------------------------------------------------------
// Reports / downloads — return direct URLs for <a> tags
// ---------------------------------------------------------------------------

export const pdfReportUrl    = (jobId: string) => `${API_BASE}/api/report/${jobId}/pdf`;
export const patchPngUrl     = (jobId: string) => `${API_BASE}/api/patch/${jobId}/png`;
export const patchPrintUrl   = (jobId: string) => `${API_BASE}/api/patch/${jobId}/printable`;

// ---------------------------------------------------------------------------
// Patch
// ---------------------------------------------------------------------------

export async function generatePatch(req: PatchGenerateRequest): Promise<PatchGenerateResponse> {
  const res = await fetch(`${API_BASE}/api/patch/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return handleResponse<PatchGenerateResponse>(res);
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

export async function compareAudits(beforeId: string, afterId: string): Promise<CompareResponse> {
  const res = await fetch(
    `${API_BASE}/api/compare?before_id=${encodeURIComponent(beforeId)}&after_id=${encodeURIComponent(afterId)}`
  );
  return handleResponse<CompareResponse>(res);
}

// ---------------------------------------------------------------------------
// WebSocket progress stream
// Opens a WS connection and calls onMessage for each frame.
// Returns a cleanup function that closes the socket.
// Falls back to polling if WS is unavailable.
// ---------------------------------------------------------------------------

export function connectProgressWS(
  jobId: string,
  onMessage: (msg: WSMessage) => void,
  onError?: (err: string) => void,
): () => void {
  let ws: WebSocket;
  let closed = false;

  try {
    ws = new WebSocket(`${WS_BASE}/ws/${jobId}`);

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        onMessage(msg);
      } catch { /* ignore malformed frame */ }
    };

    ws.onerror = () => {
      if (!closed) {
        onError?.('WebSocket connection failed — falling back to polling');
        startPollingFallback(jobId, onMessage, onError);
      }
    };

    ws.onclose = () => { closed = true; };

  } catch {
    // WebSocket constructor threw (e.g. SSR) — fall back to polling
    startPollingFallback(jobId, onMessage, onError);
  }

  return () => {
    closed = true;
    try { ws?.close(); } catch { /* ignore */ }
  };
}

function startPollingFallback(
  jobId: string,
  onMessage: (msg: WSMessage) => void,
  onError?: (err: string) => void,
) {
  let stopped = false;
  const poll = async () => {
    while (!stopped) {
      await new Promise((r) => setTimeout(r, 3000));
      try {
        const status = await getAuditStatus(jobId);
        const pct  = status.progress_pct;
        const step = status.current_step;

        if (status.status === 'failed') {
          onMessage({ type: 'error', message: status.error ?? 'Audit failed', progress_pct: pct, step_name: step });
          stopped = true;
          return;
        }
        if (status.status === 'complete') {
          onMessage({ type: 'done', message: 'Audit complete', progress_pct: 100, step_name: 'Complete',
                      data: { kvs_score: status.kvs_score ?? undefined, job_id: jobId } });
          stopped = true;
          return;
        }
        onMessage({ type: 'progress', message: step, progress_pct: pct, step_name: step });
      } catch (e) {
        onError?.(`Polling error: ${e}`);
      }
    }
  };
  poll();
  return () => { stopped = true; };
}
