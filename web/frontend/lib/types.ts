// KAAL Frontend — Shared TypeScript types
// Mirrors the Pydantic models in web/backend/models.py

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

export interface ModelUploadResponse {
  model_id: string;
  filename: string;
  framework: string;
  input_shape: number[];
  num_classes: number;
}

export interface DatasetUploadResponse {
  dataset_id: string;
  count: number;
  formats: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

export interface AuditStartRequest {
  model_id: string;
  dataset_id: string;
  attacks: string[];
  epsilon: number;
  steps: number;
  report_formats: string[];
  no_gradcam: boolean;
}

export interface AuditStartResponse {
  job_id: string;
  status: string;
}

export type JobStatus = 'pending' | 'running' | 'complete' | 'failed';

export interface AuditStatusResponse {
  job_id: string;
  status: JobStatus;
  progress_pct: number;
  current_step: string;
  kvs_score: number | null;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Audit result JSON (from GET /api/audit/result/{job_id})
// ---------------------------------------------------------------------------

export interface KVSSection {
  score: number;
  label: string;
  color: string;
  dimension_scores: Record<string, number>;
  dimensions_tested: string[];
  dimensions_skipped: string[];
  plain_english: string;
  remediation: string[];
}

export interface AuditResult {
  kvs: KVSSection;
  model: {
    name: string;
    framework: string;
    input_shape: number[];
    num_classes: number;
  };
  dataset: {
    total_images: number;
    formats: Record<string, number>;
  };
  attacks: Record<string, unknown>;
  physical_robustness?: {
    overall_survival_rate: number;
    physical_threat_rating: string;
    category_summary: Record<string, number>;
  };
  audit_duration_seconds: number;
  generated_at: string;
}

// ---------------------------------------------------------------------------
// Patch
// ---------------------------------------------------------------------------

export interface PatchGenerateRequest {
  model_id: string;
  dataset_id: string;
  target_class: number;
  patch_fraction: number;
  iterations: number;
  print_cm: number;
}

export interface PatchGenerateResponse {
  job_id: string;
}

export interface PatchResult {
  target_class: number;
  attack_success_rate: number;
  avg_confidence_on_target: number;
  patch_fraction_used: number;
  iterations_used: number;
  plain_english: string;
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

export interface CompareResponse {
  before: KVSSection;
  after: KVSSection;
  delta: {
    overall: number;
    dimensions: Record<string, number>;
  };
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export type WSMessageType = 'progress' | 'step_complete' | 'error' | 'done';

export interface WSMessage {
  type: WSMessageType;
  message: string;
  progress_pct: number;
  step_name: string;
  data?: {
    kvs_score?: number;
    job_id?: string;
  } | null;
}

// ---------------------------------------------------------------------------
// KVS risk tier
// ---------------------------------------------------------------------------

export interface RiskTier {
  label: string;
  color: string;        // Tailwind text-* class
  bg: string;           // Tailwind bg-* class
  border: string;       // Tailwind border-* class
  hex: string;          // Hex colour
}
