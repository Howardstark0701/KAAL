"""In-memory job state store — Spec 16.1 — Phase 10, Kiro Prompt 10.2.

Dict mapping job_id (uuid4) → JobState.
No database, no file persistence — in-memory only.
Jobs expire after 2 hours via a background cleanup task.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JobState
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    """Holds all state for a single audit or patch job."""

    job_id: str
    """UUID4 string."""

    job_type: str
    """'audit' | 'patch'"""

    status: str
    """'pending' | 'running' | 'complete' | 'failed'"""

    progress_pct: int
    """0–100."""

    current_step: str
    """Human-readable description of the current pipeline step."""

    # ── Result paths ──────────────────────────────────────────────────────────
    result_json_path: Optional[str] = None
    """Absolute path to the generated report.json (audit jobs only)."""

    pdf_path: Optional[str] = None
    """Absolute path to the generated report.pdf."""

    fingerprint_path: Optional[str] = None
    patch_png_path: Optional[str] = None
    patch_pdf_path: Optional[str] = None
    output_dir: Optional[str] = None

    # ── Inline result data ─────────────────────────────────────────────────────
    result_data: Optional[dict] = None
    """Populated with the full audit JSON dict when status == 'complete'."""

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str] = None

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    # ── Cached file IDs (for file retrieval) ──────────────────────────────────
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------

class JobStore:
    """Thread-safe in-memory job registry.

    Usage:
        store = JobStore()
        job_id = store.create("audit")
        store.update(job_id, status="running", progress_pct=10)
        job = store.get(job_id)
        store.cleanup_expired()
    """

    # Job TTL: 2 hours in seconds
    JOB_TTL_SECONDS = 7200

    def __init__(self):
        self._jobs: dict[str, JobState] = {}
        # Uploaded file registry: file_id → absolute path on disk
        self._files: dict[str, str] = {}

    # ── Job CRUD ──────────────────────────────────────────────────────────────

    def create(self, job_type: str = "audit") -> str:
        """Create a new job, return its job_id."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobState(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            progress_pct=0,
            current_step="Queued",
        )
        return job_id

    def get(self, job_id: str) -> Optional[JobState]:
        """Return JobState or None if not found."""
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> bool:
        """Update any JobState fields by keyword argument.

        Returns True if job found and updated, False otherwise.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        return True

    def complete(self, job_id: str, result_data: dict,
                 result_json_path: Optional[str] = None,
                 pdf_path: Optional[str] = None,
                 fingerprint_path: Optional[str] = None):
        """Mark a job complete with result data."""
        self.update(
            job_id,
            status="complete",
            progress_pct=100,
            current_step="Complete",
            result_data=result_data,
            result_json_path=result_json_path,
            pdf_path=pdf_path,
            fingerprint_path=fingerprint_path,
            completed_at=datetime.now(timezone.utc),
        )

    def fail(self, job_id: str, error: str):
        """Mark a job as failed with an error message."""
        self.update(
            job_id,
            status="failed",
            current_step="Failed",
            error=error,
            completed_at=datetime.now(timezone.utc),
        )

    def all_jobs(self) -> list[JobState]:
        return list(self._jobs.values())

    # ── File registry ─────────────────────────────────────────────────────────

    def register_file(self, file_path: str) -> str:
        """Register an uploaded file path and return its file_id."""
        file_id = str(uuid.uuid4())
        self._files[file_id] = file_path
        return file_id

    def get_file_path(self, file_id: str) -> Optional[str]:
        return self._files.get(file_id)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """Remove jobs older than JOB_TTL_SECONDS. Returns count removed."""
        now = datetime.now(timezone.utc)
        expired = [
            jid for jid, job in self._jobs.items()
            if (now - job.created_at).total_seconds() > self.JOB_TTL_SECONDS
        ]
        for jid in expired:
            del self._jobs[jid]
        return len(expired)


# ---------------------------------------------------------------------------
# Module-level singleton used by all route handlers
# ---------------------------------------------------------------------------

job_store = JobStore()
