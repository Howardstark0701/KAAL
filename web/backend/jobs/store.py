"""In-memory job state store — Spec 16.1 — Phase 10, Kiro Prompt 10.2.

Dict mapping job_id (uuid4) → JobState.
No database, no file persistence — in-memory only.
Jobs expire after 2 hours via a background cleanup task.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
# _FileEntry — registry value for an uploaded file / dataset dir
# ---------------------------------------------------------------------------

@dataclass
class _FileEntry:
    """Value stored in the _files registry for one uploaded file."""

    path: str
    """Absolute path on disk — a single model file, or a dataset directory."""

    created_at: datetime
    """Registration time — used to age out orphaned entries via the job TTL."""


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
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        # Uploaded file registry: file_id → _FileEntry (path + created_at)
        self._files: dict[str, _FileEntry] = {}

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
        Thread-safe: the read-modify-write is guarded by self._lock.
        """
        with self._lock:
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
        self._files[file_id] = _FileEntry(
            path=file_path,
            created_at=datetime.now(timezone.utc),
        )
        return file_id

    def get_file_path(self, file_id: str) -> Optional[str]:
        entry = self._files.get(file_id)
        return entry.path if entry else None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """Remove jobs older than JOB_TTL_SECONDS. Returns count removed.

        Also purges the file registry and the on-disk temp dirs:
        - files attached to the removed (expired) jobs, and
        - orphaned file entries (not referenced by any live job) older
          than the same TTL — covers uploads that never got attached.
        Thread-safe: both registries are mutated under self._lock.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            expired = [
                jid for jid, job in self._jobs.items()
                if (now - job.created_at).total_seconds() > self.JOB_TTL_SECONDS
            ]

            # File_ids attached to the jobs that are about to expire.
            expired_files: set[str] = set()
            for jid in expired:
                job = self._jobs[jid]
                if job.model_id:
                    expired_files.add(job.model_id)
                if job.dataset_id:
                    expired_files.add(job.dataset_id)

            for jid in expired:
                del self._jobs[jid]

            # Set of file_ids still referenced by live jobs.
            referenced: set[str] = set()
            for job in self._jobs.values():
                if job.model_id:
                    referenced.add(job.model_id)
                if job.dataset_id:
                    referenced.add(job.dataset_id)

            # 1. Remove the files attached to expired jobs — unless another
            #    live job still references the same file (shared uploads).
            for fid in expired_files:
                if fid in referenced:
                    continue
                entry = self._files.pop(fid, None)
                if entry is not None:
                    _remove_disk_path(entry.path)

            # 2. Remove orphaned file entries (no live job references them)
            #    once they pass the job TTL — uploads that never got attached.
            for fid, entry in list(self._files.items()):
                if fid in referenced:
                    continue
                if (now - entry.created_at).total_seconds() <= self.JOB_TTL_SECONDS:
                    continue
                self._files.pop(fid, None)
                _remove_disk_path(entry.path)

            return len(expired)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_disk_path(path: str) -> None:
    """Delete a registered file or dataset directory from disk, ignoring errors.

    Dataset uploads are directories (shutil.rmtree); model uploads are single
    files (unlink). Never raises — cleanup must not take the server down.
    """
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton used by all route handlers
# ---------------------------------------------------------------------------

job_store = JobStore()
