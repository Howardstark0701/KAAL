"""KAAL FastAPI application — Spec 16.1 — Phase 10, Kiro Prompt 10.1.

Start the server:
    uvicorn web.backend.main:app --host 127.0.0.1 --port 8080

Configuration:
    CORS:            localhost:3000 only (Next.js frontend)
    Max upload size: 500 MB
    Request timeout: 30 minutes
    Cleanup:         Expired jobs purged every 10 minutes
"""

from __future__ import annotations

import asyncio
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.backend.jobs.store import job_store
from web.backend.routes.audit import router
from web.backend.ws.progress import ws_router

# ---------------------------------------------------------------------------
# Hard timeout for running jobs
# ---------------------------------------------------------------------------

# A running audit/patch job that exceeds this wall-clock cap (monotonic) is
# marked failed by the watchdog below. Guards against NaN gradients, corrupted
# dataset samples, or attacks stuck in an infinite loop — jobs that would
# otherwise hang forever, block a worker slot, and never hit the 2h TTL.
JOB_TIMEOUT_SECONDS = 3600   # 1 hour


def _fail_timed_out_jobs() -> None:
    """Fail running jobs whose elapsed wall-clock time exceeds the timeout.

    Separated from the 30s watchdog loop so the logic can be exercised
    directly without sleeping. Only mutates the store — the attack thread /
    asyncio task is never cancelled; the slot is freed and the client gets a
    terminal status. store.fail() is idempotent, so a job that completed or
    failed between the check and the call is left untouched.
    """
    now = time.monotonic()
    for job in job_store.all_jobs():
        if (
            job.status == "running"
            and job.started_at is not None
            and (now - job.started_at) > JOB_TIMEOUT_SECONDS
        ):
            job_store.fail(
                job.job_id,
                "Audit timed out after "
                f"{JOB_TIMEOUT_SECONDS // 60} minutes. "
                "The model or dataset may have caused the "
                "attack to hang. Try a smaller dataset or "
                "fewer attack steps.",
            )


# ---------------------------------------------------------------------------
# Lifespan — start the cleanup + watchdog tasks, cancel them on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the cleanup + watchdog tasks on startup, cancel them on shutdown."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(600)    # 10 minutes
            removed = job_store.cleanup_expired()
            if removed:
                print(f"[KAAL] Cleaned up {removed} expired job(s).")

    async def _watchdog():
        while True:
            await asyncio.sleep(30)     # check every 30 seconds
            _fail_timed_out_jobs()

    cleanup_task = asyncio.create_task(_cleanup_loop())
    watchdog_task = asyncio.create_task(_watchdog())
    try:
        yield
    finally:
        cleanup_task.cancel()
        watchdog_task.cancel()


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KAAL",
    description="Adversarial Robustness Auditing Tool — REST API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — localhost:3000 only
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Body size limit — 500 MB
# ---------------------------------------------------------------------------

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    max_bytes = 500 * 1024 * 1024   # 500 MB
    content_length = request.headers.get("content-length")
    if content_length is not None:
        # Malformed Content-Length (non-integer) → 400 instead of an
        # unhandled ValueError bubbling up as a 500.
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Malformed Content-Length header."},
            )
        if declared_length > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Upload too large. Maximum size is 500 MB."},
            )
        return await call_next(request)

    # No Content-Length: a request body could still arrive (chunked
    # transfer-encoding, or a body-carrying method without a length), which
    # would bypass the size limit entirely. Reject those — do not assume
    # the request is safe just because the header is absent. Bodyless
    # methods (GET/HEAD/OPTIONS) have nothing to size-check and pass through.
    transfer_encoding = request.headers.get("transfer-encoding")
    if transfer_encoding or request.method in {"POST", "PUT", "PATCH"}:
        return JSONResponse(
            status_code=411,
            content={
                "detail": "Content-Length header required. "
                          "Maximum upload size is 500 MB."
            },
        )
    return await call_next(request)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(router)       # REST endpoints
app.include_router(ws_router)    # WebSocket /ws/{job_id}

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "KAAL", "version": "1.0.0"}

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log the full exception server-side for diagnosis, but never echo it
    # back to the client — internal paths and stack frames must not leak.
    print(f"[KAAL] Unhandled exception on {request.method} {request.url.path}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"},
    )
