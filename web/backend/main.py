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
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.backend.jobs.store import job_store
from web.backend.routes.audit import router
from web.backend.ws.progress import ws_router

# ---------------------------------------------------------------------------
# Lifespan — start the expired-job cleanup loop, cancel it on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the cleanup task on startup, cancel it on shutdown."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(600)    # 10 minutes
            removed = job_store.cleanup_expired()
            if removed:
                print(f"[KAAL] Cleaned up {removed} expired job(s).")

    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()


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
