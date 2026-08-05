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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.backend.jobs.store import job_store
from web.backend.routes.audit import router
from web.backend.ws.progress import ws_router

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KAAL",
    description="Adversarial Robustness Auditing Tool — REST API",
    version="1.0.0",
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
    if content_length and int(content_length) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": "Upload too large. Maximum size is 500 MB."},
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
# Startup: background cleanup task
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def start_cleanup_task():
    """Purge expired jobs every 10 minutes."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(600)    # 10 minutes
            removed = job_store.cleanup_expired()
            if removed:
                print(f"[KAAL] Cleaned up {removed} expired job(s).")

    asyncio.create_task(_cleanup_loop())

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
