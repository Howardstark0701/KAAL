"""WebSocket live progress streaming — Spec 16.1 — Phase 10, Kiro Prompt 10.4.

Endpoint: /ws/{job_id}

Polls JobStore every 500ms and sends progress messages to the connected
client. Closes connection after "done" or "error" message.

Message format:
    {
        "type": "progress" | "step_complete" | "error" | "done",
        "message": str,
        "progress_pct": int,
        "step_name": str,
        "data": {} | null
    }
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from web.backend.jobs.store import job_store
from web.backend.routes.audit import _JOB_ID_RE

ws_router = APIRouter()

_POLL_INTERVAL = 0.5   # seconds between store polls


@ws_router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Stream job progress to the client until complete or failed."""
    # Reject malformed job IDs before the polling loop. A 400 isn't expressible
    # over an established WebSocket, so we send the same message and close.
    if not _JOB_ID_RE.match(job_id):
        await websocket.accept()
        await _send(websocket, "error", "Invalid job ID format.", 0, "")
        return
    await websocket.accept()

    try:
        last_pct = -1

        while True:
            job = job_store.get(job_id)

            if job is None:
                await _send(websocket, "error", "Job not found.", 0, "")
                break

            pct  = job.progress_pct
            step = job.current_step

            if job.status == "failed":
                await _send(websocket, "error",
                            job.error or "Audit failed.", pct, step)
                break

            if job.status == "complete":
                # Send final summary data with the done message
                kvs_score = None
                if job.result_data:
                    kvs_score = job.result_data.get("kvs", {}).get("score")

                await _send(websocket, "done",
                            "Audit complete.", 100, "Complete",
                            data={"kvs_score": kvs_score,
                                  "job_id": job_id})
                break

            # Send progress update only when something changed
            if pct != last_pct:
                msg_type = "step_complete" if pct > last_pct and pct % 20 == 0 \
                           else "progress"
                await _send(websocket, msg_type,
                            step, pct, step)
                last_pct = pct

            await asyncio.sleep(_POLL_INTERVAL)

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


async def _send(
    ws: WebSocket,
    msg_type: str,
    message: str,
    progress_pct: int,
    step_name: str,
    data: dict | None = None,
):
    """Send a JSON message and close the socket on 'done' or 'error'."""
    payload = {
        "type":         msg_type,
        "message":      message,
        "progress_pct": progress_pct,
        "step_name":    step_name,
        "data":         data,
    }
    await ws.send_text(json.dumps(payload))

    if msg_type in ("done", "error"):
        await ws.close()
