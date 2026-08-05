"""Pydantic request/response models — Spec 16.1 — Phase 10."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload responses
# ---------------------------------------------------------------------------

class ModelUploadResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    filename: str
    framework: str
    input_shape: list[int]
    num_classes: int


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    count: int
    formats: dict[str, int]


# ---------------------------------------------------------------------------
# Audit request / response
# ---------------------------------------------------------------------------

class AuditStartRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    dataset_id: str
    attacks: list[str] = ["fgsm", "pgd", "patch", "physical"]
    epsilon: float = Field(default=0.03, ge=0.001, le=1.0)
    steps: int = Field(default=40, ge=1, le=200)
    report_formats: list[str] = ["pdf", "json"]
    no_gradcam: bool = False


class AuditStartResponse(BaseModel):
    job_id: str
    status: str = "started"


class AuditStatusResponse(BaseModel):
    job_id: str
    status: str               # pending | running | complete | failed
    progress_pct: int
    current_step: str
    kvs_score: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Patch request
# ---------------------------------------------------------------------------

class PatchGenerateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    dataset_id: str
    target_class: int = Field(default=0, ge=0)
    patch_fraction: float = Field(default=0.05, ge=0.001, le=0.5)
    iterations: int = Field(default=500, ge=1)
    print_cm: float = Field(default=15.0, gt=0)


class PatchGenerateResponse(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# Compare request
# ---------------------------------------------------------------------------

class CompareResponse(BaseModel):
    before: dict
    after: dict
    delta: dict


# ---------------------------------------------------------------------------
# WebSocket message
# ---------------------------------------------------------------------------

class WSMessage(BaseModel):
    type: str                          # progress | step_complete | error | done
    message: str
    progress_pct: int
    step_name: str
    data: Optional[dict] = None
