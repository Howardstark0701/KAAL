"""Audit route handlers — Spec 16.1 — Phase 10, Kiro Prompt 10.3.

Endpoints:
    POST /api/upload/model
    POST /api/upload/dataset
    POST /api/audit/start      (async background task)
    GET  /api/audit/status/{job_id}
    GET  /api/audit/result/{job_id}
    GET  /api/report/{job_id}/pdf
    GET  /api/patch/{job_id}/png
    GET  /api/patch/{job_id}/printable
    POST /api/patch/generate
    GET  /api/compare
"""

from __future__ import annotations

import os
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from web.backend.jobs.store import job_store
from web.backend.models import (
    AuditStartRequest,
    AuditStartResponse,
    AuditStatusResponse,
    CompareResponse,
    DatasetUploadResponse,
    ModelUploadResponse,
    PatchGenerateRequest,
    PatchGenerateResponse,
)

router = APIRouter()

# Uploads land in a temp dir per server lifetime
_UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="kaal_uploads_"))
_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="kaal_outputs_"))


# ---------------------------------------------------------------------------
# POST /api/upload/model
# ---------------------------------------------------------------------------

@router.post("/api/upload/model", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """Accept a model file upload, sniff the framework, return model_id."""
    suffix = Path(file.filename).suffix.lower()
    supported = {".h5", ".keras", ".pt", ".pth", ".onnx", ".tflite"}
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model format '{suffix}'. "
                   f"Supported: {', '.join(sorted(supported))}",
        )

    # Save to disk
    save_path = _UPLOAD_DIR / f"{file.filename}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    # Load model to get metadata
    try:
        from kaal.engine.loader import load_model
        kaal_model = load_model(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to load model: {e}")

    model_id = job_store.register_file(str(save_path))

    return ModelUploadResponse(
        model_id=model_id,
        filename=file.filename,
        framework=kaal_model.framework,
        input_shape=list(kaal_model.input_shape),
        num_classes=kaal_model.num_classes,
    )


# ---------------------------------------------------------------------------
# POST /api/upload/dataset
# ---------------------------------------------------------------------------

@router.post("/api/upload/dataset", response_model=DatasetUploadResponse)
async def upload_dataset(files: list[UploadFile] = File(...)):
    """Accept multiple image uploads, save to a dataset dir, return dataset_id."""
    dataset_dir = _UPLOAD_DIR / f"dataset_{os.urandom(4).hex()}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    formats: dict[str, int] = {}
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if suffix not in allowed:
            continue
        dest = dataset_dir / upload.filename
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(upload.file, f_out)
        formats[suffix] = formats.get(suffix, 0) + 1

    if not formats:
        raise HTTPException(status_code=400, detail="No valid image files found.")

    dataset_id = job_store.register_file(str(dataset_dir))
    total = sum(formats.values())

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        count=total,
        formats={k.lstrip("."): v for k, v in formats.items()},
    )


# ---------------------------------------------------------------------------
# POST /api/audit/start
# ---------------------------------------------------------------------------

@router.post("/api/audit/start", response_model=AuditStartResponse)
async def start_audit(req: AuditStartRequest, background_tasks: BackgroundTasks):
    """Start a full audit as a background task. Returns job_id immediately."""
    model_path = job_store.get_file_path(req.model_id)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"model_id '{req.model_id}' not found.")

    dataset_path = job_store.get_file_path(req.dataset_id)
    if not dataset_path:
        raise HTTPException(status_code=404, detail=f"dataset_id '{req.dataset_id}' not found.")

    job_id = job_store.create("audit")
    job_store.update(job_id, model_id=req.model_id, dataset_id=req.dataset_id)

    background_tasks.add_task(
        _run_audit_pipeline,
        job_id=job_id,
        model_path=model_path,
        dataset_path=dataset_path,
        attacks=req.attacks,
        epsilon=req.epsilon,
        steps=req.steps,
        report_formats=req.report_formats,
        no_gradcam=req.no_gradcam,
    )

    return AuditStartResponse(job_id=job_id, status="started")


# ---------------------------------------------------------------------------
# GET /api/audit/status/{job_id}
# ---------------------------------------------------------------------------

@router.get("/api/audit/status/{job_id}", response_model=AuditStatusResponse)
async def get_audit_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    kvs_score = None
    if job.status == "complete" and job.result_data:
        kvs_score = job.result_data.get("kvs", {}).get("score")

    return AuditStatusResponse(
        job_id=job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        current_step=job.current_step,
        kvs_score=kvs_score,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /api/audit/result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/api/audit/result/{job_id}")
async def get_audit_result(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete (status: {job.status}).",
        )
    return JSONResponse(content=job.result_data or {})


# ---------------------------------------------------------------------------
# GET /api/report/{job_id}/pdf
# ---------------------------------------------------------------------------

@router.get("/api/report/{job_id}/pdf")
async def download_pdf(job_id: str):
    job = job_store.get(job_id)
    if not job or job.status != "complete":
        raise HTTPException(status_code=404, detail="Report not available.")
    if not job.pdf_path or not os.path.exists(job.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(
        job.pdf_path,
        media_type="application/pdf",
        filename=f"kaal_report_{job_id[:8]}.pdf",
    )


# ---------------------------------------------------------------------------
# GET /api/patch/{job_id}/png
# ---------------------------------------------------------------------------

@router.get("/api/patch/{job_id}/png")
async def download_patch_png(job_id: str):
    job = job_store.get(job_id)
    if not job or not job.patch_png_path or not os.path.exists(job.patch_png_path):
        raise HTTPException(status_code=404, detail="Patch PNG not found.")
    return FileResponse(job.patch_png_path, media_type="image/png",
                        filename=f"kaal_patch_{job_id[:8]}.png")


# ---------------------------------------------------------------------------
# GET /api/patch/{job_id}/printable
# ---------------------------------------------------------------------------

@router.get("/api/patch/{job_id}/printable")
async def download_patch_printable(job_id: str):
    job = job_store.get(job_id)
    if not job or not job.patch_pdf_path or not os.path.exists(job.patch_pdf_path):
        raise HTTPException(status_code=404, detail="Printable patch PDF not found.")
    return FileResponse(job.patch_pdf_path, media_type="application/pdf",
                        filename=f"kaal_patch_print_{job_id[:8]}.pdf")


# ---------------------------------------------------------------------------
# POST /api/patch/generate
# ---------------------------------------------------------------------------

@router.post("/api/patch/generate", response_model=PatchGenerateResponse)
async def generate_patch_endpoint(req: PatchGenerateRequest,
                                   background_tasks: BackgroundTasks):
    model_path = job_store.get_file_path(req.model_id)
    if not model_path:
        raise HTTPException(status_code=404, detail="model_id not found.")
    dataset_path = job_store.get_file_path(req.dataset_id)
    if not dataset_path:
        raise HTTPException(status_code=404, detail="dataset_id not found.")

    job_id = job_store.create("patch")
    background_tasks.add_task(
        _run_patch_pipeline,
        job_id=job_id,
        model_path=model_path,
        dataset_path=dataset_path,
        target_class=req.target_class,
        patch_fraction=req.patch_fraction,
        iterations=req.iterations,
        print_cm=req.print_cm,
    )
    return PatchGenerateResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# GET /api/compare
# ---------------------------------------------------------------------------

@router.get("/api/compare")
async def compare_audits(
    before_id: str = Query(...),
    after_id:  str = Query(...),
):
    job_a = job_store.get(before_id)
    job_b = job_store.get(after_id)

    if not job_a or job_a.status != "complete":
        raise HTTPException(status_code=404, detail=f"Job '{before_id}' not ready.")
    if not job_b or job_b.status != "complete":
        raise HTTPException(status_code=404, detail=f"Job '{after_id}' not ready.")

    kvs_a = (job_a.result_data or {}).get("kvs", {})
    kvs_b = (job_b.result_data or {}).get("kvs", {})

    dims_a = kvs_a.get("dimension_scores", {})
    dims_b = kvs_b.get("dimension_scores", {})
    all_dims = sorted(set(list(dims_a) + list(dims_b)))

    delta = {
        "overall": round(kvs_b.get("score", 0) - kvs_a.get("score", 0), 4),
        "dimensions": {
            d: round(dims_b.get(d, 0) - dims_a.get(d, 0), 4)
            for d in all_dims
        },
    }
    return CompareResponse(before=kvs_a, after=kvs_b, delta=delta)


# ---------------------------------------------------------------------------
# Background task: full audit pipeline
# ---------------------------------------------------------------------------

def _run_audit_pipeline(
    job_id: str,
    model_path: str,
    dataset_path: str,
    attacks: list[str],
    epsilon: float,
    steps: int,
    report_formats: list[str],
    no_gradcam: bool,
):
    """Run the full KAAL audit pipeline as a FastAPI background task."""
    import time
    start_time = time.time()

    try:
        job_store.update(job_id, status="running", progress_pct=5,
                         current_step="Loading model and dataset")

        from kaal.engine.loader import load_model
        from kaal.engine.dataset import load_dataset

        kaal_model   = load_model(model_path)
        kaal_dataset = load_dataset(dataset_path, input_shape=kaal_model.input_shape)

        output_dir = _OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        fgsm_agg = pgd_agg = patch_result = phys_result = None
        adv_tensors: list = []
        orig_classes: list = []
        attack_list = [a.lower() for a in attacks]
        n_attacks = len(attack_list)
        pct_per_step = 70 // max(n_attacks, 1)
        current_pct = 10

        # FGSM
        if "fgsm" in attack_list:
            job_store.update(job_id, progress_pct=current_pct,
                             current_step="Running FGSM attack")
            from kaal.attacks.fgsm import fgsm_attack_dataset
            fgsm_agg = fgsm_attack_dataset(kaal_model, kaal_dataset, epsilon=epsilon)
            for r in fgsm_agg["results"]:
                adv_tensors.append(r.adversarial_tensor)
                orig_classes.append(r.original_class)
            current_pct += pct_per_step

        # PGD
        if "pgd" in attack_list:
            job_store.update(job_id, progress_pct=current_pct,
                             current_step="Running PGD attack")
            from kaal.attacks.pgd import pgd_attack_dataset
            pgd_agg = pgd_attack_dataset(kaal_model, kaal_dataset,
                                          epsilon=epsilon, steps=steps)
            if not adv_tensors:
                for r in pgd_agg["results"]:
                    adv_tensors.append(r.adversarial_tensor)
                    orig_classes.append(r.original_class)
            current_pct += pct_per_step

        # Patch
        if "patch" in attack_list:
            job_store.update(job_id, progress_pct=current_pct,
                             current_step="Generating adversarial patch")
            from kaal.attacks.patch import generate_patch
            patch_result = generate_patch(
                kaal_model, kaal_dataset,
                target_class=0, patch_fraction=0.05,
                iterations=500, output_dir=str(output_dir), verbose=False,
            )
            current_pct += pct_per_step

        # Physical
        if "physical" in attack_list and adv_tensors:
            job_store.update(job_id, progress_pct=current_pct,
                             current_step="Running physical robustness tests")
            from kaal.attacks.physical import test_physical_robustness_batch
            phys_result = test_physical_robustness_batch(
                kaal_model,
                adv_tensors[:min(len(adv_tensors), 20)],
                orig_classes[:min(len(orig_classes), 20)],
            )
            current_pct += pct_per_step

        # Explainability + scoring + reports
        job_store.update(job_id, progress_pct=80,
                         current_step="Generating scores and reports")

        collapse_path = fingerprint_path = None

        if pgd_agg and pgd_agg["results"]:
            try:
                from kaal.explainability.confidence import generate_collapse_curve
                collapse_path = str(output_dir / "collapse_curve.png")
                generate_collapse_curve(pgd_agg["results"][0], collapse_path)
            except Exception:
                collapse_path = None

        from kaal.scoring.kvs import calculate_kvs
        kvs_result = calculate_kvs(
            fgsm_result=fgsm_agg,
            pgd_result=pgd_agg,
            physical_result=phys_result,
            min_epsilon=epsilon,
        )

        try:
            from kaal.fingerprint.radar import generate_fingerprint
            fingerprint_path = str(output_dir / "fingerprint.png")
            from pathlib import Path as P
            generate_fingerprint(kvs_result, P(model_path).stem, fingerprint_path)
        except Exception:
            fingerprint_path = None

        model_info = {
            "path": model_path,
            "name": Path(model_path).stem,
            "framework": kaal_model.framework,
            "input_shape": list(kaal_model.input_shape),
            "num_classes": kaal_model.num_classes,
        }
        dataset_info = {
            "path": dataset_path,
            "total_images": len(kaal_dataset),
            "formats": kaal_dataset.format_counts,
        }

        duration = time.time() - start_time

        # JSON report
        json_path = pdf_path = None
        if "json" in report_formats or "all" in report_formats:
            from kaal.reporting.json_report import generate_json_report
            json_path = generate_json_report(
                output_path=str(output_dir / "report.json"),
                model_info=model_info,
                dataset_info=dataset_info,
                kvs_result=kvs_result,
                fgsm_result=fgsm_agg,
                pgd_result=pgd_agg,
                patch_result=patch_result,
                physical_result=phys_result,
                audit_duration_seconds=duration,
            )

        if "pdf" in report_formats or "all" in report_formats:
            from kaal.reporting.pdf import generate_pdf_report
            pdf_path = generate_pdf_report(
                output_path=str(output_dir / "report.pdf"),
                model_info=model_info,
                dataset_info=dataset_info,
                kvs_result=kvs_result,
                fgsm_result=fgsm_agg["results"][0] if fgsm_agg and fgsm_agg["results"] else None,
                pgd_result=pgd_agg["results"][0] if pgd_agg and pgd_agg["results"] else None,
                patch_result=patch_result,
                physical_result=phys_result,
                collapse_curve_path=collapse_path,
                fingerprint_path=fingerprint_path,
                audit_duration_seconds=duration,
            )

        # Load result JSON for storage
        result_data = {}
        if json_path and Path(json_path).exists():
            with open(json_path) as f:
                result_data = json.load(f)

        patch_png  = str(output_dir / "patch.png") if patch_result else None
        patch_pdf  = patch_result.patch_printable_pdf_path if patch_result else None

        job_store.complete(
            job_id,
            result_data=result_data,
            result_json_path=json_path,
            pdf_path=pdf_path,
            fingerprint_path=fingerprint_path,
        )
        job_store.update(
            job_id,
            output_dir=str(output_dir),
            patch_png_path=patch_png,
            patch_pdf_path=patch_pdf,
        )

    except Exception as exc:
        job_store.fail(job_id, error=str(exc) + "\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Background task: patch-only pipeline
# ---------------------------------------------------------------------------

def _run_patch_pipeline(
    job_id: str,
    model_path: str,
    dataset_path: str,
    target_class: int,
    patch_fraction: float,
    iterations: int,
    print_cm: float,
):
    try:
        job_store.update(job_id, status="running", progress_pct=5,
                         current_step="Loading model and dataset")

        from kaal.engine.loader import load_model
        from kaal.engine.dataset import load_dataset
        from kaal.attacks.patch import generate_patch

        kaal_model   = load_model(model_path)
        kaal_dataset = load_dataset(dataset_path, input_shape=kaal_model.input_shape)
        output_dir = _OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        job_store.update(job_id, progress_pct=20,
                         current_step="Training adversarial patch")
        result = generate_patch(
            kaal_model, kaal_dataset,
            target_class=target_class,
            patch_fraction=patch_fraction,
            iterations=iterations,
            output_dir=str(output_dir),
            print_size_cm=print_cm,
            verbose=False,
        )

        job_store.complete(job_id, result_data={
            "target_class": target_class,
            "attack_success_rate": result.attack_success_rate,
            "avg_confidence_on_target": result.avg_confidence_on_target,
            "patch_fraction_used": result.patch_fraction_used,
            "iterations_used": result.iterations_used,
            "plain_english": result.plain_english,
        })
        job_store.update(
            job_id,
            output_dir=str(output_dir),
            patch_png_path=str(output_dir / "patch.png"),
            patch_pdf_path=result.patch_printable_pdf_path,
        )

    except Exception as exc:
        job_store.fail(job_id, error=str(exc))
