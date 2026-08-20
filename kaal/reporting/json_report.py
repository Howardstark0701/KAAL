"""JSON Report Generator — Spec 14.2 — Phase 8, Kiro Prompt 8.1.

Produces a structured JSON file summarising a complete KAAL audit.
Schema matches Spec 14.2 exactly.

Serialisation rules:
    torch.Tensor  → Python list via .tolist()
    PIL.Image     → saved to output dir, path string stored in JSON
    dataclasses   → converted to dict recursively
    float         → rounded to 4 decimal places
    All paths     → relative to output directory
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kaal.engine.utils import ensure_dir


# ---------------------------------------------------------------------------
# generate_json_report() — main entry point
# ---------------------------------------------------------------------------

def generate_json_report(
    output_path: str,
    model_info: dict,
    dataset_info: dict,
    kvs_result=None,
    fgsm_result=None,
    pgd_result=None,
    patch_result=None,
    blackbox_result=None,
    physical_result=None,
    epsilon_curve=None,
    audit_duration_seconds: float = 0.0,
    kaal_version: str = "1.0.0",
    audit_id: Optional[str] = None,
) -> str:
    """Generate a structured JSON audit report and save it to disk.

    Args:
        output_path:             Full path to the output JSON file.
        model_info:              Dict with keys: path, name, framework,
                                 input_shape, num_classes.
        dataset_info:            Dict with keys: path, total_images, formats.
        kvs_result:              KVSResult from calculate_kvs().
        fgsm_result:             FGSMResult or aggregate dict from fgsm_attack_dataset().
        pgd_result:              PGDResult or aggregate dict from pgd_attack_dataset().
        patch_result:            PatchResult from generate_patch().
        blackbox_result:         BlackBoxResult, or the dataset aggregate from
                                 blackbox_attack_dataset().
        epsilon_curve:           dict from epsilon_robustness_curve(). Recorded
                                 as the evidence behind KVS Dim 3a.
        physical_result:         PhysicalRobustnessResult.
        audit_duration_seconds:  How long the audit took.
        kaal_version:            KAAL version string.
        audit_id:                UUID string. Auto-generated if None.

    Returns:
        Absolute path to the saved JSON file.
    """
    output_path = os.path.abspath(output_path)
    ensure_dir(os.path.dirname(output_path))
    output_dir = os.path.dirname(output_path)

    if audit_id is None:
        audit_id = str(uuid.uuid4())

    timestamp = datetime.now(timezone.utc).isoformat()

    doc: dict[str, Any] = {
        "meta": _build_meta(kaal_version, audit_id, timestamp, audit_duration_seconds),
        "model": _build_model_section(model_info),
        "dataset": _build_dataset_section(dataset_info),
        "kvs": _build_kvs_section(kvs_result),
        "attacks": _build_attacks_section(
            fgsm_result, pgd_result, patch_result, blackbox_result
        ),
        "epsilon_robustness": _build_epsilon_curve_section(epsilon_curve),
        "physical_robustness": _build_physical_section(physical_result),
        "output_files": _build_output_files_section(output_dir, patch_result),
        "remediation": _build_remediation_section(kvs_result),
    }

    # Write JSON — pretty-printed, 2-space indent
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=_json_serialiser, ensure_ascii=False)

    return output_path


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_epsilon_curve_section(curve) -> dict:
    """Measured perturbation threshold — the evidence behind KVS Dim 3a.

    Empty dict when no curve was measured, so the key is always present and a
    consumer can distinguish "not measured" from "measured, model never broke"
    (which reports epsilon_50: null with a populated curve).
    """
    if not curve:
        return {}
    get = curve.get if isinstance(curve, dict) else \
        (lambda k, d=None: getattr(curve, k, d))
    # Epsilons keep 6 decimals: the ladder is n/255, so 4 would collapse
    # 0.5/255 (0.001961) and 1/255 (0.003922) toward each other on display.
    return {
        "epsilons":       [round(float(e), 6) for e in (get("epsilons") or [])],
        "success_rates":  [_r4(r) for r in (get("success_rates") or [])],
        "epsilon_50":     (round(float(get("epsilon_50")), 6)
                           if get("epsilon_50") is not None else None),
        "target_rate":    _r4(get("target_rate", 0.5) or 0.5),
        "images_sampled": int(get("images_sampled", 0) or 0),
    }


def _build_meta(
    version: str,
    audit_id: str,
    timestamp: str,
    duration: float,
) -> dict:
    return {
        "kaal_version":      version,
        "audit_id":          audit_id,
        "timestamp":         timestamp,
        "duration_seconds":  round(duration, 1),
    }


def _build_model_section(model_info: dict) -> dict:
    return {
        "path":        str(model_info.get("path", "")),
        "name":        str(model_info.get("name", "")),
        "framework":   str(model_info.get("framework", "")),
        "input_shape": list(model_info.get("input_shape", [])),
        "num_classes": int(model_info.get("num_classes", 0)),
    }


def _build_dataset_section(dataset_info: dict) -> dict:
    return {
        "path":         str(dataset_info.get("path", "")),
        "total_images": int(dataset_info.get("total_images", 0)),
        "formats":      {
            str(k): int(v)
            for k, v in dataset_info.get("formats", {}).items()
        },
    }


def _build_kvs_section(kvs_result) -> dict:
    if kvs_result is None:
        return {}
    return {
        "score":            _r4(getattr(kvs_result, "score", 0.0)),
        "label":            str(getattr(kvs_result, "label", "")),
        "color":            str(getattr(kvs_result, "color", "")),
        "dimension_scores": {
            k: _r4(v)
            for k, v in getattr(kvs_result, "dimension_scores", {}).items()
        },
        "dimensions_tested":  list(getattr(kvs_result, "dimensions_tested", [])),
        "dimensions_skipped": list(getattr(kvs_result, "dimensions_skipped", [])),
    }


def _build_attacks_section(fgsm, pgd, patch, blackbox) -> dict:
    out: dict[str, Any] = {}

    # FGSM
    if fgsm is not None:
        if isinstance(fgsm, dict):
            out["fgsm"] = {
                "epsilon":         _r4(fgsm.get("epsilon_used", 0.0)),
                "success_rate":    _r4(fgsm.get("success_rate", 0.0)),
                "avg_confidence_delta": _r4(fgsm.get("avg_confidence_delta", 0.0)),
                "plain_english":   "",
            }
        else:
            out["fgsm"] = {
                "epsilon":              _r4(getattr(fgsm, "epsilon_used", 0.0)),
                "success_rate":         _r4(1.0 if getattr(fgsm, "success", False) else 0.0),
                "avg_confidence_delta": _r4(getattr(fgsm, "confidence_delta", 0.0)),
                "plain_english":        str(getattr(fgsm, "plain_english", "")),
            }

    # PGD
    if pgd is not None:
        if isinstance(pgd, dict):
            out["pgd"] = {
                "epsilon":              _r4(pgd.get("epsilon_used", 0.0)),
                "alpha":                _r4(pgd.get("alpha_used", 0.0)),
                "steps":                int(pgd.get("steps_used", 0)),
                "success_rate":         _r4(pgd.get("success_rate", 0.0)),
                "avg_steps_to_success": _r4(pgd.get("avg_steps_to_success", -1)),
                "plain_english":        "",
            }
        else:
            out["pgd"] = {
                "epsilon":              _r4(getattr(pgd, "epsilon_used", 0.0)),
                "alpha":                _r4(getattr(pgd, "alpha_used", 0.0)),
                "steps":                int(getattr(pgd, "steps_used", 0)),
                "success_rate":         _r4(1.0 if getattr(pgd, "success", False) else 0.0),
                "avg_steps_to_success": int(getattr(pgd, "steps_to_success", -1)),
                "plain_english":        str(getattr(pgd, "plain_english", "")),
            }

    # Patch
    if patch is not None:
        pdf_path = getattr(patch, "patch_printable_pdf_path", "")
        out["patch"] = {
            "patch_fraction":   _r4(getattr(patch, "patch_fraction_used", 0.0)),
            "iterations":       int(getattr(patch, "iterations_used", 0)),
            "success_rate":     _r4(getattr(patch, "attack_success_rate", 0.0)),
            "avg_confidence_on_target": _r4(getattr(patch, "avg_confidence_on_target", 0.0)),
            "target_class":     int(getattr(patch, "target_class", 0)),
            "physical_threat_rating": "",
            "plain_english":    str(getattr(patch, "plain_english", "")),
            "printable_pdf":    pdf_path,
        }

    # Black-box
    #
    # Two shapes reach here. Every audit pipeline passes the dataset-level
    # aggregate from blackbox_attack_dataset(), which reports averages and has
    # no single .success / .queries_used. A bare BlackBoxResult from a
    # single-image call has those instead. Read whichever is present rather
    # than defaulting the aggregate's fields to zero.
    if blackbox is not None:
        if isinstance(blackbox, dict):
            get = blackbox.get
        else:
            get = lambda k, d=None: getattr(blackbox, k, d)

        is_aggregate = get("avg_query_efficiency") is not None or \
                       get("avg_queries_used") is not None

        if is_aggregate:
            out["blackbox"] = {
                "max_queries":      int(get("max_queries", 0) or 0),
                "success_rate":     _r4(get("success_rate", 0.0) or 0.0),
                "avg_queries_used": _r4(get("avg_queries_used", 0.0) or 0.0),
                "query_efficiency": _r4(get("avg_query_efficiency")
                                        or get("query_efficiency") or 0.0),
                "total_images":     int(get("total_images", 0) or 0),
                "successful_attacks": int(get("successful_attacks", 0) or 0),
                "plain_english":    str(get("plain_english", "") or ""),
            }
        else:
            out["blackbox"] = {
                "max_queries":      int(get("max_queries", 0) or 0),
                "success_rate":     _r4(1.0 if get("success", False) else 0.0),
                "queries_used":     int(get("queries_used", 0) or 0),
                "query_efficiency": _r4(get("query_efficiency", 0.0) or 0.0),
                "plain_english":    str(get("plain_english", "") or ""),
            }

    return out


def _build_physical_section(physical) -> dict:
    if physical is None:
        return {}

    per_transform = {}
    for name, tr in getattr(physical, "per_transform_results", {}).items():
        per_transform[name] = {
            "success_rate": _r4(getattr(tr, "success_rate", 0.0)),
            "category":     str(getattr(tr, "category", "")),
            "total_tested": int(getattr(tr, "total_tested", 0)),
        }

    return {
        "overall_survival_rate":  _r4(getattr(physical, "overall_survival_rate", 0.0)),
        "physical_threat_rating": str(getattr(physical, "physical_threat_rating", "")),
        "most_robust_transform":  str(getattr(physical, "most_robust_transform", "")),
        "least_robust_transform": str(getattr(physical, "least_robust_transform", "")),
        "category_summary": {
            k: _r4(v)
            for k, v in getattr(physical, "category_summary", {}).items()
        },
        "per_transform": per_transform,
        "plain_english": str(getattr(physical, "plain_english", "")),
    }


def _build_output_files_section(output_dir: str, patch_result) -> dict:
    def rel(p: str) -> str:
        """Return path relative to output_dir, or the path as-is if not under it."""
        try:
            return os.path.relpath(p, output_dir)
        except ValueError:
            return p

    out = {
        "pdf_report":           rel(os.path.join(output_dir, "report.pdf")),
        "json_report":          rel(os.path.join(output_dir, "report.json")),
        "fingerprint_chart":    rel(os.path.join(output_dir, "fingerprint.png")),
        "adversarial_examples": rel(os.path.join(output_dir, "adversarial/")),
        "gradcam_comparisons":  rel(os.path.join(output_dir, "gradcam/")),
    }

    if patch_result is not None:
        out["adversarial_patch"] = rel(os.path.join(output_dir, "patch.png"))
        pdf = getattr(patch_result, "patch_printable_pdf_path", "")
        out["patch_printable"]  = rel(pdf) if pdf else rel(os.path.join(output_dir, "patch_print.pdf"))

    return out


def _build_remediation_section(kvs_result) -> list:
    if kvs_result is None:
        return []
    return list(getattr(kvs_result, "remediation", []))


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _r4(v) -> float:
    """Round a numeric value to 4 decimal places."""
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return 0.0


def _json_serialiser(obj: Any) -> Any:
    """Fallback JSON serialiser for non-standard types."""
    # torch.Tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass

    # PIL Image → skip (images are saved separately)
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return None
    except ImportError:
        pass

    # numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 4)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    # dataclasses
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
    except Exception:
        pass

    # Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Fallback
    return str(obj)
