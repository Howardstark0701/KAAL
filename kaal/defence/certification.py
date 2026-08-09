"""Model certification — kaal/defence/certification.py

Runs a full KAAL audit, fingerprints the model, checks determinism,
generates a certification badge SVG, and writes a certificate.json.

Usage:
    from kaal.defence.certification import certify_model

    bundle = certify_model(
        model_path  = "resnet50.pt",
        dataset_dir = "./images/",
        output_dir  = "./kaal_cert/",
        attacks     = ["fgsm", "pgd"],
        org_name    = "Acme Corp",
    )
    print(bundle.kvs_score, bundle.kvs_label)
    print(bundle.badge_svg_path)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kaal.defence.fingerprint import fingerprint_model, ModelFingerprint
from kaal.engine.utils import resolve_input_shape


# ---------------------------------------------------------------------------
# Attack-name validation
# ---------------------------------------------------------------------------

_VALID_ATTACKS = {"fgsm", "pgd", "patch", "blackbox", "physical"}


def _validate_attacks(attacks) -> set[str]:
    """Validate attack names against the supported set.

    Raises:
        ValueError: If any name is unknown, or the set is empty.
    """
    attack_set = {a.lower().strip() for a in attacks if a.strip()}
    unknown = attack_set - _VALID_ATTACKS
    if unknown:
        raise ValueError(
            f"Unknown attack name(s): {', '.join(sorted(unknown))}. "
            f"Supported attacks: {', '.join(sorted(_VALID_ATTACKS))}."
        )
    if not attack_set:
        raise ValueError(
            "No attacks specified. "
            f"Supported attacks: {', '.join(sorted(_VALID_ATTACKS))}."
        )
    return attack_set


# ---------------------------------------------------------------------------
# Operational impact sentences by risk tier
# ---------------------------------------------------------------------------

_OPERATIONAL_IMPACT: dict[str, str] = {
    "Robust": (
        "Model is Robust — attack success rates are near zero under standard "
        "adversarial conditions; no immediate remediation required."
    ),
    "Low Risk": (
        "Model is Low Risk — minor susceptibility detected; adversarial "
        "misclassification is possible but requires significant perturbation."
    ),
    "Medium Risk": (
        "Model is Medium Risk — meaningful vulnerability across multiple "
        "attack vectors; remediation recommended before production deployment."
    ),
    "High Risk": (
        "Model is High Risk — reliable misclassification achievable under "
        "realistic conditions with low perturbation; remediation required."
    ),
    "Critical": (
        "Model is Critical — highly vulnerable across multiple dimensions; "
        "adversarial attacks are trivially effective; immediate remediation required."
    ),
    "Catastrophic": (
        "Model is Catastrophic — collapses under minimal perturbation across "
        "all tested dimensions; not suitable for security-sensitive deployment."
    ),
}


# ---------------------------------------------------------------------------
# CertificationBundle
# ---------------------------------------------------------------------------

@dataclass
class CertificationBundle:
    """Full certification result for one model."""

    model_fingerprint:    ModelFingerprint
    kvs_score:            float
    kvs_label:            str
    audit_timestamp:      str
    re_audit_kvs_score:   float
    is_deterministic:     bool
    org_name:             str
    badge_svg_path:       str
    certificate_json_path: str
    operational_impact:   str

    # ── Per-attack success rates (None if attack not run) ─────────────────
    fgsm_success_rate:    Optional[float] = None
    pgd_success_rate:     Optional[float] = None
    patch_success_rate:   Optional[float] = None


# ---------------------------------------------------------------------------
# certify_model()
# ---------------------------------------------------------------------------

def certify_model(
    model_path:  str,
    dataset_dir: str,
    output_dir:  str = "./kaal_cert/",
    attacks:     list[str] = None,
    org_name:    str = "Unknown",
    input_shape: Optional[tuple] = None,
) -> CertificationBundle:
    """Run a full KAAL audit and produce a certification bundle.

    Steps:
        1. Fingerprint the model.
        2. Run full audit (primary).
        3. Run full audit again (re-audit) for determinism check.
        4. Generate badge SVG.
        5. Write certificate.json.
        6. Return CertificationBundle.

    Args:
        model_path:  Path to the model file (.pt, .h5, .onnx, etc.)
        dataset_dir: Path to a directory of test images.
        output_dir:  Where to write badge.svg and certificate.json.
        attacks:     List of attack names. Default: ["fgsm", "pgd", "patch"].
        org_name:    Organisation name shown on the badge and certificate.
        input_shape: Optional explicit (H, W) or (C, H, W) override for the
                     dataset, for dynamic-shape ONNX/TFLite models whose own
                     input_shape has None spatial dims. Defaults to the model's
                     own input shape.

    Returns:
        CertificationBundle with all certification data.

    Raises:
        FileNotFoundError: model_path or dataset_dir does not exist.
    """
    if attacks is None:
        attacks = ["fgsm", "pgd", "patch"]

    model_path  = str(Path(model_path).resolve())
    dataset_dir = str(Path(dataset_dir).resolve())
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: '{model_path}'")
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(f"Dataset directory not found: '{dataset_dir}'")

    # ── Step 1: Fingerprint ───────────────────────────────────────────────
    fingerprint = fingerprint_model(model_path)

    # ── Step 2: Primary audit ─────────────────────────────────────────────
    print(f"[KAAL Certify] Primary audit ({', '.join(attacks).upper()})…")
    kvs1, fgsm_rate, pgd_rate, patch_rate = _run_audit(
        model_path, dataset_dir, attacks, input_shape=input_shape
    )
    audit_ts = datetime.now(timezone.utc).isoformat()

    # ── Step 3: Re-audit for determinism ─────────────────────────────────
    print("[KAAL Certify] Re-audit (determinism check)…")
    kvs2, _, _, _ = _run_audit(model_path, dataset_dir, attacks, input_shape=input_shape)

    is_deterministic = abs(kvs1.score - kvs2.score) <= 0.1

    if not is_deterministic:
        print(
            f"[KAAL Certify] WARNING: Non-deterministic — "
            f"audit 1={kvs1.score:.1f}, audit 2={kvs2.score:.1f}"
        )

    # ── Step 4: Badge SVG ─────────────────────────────────────────────────
    badge_path = str(output_path / "badge.svg")
    _generate_badge_svg(
        output_path=badge_path,
        kvs_score=kvs1.score,
        kvs_label=kvs1.label,
        kvs_color=kvs1.color,
        org_name=org_name,
        is_deterministic=is_deterministic,
    )
    print(f"[KAAL Certify] Badge saved → {badge_path}")

    # ── Step 5: Certificate JSON ──────────────────────────────────────────
    cert_path = str(output_path / "certificate.json")
    operational_impact = _OPERATIONAL_IMPACT.get(
        kvs1.label,
        f"Model is {kvs1.label} — review audit report for details.",
    )

    bundle = CertificationBundle(
        model_fingerprint=fingerprint,
        kvs_score=kvs1.score,
        kvs_label=kvs1.label,
        audit_timestamp=audit_ts,
        re_audit_kvs_score=kvs2.score,
        is_deterministic=is_deterministic,
        org_name=org_name,
        badge_svg_path=badge_path,
        certificate_json_path=cert_path,
        operational_impact=operational_impact,
        fgsm_success_rate=fgsm_rate,
        pgd_success_rate=pgd_rate,
        patch_success_rate=patch_rate,
    )

    _write_certificate_json(bundle, cert_path)
    print(f"[KAAL Certify] Certificate saved → {cert_path}")

    return bundle


# ---------------------------------------------------------------------------
# Internal: audit runner (thin wrapper over existing KAAL attack modules)
# ---------------------------------------------------------------------------

def _run_audit(model_path: str, dataset_dir: str, attacks: list[str],
               input_shape: Optional[tuple] = None):
    """Run FGSM/PGD/patch attacks and return (KVSResult, fgsm_rate, pgd_rate, patch_rate)."""
    from kaal.engine.loader import load_model
    from kaal.engine.dataset import load_dataset
    from kaal.scoring.kvs import calculate_kvs

    attack_set = _validate_attacks(attacks)

    km = load_model(model_path)
    ds = load_dataset(
        dataset_dir,
        input_shape=resolve_input_shape(km.input_shape, input_shape),
        max_images=50,
    )

    fgsm_agg     = None
    pgd_agg      = None
    patch_result = None
    blackbox_result = None

    if "fgsm" in attack_set:
        from kaal.attacks.fgsm import fgsm_attack_dataset
        fgsm_agg = fgsm_attack_dataset(km, ds, epsilon=0.03)

    if "pgd" in attack_set:
        from kaal.attacks.pgd import pgd_attack_dataset
        pgd_agg = pgd_attack_dataset(km, ds, epsilon=0.03, steps=20)

    if "patch" in attack_set:
        from kaal.attacks.patch import generate_patch
        try:
            patch_result = generate_patch(
                km, ds, target_class=0,
                patch_fraction=0.05, iterations=100, verbose=False,
            )
        except Exception:
            pass

    if "blackbox" in attack_set:
        from kaal.attacks.blackbox import blackbox_attack_dataset
        try:
            bb_agg = blackbox_attack_dataset(km, ds, epsilon=0.03, max_images=50)
            from types import SimpleNamespace
            # KVS Dim 5 reads `.query_efficiency`, so expose the dataset
            # aggregate as an object rather than the raw dict.
            blackbox_result = SimpleNamespace(
                query_efficiency=bb_agg["avg_query_efficiency"],
                success_rate=bb_agg["success_rate"],
            )
        except Exception:
            pass

    kvs = calculate_kvs(
        fgsm_result=fgsm_agg,
        pgd_result=pgd_agg,
        patch_result=patch_result,
        blackbox_result=blackbox_result,
    )

    fgsm_rate  = fgsm_agg["success_rate"]  if fgsm_agg   else None
    pgd_rate   = pgd_agg["success_rate"]   if pgd_agg    else None
    patch_rate = patch_result.attack_success_rate if patch_result else None

    return kvs, fgsm_rate, pgd_rate, patch_rate


# ---------------------------------------------------------------------------
# Internal: badge SVG generator (pure string, no SVG library)
# ---------------------------------------------------------------------------

_RISK_COLORS: dict[str, str] = {
    "Robust":       "#22C55E",
    "Low Risk":     "#EAB308",
    "Medium Risk":  "#F97316",
    "High Risk":    "#EF4444",
    "Critical":     "#CC0000",
    "Catastrophic": "#7F0000",
}


def _generate_badge_svg(
    output_path: str,
    kvs_score:   float,
    kvs_label:   str,
    kvs_color:   str,
    org_name:    str,
    is_deterministic: bool,
) -> None:
    """Write a 320×120 dark certification badge SVG to output_path."""
    date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    risk_hex  = _RISK_COLORS.get(kvs_label, kvs_color)
    det_mark  = "" if is_deterministic else " ⚠ non-det"

    # Escape org_name for XML
    safe_org = (
        org_name
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="120" role="img"
     aria-label="KAAL Certification Badge — KVS {kvs_score:.1f} {kvs_label}">
  <title>KAAL Certification Badge</title>

  <!-- Background -->
  <rect width="320" height="120" rx="8" ry="8" fill="#0A0A0A"/>
  <rect width="320" height="120" rx="8" ry="8" fill="none"
        stroke="#1F1F1F" stroke-width="1.5"/>

  <!-- Divider line -->
  <line x1="128" y1="12" x2="128" y2="108" stroke="#1F1F1F" stroke-width="1"/>

  <!-- Left: KAAL branding -->
  <text x="64" y="44" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="22" font-weight="bold" fill="#CC0000">KAAL</text>

  <text x="64" y="64" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="9" fill="#888888" letter-spacing="2">ADVERSARIAL</text>

  <text x="64" y="76" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="9" fill="#888888" letter-spacing="2">AUDIT</text>

  <!-- Left: determinism indicator -->
  <text x="64" y="96" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="8" fill="{'#22C55E' if is_deterministic else '#EF4444'}">
    {'deterministic' if is_deterministic else 'non-deterministic'}
  </text>

  <!-- Right: KVS score -->
  <text x="224" y="50" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="34" font-weight="bold" fill="{risk_hex}">{kvs_score:.1f}</text>

  <!-- Right: risk label badge -->
  <rect x="148" y="58" width="152" height="22" rx="4" ry="4"
        fill="{risk_hex}22" stroke="{risk_hex}" stroke-width="1"/>
  <text x="224" y="74" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="11" font-weight="bold" fill="{risk_hex}">{kvs_label}</text>

  <!-- Footer: certified date + org -->
  <line x1="8" y1="98" x2="312" y2="98" stroke="#1F1F1F" stroke-width="1"/>
  <text x="160" y="112" text-anchor="middle"
        font-family="'JetBrains Mono', Courier New, monospace"
        font-size="8" fill="#555555">
    Certified&#160;&#183;&#160;{date_str}&#160;&#183;&#160;{safe_org}
  </text>
</svg>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(svg, encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal: certificate.json writer
# ---------------------------------------------------------------------------

def _write_certificate_json(bundle: CertificationBundle, path: str) -> None:
    """Serialise the CertificationBundle to certificate.json."""
    doc = {
        "kaal_version":         "1.0.0",
        "certificate_type":     "adversarial_robustness_audit",
        "org_name":             bundle.org_name,
        "audit_timestamp":      bundle.audit_timestamp,
        "kvs": {
            "score":            bundle.kvs_score,
            "label":            bundle.kvs_label,
            "re_audit_score":   bundle.re_audit_kvs_score,
            "is_deterministic": bundle.is_deterministic,
        },
        "attack_results": {
            "fgsm_success_rate":  bundle.fgsm_success_rate,
            "pgd_success_rate":   bundle.pgd_success_rate,
            "patch_success_rate": bundle.patch_success_rate,
        },
        "operational_impact":   bundle.operational_impact,
        "model_fingerprint": {
            "file_hash":        bundle.model_fingerprint.file_hash,
            "weight_hash":      bundle.model_fingerprint.weight_hash,
            "file_size_bytes":  bundle.model_fingerprint.file_size_bytes,
            "modified_at":      bundle.model_fingerprint.modified_at,
            "model_path":       bundle.model_fingerprint.model_path,
            "generated_at":     bundle.model_fingerprint.generated_at,
        },
        "outputs": {
            "badge_svg":        bundle.badge_svg_path,
            "certificate_json": bundle.certificate_json_path,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
