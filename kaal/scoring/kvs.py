"""KVS Scoring Engine — Spec 12.1 — Phase 7, Kiro Prompt 7.1.

KVS = KAAL Vulnerability Score
Scale: 0.0 to 10.0
Purpose: Single number summarising a model's adversarial robustness
         across five weighted dimensions.

Five Dimensions and Weights:
    Dim 1 — FGSM Susceptibility      20%   fgsm_success_rate × 10
    Dim 2 — PGD  Susceptibility      30%   pgd_success_rate  × 10
    Dim 3 — Perturbation Threshold   20%   (1 − min_epsilon) × 10
    Dim 4 — Physical Survivability   20%   physical_survival_rate × 10
    Dim 5 — Black-Box Efficiency     10%   query_efficiency × 10

Final score:
    kvs = 0.20×dim1 + 0.30×dim2 + 0.20×dim3 + 0.20×dim4 + 0.10×dim5
    kvs = round(kvs, 1), clamped to [0.0, 10.0]

Any dimension whose result is None is skipped and its weight is
redistributed proportionally among the tested dimensions.

KVS Labels:
    ≤ 2.0  Robust
    ≤ 4.0  Low Risk
    ≤ 6.0  Medium Risk
    ≤ 8.0  High Risk
    ≤ 9.5  Critical
    > 9.5  Catastrophic

Remediation rules:
    high_fgsm        if dim1 score > 6.0
    high_pgd         if dim2 score > 6.0
    low_threshold    if dim3 score > 7.0
    high_physical    if dim4 score > 6.0
    high_blackbox    if dim5 score > 6.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Remediation mapping — Spec 12.1
# ---------------------------------------------------------------------------

REMEDIATION_MAP: dict[str, str] = {
    "high_fgsm": (
        "Apply input preprocessing: JPEG compression or random noise addition "
        "before inference."
    ),
    "high_pgd": (
        "Implement adversarial training: retrain model with PGD-generated "
        "examples included in training set."
    ),
    "low_threshold": (
        "Model is sensitive to minimal perturbations. "
        "Consider ensemble methods or certified defenses."
    ),
    "high_physical": (
        "Attack survives real-world conditions. "
        "Physical patch detection layer recommended."
    ),
    "high_blackbox": (
        "Model vulnerable to query-based attacks. "
        "Implement query rate limiting and output confidence rounding."
    ),
}

# Dimension weights — must sum to 1.0
_DIM_WEIGHTS: dict[str, float] = {
    "fgsm_susceptibility":   0.20,
    "pgd_susceptibility":    0.30,
    "perturbation_threshold": 0.20,
    "physical_survivability": 0.20,
    "blackbox_efficiency":   0.10,
}

# Remediation thresholds
_THRESH_HIGH_FGSM      = 6.0
_THRESH_HIGH_PGD       = 6.0
_THRESH_LOW_THRESHOLD  = 7.0
_THRESH_HIGH_PHYSICAL  = 6.0
_THRESH_HIGH_BLACKBOX  = 6.0


# ---------------------------------------------------------------------------
# Result dataclass — Spec 12.1
# ---------------------------------------------------------------------------

@dataclass
class KVSResult:
    """Complete KVS scoring result."""

    score: float
    """Overall KVS score, 0.0 to 10.0."""

    label: str
    """Human-readable risk label: 'Robust' → 'Catastrophic'."""

    color: str
    """Hex color for UI display, corresponding to the label."""

    dimension_scores: dict[str, float]
    """Per-dimension scores (0–10), keyed by dimension name.
    Only contains dimensions that were actually tested."""

    dimensions_tested: list[str]
    """Names of dimensions that contributed to the score."""

    dimensions_skipped: list[str]
    """Names of dimensions skipped due to missing attack results."""

    plain_english: str
    """One-sentence summary. No drama."""

    remediation: list[str]
    """Ordered list of recommended remediation actions."""


# ---------------------------------------------------------------------------
# calculate_kvs() — main entry point
# ---------------------------------------------------------------------------

def calculate_kvs(
    fgsm_result=None,
    pgd_result=None,
    physical_result=None,
    blackbox_result=None,
    min_epsilon: Optional[float] = None,
    fgsm_success_rate: Optional[float] = None,
    pgd_success_rate: Optional[float] = None,
) -> KVSResult:
    """Calculate the KAAL Vulnerability Score from attack results.

    Pass the result objects from the attack modules. Any that are None
    are skipped and their weight is redistributed proportionally.

    Args:
        fgsm_result:       FGSMResult from fgsm_attack_dataset(), or a dict
                           with key 'success_rate'. Pass None to skip.
        pgd_result:        PGDResult from pgd_attack_dataset(), or a dict
                           with key 'success_rate'. Pass None to skip.
        physical_result:   PhysicalRobustnessResult. Uses overall_survival_rate.
        blackbox_result:   BlackBoxResult (stub — not yet implemented).
                           Pass None to skip.
        min_epsilon:       Minimum epsilon at which FGSM achieves ≥50% success
                           across the dataset. Used for Dim 3.
                           If None, estimated from fgsm_result.epsilon_used.
        fgsm_success_rate: Override the FGSM success rate (0.0–1.0) if passing
                           a pre-computed value instead of a result object.
        pgd_success_rate:  Override the PGD success rate similarly.

    Returns:
        KVSResult with score, label, color, per-dim scores, and remediation.

    Example:
        fgsm_agg = fgsm_attack_dataset(model, dataset, epsilon=0.03)
        pgd_agg  = pgd_attack_dataset(model, dataset, epsilon=0.03, steps=40)
        phys     = test_physical_robustness(model, adv_tensor, orig_class)

        kvs = calculate_kvs(
            fgsm_result=fgsm_agg,
            pgd_result=pgd_agg,
            physical_result=phys,
            min_epsilon=0.03,
        )
        print(kvs.score, kvs.label)
    """
    dim_scores: dict[str, Optional[float]] = {
        "fgsm_susceptibility":    None,
        "pgd_susceptibility":     None,
        "perturbation_threshold": None,
        "physical_survivability": None,
        "blackbox_efficiency":    None,
    }

    # ── Dim 1 — FGSM Susceptibility (weight 20%) ────────────────────────────
    fgsm_rate = _extract_success_rate(fgsm_result, override=fgsm_success_rate)
    if fgsm_rate is not None:
        dim_scores["fgsm_susceptibility"] = fgsm_rate * 10.0

    # ── Dim 2 — PGD Susceptibility (weight 30%) ─────────────────────────────
    pgd_rate = _extract_success_rate(pgd_result, override=pgd_success_rate)
    if pgd_rate is not None:
        dim_scores["pgd_susceptibility"] = pgd_rate * 10.0

    # ── Dim 3 — Perturbation Threshold (weight 20%) ─────────────────────────
    # Score = (1 − min_epsilon) × 10
    # Lower epsilon needed → higher score → more vulnerable
    # min_epsilon clamped to [0.001, 1.0] before calculation
    eps = _resolve_min_epsilon(min_epsilon, fgsm_result, pgd_result)
    if eps is not None:
        eps_clamped = max(0.001, min(1.0, eps))
        dim_scores["perturbation_threshold"] = (1.0 - eps_clamped) * 10.0

    # ── Dim 4 — Physical Survivability (weight 20%) ─────────────────────────
    if physical_result is not None:
        survival = getattr(physical_result, "overall_survival_rate", None)
        if survival is not None:
            dim_scores["physical_survivability"] = float(survival) * 10.0

    # ── Dim 5 — Black-Box Efficiency (weight 10%) ───────────────────────────
    if blackbox_result is not None:
        qe = getattr(blackbox_result, "query_efficiency", None)
        if qe is not None:
            dim_scores["blackbox_efficiency"] = float(qe) * 10.0

    # ── Aggregate with weight redistribution ────────────────────────────────
    tested   = [d for d, v in dim_scores.items() if v is not None]
    skipped  = [d for d, v in dim_scores.items() if v is None]

    if not tested:
        # Nothing to score — return neutral 0.0
        return KVSResult(
            score=0.0,
            label=get_kvs_label(0.0),
            color=get_kvs_color(0.0),
            dimension_scores={},
            dimensions_tested=[],
            dimensions_skipped=list(dim_scores.keys()),
            plain_english="No attack results provided; KVS score cannot be computed.",
            remediation=[],
        )

    # Redistribute skipped weights proportionally
    total_skipped_weight = sum(_DIM_WEIGHTS[d] for d in skipped)
    total_tested_weight  = sum(_DIM_WEIGHTS[d] for d in tested)

    kvs_raw = 0.0
    scored_dims: dict[str, float] = {}

    for dim in tested:
        raw_score = dim_scores[dim]
        # Effective weight = original weight + proportional share of skipped weight
        effective_weight = (
            _DIM_WEIGHTS[dim]
            + _DIM_WEIGHTS[dim] / total_tested_weight * total_skipped_weight
        )
        kvs_raw += effective_weight * raw_score
        scored_dims[dim] = round(raw_score, 2)

    kvs = round(max(0.0, min(10.0, kvs_raw)), 1)

    # ── Remediation ─────────────────────────────────────────────────────────
    remediation = _build_remediation(scored_dims)

    # ── plain_english ────────────────────────────────────────────────────────
    label = get_kvs_label(kvs)
    plain_english = _build_plain_english(kvs, label, tested, skipped)

    return KVSResult(
        score=kvs,
        label=label,
        color=get_kvs_color(kvs),
        dimension_scores=scored_dims,
        dimensions_tested=tested,
        dimensions_skipped=skipped,
        plain_english=plain_english,
        remediation=remediation,
    )


# ---------------------------------------------------------------------------
# Label and color helpers — Spec 12.1
# ---------------------------------------------------------------------------

def get_kvs_label(kvs: float) -> str:
    """Return the risk label for a given KVS score."""
    if kvs <= 2.0:
        return "Robust"
    elif kvs <= 4.0:
        return "Low Risk"
    elif kvs <= 6.0:
        return "Medium Risk"
    elif kvs <= 8.0:
        return "High Risk"
    elif kvs <= 9.5:
        return "Critical"
    else:
        return "Catastrophic"


def get_kvs_color(kvs: float) -> str:
    """Return the hex UI color for a given KVS score."""
    if kvs <= 2.0:
        return "#4ADE80"   # green
    elif kvs <= 4.0:
        return "#A3E635"   # yellow-green
    elif kvs <= 6.0:
        return "#FACC15"   # yellow
    elif kvs <= 8.0:
        return "#FB923C"   # orange
    else:
        return "#CC0000"   # KAAL red


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_success_rate(
    result,
    override: Optional[float] = None,
) -> Optional[float]:
    """Pull a success_rate float from a result object or aggregate dict.

    Priority: explicit override > result dict key > result object attribute.
    """
    if override is not None:
        return float(max(0.0, min(1.0, override)))

    if result is None:
        return None

    # Dict form (from fgsm_attack_dataset / pgd_attack_dataset)
    if isinstance(result, dict):
        rate = result.get("success_rate")
        return float(rate) if rate is not None else None

    # Dataclass / object form (single result — treat success as binary 0/1)
    success = getattr(result, "success", None)
    if success is not None:
        return 1.0 if success else 0.0

    return None


def _resolve_min_epsilon(
    min_epsilon: Optional[float],
    fgsm_result,
    pgd_result,
) -> Optional[float]:
    """Resolve the minimum epsilon for Dim 3.

    Priority:
        1. Explicit min_epsilon argument
        2. epsilon_used from fgsm_result (aggregate dict or single result)
        3. epsilon_used from pgd_result
        4. None if nothing available
    """
    if min_epsilon is not None:
        return float(min_epsilon)

    for result in (fgsm_result, pgd_result):
        if result is None:
            continue
        if isinstance(result, dict):
            eps = result.get("epsilon_used")
        else:
            eps = getattr(result, "epsilon_used", None)
        if eps is not None:
            return float(eps)

    return None


def _build_remediation(dim_scores: dict[str, float]) -> list[str]:
    """Select relevant remediation actions based on per-dimension scores."""
    actions = []

    if dim_scores.get("fgsm_susceptibility", 0.0) > _THRESH_HIGH_FGSM:
        actions.append(REMEDIATION_MAP["high_fgsm"])

    if dim_scores.get("pgd_susceptibility", 0.0) > _THRESH_HIGH_PGD:
        actions.append(REMEDIATION_MAP["high_pgd"])

    if dim_scores.get("perturbation_threshold", 0.0) > _THRESH_LOW_THRESHOLD:
        actions.append(REMEDIATION_MAP["low_threshold"])

    if dim_scores.get("physical_survivability", 0.0) > _THRESH_HIGH_PHYSICAL:
        actions.append(REMEDIATION_MAP["high_physical"])

    if dim_scores.get("blackbox_efficiency", 0.0) > _THRESH_HIGH_BLACKBOX:
        actions.append(REMEDIATION_MAP["high_blackbox"])

    return actions


def _build_plain_english(
    kvs: float,
    label: str,
    tested: list[str],
    skipped: list[str],
) -> str:
    """One factual sentence. No drama."""
    n_tested  = len(tested)
    n_skipped = len(skipped)

    if n_skipped == 0:
        return (
            f"Model scored {kvs:.1f}/10 ({label}) across all five "
            f"vulnerability dimensions."
        )
    return (
        f"Model scored {kvs:.1f}/10 ({label}) across {n_tested} of five "
        f"vulnerability dimensions; {n_skipped} dimension(s) were not tested."
    )
