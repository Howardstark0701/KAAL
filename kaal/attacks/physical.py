"""Physical Robustness Simulator.

Spec 10.5 — Phase 5, Kiro Prompt 5.1.

Tests whether adversarial examples survive real-world image transformations.
This is KAAL's key technical differentiator — simulates what happens when
an adversarial image is printed, photographed, or transmitted through a
lossy channel.

Seven transform categories (19 variants total):
    1. JPEG Compression     — quality 90, 75, 60, 40
    2. Gaussian Noise       — sigma 0.01, 0.02, 0.05
    3. Brightness Variation — factors 0.5, 0.75, 1.25, 1.5
    4. Contrast Variation   — factors 0.5, 0.75, 1.25, 1.5
    5. Rotation             — 5°, 10°, 15°, 30°
    6. Scaling              — factors 0.8, 0.9, 1.1, 1.2
    7. Gaussian Blur        — kernel sizes 3, 5, 7

Physical Threat Ratings:
    "Lab Only"    — <30% survival  — works digitally, fails in physical world
    "Limited"     — 30–70% survival — survives some real-world conditions
    "Field Ready" — >70% survival  — realistic deployment threat
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

from kaal.engine.loader import KaalModel
from kaal.engine.utils import tensor_to_pil, pil_to_tensor


# ---------------------------------------------------------------------------
# Transform registry — all 19 variants defined here
# ---------------------------------------------------------------------------

# Each entry: (transform_name, category, callable: PIL.Image → PIL.Image)
_TRANSFORMS: list[tuple[str, str, callable]] = [
    # 1. JPEG Compression
    ("jpeg_90",   "jpeg_compression",  lambda img: _jpeg(img, 90)),
    ("jpeg_75",   "jpeg_compression",  lambda img: _jpeg(img, 75)),
    ("jpeg_60",   "jpeg_compression",  lambda img: _jpeg(img, 60)),
    ("jpeg_40",   "jpeg_compression",  lambda img: _jpeg(img, 40)),

    # 2. Gaussian Noise
    ("noise_001", "gaussian_noise",    lambda img: _gaussian_noise(img, 0.01)),
    ("noise_002", "gaussian_noise",    lambda img: _gaussian_noise(img, 0.02)),
    ("noise_005", "gaussian_noise",    lambda img: _gaussian_noise(img, 0.05)),

    # 3. Brightness Variation
    ("brightness_050", "brightness",   lambda img: _brightness(img, 0.50)),
    ("brightness_075", "brightness",   lambda img: _brightness(img, 0.75)),
    ("brightness_125", "brightness",   lambda img: _brightness(img, 1.25)),
    ("brightness_150", "brightness",   lambda img: _brightness(img, 1.50)),

    # 4. Contrast Variation
    ("contrast_050", "contrast",       lambda img: _contrast(img, 0.50)),
    ("contrast_075", "contrast",       lambda img: _contrast(img, 0.75)),
    ("contrast_125", "contrast",       lambda img: _contrast(img, 1.25)),
    ("contrast_150", "contrast",       lambda img: _contrast(img, 1.50)),

    # 5. Rotation
    ("rotation_05",  "rotation",       lambda img: _rotate(img,  5)),
    ("rotation_10",  "rotation",       lambda img: _rotate(img, 10)),
    ("rotation_15",  "rotation",       lambda img: _rotate(img, 15)),
    ("rotation_30",  "rotation",       lambda img: _rotate(img, 30)),

    # 6. Scaling
    ("scale_080", "scaling",           lambda img: _scale(img, 0.8)),
    ("scale_090", "scaling",           lambda img: _scale(img, 0.9)),
    ("scale_110", "scaling",           lambda img: _scale(img, 1.1)),
    ("scale_120", "scaling",           lambda img: _scale(img, 1.2)),

    # 7. Gaussian Blur
    ("blur_3",    "gaussian_blur",     lambda img: _blur(img, 3)),
    ("blur_5",    "gaussian_blur",     lambda img: _blur(img, 5)),
    ("blur_7",    "gaussian_blur",     lambda img: _blur(img, 7)),
]

# Map transform_name → (category, callable)
_TRANSFORM_MAP: dict[str, tuple[str, callable]] = {
    name: (cat, fn) for name, cat, fn in _TRANSFORMS
}

# All available transform names
ALL_TRANSFORM_NAMES: list[str] = [name for name, _, _ in _TRANSFORMS]

# Physical threat rating thresholds — Spec 10.5
_THRESHOLD_FIELD_READY = 0.70   # >70%  survival
_THRESHOLD_LIMITED     = 0.30   # 30–70% survival
# <30% = Lab Only


# ---------------------------------------------------------------------------
# Per-transform result
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    """Result for a single transform variant across all test images."""

    transform_name: str
    """e.g. 'jpeg_75' or 'brightness_050'"""

    category: str
    """Transform category: 'jpeg_compression', 'gaussian_noise', etc."""

    success_rate: float
    """Fraction of images where adversarial effect survived this transform."""

    total_tested: int
    """Number of images tested under this transform."""

    successful: int
    """Number where the attack still succeeded post-transform."""


# ---------------------------------------------------------------------------
# Main result dataclass — Spec 10.5
# ---------------------------------------------------------------------------

@dataclass
class PhysicalRobustnessResult:
    """Complete physical robustness test result."""

    overall_survival_rate: float
    """Mean attack survival rate across ALL transform variants tested (0.0–1.0)."""

    per_transform_results: dict[str, TransformResult]
    """Mapping transform_name → TransformResult."""

    most_robust_transform: str
    """Transform variant where the attack best survives (highest success_rate)."""

    least_robust_transform: str
    """Transform variant that most effectively breaks the attack (lowest success_rate)."""

    physical_threat_rating: str
    """'Lab Only' | 'Limited' | 'Field Ready' — based on overall_survival_rate."""

    transforms_tested: list[str]
    """Names of all transform variants that were run."""

    category_summary: dict[str, float]
    """Average survival rate per category (e.g. {'jpeg_compression': 0.72, ...})."""

    plain_english: str
    """One factual sentence. No drama, no exclamation marks."""


# ---------------------------------------------------------------------------
# test_physical_robustness() — main entry point
# ---------------------------------------------------------------------------

def test_physical_robustness(
    model: KaalModel,
    adversarial_tensor: torch.Tensor,
    original_class: int,
    transformations: Optional[list[str]] = None,
) -> PhysicalRobustnessResult:
    """Test whether an adversarial image survives real-world transformations.

    Takes a single already-attacked adversarial image and tests it under
    every specified transform. For each transform, the model is queried on
    the transformed image and we check if the adversarial effect persists
    (i.e., the model still predicts something other than original_class).

    Args:
        model:               KaalModel from kaal.engine.loader.
        adversarial_tensor:  Already-attacked image tensor (C, H, W).
                             Should be the adversarial_tensor from FGSMResult
                             or PGDResult.
        original_class:      The class the model correctly predicts on the
                             clean image. Attack "survives" when the model
                             does NOT predict this class post-transform.
        transformations:     List of transform names to test.
                             Pass None to run all 26 variants.
                             Use ALL_TRANSFORM_NAMES to get the full list.

    Returns:
        PhysicalRobustnessResult with per-transform breakdown and
        overall physical threat rating.

    Raises:
        ValueError: Unknown transform name in transformations list.

    Example:
        fgsm_result = fgsm_attack(model, tensor, epsilon=0.03)
        phys = test_physical_robustness(
            model,
            fgsm_result.adversarial_tensor,
            fgsm_result.original_class,
        )
        print(phys.physical_threat_rating)
        print(phys.plain_english)
    """
    # Squeeze batch dim
    if adversarial_tensor.dim() == 4:
        adversarial_tensor = adversarial_tensor.squeeze(0)

    # Resolve transform list
    if transformations is None:
        transforms_to_run = ALL_TRANSFORM_NAMES
    else:
        unknown = [t for t in transformations if t not in _TRANSFORM_MAP]
        if unknown:
            raise ValueError(
                f"Unknown transform name(s): {unknown}\n"
                f"→ Valid names: {ALL_TRANSFORM_NAMES}"
            )
        transforms_to_run = transformations

    # Convert adversarial tensor to PIL once — apply transforms in PIL space
    adv_pil = tensor_to_pil(adversarial_tensor)
    original_size = adv_pil.size  # (W, H) — needed for scale-back after transforms

    # --- Run each transform -------------------------------------------------
    per_transform: dict[str, TransformResult] = {}

    for t_name in transforms_to_run:
        category, transform_fn = _TRANSFORM_MAP[t_name]

        # Apply transform to the adversarial PIL image
        try:
            transformed_pil = transform_fn(adv_pil)
        except Exception as exc:
            # If a transform fails (e.g. extreme params), record as 0% survival
            per_transform[t_name] = TransformResult(
                transform_name=t_name,
                category=category,
                success_rate=0.0,
                total_tested=1,
                successful=0,
            )
            continue

        # Resize back to original dimensions if transform changed them
        if transformed_pil.size != original_size:
            transformed_pil = transformed_pil.resize(
                original_size, Image.LANCZOS
            )

        # Normalise back to model input tensor
        try:
            transformed_tensor = pil_to_tensor(transformed_pil)
        except Exception:
            per_transform[t_name] = TransformResult(
                transform_name=t_name,
                category=category,
                success_rate=0.0,
                total_tested=1,
                successful=0,
            )
            continue

        # Query model on transformed adversarial image
        pred = model.predict(transformed_tensor)
        # Attack "survives" if model still does NOT predict the original class
        survived = int(pred["class_idx"] != original_class)

        per_transform[t_name] = TransformResult(
            transform_name=t_name,
            category=category,
            success_rate=float(survived),
            total_tested=1,
            successful=survived,
        )

    # --- Aggregate results --------------------------------------------------
    if not per_transform:
        return _empty_result()

    survival_rates = [r.success_rate for r in per_transform.values()]
    overall_survival = float(np.mean(survival_rates))

    # Best/worst transforms
    most_robust   = max(per_transform, key=lambda k: per_transform[k].success_rate)
    least_robust  = min(per_transform, key=lambda k: per_transform[k].success_rate)

    # Category averages
    category_sums: dict[str, list[float]] = {}
    for r in per_transform.values():
        category_sums.setdefault(r.category, []).append(r.success_rate)
    category_summary = {
        cat: round(float(np.mean(vals)), 4)
        for cat, vals in category_sums.items()
    }

    # Physical threat rating
    rating = _compute_threat_rating(overall_survival)

    plain_english = _build_plain_english(
        overall_survival=overall_survival,
        rating=rating,
        most_robust=most_robust,
        least_robust=least_robust,
        n_transforms=len(per_transform),
    )

    return PhysicalRobustnessResult(
        overall_survival_rate=round(overall_survival, 4),
        per_transform_results=per_transform,
        most_robust_transform=most_robust,
        least_robust_transform=least_robust,
        physical_threat_rating=rating,
        transforms_tested=list(per_transform.keys()),
        category_summary=category_summary,
        plain_english=plain_english,
    )

# Prevent pytest from collecting this as a test function
test_physical_robustness.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Batch variant — run across a full dataset of adversarial images
# ---------------------------------------------------------------------------

def test_physical_robustness_batch(
    model: KaalModel,
    adversarial_tensors: list[torch.Tensor],
    original_classes: list[int],
    transformations: Optional[list[str]] = None,
) -> PhysicalRobustnessResult:
    """Run physical robustness tests across multiple adversarial images.

    Aggregates per-transform success rates across all images, giving a
    statistically meaningful result vs. a single-image test.

    Args:
        model:                KaalModel instance.
        adversarial_tensors:  List of adversarial image tensors.
        original_classes:     Corresponding original class indices.
        transformations:      Transform names to test (None = all).

    Returns:
        Aggregated PhysicalRobustnessResult across all images.
    """
    if len(adversarial_tensors) != len(original_classes):
        raise ValueError(
            f"adversarial_tensors length ({len(adversarial_tensors)}) must match "
            f"original_classes length ({len(original_classes)})."
        )
    if not adversarial_tensors:
        return _empty_result()

    # Resolve transforms once
    if transformations is None:
        transforms_to_run = ALL_TRANSFORM_NAMES
    else:
        unknown = [t for t in transformations if t not in _TRANSFORM_MAP]
        if unknown:
            raise ValueError(f"Unknown transform name(s): {unknown}")
        transforms_to_run = transformations

    # Accumulate successes and totals per transform
    successes: dict[str, int] = {t: 0 for t in transforms_to_run}
    totals:    dict[str, int] = {t: 0 for t in transforms_to_run}

    for adv_tensor, orig_class in zip(adversarial_tensors, original_classes):
        single = test_physical_robustness(
            model, adv_tensor, orig_class, transformations=transforms_to_run
        )
        for t_name, res in single.per_transform_results.items():
            successes[t_name] += res.successful
            totals[t_name]    += res.total_tested

    # Build aggregated per_transform_results
    per_transform: dict[str, TransformResult] = {}
    for t_name in transforms_to_run:
        cat, _ = _TRANSFORM_MAP[t_name]
        total = totals[t_name]
        succ  = successes[t_name]
        per_transform[t_name] = TransformResult(
            transform_name=t_name,
            category=cat,
            success_rate=round(succ / total, 4) if total > 0 else 0.0,
            total_tested=total,
            successful=succ,
        )

    survival_rates   = [r.success_rate for r in per_transform.values()]
    overall_survival = float(np.mean(survival_rates))
    most_robust      = max(per_transform, key=lambda k: per_transform[k].success_rate)
    least_robust     = min(per_transform, key=lambda k: per_transform[k].success_rate)

    category_sums: dict[str, list[float]] = {}
    for r in per_transform.values():
        category_sums.setdefault(r.category, []).append(r.success_rate)
    category_summary = {
        cat: round(float(np.mean(vals)), 4)
        for cat, vals in category_sums.items()
    }

    rating = _compute_threat_rating(overall_survival)
    plain_english = _build_plain_english(
        overall_survival=overall_survival,
        rating=rating,
        most_robust=most_robust,
        least_robust=least_robust,
        n_transforms=len(per_transform),
    )

    return PhysicalRobustnessResult(
        overall_survival_rate=round(overall_survival, 4),
        per_transform_results=per_transform,
        most_robust_transform=most_robust,
        least_robust_transform=least_robust,
        physical_threat_rating=rating,
        transforms_tested=list(per_transform.keys()),
        category_summary=category_summary,
        plain_english=plain_english,
    )

# Prevent pytest from collecting this as a test function
test_physical_robustness_batch.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Individual PIL transform implementations
# ---------------------------------------------------------------------------

def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    """JPEG encode at given quality then decode back — simulates compression."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() detaches from BytesIO


def _gaussian_noise(img: Image.Image, sigma: float,
                    seed: Optional[int] = None) -> Image.Image:
    """Add zero-mean Gaussian noise with given sigma (in [0,1] pixel scale).

    Pass seed to make the noise reproducible (np.random is seeded first).
    """
    if seed is not None:
        np.random.seed(seed)
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def _brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness by factor (1.0 = original, <1 = darker, >1 = brighter)."""
    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast by factor (1.0 = original)."""
    return ImageEnhance.Contrast(img).enhance(factor)


def _rotate(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image by angle degrees, filling edges with reflected content."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def _scale(img: Image.Image, factor: float) -> Image.Image:
    """Scale image by factor and crop/pad back to original size.

    factor < 1.0 → zoom out (shrink, pad with black edges)
    factor > 1.0 → zoom in (crop centre)
    """
    w, h = img.size
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Paste into a black canvas of original size
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    offset_x = (w - new_w) // 2
    offset_y = (h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _blur(img: Image.Image, kernel_size: int) -> Image.Image:
    """Apply Gaussian blur with given kernel size (must be odd)."""
    radius = (kernel_size - 1) / 2.0
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_threat_rating(survival_rate: float) -> str:
    """Map overall survival rate to a physical threat rating label."""
    if survival_rate > _THRESHOLD_FIELD_READY:
        return "Field Ready"
    elif survival_rate >= _THRESHOLD_LIMITED:
        return "Limited"
    else:
        return "Lab Only"


def _build_plain_english(
    overall_survival: float,
    rating: str,
    most_robust: str,
    least_robust: str,
    n_transforms: int,
) -> str:
    """One factual sentence. No exclamation marks, no threat language."""
    return (
        f"Adversarial effect survived {overall_survival:.0%} of {n_transforms} "
        f"real-world transforms tested, rating '{rating}'; "
        f"most robust under '{most_robust}', "
        f"least robust under '{least_robust}'."
    )


def _empty_result() -> PhysicalRobustnessResult:
    return PhysicalRobustnessResult(
        overall_survival_rate=0.0,
        per_transform_results={},
        most_robust_transform="",
        least_robust_transform="",
        physical_threat_rating="Lab Only",
        transforms_tested=[],
        category_summary={},
        plain_english="No transforms were tested.",
    )


# ---------------------------------------------------------------------------
# Convenience: list all transforms grouped by category
# ---------------------------------------------------------------------------

def list_transforms() -> dict[str, list[str]]:
    """Return all transform names grouped by category."""
    out: dict[str, list[str]] = {}
    for name, cat, _ in _TRANSFORMS:
        out.setdefault(cat, []).append(name)
    return out
