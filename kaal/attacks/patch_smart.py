"""GradCAM-guided adversarial patch generator.

kaal/attacks/patch_smart.py

Extends generate_patch() from kaal/attacks/patch.py with saliency-guided
initialization and saliency-weighted position sampling. The patch is
initialized from GradCAM activations and placed preferentially over
high-attention regions during training, producing a stronger attack
with the same iteration budget.

Key differences from generate_patch():
    1. Reference image analysed with GradCAM before training begins.
    2. Patch initialized as saliency-weighted noise (not pure random).
    3. 70% of training positions chosen from top-5 saliency hotspots;
       30% random (for position generalization).
    4. Baseline comparison — runs generate_patch() at identical params
       and reports the improvement (skipped in fast mode).
    5. Returns SmartPatchResult with all original PatchResult fields plus
       saliency diagnostics.

Non-PyTorch fallback:
    If model.framework != "pytorch", calls generate_patch() directly,
    emits a RuntimeWarning, and wraps the result in SmartPatchResult
    with saliency fields set to None and layer_name "N/A (non-PyTorch model)".
"""

from __future__ import annotations

import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kaal.attacks.patch import (
    PatchResult,
    apply_patch,
    generate_patch,
    patch_to_printable,
    _get_chw,
    _sample_positions,
    _NORM_MIN,
    _NORM_MAX,
    _build_plain_english,
)
from kaal.attacks.gradcam import extract_saliency
from kaal.engine.loader import KaalModel
from kaal.engine.utils import IMAGENET_MEAN, IMAGENET_STD, tensor_to_pil, ensure_dir


# ---------------------------------------------------------------------------
# SmartPatchResult
# ---------------------------------------------------------------------------

@dataclass
class SmartPatchResult:
    """Result of GradCAM-guided adversarial patch generation.

    Contains all PatchResult fields plus saliency diagnostics.
    Saliency fields are None when model is not PyTorch (fallback mode).
    """

    # ── All PatchResult fields ────────────────────────────────────────────────

    patch_tensor: torch.Tensor
    """Trained patch in normalized ImageNet space, shape (3, H_p, W_p)."""

    patch_pil: Image.Image
    """Screen-resolution RGB PIL Image of the patch (denormalized)."""

    patch_printable_pdf_path: str
    """Absolute path to the generated print-ready PDF. Empty if no output_dir."""

    attack_success_rate: float
    """Fraction of dataset images misclassified with patch (0.0–1.0)."""

    avg_confidence_on_target: float
    """Average model confidence on target_class with patch applied."""

    target_class: int
    """Class index the patch steers the model toward."""

    patch_fraction_used: float
    """Patch area as fraction of total image area."""

    iterations_used: int
    """Number of gradient ascent iterations used."""

    plain_english: str
    """One factual sentence describing the result. No drama."""

    # ── SmartPatch-specific fields ────────────────────────────────────────────

    saliency_map: Optional[np.ndarray]
    """2D float32 array (H, W), values in [0, 1]. None in fallback mode."""

    target_layer_name: str
    """Dotted name of the hooked Conv2d layer, e.g. 'layer4.1.conv2'.
    'N/A (non-PyTorch model)' in fallback mode."""

    saliency_coverage: Optional[float]
    """Fraction of top-20% saliency pixels covered by patch at best position.
    0.0–1.0. None in fallback mode."""

    baseline_success_rate: Optional[float]
    """Success rate of a standard (non-guided) patch at identical params.
    None when fast=True (baseline comparison skipped) or fallback mode."""

    improvement_pct: Optional[float]
    """(smart_rate - baseline_rate) / baseline_rate * 100.
    None when fast=True or fallback mode."""


# ---------------------------------------------------------------------------
# generate_smart_patch() — public entry point
# ---------------------------------------------------------------------------

def generate_smart_patch(
    model: KaalModel,
    dataset,
    target_class: int,
    patch_fraction: float = 0.05,
    iterations: int = 500,
    learning_rate: float = 0.01,
    output_dir: Optional[str] = None,
    seed: int = 42,
    print_size_cm: float = 15.0,
    verbose: bool = True,
    fast: bool = False,
) -> SmartPatchResult:
    """Train a GradCAM-guided adversarial patch.

    Args:
        model:          KaalModel (PyTorch strongly recommended; falls back to
                        generate_patch() for other frameworks with a warning).
        dataset:        KaalDataset from load_dataset().
        target_class:   Class index the patch should steer the model toward.
        patch_fraction: Patch area as fraction of image area. Default 0.05.
        iterations:     Gradient ascent steps. Capped at 100 when fast=True.
        learning_rate:  Gradient step size. Default 0.01.
        output_dir:     Directory to save patch PNG and printable PDF.
                        None = no files written.
        seed:           Random seed for reproducibility.
        print_size_cm:  Physical print size in cm for the printable PDF.
        verbose:        Print progress every 50 iterations.
        fast:           If True, cap iterations at 100 and skip baseline
                        comparison (useful for quick demos / unit tests).

    Returns:
        SmartPatchResult with all PatchResult fields plus saliency diagnostics.
    """
    # ── Non-PyTorch fallback ─────────────────────────────────────────────────
    if model.framework != "pytorch":
        warnings.warn(
            f"generate_smart_patch() received a '{model.framework}' model. "
            "GradCAM-guided initialization requires PyTorch. "
            "Falling back to standard generate_patch() with no saliency guidance.",
            RuntimeWarning,
            stacklevel=2,
        )
        base = generate_patch(
            model, dataset, target_class=target_class,
            patch_fraction=patch_fraction,
            iterations=min(iterations, 100) if fast else iterations,
            learning_rate=learning_rate,
            output_dir=output_dir,
            seed=seed,
            print_size_cm=print_size_cm,
            verbose=verbose,
        )
        return _wrap_fallback(base)

    # ── Cap iterations in fast mode ──────────────────────────────────────────
    if fast:
        iterations = min(iterations, 100)

    # ── Preload dataset ──────────────────────────────────────────────────────
    images = list(dataset)
    if not images:
        raise ValueError("Dataset is empty — cannot train patch.")

    # ── Step 1: GradCAM on reference image ───────────────────────────────────
    reference_tensor, _, _ = images[0]
    if reference_tensor.dim() == 4:
        reference_tensor = reference_tensor.squeeze(0)

    if verbose:
        print("[KAAL SmartPatch] Running GradCAM on reference image...")

    saliency_raw, layer_name = extract_saliency(
        model, reference_tensor, target_class
    )

    # ── Step 2: Resize saliency to model input (H, W) ────────────────────────
    c, img_h, img_w = _get_chw(model.input_shape)
    saliency_map = _resize_saliency(saliency_raw, img_h, img_w)

    if verbose:
        print(f"[KAAL SmartPatch] Hooked layer: {layer_name}  "
              f"| Saliency range: [{saliency_map.min():.3f}, {saliency_map.max():.3f}]")

    # ── Step 3: Patch size ────────────────────────────────────────────────────
    img_area  = img_h * img_w
    patch_side = max(1, int(math.sqrt(int(img_area * patch_fraction))))
    patch_h = patch_side
    patch_w = patch_side

    if verbose:
        print(f"[KAAL SmartPatch] Image: {img_h}×{img_w} | Patch: {patch_h}×{patch_w} "
              f"({patch_fraction*100:.1f}%)")

    # ── Step 4: Precompute top-5 saliency positions ───────────────────────────
    top5_positions = _top_saliency_positions(
        saliency_map, patch_h, patch_w, n=5
    )

    # ── Step 5: Saliency-weighted patch initialization ───────────────────────
    torch.manual_seed(seed)
    random.seed(seed)

    patch_norm = _saliency_init_patch(
        saliency_map, patch_h, patch_w,
        img_h, img_w, seed=seed,
    )
    patch_norm = patch_norm.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([patch_norm], lr=learning_rate)

    if verbose:
        print(f"[KAAL SmartPatch] Training {iterations} iterations  "
              f"(70% saliency positions, 30% random)...")

    # ── Step 6: Training loop with saliency-guided positions ─────────────────
    model._model.eval()

    for iteration in range(1, iterations + 1):
        tensor, _, _ = random.choice(images)

        # 70% saliency-guided, 30% random
        if top5_positions and random.random() < 0.70:
            px, py = random.choice(top5_positions)
        else:
            max_x = max(0, img_w - patch_w)
            max_y = max(0, img_h - patch_h)
            px = random.randint(0, max_x) if max_x > 0 else 0
            py = random.randint(0, max_y) if max_y > 0 else 0

        patched = apply_patch(tensor.detach(), patch_norm, px, py)
        inp     = patched.unsqueeze(0) if patched.dim() == 3 else patched

        logits = model._model(inp)
        loss   = -logits[0, target_class]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp to valid normalized range
        with torch.no_grad():
            nm = _NORM_MIN.expand_as(patch_norm)
            nx = _NORM_MAX.expand_as(patch_norm)
            patch_norm.data.clamp_(nm, nx)

        if verbose and iteration % 50 == 0:
            conf = float(F.softmax(logits, dim=1)[0, target_class].item())
            print(f"[KAAL SmartPatch] Iter {iteration:>4}/{iterations} | "
                  f"loss={loss.item():+.4f} | target_conf={conf:.3f}")

    # ── Step 7: Evaluate smart patch ─────────────────────────────────────────
    patch_final = patch_norm.detach()
    smart_rate, avg_conf = _evaluate_patch(
        model, images, patch_final, target_class, img_h, img_w, patch_h, patch_w
    )

    if verbose:
        print(f"[KAAL SmartPatch] Smart patch success rate: {smart_rate:.1%} "
              f"| Avg target conf: {avg_conf:.3f}")

    # ── Step 8: Baseline comparison (skipped in fast mode) ───────────────────
    baseline_rate: Optional[float] = None
    improvement:   Optional[float] = None

    if not fast:
        if verbose:
            print("[KAAL SmartPatch] Running baseline patch for comparison...")
        base_result = generate_patch(
            model, dataset, target_class=target_class,
            patch_fraction=patch_fraction,
            iterations=iterations,
            learning_rate=learning_rate,
            output_dir=None,
            seed=seed,
            verbose=False,
        )
        baseline_rate = base_result.attack_success_rate
        if baseline_rate > 0:
            improvement = (smart_rate - baseline_rate) / baseline_rate * 100.0
        else:
            improvement = None   # avoid division-by-zero when baseline is 0%
        if verbose:
            print(f"[KAAL SmartPatch] Baseline: {baseline_rate:.1%} | "
                  f"Improvement: "
                  f"{f'{improvement:+.1f}%' if improvement is not None else 'N/A (baseline=0)'}")

    # ── Step 9: Saliency coverage ─────────────────────────────────────────────
    saliency_coverage = _compute_saliency_coverage(
        saliency_map, patch_h, patch_w, top5_positions
    )

    # ── Step 10: Build PIL + save files ──────────────────────────────────────
    patch_pil = tensor_to_pil(patch_final)
    pdf_path  = ""
    if output_dir is not None:
        ensure_dir(output_dir)
        png_path = os.path.join(output_dir, "smart_patch.png")
        patch_pil.save(png_path)
        pdf_path = os.path.join(output_dir, "smart_patch_print.pdf")
        pdf_path = patch_to_printable(
            patch_tensor=patch_final,
            size_cm=print_size_cm,
            output_path=pdf_path,
        )

    # ── plain_english ─────────────────────────────────────────────────────────
    plain_english = _build_smart_plain_english(
        smart_rate=smart_rate,
        avg_conf=avg_conf,
        target_class=target_class,
        patch_fraction=patch_fraction,
        iterations=iterations,
        layer_name=layer_name,
        baseline_rate=baseline_rate,
        improvement=improvement,
    )

    return SmartPatchResult(
        # PatchResult fields
        patch_tensor=patch_final,
        patch_pil=patch_pil,
        patch_printable_pdf_path=pdf_path,
        attack_success_rate=round(smart_rate, 4),
        avg_confidence_on_target=round(avg_conf, 4),
        target_class=target_class,
        patch_fraction_used=patch_fraction,
        iterations_used=iterations,
        plain_english=plain_english,
        # SmartPatch-specific fields
        saliency_map=saliency_map,
        target_layer_name=layer_name,
        saliency_coverage=round(float(saliency_coverage), 4),
        baseline_success_rate=round(float(baseline_rate), 4) if baseline_rate is not None else None,
        improvement_pct=round(float(improvement), 2) if improvement is not None else None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resize_saliency(saliency: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2D saliency map to (h, w) using PIL LANCZOS."""
    if saliency.shape == (h, w):
        return saliency.astype(np.float32)
    pil_img = Image.fromarray((saliency * 255).astype(np.uint8), mode="L")
    pil_img = pil_img.resize((w, h), Image.LANCZOS)
    result  = np.array(pil_img, dtype=np.float32) / 255.0
    # Re-normalise in case rounding drifted the range
    r_min, r_max = result.min(), result.max()
    if r_max - r_min > 1e-8:
        result = (result - r_min) / (r_max - r_min)
    return result


def _top_saliency_positions(
    saliency: np.ndarray,
    patch_h: int,
    patch_w: int,
    n: int = 5,
) -> list[tuple[int, int]]:
    """Find the n patch positions with highest mean saliency underneath.

    Slides a (patch_h × patch_w) window across the saliency map and
    returns the top-n (x, y) positions sorted by descending mean saliency.
    """
    img_h, img_w = saliency.shape
    max_x = max(0, img_w - patch_w)
    max_y = max(0, img_h - patch_h)

    if max_x == 0 and max_y == 0:
        return [(0, 0)]

    # Step size — balance accuracy vs speed; use stride of patch_side//2
    stride = max(1, min(patch_h, patch_w) // 2)
    scores: list[tuple[float, int, int]] = []

    for y in range(0, max_y + 1, stride):
        for x in range(0, max_x + 1, stride):
            region = saliency[y:y + patch_h, x:x + patch_w]
            scores.append((float(region.mean()), x, y))

    scores.sort(reverse=True)
    return [(x, y) for _, x, y in scores[:n]]


def _saliency_init_patch(
    saliency: np.ndarray,
    patch_h: int,
    patch_w: int,
    img_h: int,
    img_w: int,
    seed: int = 42,
) -> torch.Tensor:
    """Initialize patch pixels using saliency as a weighting signal.

    High-saliency positions → pixel values initialized near 0.5 (mid-gray,
    which is a perturbed but visually subtle starting point).
    Low-saliency positions  → uniform random noise in [0, 1].

    The blend is:  pixel = s * 0.5 + (1 - s) * rand   where s = saliency value.

    Returns:
        Patch tensor in normalized ImageNet space, shape (3, patch_h, patch_w).
    """
    rng = np.random.default_rng(seed)

    # Crop the saliency map to patch size from the top-left (or scale down)
    sal_crop = saliency[:patch_h, :patch_w]
    if sal_crop.shape != (patch_h, patch_w):
        sal_crop = _resize_saliency(saliency, patch_h, patch_w)

    # Random noise in [0, 1]
    noise = rng.uniform(0.0, 1.0, (3, patch_h, patch_w)).astype(np.float32)

    # Saliency weights: shape (1, H, W) — same weight applied to all channels
    s = sal_crop[np.newaxis, :, :]  # (1, patch_h, patch_w)

    # Blend: high saliency → 0.5, low saliency → random
    pixels = s * 0.5 + (1.0 - s) * noise  # (3, patch_h, patch_w) via broadcast

    # Convert pixel space [0,1] → normalized ImageNet space
    pixels_t = torch.tensor(pixels)
    mean = IMAGENET_MEAN          # (3, 1, 1)
    std  = IMAGENET_STD           # (3, 1, 1)
    patch_norm = (pixels_t - mean) / std

    # Clamp to valid normalized range
    nm = _NORM_MIN.expand_as(patch_norm)
    nx = _NORM_MAX.expand_as(patch_norm)
    patch_norm = patch_norm.clamp(nm, nx)

    return patch_norm


def _evaluate_patch(
    model: KaalModel,
    images: list,
    patch_tensor: torch.Tensor,
    target_class: int,
    img_h: int, img_w: int,
    patch_h: int, patch_w: int,
) -> tuple[float, float]:
    """Evaluate patch attack success rate and avg target confidence.

    Returns (success_rate, avg_confidence_on_target).
    """
    successes = 0
    total_conf = 0.0
    total = 0

    positions = _sample_positions(img_h, img_w, patch_h, patch_w, n=5)

    for tensor, _, _ in images:
        best_conf = 0.0
        succeeded = False
        for px, py in positions:
            patched = apply_patch(tensor, patch_tensor, px, py)
            pred    = model.predict(patched)
            conf    = pred["all_confidences"][target_class]
            best_conf = max(best_conf, conf)
            if pred["class_idx"] == target_class:
                succeeded = True
        successes  += int(succeeded)
        total_conf += best_conf
        total      += 1

    rate = successes / total if total > 0 else 0.0
    conf = total_conf / total if total > 0 else 0.0
    return rate, conf


def _compute_saliency_coverage(
    saliency: np.ndarray,
    patch_h: int,
    patch_w: int,
    top_positions: list[tuple[int, int]],
) -> float:
    """Fraction of top-20% saliency pixels covered by patch at best position.

    The 'best position' is the first entry in top_positions (highest mean
    saliency under the patch).

    Returns a float in [0, 1].
    """
    if not top_positions:
        return 0.0

    # Top-20% saliency mask
    threshold = float(np.percentile(saliency, 80))   # 80th percentile → top 20%
    top_mask  = (saliency >= threshold)               # (H, W) bool
    n_top_pixels = int(top_mask.sum())

    if n_top_pixels == 0:
        return 0.0

    # Place patch at best position and count overlap with top-20% mask
    img_h, img_w = saliency.shape
    best_x, best_y = top_positions[0]
    x_end = min(best_x + patch_w, img_w)
    y_end = min(best_y + patch_h, img_h)

    patch_region_mask = np.zeros_like(top_mask)
    patch_region_mask[best_y:y_end, best_x:x_end] = True

    overlap = int((top_mask & patch_region_mask).sum())
    return overlap / n_top_pixels


def _wrap_fallback(base: PatchResult) -> SmartPatchResult:
    """Wrap a standard PatchResult in SmartPatchResult for the fallback case."""
    return SmartPatchResult(
        patch_tensor=base.patch_tensor,
        patch_pil=base.patch_pil,
        patch_printable_pdf_path=base.patch_printable_pdf_path,
        attack_success_rate=base.attack_success_rate,
        avg_confidence_on_target=base.avg_confidence_on_target,
        target_class=base.target_class,
        patch_fraction_used=base.patch_fraction_used,
        iterations_used=base.iterations_used,
        plain_english=base.plain_english,
        saliency_map=None,
        target_layer_name="N/A (non-PyTorch model)",
        saliency_coverage=None,
        baseline_success_rate=None,
        improvement_pct=None,
    )


def _build_smart_plain_english(
    smart_rate: float,
    avg_conf: float,
    target_class: int,
    patch_fraction: float,
    iterations: int,
    layer_name: str,
    baseline_rate: Optional[float],
    improvement: Optional[float],
) -> str:
    """One factual sentence. No drama."""
    pct = int(patch_fraction * 100)
    base = (
        f"GradCAM-guided patch ({pct}% of image, {iterations} iterations, "
        f"guided by '{layer_name}') achieved {smart_rate:.0%} attack success rate "
        f"against class {target_class} with average target confidence {avg_conf:.2f}"
    )
    if baseline_rate is not None and improvement is not None:
        sign = "+" if improvement >= 0 else ""
        base += (
            f"; {sign}{improvement:.1f}% vs standard patch baseline "
            f"({baseline_rate:.0%})"
        )
    return base + "."
