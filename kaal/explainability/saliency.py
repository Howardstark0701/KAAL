"""Saliency map generation — Spec 11.2 — Phase 6, Kiro Prompt 6.2.

Pixel-level sensitivity map. Shows which individual pixels most affect
model output when perturbed.

Method: Vanilla gradient saliency — magnitude of gradient of the target
class score with respect to each input pixel.

    saliency(x) = |∂ score(x) / ∂ x|

The absolute gradient magnitude shows how sensitive the model's output
is to a small change at each pixel. High values → model cares about that pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kaal.engine.loader import KaalModel
from kaal.engine.utils import tensor_to_pil, denormalize


# ---------------------------------------------------------------------------
# Result dataclass — Spec 11.2
# ---------------------------------------------------------------------------

@dataclass
class SaliencyResult:
    """Saliency map result for a single image."""

    saliency_array: np.ndarray
    """Raw saliency map (H × W float32, values in [0, 1]).
    Higher value = pixel has more influence on model output."""

    saliency_pil: Image.Image
    """Grayscale PIL Image. Brighter pixels = higher sensitivity."""

    overlay_pil: Image.Image
    """Original image with saliency overlaid as a red-tinted heatmap at 50% opacity."""

    top_sensitive_pixels_pct: float
    """Percentage of pixels (0.0–1.0) that collectively drive >80% of the
    total saliency signal. Lower = model relies on fewer, more focused pixels."""


# ---------------------------------------------------------------------------
# generate_saliency() — main entry point
# ---------------------------------------------------------------------------

def generate_saliency(
    model: KaalModel,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> SaliencyResult:
    """Compute a pixel-level saliency map for one image.

    Args:
        model:        KaalModel (PyTorch only).
        image_tensor: Normalized tensor (C, H, W) or (1, C, H, W).
        target_class: Class to compute saliency for. None = model's prediction.

    Returns:
        SaliencyResult with raw array, grayscale PIL, overlay PIL, and
        the percentage of pixels driving >80% of the saliency.

    Raises:
        NotImplementedError: Model framework is not PyTorch.
    """
    if model.framework != "pytorch":
        raise NotImplementedError(
            f"Saliency requires a PyTorch model, got '{model.framework}'.\n"
            "→ Convert your model to PyTorch (.pt) to use saliency maps."
        )

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    # Resolve target class
    if target_class is None:
        pred = model.predict(image_tensor)
        target_class = pred["class_idx"]

    # Compute raw gradient saliency
    saliency_raw = _compute_saliency(model._model, image_tensor, target_class)

    # Normalise to [0, 1]
    s_min, s_max = saliency_raw.min(), saliency_raw.max()
    if s_max - s_min > 1e-10:
        saliency_norm = (saliency_raw - s_min) / (s_max - s_min)
    else:
        saliency_norm = np.zeros_like(saliency_raw)

    saliency_norm = saliency_norm.astype(np.float32)

    # Build grayscale PIL
    saliency_pil = _to_grayscale_pil(saliency_norm)

    # Build red-tinted overlay on original image
    overlay_pil = _make_saliency_overlay(image_tensor, saliency_norm)

    # Compute top_sensitive_pixels_pct
    top_pct = _top_pixels_for_80pct(saliency_norm)

    return SaliencyResult(
        saliency_array=saliency_norm,
        saliency_pil=saliency_pil,
        overlay_pil=overlay_pil,
        top_sensitive_pixels_pct=round(float(top_pct), 4),
    )


# ---------------------------------------------------------------------------
# Core saliency computation
# ---------------------------------------------------------------------------

def _compute_saliency(
    raw_model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Compute |∂ score[target_class] / ∂ x| and collapse to (H, W) via max over channels.

    Returns a 2D float32 numpy array (H, W).
    """
    raw_model.eval()
    inp = image_tensor.unsqueeze(0).requires_grad_(True)

    logits = raw_model(inp)
    raw_model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    # Gradient: (1, C, H, W) → take absolute value → max over channels → (H, W)
    grad = inp.grad.data                    # (1, C, H, W)
    saliency = grad.abs().squeeze(0)        # (C, H, W)
    saliency_2d, _ = saliency.max(dim=0)    # (H, W) — most sensitive channel per pixel

    return saliency_2d.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _to_grayscale_pil(saliency: np.ndarray) -> Image.Image:
    """Convert a normalised [0,1] saliency array to a grayscale PIL Image.

    Brighter = more sensitive.
    """
    arr_uint8 = (saliency * 255).astype(np.uint8)
    return Image.fromarray(arr_uint8, mode="L")


def _make_saliency_overlay(
    image_tensor: torch.Tensor,
    saliency: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay saliency on the original image as a red-channel heat.

    The saliency is mapped to a red tint: high saliency → more red.
    Blended 50/50 with the original image.

    Args:
        image_tensor: Normalized image tensor (C, H, W).
        saliency:     Normalised saliency array (H, W) in [0, 1].
        alpha:        Blend factor for the saliency overlay (default 0.5).
    """
    # Original in [0,1] float
    orig_np = denormalize(image_tensor).permute(1, 2, 0).numpy()  # (H,W,3)

    # Red-channel heatmap: (H,W,3) with full red, zero G/B
    red_map = np.zeros((*saliency.shape, 3), dtype=np.float32)
    red_map[..., 0] = saliency   # R channel = saliency intensity

    blended = (1.0 - alpha) * orig_np + alpha * red_map
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")


def _top_pixels_for_80pct(saliency: np.ndarray) -> float:
    """Compute the fraction of pixels that together account for ≥80% of total saliency.

    A lower value means the model is more focused (relies on fewer pixels).
    A value of 1.0 means saliency is spread uniformly across all pixels.
    """
    flat = saliency.flatten()
    total = float(flat.sum())
    if total < 1e-10:
        return 1.0

    # Sort pixels by saliency descending
    sorted_desc = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_desc)

    # Find how many pixels needed to reach 80% of total
    threshold = 0.80 * total
    n_pixels = int(np.searchsorted(cumsum, threshold)) + 1
    n_pixels = min(n_pixels, len(flat))

    return n_pixels / len(flat)
