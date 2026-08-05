"""GradCAM visualization — Spec 11.1 — Phase 6, Kiro Prompt 6.1.

Generates a heatmap showing which image regions the model focuses on
when making a prediction. Run on BOTH clean and adversarial images to
show how an attack shifts model attention.

Implementation:
    - Uses vanilla GradCAM computed manually (no external grad-cam library
      required at import time — falls back cleanly if pytorch_grad_cam is
      not installed).
    - Works by hooking the last convolutional layer's activations and
      gradients, then computing a weighted sum to produce the heatmap.
    - attention_shift_score = 1 − cosine_similarity(clean_heatmap, adv_heatmap)
      normalised to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from kaal.engine.loader import KaalModel
from kaal.engine.utils import tensor_to_pil, denormalize


# ---------------------------------------------------------------------------
# Result dataclasses — Spec 11.1
# ---------------------------------------------------------------------------

@dataclass
class GradCAMResult:
    """GradCAM result for a single image."""

    heatmap_array: np.ndarray
    """Raw heatmap (H × W float32 array, values in [0, 1])."""

    overlay_pil: Image.Image
    """Original image with heatmap overlaid. Red = high attention, Blue = low."""

    target_class_used: int
    """Class index GradCAM was computed for."""

    top_attention_region: str
    """Human-readable description: 'top-left' | 'top-center' | 'top-right' |
    'center-left' | 'center' | 'center-right' |
    'bottom-left' | 'bottom-center' | 'bottom-right'."""


@dataclass
class GradCAMComparisonResult:
    """GradCAM run on both clean and adversarial images."""

    clean_gradcam: GradCAMResult
    """GradCAM on the original clean image."""

    adversarial_gradcam: GradCAMResult
    """GradCAM on the adversarial image."""

    side_by_side_pil: Image.Image
    """Both overlays in one image, side by side, separated by a 2px white border."""

    attention_shift_score: float
    """0–1. How much model attention shifted between clean and adversarial.
    0 = identical attention, 1 = completely different attention."""

    plain_english: str
    """One sentence describing the attention shift. No drama."""


# ---------------------------------------------------------------------------
# generate_gradcam() — single image
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: KaalModel,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> GradCAMResult:
    """Generate a GradCAM heatmap for one image.

    Args:
        model:        KaalModel (PyTorch only — raises NotImplementedError otherwise).
        image_tensor: Normalized tensor (C, H, W) or (1, C, H, W).
        target_class: Class to compute GradCAM for. None = use model's prediction.

    Returns:
        GradCAMResult with heatmap array, coloured overlay, and attention region.

    Raises:
        NotImplementedError: Model framework is not PyTorch.
    """
    if model.framework != "pytorch":
        raise NotImplementedError(
            f"GradCAM requires a PyTorch model, got '{model.framework}'.\n"
            "→ Convert your model to PyTorch (.pt) to use GradCAM."
        )

    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    # Resolve target class
    if target_class is None:
        pred = model.predict(image_tensor)
        target_class = pred["class_idx"]

    # Compute heatmap
    heatmap = _compute_gradcam(model._model, image_tensor, target_class)

    # Resize heatmap to image spatial size
    _, H, W = image_tensor.shape
    heatmap_resized = _resize_heatmap(heatmap, H, W)

    # Build coloured overlay
    overlay = _make_overlay(image_tensor, heatmap_resized)

    return GradCAMResult(
        heatmap_array=heatmap_resized,
        overlay_pil=overlay,
        target_class_used=target_class,
        top_attention_region=_attention_region(heatmap_resized),
    )


# ---------------------------------------------------------------------------
# generate_gradcam_comparison() — clean vs adversarial
# ---------------------------------------------------------------------------

def generate_gradcam_comparison(
    model: KaalModel,
    clean_tensor: torch.Tensor,
    adversarial_tensor: torch.Tensor,
) -> GradCAMComparisonResult:
    """Run GradCAM on both a clean and adversarial image and compare.

    Args:
        model:               KaalModel (PyTorch only).
        clean_tensor:        Original clean image tensor (C, H, W).
        adversarial_tensor:  Adversarial image tensor (C, H, W).

    Returns:
        GradCAMComparisonResult with both heatmaps, side-by-side PIL,
        attention shift score, and a plain-English finding.
    """
    # Use the model's prediction on the clean image as target class for both
    clean_pred  = model.predict(clean_tensor)
    target_class = clean_pred["class_idx"]

    clean_result = generate_gradcam(model, clean_tensor,  target_class=target_class)
    adv_result   = generate_gradcam(model, adversarial_tensor, target_class=target_class)

    # Side-by-side PIL: [clean overlay | 2px white border | adv overlay]
    side_by_side = _make_side_by_side(
        clean_result.overlay_pil,
        adv_result.overlay_pil,
        border_px=2,
    )

    # Attention shift score = 1 − cosine_similarity(flat heatmaps)
    shift = _attention_shift(clean_result.heatmap_array, adv_result.heatmap_array)

    plain_english = _build_comparison_plain_english(
        shift=shift,
        clean_region=clean_result.top_attention_region,
        adv_region=adv_result.top_attention_region,
    )

    return GradCAMComparisonResult(
        clean_gradcam=clean_result,
        adversarial_gradcam=adv_result,
        side_by_side_pil=side_by_side,
        attention_shift_score=round(float(shift), 4),
        plain_english=plain_english,
    )


# ---------------------------------------------------------------------------
# Core GradCAM computation
# ---------------------------------------------------------------------------

def _compute_gradcam(
    raw_model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Compute raw GradCAM heatmap (spatial size of last conv layer).

    Returns a 2D float32 numpy array normalised to [0, 1].
    """
    raw_model.eval()

    # Find the last convolutional layer
    last_conv = _find_last_conv(raw_model)

    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    # Register hooks
    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_fwd = last_conv.register_forward_hook(fwd_hook)
    h_bwd = last_conv.register_full_backward_hook(bwd_hook)

    try:
        inp = image_tensor.unsqueeze(0).requires_grad_(True)
        logits = raw_model(inp)
        raw_model.zero_grad()
        score = logits[0, target_class]
        score.backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    if not activations or not gradients:
        # Fallback: return uniform heatmap if hooks failed
        _, H, W = image_tensor.shape
        return np.ones((H // 32 or 1, W // 32 or 1), dtype=np.float32)

    act  = activations[0].squeeze(0)   # (C, h, w)
    grad = gradients[0].squeeze(0)     # (C, h, w)

    # Global average pool the gradients → channel weights
    weights = grad.mean(dim=(1, 2))    # (C,)

    # Weighted sum of activations
    cam = (weights[:, None, None] * act).sum(dim=0)  # (h, w)
    cam = F.relu(cam)                                  # keep positive contributions

    cam_np = cam.cpu().numpy()
    # Normalise to [0, 1]
    cam_min, cam_max = cam_np.min(), cam_np.max()
    if cam_max - cam_min > 1e-8:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np)

    return cam_np.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_last_conv(model: nn.Module) -> nn.Module:
    """Walk model layers and return the last Conv2d layer found."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found in model.\n"
            "→ GradCAM requires a model with at least one convolutional layer."
        )
    return last_conv


def _resize_heatmap(heatmap: np.ndarray, H: int, W: int) -> np.ndarray:
    """Upsample heatmap to (H, W) using bilinear interpolation."""
    hmap_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    resized = F.interpolate(hmap_t, size=(H, W), mode="bilinear", align_corners=False)
    return resized.squeeze().numpy().astype(np.float32)


def _make_overlay(image_tensor: torch.Tensor, heatmap: np.ndarray) -> Image.Image:
    """Overlay heatmap on original image using a red-blue colormap.

    Red   = high attention (heatmap close to 1)
    Blue  = low  attention (heatmap close to 0)
    """
    # Original image as float [0,1] (H,W,3)
    orig = denormalize(image_tensor)
    orig_np = orig.permute(1, 2, 0).numpy()  # (H,W,3)

    # Convert heatmap to RGB colour via jet-like colormap (red-yellow-green-blue)
    h = heatmap  # (H,W) in [0,1]
    r = np.clip(1.5 - np.abs(h * 4.0 - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(h * 4.0 - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(h * 4.0 - 1.0), 0, 1)
    colormap = np.stack([r, g, b], axis=-1)  # (H,W,3)

    # Blend: 50% original + 50% colormap
    blended = 0.5 * orig_np + 0.5 * colormap
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8), mode="RGB")


def _make_side_by_side(
    left: Image.Image,
    right: Image.Image,
    border_px: int = 2,
) -> Image.Image:
    """Place two PIL Images side by side with a white border between them."""
    # Ensure same height
    h = max(left.height, right.height)
    if left.height != h:
        left = left.resize((left.width, h), Image.LANCZOS)
    if right.height != h:
        right = right.resize((right.width, h), Image.LANCZOS)

    total_w = left.width + border_px + right.width
    canvas = Image.new("RGB", (total_w, h), (255, 255, 255))
    canvas.paste(left,  (0, 0))
    canvas.paste(right, (left.width + border_px, 0))
    return canvas


def _attention_shift(heatmap_a: np.ndarray, heatmap_b: np.ndarray) -> float:
    """Cosine distance between two flattened heatmaps, in [0, 1].

    0 = identical attention patterns.
    1 = completely orthogonal (maximally different).
    """
    a = heatmap_a.flatten().astype(np.float64)
    b = heatmap_b.flatten().astype(np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
    cosine_sim = max(-1.0, min(1.0, cosine_sim))
    # cosine distance ∈ [0, 2], normalised to [0, 1]
    return (1.0 - cosine_sim) / 2.0


def _attention_region(heatmap: np.ndarray) -> str:
    """Return a human-readable label for where the peak attention falls.

    Divides the heatmap into a 3×3 grid and returns the label of the
    cell containing the highest mean activation.
    """
    H, W = heatmap.shape
    h3 = H // 3 or 1
    w3 = W // 3 or 1

    row_labels = ["top", "center", "bottom"]
    col_labels = ["left", "center", "right"]

    best_val = -1.0
    best_label = "center"

    for ri, rl in enumerate(row_labels):
        for ci, cl in enumerate(col_labels):
            r0, r1 = ri * h3, (ri + 1) * h3
            c0, c1 = ci * w3, (ci + 1) * w3
            region_mean = float(heatmap[r0:r1, c0:c1].mean())
            if region_mean > best_val:
                best_val = region_mean
                if rl == "center" and cl == "center":
                    best_label = "center"
                elif rl == "center":
                    best_label = f"center-{cl}"
                elif cl == "center":
                    best_label = f"{rl}-center"
                else:
                    best_label = f"{rl}-{cl}"

    return best_label


def _build_comparison_plain_english(
    shift: float,
    clean_region: str,
    adv_region: str,
) -> str:
    """One factual sentence about the attention shift."""
    shift_pct = int(shift * 100)
    if shift < 0.1:
        magnitude = "minimal"
    elif shift < 0.3:
        magnitude = "moderate"
    elif shift < 0.6:
        magnitude = "significant"
    else:
        magnitude = "substantial"

    if clean_region == adv_region:
        return (
            f"Model attention shows {magnitude} shift (score {shift:.2f}) "
            f"under adversarial perturbation, remaining focused on the "
            f"{clean_region} region."
        )
    return (
        f"Model attention shifted {magnitude}ly (score {shift:.2f}) from the "
        f"{clean_region} region on the clean image to the {adv_region} region "
        f"on the adversarial image."
    )
