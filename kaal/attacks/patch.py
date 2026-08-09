"""Adversarial Patch Generator.

Spec 10.3 — Phase 4, Kiro Prompt 4.1.

How patch training works:
    Initialize patch P as random noise (small square tensor)
    For iteration i = 1 to N:
        Sample random location (x, y) in image
        Apply patch at (x, y): image_with_patch = apply_patch(image, P, x, y)
        Forward pass: logits = model(image_with_patch)
        Loss = -logits[target_class]        ← maximize target class score
        Backward: compute ∇P
        Update: P = P + lr × ∇P
        Clip P to valid pixel range [0, 1]  (in raw pixel space)
    Return trained patch P

The patch is model-agnostic in position — it causes misclassification
regardless of where in the frame it appears.
"""

from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from kaal.engine.loader import KaalModel
from kaal.engine.utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    denormalize,
    tensor_to_pil,
    ensure_dir,
)


# ---------------------------------------------------------------------------
# ImageNet normalization bounds for patch clamping
# ---------------------------------------------------------------------------
# Raw pixel [0,1] → normalized via (pixel - mean) / std
# min normalized ≈ (0 - 0.485) / 0.229 ≈ -2.118
# max normalized ≈ (1 - 0.406) / 0.225 ≈  2.640
_NORM_MIN = ((torch.zeros(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD)
_NORM_MAX = ((torch.ones(3,  1, 1) - IMAGENET_MEAN) / IMAGENET_STD)


# ---------------------------------------------------------------------------
# Result dataclass — Spec 10.3
# ---------------------------------------------------------------------------

@dataclass
class PatchResult:
    """Complete result of adversarial patch generation."""

    patch_tensor: torch.Tensor
    """Trained patch in normalized ImageNet space, shape (3, H_p, W_p)."""

    patch_pil: Image.Image
    """Screen-resolution RGB PIL Image of the patch (denormalized)."""

    patch_printable_pdf_path: str
    """Absolute path to the generated print-ready PDF file.
    Empty string if output_dir was not provided."""

    attack_success_rate: float
    """Fraction of dataset images misclassified with patch in frame (0.0–1.0)."""

    avg_confidence_on_target: float
    """Average model confidence on target_class across dataset with patch applied."""

    target_class: int
    """Class index the patch steers the model toward."""

    patch_fraction_used: float
    """Patch area as fraction of total image area (e.g. 0.05 = 5%)."""

    iterations_used: int
    """Number of gradient ascent iterations used during training."""

    plain_english: str
    """One factual sentence. No drama, no exclamation marks."""


# ---------------------------------------------------------------------------
# apply_patch() — place patch onto image tensor at position (x, y)
# ---------------------------------------------------------------------------

def apply_patch(
    image_tensor: torch.Tensor,
    patch_tensor: torch.Tensor,
    x: int,
    y: int,
) -> torch.Tensor:
    """Overlay patch_tensor onto image_tensor at top-left corner (x, y).

    The patch is clipped to image bounds — no wrap-around.

    Args:
        image_tensor: Image in normalized space, shape (C, H, W) or (1, C, H, W).
        patch_tensor: Patch in normalized space, shape (C, H_p, W_p).
        x:            Horizontal offset (column index) for patch top-left.
        y:            Vertical offset (row index) for patch top-left.

    Returns:
        New tensor (same shape as image_tensor, no batch dim) with patch applied.
    """
    # Remove batch dim if present
    squeeze = False
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
        squeeze = True

    _, img_h, img_w = image_tensor.shape
    _, pat_h, pat_w = patch_tensor.shape

    # Clip patch to image boundaries
    x_end = min(x + pat_w, img_w)
    y_end = min(y + pat_h, img_h)
    patch_w_clipped = x_end - x
    patch_h_clipped = y_end - y

    if patch_w_clipped <= 0 or patch_h_clipped <= 0:
        return image_tensor.clone()

    result = image_tensor.clone()
    result[:, y:y_end, x:x_end] = patch_tensor[:, :patch_h_clipped, :patch_w_clipped]

    if squeeze:
        result = result.unsqueeze(0)  # restore original batch dim
    return result


# ---------------------------------------------------------------------------
# generate_patch() — main training loop
# ---------------------------------------------------------------------------

def generate_patch(
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
) -> PatchResult:
    """Train an adversarial patch via gradient ascent.

    The patch is trained to cause the model to predict target_class
    regardless of where in the frame it appears.

    Args:
        model:          KaalModel from kaal.engine.loader.
        dataset:        KaalDataset — images to train the patch on.
        target_class:   Class index the patch should steer the model toward.
        patch_fraction: Patch area as fraction of image area. Default 0.05 (5%).
                        0.05 on a 224×224 image → ~22×22 pixel patch.
        iterations:     Gradient ascent steps. More = stronger patch. Default 500.
        learning_rate:  Gradient step size. Default 0.01.
        output_dir:     Directory to save patch PNG and printable PDF.
                        If None, no files are written.
        seed:           Random seed for reproducibility.
        print_size_cm:  Physical print size in cm for the printable PDF.
        verbose:        Print progress every 50 iterations.

    Returns:
        PatchResult with trained patch tensor, PIL image, success rate,
        average confidence, and path to printable PDF.

    Raises:
        ValueError: Invalid parameters.
        NotImplementedError: Model framework doesn't support gradients.
    """
    # Patch training relies on torch autograd (model._model + loss.backward()).
    # Other frameworks expose no torch-compatible gradients — fail fast with a
    # clear message instead of an opaque crash deep in the training loop.
    if model.framework != "pytorch":
        raise NotImplementedError(
            "generate_patch requires a PyTorch model. "
            "Note: patch_smart also requires PyTorch for gradient-based patch training."
        )

    # --- Validate inputs -----------------------------------------------------
    if not (0.001 <= patch_fraction <= 0.5):
        raise ValueError(
            f"patch_fraction must be between 0.001 and 0.5, got {patch_fraction}.\n"
            "→ Typical value: 0.05 (5% of image area)."
        )
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}.")
    if not (0 < learning_rate < 10):
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

    # --- Determine patch size from model input shape -------------------------
    c, img_h, img_w = _get_chw(model.input_shape)
    img_area = img_h * img_w
    patch_area = int(img_area * patch_fraction)
    patch_side = max(1, int(math.sqrt(patch_area)))  # square patch
    patch_h = patch_side
    patch_w = patch_side

    if verbose:
        print(f"[KAAL Patch] Image: {img_h}×{img_w} | Patch: {patch_h}×{patch_w} "
              f"({patch_fraction*100:.1f}% of image)")
        print(f"[KAAL Patch] Target class: {target_class} | "
              f"Iterations: {iterations} | LR: {learning_rate}")

    # --- Initialize patch as random noise in [0,1] pixel space ---------------
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    # Initialize in pixel space [0,1], then convert to normalized space
    patch_pixels = torch.rand(3, patch_h, patch_w)
    patch_norm = (patch_pixels - IMAGENET_MEAN) / IMAGENET_STD
    patch_norm = patch_norm.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([patch_norm], lr=learning_rate)

    # --- Preload images into memory (avoids re-opening in hot loop) ----------
    images = list(dataset)
    if not images:
        raise ValueError("Dataset is empty — cannot train patch.")

    # --- Gradient ascent training loop ---------------------------------------
    for iteration in range(1, iterations + 1):
        # Sample a random image from the dataset
        tensor, _, _ = random.choice(images)

        # Sample random patch position
        max_x = max(0, img_w - patch_w)
        max_y = max(0, img_h - patch_h)
        px = random.randint(0, max_x)
        py = random.randint(0, max_y)

        # Apply current patch to image.
        # apply_patch returns a new tensor that shares patch_norm in the graph.
        patched = apply_patch(tensor.detach(), patch_norm, px, py)

        # Forward pass — batch dim required by model
        inp = patched.unsqueeze(0) if patched.dim() == 3 else patched

        # Gradient flows through patch_norm via the patched region of inp
        model._model.eval()
        logits = model._model(inp)

        # Loss = -logit[target_class]  → maximizing target class score
        loss = -logits[0, target_class]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp patch back to valid normalized range (equivalent to pixel [0,1])
        with torch.no_grad():
            norm_min = _NORM_MIN.expand_as(patch_norm)
            norm_max = _NORM_MAX.expand_as(patch_norm)
            patch_norm.data.clamp_(norm_min, norm_max)

        if verbose and iteration % 50 == 0:
            conf = float(F.softmax(logits, dim=1)[0, target_class].item())
            print(f"[KAAL Patch] Iter {iteration:>4}/{iterations} | "
                  f"loss={loss.item():+.4f} | "
                  f"target_conf={conf:.3f}")

    # --- Evaluate patch attack success rate on full dataset ------------------
    patch_final = patch_norm.detach()
    successes = 0
    total_target_conf = 0.0
    total = 0

    for tensor, _, _ in images:
        # Test at multiple random positions
        positions = _sample_positions(img_h, img_w, patch_h, patch_w, n=5)
        image_succeeded = False
        best_conf = 0.0

        for px, py in positions:
            patched = apply_patch(tensor, patch_final, px, py)
            pred = model.predict(patched)
            conf_on_target = pred["all_confidences"][target_class]
            best_conf = max(best_conf, conf_on_target)
            if pred["class_idx"] == target_class:
                image_succeeded = True

        successes += int(image_succeeded)
        total_target_conf += best_conf
        total += 1

    success_rate = successes / total if total > 0 else 0.0
    avg_conf = total_target_conf / total if total > 0 else 0.0

    if verbose:
        print(f"[KAAL Patch] Done. Success rate: {success_rate:.1%} | "
              f"Avg target conf: {avg_conf:.3f}")

    # --- Build output PIL images ---------------------------------------------
    patch_pil = tensor_to_pil(patch_final)

    # --- Save files if output_dir given --------------------------------------
    pdf_path = ""
    if output_dir is not None:
        ensure_dir(output_dir)
        png_path = os.path.join(output_dir, "patch.png")
        patch_pil.save(png_path)

        pdf_path = os.path.join(output_dir, "patch_print.pdf")
        pdf_path = patch_to_printable(
            patch_tensor=patch_final,
            size_cm=print_size_cm,
            output_path=pdf_path,
        )

    # --- plain_english -------------------------------------------------------
    plain_english = _build_plain_english(
        success_rate=success_rate,
        avg_conf=avg_conf,
        target_class=target_class,
        patch_fraction=patch_fraction,
        iterations=iterations,
    )

    return PatchResult(
        patch_tensor=patch_final,
        patch_pil=patch_pil,
        patch_printable_pdf_path=pdf_path,
        attack_success_rate=round(success_rate, 4),
        avg_confidence_on_target=round(avg_conf, 4),
        target_class=target_class,
        patch_fraction_used=patch_fraction,
        iterations_used=iterations,
        plain_english=plain_english,
    )


# ---------------------------------------------------------------------------
# patch_to_printable() — print-ready PDF output
# ---------------------------------------------------------------------------

def patch_to_printable(
    patch_tensor: torch.Tensor,
    size_cm: float = 15.0,
    dpi: int = 300,
    output_path: Optional[str] = None,
) -> str:
    """Convert a patch tensor to a print-ready PDF.

    The PDF contains:
        - Patch image at physical size_cm × size_cm at specified DPI
        - Corner calibration crosshair marks (5mm)
        - Size reference ruler along the bottom edge
        - Disclaimer text below the patch

    Args:
        patch_tensor: Trained patch tensor in normalized space (3, H, W).
        size_cm:      Physical print size in centimetres. Default 15.0.
        dpi:          Print resolution. Default 300 DPI.
        output_path:  Where to save the PDF. If None, saves to ./patch_print.pdf.

    Returns:
        Absolute path to the saved PDF file.

    Raises:
        ImportError: reportlab is not installed.
        ValueError: Invalid size_cm or dpi.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm, mm
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise ImportError(
            "reportlab is not installed.\n"
            "→ Install it with: pip install reportlab==4.1.0"
        )

    if size_cm <= 0:
        raise ValueError(f"size_cm must be > 0, got {size_cm}.")
    if dpi < 72:
        raise ValueError(f"dpi must be >= 72, got {dpi}.")

    if output_path is None:
        output_path = os.path.abspath("./patch_print.pdf")
    output_path = os.path.abspath(output_path)
    ensure_dir(os.path.dirname(output_path))

    # --- Upscale patch to print resolution ----------------------------------
    # size_cm at dpi → size_px = (size_cm / 2.54) * dpi
    size_px = int((size_cm / 2.54) * dpi)
    patch_pil = tensor_to_pil(patch_tensor)
    patch_print = patch_pil.resize((size_px, size_px), Image.LANCZOS)

    # --- Build PDF ----------------------------------------------------------
    PAGE_W, PAGE_H = A4  # 595.27 × 841.89 pts

    # Convert cm to reportlab points (1 cm = 28.346 pts)
    patch_pt = size_cm * cm          # patch width/height in pts
    margin = 2.0 * cm                # page margin
    ruler_h = 0.6 * cm               # ruler height
    calib_size = 0.5 * cm            # calibration mark size
    text_gap = 0.4 * cm

    # Patch origin (bottom-left in reportlab coords, which start from bottom)
    # We place patch centered horizontally, near top
    patch_x = (PAGE_W - patch_pt) / 2
    patch_y = PAGE_H - margin - patch_pt

    c = rl_canvas.Canvas(output_path, pagesize=A4)

    # Background
    c.setFillColorRGB(0.039, 0.039, 0.039)   # #0A0A0A
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # --- Draw patch image ---------------------------------------------------
    patch_bytes = io.BytesIO()
    patch_print.save(patch_bytes, format="PNG")
    patch_bytes.seek(0)
    img_reader = ImageReader(patch_bytes)
    c.drawImage(img_reader, patch_x, patch_y, width=patch_pt, height=patch_pt)

    # --- Corner calibration crosshairs (5mm each arm) -----------------------
    c.setStrokeColorRGB(0.8, 0.0, 0.0)   # #CC0000
    c.setLineWidth(0.5)
    arm = 0.5 * cm  # 5mm arm length
    gap = 0.15 * cm  # gap between crosshair and patch corner

    corners = [
        (patch_x - gap,        patch_y + patch_pt + gap),  # top-left
        (patch_x + patch_pt + gap, patch_y + patch_pt + gap),  # top-right
        (patch_x - gap,        patch_y - gap),              # bottom-left
        (patch_x + patch_pt + gap, patch_y - gap),          # bottom-right
    ]
    for cx, cy in corners:
        # Horizontal arm
        c.line(cx - arm, cy, cx + arm, cy)
        # Vertical arm
        c.line(cx, cy - arm, cx, cy + arm)

    # --- Size reference ruler (bottom edge of patch) ------------------------
    ruler_y = patch_y - ruler_h - 0.3 * cm
    ruler_x = patch_x
    ruler_w = patch_pt

    # Ruler background bar
    c.setFillColorRGB(0.067, 0.067, 0.067)   # #111111
    c.setStrokeColorRGB(0.122, 0.122, 0.122)  # #1F1F1F
    c.setLineWidth(0.3)
    c.rect(ruler_x, ruler_y, ruler_w, ruler_h, fill=1, stroke=1)

    # Ruler tick marks every 1 cm
    n_ticks = int(size_cm)
    tick_spacing = patch_pt / size_cm  # pts per cm
    c.setStrokeColorRGB(0.949, 0.949, 0.949)   # #F2F2F2
    c.setFillColorRGB(0.949, 0.949, 0.949)
    c.setLineWidth(0.3)
    c.setFont("Courier", 5)

    for i in range(n_ticks + 1):
        tx = ruler_x + i * tick_spacing
        tick_h = ruler_h * (0.6 if i % 5 == 0 else 0.35)
        c.line(tx, ruler_y, tx, ruler_y + tick_h)
        if i % 5 == 0 and i > 0:
            c.drawCentredString(tx, ruler_y + ruler_h + 0.1 * cm, f"{i}cm")

    # Left label
    c.drawString(ruler_x, ruler_y + ruler_h + 0.1 * cm, "0")

    # --- Title ---------------------------------------------------------------
    c.setFillColorRGB(0.949, 0.949, 0.949)  # #F2F2F2
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(PAGE_W / 2, PAGE_H - margin * 0.6,
                        "KAAL Adversarial Patch")

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.533, 0.533, 0.533)  # #888888
    c.drawCentredString(PAGE_W / 2, PAGE_H - margin * 0.6 - 0.5 * cm,
                        f"Print at 100% scale — patch size: {size_cm:.0f} cm × {size_cm:.0f} cm")

    # --- Metadata block ------------------------------------------------------
    meta_y = ruler_y - 1.2 * cm
    c.setFont("Courier", 7)
    c.setFillColorRGB(0.4, 0.4, 0.4)  # #666666
    meta_lines = [
        f"Physical size  : {size_cm:.1f} cm × {size_cm:.1f} cm",
        f"Print DPI      : {dpi}",
        f"Patch pixels   : {patch_tensor.shape[1]} × {patch_tensor.shape[2]} (trained)",
        f"Print pixels   : {size_px} × {size_px}",
    ]
    for i, line in enumerate(meta_lines):
        c.drawCentredString(PAGE_W / 2, meta_y - i * 0.35 * cm, line)

    # --- Disclaimer text -----------------------------------------------------
    disclaimer_y = meta_y - len(meta_lines) * 0.35 * cm - 0.5 * cm
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0.267, 0.267, 0.267)  # #444444
    c.drawCentredString(
        PAGE_W / 2, disclaimer_y,
        "KAAL Adversarial Patch — For Security Research Only"
    )
    c.drawCentredString(
        PAGE_W / 2, disclaimer_y - 0.35 * cm,
        "Generated patches are for authorized adversarial robustness testing and model evaluation only."
    )

    c.save()
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_chw(input_shape: tuple) -> tuple[int, int, int]:
    """Extract (C, H, W) from input_shape regardless of convention."""
    if len(input_shape) == 3:
        a, b, c = input_shape
        # PyTorch (C, H, W): first dim is small (channels 1/3/4)
        if a in (1, 3, 4) and b > 4 and c > 4:
            return a, b, c
        # TF (H, W, C): last dim is small
        return c, a, b
    raise ValueError(f"Cannot parse input_shape {input_shape}.")


def _sample_positions(
    img_h: int, img_w: int,
    patch_h: int, patch_w: int,
    n: int = 5,
) -> list[tuple[int, int]]:
    """Return n random (x, y) positions for placing the patch."""
    max_x = max(0, img_w - patch_w)
    max_y = max(0, img_h - patch_h)
    positions = set()
    # Always include corners + center for deterministic coverage
    for px, py in [
        (0, 0),
        (max_x, 0),
        (0, max_y),
        (max_x, max_y),
        (max_x // 2, max_y // 2),
    ]:
        positions.add((px, py))
    while len(positions) < n and max_x > 0 and max_y > 0:
        positions.add((random.randint(0, max_x), random.randint(0, max_y)))
    return list(positions)[:n]


def _build_plain_english(
    success_rate: float,
    avg_conf: float,
    target_class: int,
    patch_fraction: float,
    iterations: int,
) -> str:
    """One factual sentence. No drama."""
    pct = int(patch_fraction * 100)
    return (
        f"Adversarial patch occupying {pct}% of image area trained over "
        f"{iterations} iterations achieved a {success_rate:.0%} attack success rate "
        f"against class {target_class} with average target confidence {avg_conf:.2f}."
    )
