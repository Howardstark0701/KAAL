"""Shared utilities for the KAAL core engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# File / directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_supported_model(path: str | Path) -> bool:
    """Return True if file extension is a supported model format."""
    supported = {".h5", ".keras", ".pt", ".pth", ".onnx", ".tflite"}
    return Path(path).suffix.lower() in supported


def is_supported_image(path: str | Path) -> bool:
    """Return True if file extension is a supported image format."""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return Path(path).suffix.lower() in supported


def resolve_input_shape(
    model_input_shape: tuple,
    override: Optional[tuple] = None,
) -> tuple:
    """Return the explicit input-shape override when given, else the model's own shape.

    Dynamic-shape ONNX/TFLite models report input_shape with None spatial
    dimensions, which load_dataset cannot use to resize images. The CLI's
    --input-size override (parsed to an (H, W) or (C, H, W) tuple) supplies
    concrete dimensions in that case.

    Args:
        model_input_shape: The shape reported by the loaded model.
        override:          Optional explicit (H, W) or (C, H, W) tuple.

    Returns:
        model_input_shape when override is None, else override.
    """
    return override if override is not None else model_input_shape


# ---------------------------------------------------------------------------
# ImageNet normalization constants (used by dataset.py and attack modules)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Tensor ↔ PIL conversion helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization, clamp result to [0, 1].

    Args:
        tensor: Normalized image tensor (C, H, W) or (1, C, H, W).

    Returns:
        Pixel tensor in [0, 1] range, same shape as input.
    """
    t = tensor.detach().cpu()
    squeeze = False
    if t.dim() == 4:
        t = t.squeeze(0)
        squeeze = True

    mean = IMAGENET_MEAN.to(t.device)
    std  = IMAGENET_STD.to(t.device)
    out = t * std + mean
    out = out.clamp(0.0, 1.0)

    if squeeze:
        out = out.unsqueeze(0)
    return out


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized image tensor to a PIL Image.

    Handles ImageNet denormalization automatically.

    Args:
        tensor: Normalized image tensor (C, H, W) or (1, C, H, W).

    Returns:
        PIL Image in RGB mode.
    """
    t = denormalize(tensor)
    if t.dim() == 4:
        t = t.squeeze(0)
    # (C, H, W) float [0,1] → (H, W, C) uint8
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a normalized ImageNet tensor (C, H, W).

    Equivalent to torchvision ToTensor() + Normalize(ImageNet).
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(pil_image.convert("RGB"))


def perturbation_to_pil(perturbation: torch.Tensor, amplify: float = 10.0) -> Image.Image:
    """Convert a perturbation tensor to a visible PIL Image.

    Amplifies the noise so it's visible to the human eye.
    The raw perturbation is typically imperceptible without amplification.

    Args:
        perturbation: Difference tensor (C, H, W).
        amplify:      Scaling factor. Default 10× makes ε=0.03 noise visible.

    Returns:
        PIL Image showing the amplified noise pattern.
    """
    t = perturbation.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    # Scale and shift to [0, 1] for visualization
    t_vis = (t * amplify + 0.5).clamp(0.0, 1.0)
    arr = (t_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
