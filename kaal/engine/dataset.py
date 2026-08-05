"""Dataset loader — Spec 9.2 (Phase 1, Kiro Prompt 1.3).

Loads images from a directory, auto-resizes to model input shape,
and normalizes with ImageNet stats.

Returns an iterator of (image_tensor, image_path, original_pil_image) tuples.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import torch
from PIL import Image
from torchvision import transforms


# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Supported image formats
_SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class KaalDataset:
    """Image dataset loader for KAAL audits.

    Usage:
        dataset = load_dataset("./images/", input_shape=(3, 224, 224))
        for tensor, path, pil_image in dataset:
            prediction = model.predict(tensor)

        len(dataset)       # total number of images found
        dataset.summary()  # prints breakdown to console
    """

    def __init__(
        self,
        directory: str,
        input_shape: tuple,
        max_images: Optional[int] = None,
    ):
        """
        Args:
            directory:   Path to directory containing images.
            input_shape: Target model input shape.
                         PyTorch convention: (C, H, W) e.g. (3, 224, 224).
                         TF convention:      (H, W, C) e.g. (224, 224, 3).
                         Both are handled — height and width are extracted correctly.
            max_images:  Optional cap on number of images to load.
        """
        self._directory = Path(directory)
        self._input_shape = input_shape
        self._max_images = max_images

        # Parse height and width regardless of channel ordering
        self._height, self._width = _extract_hw(input_shape)

        # Discover all supported image files
        self._image_paths = self._discover_images()
        if max_images is not None:
            self._image_paths = self._image_paths[:max_images]

        # Build torchvision transform pipeline
        self._transform = transforms.Compose([
            transforms.Resize(
                (self._height, self._width),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # Track format breakdown for summary()
        self._format_counts: dict[str, int] = {}
        for p in self._image_paths:
            ext = Path(p).suffix.lower()
            self._format_counts[ext] = self._format_counts.get(ext, 0) + 1

    # ------------------------------------------------------------------
    # Core iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[torch.Tensor, str, Image.Image]]:
        """Yield (normalized_tensor, path_str, original_pil) for each image.

        Skips images that cannot be opened and prints a warning.
        """
        for path in self._image_paths:
            try:
                pil_image = Image.open(path).convert("RGB")
                tensor = self._transform(pil_image)
                yield tensor, str(path), pil_image
            except Exception as exc:
                print(f"[KAAL] Warning: skipping '{path}' — {exc}")
                continue

    def __len__(self) -> int:
        """Total number of images found in the directory."""
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, Image.Image]:
        """Load a single image by index."""
        path = self._image_paths[idx]
        pil_image = Image.open(path).convert("RGB")
        tensor = self._transform(pil_image)
        return tensor, str(path), pil_image

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a summary of the dataset to stdout."""
        print(f"\n[KAAL Dataset]")
        print(f"  Directory : {self._directory}")
        print(f"  Images    : {len(self._image_paths)}")
        print(f"  Resize to : {self._height}×{self._width}")
        if self._max_images is not None:
            print(f"  Limit     : {self._max_images}")
        print(f"  Formats   :", end="")
        if self._format_counts:
            parts = [f"{ext.lstrip('.')}={n}" for ext, n in sorted(self._format_counts.items())]
            print("  " + ", ".join(parts))
        else:
            print("  (none found)")
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_images(self) -> list[Path]:
        """Scan directory recursively and return sorted list of image paths."""
        if not self._directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: '{self._directory}'\n"
                "→ Check the path is correct and the directory exists."
            )
        if not self._directory.is_dir():
            raise ValueError(
                f"'{self._directory}' is not a directory.\n"
                "→ Provide the path to a folder containing images, not a file."
            )

        paths = [
            p for p in sorted(self._directory.rglob("*"))
            if p.is_file() and p.suffix.lower() in _SUPPORTED_FORMATS
        ]

        if not paths:
            raise ValueError(
                f"No supported images found in '{self._directory}'.\n"
                f"→ Supported formats: {', '.join(sorted(_SUPPORTED_FORMATS))}\n"
                "→ Ensure the directory contains JPEG, PNG, BMP, or WebP files."
            )

        return paths

    @property
    def image_paths(self) -> list[str]:
        """List of all discovered image path strings."""
        return [str(p) for p in self._image_paths]

    @property
    def format_counts(self) -> dict[str, int]:
        """Dict mapping file extension to count, e.g. {'.jpg': 87, '.png': 13}."""
        return dict(self._format_counts)

    @property
    def input_shape(self) -> tuple:
        """The input shape this dataset was configured for."""
        return self._input_shape


# ---------------------------------------------------------------------------
# load_dataset() — public entry point
# ---------------------------------------------------------------------------

def load_dataset(
    directory: str,
    input_shape: tuple = (3, 224, 224),
    max_images: Optional[int] = None,
) -> KaalDataset:
    """Load images from a directory and prepare them for model input.

    Args:
        directory:   Path to a directory containing image files.
        input_shape: Target model input shape.
                     Default: (3, 224, 224) — PyTorch ImageNet standard.
                     TF format (224, 224, 3) is also accepted.
        max_images:  Optional cap on number of images. Useful for quick tests.

    Returns:
        KaalDataset object — iterable of (tensor, path, pil_image) tuples.

    Raises:
        FileNotFoundError: Directory does not exist.
        ValueError:        Directory contains no supported images,
                           or path is not a directory.

    Example:
        dataset = load_dataset("./images/", input_shape=(3, 224, 224))
        for tensor, path, pil in dataset:
            result = model.predict(tensor)
    """
    return KaalDataset(directory, input_shape, max_images)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_hw(input_shape: tuple) -> tuple[int, int]:
    """Extract (height, width) from an input shape tuple.

    Handles both:
        PyTorch convention: (C, H, W) — e.g. (3, 224, 224)
        TF convention:      (H, W, C) — e.g. (224, 224, 3)

    Heuristic: the channel dimension is typically 1, 3, or 4.
    """
    if len(input_shape) == 3:
        c, h, w = input_shape
        # If first dim looks like channels (1/3/4), treat as (C, H, W)
        if c in (1, 3, 4) and h > 4 and w > 4:
            return h, w
        # Otherwise treat as (H, W, C)
        return c, h
    elif len(input_shape) == 2:
        return input_shape[0], input_shape[1]
    else:
        raise ValueError(
            f"Cannot parse input_shape {input_shape}. "
            "Expected (C, H, W) or (H, W, C) or (H, W)."
        )
