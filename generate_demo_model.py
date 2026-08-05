"""Generate demo model and images for KAAL demo recording.

Usage:
    python generate_demo_model.py

Creates:
    demo_model.pt      — pretrained ResNet18 saved as full model
    demo_images/       — 10 synthetic images (224x224 RGB)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision.models as tv
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent

# ── Model ────────────────────────────────────────────────────────────────────
print("Downloading ResNet18 (pretrained)...")
model = tv.resnet18(weights="IMAGENET1K_V1")
model.eval()
torch.save(model, ROOT / "demo_model.pt")
print(f"  Saved → demo_model.pt")

# ── Demo images ───────────────────────────────────────────────────────────────
demo_dir = ROOT / "demo_images"
demo_dir.mkdir(exist_ok=True)

# 10 synthetic images with varied colour distributions
# (realistic enough for the demo — the attack works on any pixel data)
rng = np.random.default_rng(42)
palettes = [
    (220, 180, 140),   # warm sandy
    ( 80, 120, 200),   # cool blue
    (160,  60,  60),   # brick red
    ( 90, 160, 100),   # forest green
    (200, 200,  80),   # yellow
    (140,  80, 180),   # purple
    ( 50, 180, 200),   # teal
    (230, 130,  50),   # orange
    (180, 180, 180),   # grey
    ( 60,  60,  60),   # dark
]

for i, (r, g, b) in enumerate(palettes):
    arr = rng.integers(
        [max(0, r-40), max(0, g-40), max(0, b-40)],
        [min(255, r+40), min(255, g+40), min(255, b+40)],
        size=(224, 224, 3), dtype=np.uint8,
    )
    img = Image.fromarray(arr)
    path = demo_dir / f"img_{i+1:02d}.jpg"
    img.save(path, quality=92)

print(f"  Created {len(palettes)} demo images → demo_images/")
print("Done. Ready to record.")
