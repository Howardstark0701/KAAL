"""KAAL Quick Start — try KAAL in under 2 minutes.

Downloads ResNet50, runs an FGSM attack, prints results, saves output.

Usage:
    python examples/quick_start.py
"""

import os
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR   = Path(__file__).parent / "models"
MODEL_PATH   = MODELS_DIR / "test_resnet50.pt"
IMAGE_PATH   = Path(__file__).parent / "test_image.jpg"
OUTPUT_PATH  = Path(__file__).parent / "quick_start_output.png"
MODELS_DIR.mkdir(exist_ok=True)

console = Console()

# ---------------------------------------------------------------------------
# Step 1 — Download / load ResNet50
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    console.print("[bold cyan]Downloading ResNet50 weights...[/]")
    import torchvision.models as tv
    m = tv.resnet50(weights="IMAGENET1K_V1")
    torch.save(m, MODEL_PATH)
    console.print(f"[green]Model saved → {MODEL_PATH}[/]")
else:
    console.print(f"[dim]Model already present: {MODEL_PATH}[/]")

# ---------------------------------------------------------------------------
# Step 2 — Obtain a test image (try URL, fall back to synthetic)
# ---------------------------------------------------------------------------
if not IMAGE_PATH.exists():
    try:
        import urllib.request
        url = "https://farm1.staticflickr.com/1/1_5c1e1e1e1_m.jpg"
        urllib.request.urlretrieve(url, IMAGE_PATH)
        console.print(f"[green]Test image downloaded → {IMAGE_PATH}[/]")
    except Exception:
        # Synthetic fallback — random 224×224 RGB image
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(IMAGE_PATH)
        console.print(f"[yellow]Synthetic test image created → {IMAGE_PATH}[/]")
else:
    console.print(f"[dim]Test image already present: {IMAGE_PATH}[/]")

# ---------------------------------------------------------------------------
# Step 3 — Load model and dataset via KAAL engine
# ---------------------------------------------------------------------------
from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset

model   = load_model(str(MODEL_PATH))
dataset = load_dataset(str(IMAGE_PATH.parent), input_shape=model.input_shape, max_images=1)
console.print(f"[cyan]Loaded:[/] {model.framework} model · {model.num_classes} classes")

# ---------------------------------------------------------------------------
# Step 4 — Run FGSM attack
# ---------------------------------------------------------------------------
from kaal.attacks.fgsm import fgsm_attack

tensor, path, pil = next(iter(dataset))
console.print("[bold cyan]Running FGSM attack (ε=0.03)...[/]")
result = fgsm_attack(model, tensor, epsilon=0.03)

# ---------------------------------------------------------------------------
# Step 5 — Print results using Rich table
# ---------------------------------------------------------------------------
table = Table(title="FGSM Result", show_header=True, header_style="bold magenta")
table.add_column("Field",  style="cyan",  no_wrap=True)
table.add_column("Value",  style="white")

table.add_row("success",               str(result.success))
table.add_row("original_class",        str(result.original_class))
table.add_row("original_confidence",   f"{result.original_confidence:.4f}")
table.add_row("adversarial_class",     str(result.adversarial_class))
table.add_row("adversarial_confidence",f"{result.adversarial_confidence:.4f}")
table.add_row("confidence_delta",      f"{result.confidence_delta:.4f}")
table.add_row("epsilon_used",          str(result.epsilon_used))
table.add_row("perturbation_shape",    str(tuple(result.perturbation_tensor.shape)))
table.add_row("plain_english",         result.plain_english)

console.print(table)

# ---------------------------------------------------------------------------
# Step 6 — Save adversarial image
# ---------------------------------------------------------------------------
result.adversarial_pil.save(OUTPUT_PATH)
console.print(f"[green]Adversarial image saved → {OUTPUT_PATH}[/]")
