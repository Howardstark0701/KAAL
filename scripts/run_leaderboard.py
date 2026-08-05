"""KAAL Leaderboard Runner — scripts/run_leaderboard.py

Downloads 4 popular torchvision models, audits each with KAAL FGSM + PGD
attacks, and generates a leaderboard HTML page.

Usage:
    python scripts/run_leaderboard.py                 # FGSM + PGD, 20 images
    python scripts/run_leaderboard.py --full          # + patch, 50 images
    python scripts/run_leaderboard.py --device cpu
    python scripts/run_leaderboard.py --device gpu
    python scripts/run_leaderboard.py --full --device cpu
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Parse flags — no argparse needed
# ---------------------------------------------------------------------------

FULL_MODE   = "--full"   in sys.argv
DEVICE_FLAG = None
for i, arg in enumerate(sys.argv):
    if arg == "--device" and i + 1 < len(sys.argv):
        DEVICE_FLAG = sys.argv[i + 1]

# ---------------------------------------------------------------------------
# Directory layout (relative to repo root, not scripts/)
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent.resolve()
IMG_DIR     = REPO_ROOT / "scripts" / "leaderboard_images"
MODEL_DIR   = REPO_ROOT / "scripts" / "leaderboard_models"
RESULTS_DIR = REPO_ROOT / "scripts" / "leaderboard_results"
HTML_OUT    = REPO_ROOT / "leaderboard.html"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

from kaal.config import resolve_device, get_defaults

device   = resolve_device(DEVICE_FLAG)
defaults = get_defaults(device)

# ---------------------------------------------------------------------------
# Attack config
# ---------------------------------------------------------------------------

ATTACKS    = ["fgsm", "pgd", "patch"] if FULL_MODE else ["fgsm", "pgd"]
MAX_IMAGES = 50 if FULL_MODE else 20

# ---------------------------------------------------------------------------
# Print header + time estimate
# ---------------------------------------------------------------------------

print()
print("=" * 56)
print("  KAAL Leaderboard Runner")
print("=" * 56)
print(f"  Device   : {device.upper()}")
print(f"  Attacks  : {', '.join(a.upper() for a in ATTACKS)}")
print(f"  Images   : {MAX_IMAGES} per model")
print(f"  Models   : 4 (ResNet-50, MobileNet-V2, EfficientNet-B0, DenseNet-121)")
print()

if device == "gpu":
    if FULL_MODE:
        print("  Estimated time: 4-6 minutes (GPU)")
    else:
        print("  Estimated time: 2-3 minutes (GPU)")
else:
    if FULL_MODE:
        print("  Estimated time: 25-40 minutes (CPU, full mode)")
    else:
        print("  Estimated time: 8-12 minutes (FGSM + PGD, 20 images x 4 models)")
print()

# ---------------------------------------------------------------------------
# Step 1 — Download 20 ImageNet sample images
# ---------------------------------------------------------------------------
# Source: github.com/EliSchwartz/imagenet-sample-images
# Files follow the pattern: n<synset_id>_<name>.JPEG

_RAW_BASE = (
    "https://raw.githubusercontent.com/"
    "EliSchwartz/imagenet-sample-images/master/"
)

# 20 known filenames from the EliSchwartz repo
_SAMPLE_IMAGES = [
    "n01440764_tench.JPEG",
    "n01443537_goldfish.JPEG",
    "n01484850_great_white_shark.JPEG",
    "n01491361_tiger_shark.JPEG",
    "n01494475_hammerhead.JPEG",
    "n01496331_electric_ray.JPEG",
    "n01498041_stingray.JPEG",
    "n01514668_cock.JPEG",
    "n01514859_hen.JPEG",
    "n01518878_ostrich.JPEG",
    "n01530575_brambling.JPEG",
    "n01531178_goldfinch.JPEG",
    "n01532829_house_finch.JPEG",
    "n01534433_junco.JPEG",
    "n01537544_indigo_bunting.JPEG",
    "n01558993_robin.JPEG",
    "n01560419_bulbul.JPEG",
    "n01580077_jay.JPEG",
    "n01582220_magpie.JPEG",
    "n01592084_chickadee.JPEG",
]

print("[1/4] Downloading sample images...")
downloaded = 0
for fname in _SAMPLE_IMAGES:
    dest = IMG_DIR / fname
    if dest.exists():
        downloaded += 1
        continue
    url = _RAW_BASE + fname
    try:
        urllib.request.urlretrieve(url, dest)
        downloaded += 1
        print(f"  ✓ {fname}")
    except Exception as e:
        print(f"  ! Failed to download {fname}: {e} — generating synthetic fallback")
        # Synthetic fallback: random 224x224 RGB image
        try:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            # Save with .jpg extension so KAAL dataset loader accepts it
            fallback_name = fname.replace(".JPEG", "_synthetic.jpg")
            img.save(IMG_DIR / fallback_name)
            downloaded += 1
        except Exception as fe:
            print(f"    Synthetic fallback also failed: {fe}")

print(f"  {downloaded} images ready in {IMG_DIR}")
print()

# Verify at least some images exist
image_files = list(IMG_DIR.glob("*.JPEG")) + list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
if not image_files:
    print("ERROR: No images available. Cannot run benchmark.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2 — Download / save models
# ---------------------------------------------------------------------------

import torch
import torchvision.models as tv

_MODELS = [
    ("resnet50",        "ResNet-50",        lambda: tv.resnet50(weights="IMAGENET1K_V1")),
    ("mobilenet_v2",    "MobileNet-V2",     lambda: tv.mobilenet_v2(weights="IMAGENET1K_V1")),
    ("efficientnet_b0", "EfficientNet-B0",  lambda: tv.efficientnet_b0(weights="IMAGENET1K_V1")),
    ("densenet121",     "DenseNet-121",     lambda: tv.densenet121(weights="IMAGENET1K_V1")),
]

print("[2/4] Preparing models...")
model_specs: list[tuple[str, str]] = []

for key, display_name, factory in _MODELS:
    path = MODEL_DIR / f"{key}.pt"
    if path.exists():
        print(f"  ✓ {display_name} (cached)")
    else:
        print(f"  Downloading {display_name}...", end=" ", flush=True)
        try:
            m = factory()
            m.eval()
            torch.save(m, path)
            print("saved.")
        except Exception as e:
            print(f"FAILED: {e}")
            continue
    model_specs.append((str(path), display_name))

print(f"  {len(model_specs)} models ready.")
print()

if not model_specs:
    print("ERROR: No models available. Cannot run benchmark.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3 — Run benchmark
# ---------------------------------------------------------------------------

from kaal.benchmark.runner import run_benchmark, BenchmarkEntry

print(f"[3/4] Running benchmark ({', '.join(a.upper() for a in ATTACKS)})...")
t0 = time.time()

entries: list[BenchmarkEntry] = []

for model_path, display_name in model_specs:
    print(f"\n  Auditing {display_name}...")
    try:
        partial = run_benchmark(
            model_specs=[(model_path, display_name)],
            dataset_dir=str(IMG_DIR),
            attacks=ATTACKS,
            output_dir=str(RESULTS_DIR),
            max_images=MAX_IMAGES,
        )
        entries.extend(partial)
    except Exception as e:
        warnings.warn(
            f"Audit failed for {display_name}: {e} — skipping this model.",
            RuntimeWarning,
            stacklevel=1,
        )
        print(f"  ! WARNING: {display_name} audit failed: {e}")
        print(f"    Continuing with remaining models...")

elapsed = time.time() - t0
m, s = divmod(int(elapsed), 60)
print()
print(f"  Benchmark complete. Duration: {m}m {s:02d}s")
print()

if not entries:
    print("ERROR: All model audits failed. No results to display.")
    sys.exit(1)

# Sort by kvs_score descending
entries.sort(key=lambda e: e.kvs_score, reverse=True)

# ---------------------------------------------------------------------------
# Step 4 — Generate leaderboard HTML
# ---------------------------------------------------------------------------

from kaal.benchmark.leaderboard_page import generate_leaderboard_html

print("[4/4] Generating leaderboard HTML...")
html_path = generate_leaderboard_html(entries, output_path=str(HTML_OUT))

print()
print("=" * 56)
print("  Leaderboard saved to leaderboard.html")
print(f"  {len(entries)} model(s) ranked")
print()
print("  Results:")
for rank, e in enumerate(entries, 1):
    fgsm_str  = f"FGSM {e.fgsm_success_rate:.0%}"  if e.fgsm_success_rate  is not None else "FGSM —"
    pgd_str   = f"PGD {e.pgd_success_rate:.0%}"    if e.pgd_success_rate   is not None else "PGD —"
    patch_str = f"Patch {e.patch_success_rate:.0%}" if e.patch_success_rate is not None else ""
    print(
        f"  {rank}. {e.model_name:<18}  KVS {e.kvs_score:.1f}  "
        f"[{e.kvs_label}]  {fgsm_str}  {pgd_str}  {patch_str}"
    )
print()
print(f"  Open: {html_path}")
print("=" * 56)
print()
