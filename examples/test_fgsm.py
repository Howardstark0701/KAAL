"""FGSM verification script — Phase 2, Kiro Prompt 2.2.

Downloads ResNet50 pretrained, fetches one COCO validation image,
runs fgsm_attack() with epsilon=0.03, prints results, saves output.

NOTE: This script is for Phase 2 verification only — not for production use.

Usage:
    python examples/test_fgsm.py
"""

import os
import sys
import tempfile
import urllib.request

import torch
import torchvision.models as models

# Make sure kaal package is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.fgsm import fgsm_attack, fgsm_attack_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH = "./fgsm_test_output.png"
MODEL_PATH  = "./fgsm_test_model.pt"
EPSILON     = 0.03

# One COCO validation image (small, publicly available)
TEST_IMAGE_URL = (
    "http://images.cocodataset.org/val2017/000000039769.jpg"
)

# ---------------------------------------------------------------------------
# Step 1 — Download / cache ResNet50
# ---------------------------------------------------------------------------

print("\n[1/4] Loading ResNet50 pretrained...")
if not os.path.exists(MODEL_PATH):
    model_pt = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model_pt.eval()
    torch.save(model_pt, MODEL_PATH)
    print(f"      Saved to {MODEL_PATH}")
else:
    print(f"      Using cached model at {MODEL_PATH}")

kaal_model = load_model(MODEL_PATH)
print(f"      Framework  : {kaal_model.framework}")
print(f"      Input shape: {kaal_model.input_shape}")
print(f"      Classes    : {kaal_model.num_classes}")

# ---------------------------------------------------------------------------
# Step 2 — Download one COCO validation image
# ---------------------------------------------------------------------------

print("\n[2/4] Downloading test image...")
tmp_dir = tempfile.mkdtemp(prefix="kaal_fgsm_test_")
img_path = os.path.join(tmp_dir, "test.jpg")

try:
    urllib.request.urlretrieve(TEST_IMAGE_URL, img_path)
    print(f"      Saved to {img_path}")
except Exception as e:
    print(f"      Download failed: {e}")
    print("      Falling back to random noise image...")
    import numpy as np
    from PIL import Image
    arr = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

# ---------------------------------------------------------------------------
# Step 3 — Run FGSM attack
# ---------------------------------------------------------------------------

print(f"\n[3/4] Running FGSM attack (ε={EPSILON})...")
dataset = load_dataset(tmp_dir, input_shape=kaal_model.input_shape)

tensor, path, pil_img = next(iter(dataset))
result = fgsm_attack(kaal_model, tensor, epsilon=EPSILON)

print(f"\n      {'─' * 50}")
print(f"      Original class      : {result.original_class}")
print(f"      Original confidence : {result.original_confidence:.4f}")
print(f"      Adversarial class   : {result.adversarial_class}")
print(f"      Adversarial conf.   : {result.adversarial_confidence:.4f}")
print(f"      Confidence delta    : {result.confidence_delta:+.4f}")
print(f"      Epsilon used        : {result.epsilon_used}")
print(f"      Attack success      : {result.success}")
print(f"      Plain English       : {result.plain_english}")
print(f"      {'─' * 50}")

# ---------------------------------------------------------------------------
# Step 4 — Save adversarial image
# ---------------------------------------------------------------------------

print(f"\n[4/4] Saving adversarial image...")
result.adversarial_pil.save(OUTPUT_PATH)
print(f"      Adversarial image saved to: {OUTPUT_PATH}")

# Also save a side-by-side comparison
from PIL import Image as PILImage
side_by_side = PILImage.new("RGB", (pil_img.width * 2 + 4, pil_img.height))
side_by_side.paste(pil_img, (0, 0))
side_by_side.paste(result.adversarial_pil.resize(pil_img.size), (pil_img.width + 4, 0))
comparison_path = OUTPUT_PATH.replace(".png", "_comparison.png")
side_by_side.save(comparison_path)
print(f"      Comparison image saved to : {comparison_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'═' * 52}")
print(f"  FGSM Phase 2 Verification {'PASSED' if result.success else 'COMPLETE'}")
if not result.success:
    print(f"  Note: attack did not cause misclassification at ε={EPSILON}")
    print(f"  Try a higher epsilon (e.g. 0.1) for a stronger effect.")
print(f"{'═' * 52}\n")
