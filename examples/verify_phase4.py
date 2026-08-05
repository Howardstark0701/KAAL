"""Phase 4 PRD verification — Adversarial Patch Generator."""
import os, sys, tempfile, torch, torchvision.models as models
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.patch import generate_patch, apply_patch, patch_to_printable

# Setup
tmp = tempfile.mkdtemp()
img_dir = os.path.join(tmp, "images")
os.makedirs(img_dir)
for i in range(6):
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(os.path.join(img_dir, f"img_{i}.jpg"))

m = models.resnet18(weights=None)
m.eval()
model_path = os.path.join(tmp, "r18.pt")
torch.save(m, model_path)
kaal_model = load_model(model_path)

out_dir = os.path.join(tmp, "patch_output")
dataset = load_dataset(img_dir, input_shape=kaal_model.input_shape)

print("Training adversarial patch (50 iterations)...")
result = generate_patch(
    kaal_model, dataset,
    target_class=123,
    patch_fraction=0.05,
    iterations=50,
    output_dir=out_dir,
    verbose=True,
)

print()
print("PatchResult fields:")
print(f"  patch_tensor shape       : {result.patch_tensor.shape}")
print(f"  patch_pil size           : {result.patch_pil.size} mode={result.patch_pil.mode}")
print(f"  attack_success_rate      : {result.attack_success_rate:.1%}")
print(f"  avg_confidence_on_target : {result.avg_confidence_on_target:.4f}")
print(f"  target_class             : {result.target_class}")
print(f"  patch_fraction_used      : {result.patch_fraction_used}")
print(f"  iterations_used          : {result.iterations_used}")
print(f"  plain_english            : {result.plain_english}")
print(f"  patch_printable_pdf_path : {result.patch_printable_pdf_path}")

# Verify PNG was saved
png_path = os.path.join(out_dir, "patch.png")
assert os.path.exists(png_path), "patch.png not found"
pil = Image.open(png_path)
print(f"\npatch.png: {pil.size} {pil.mode} — OK")

# Verify PDF
assert os.path.exists(result.patch_printable_pdf_path), "PDF not found"
with open(result.patch_printable_pdf_path, "rb") as f:
    header = f.read(5)
assert header == b"%PDF-", f"Bad PDF header: {header}"
size_kb = os.path.getsize(result.patch_printable_pdf_path) / 1024
print(f"patch_print.pdf: {size_kb:.1f} KB — valid PDF header — OK")

# Verify apply_patch geometry
import torch
img = torch.randn(3, 224, 224)
patched = apply_patch(img, result.patch_tensor, x=10, y=10)
assert patched.shape == img.shape
print(f"apply_patch geometry: {patched.shape} — OK")

# Check plain_english rules
assert "!" not in result.plain_english
assert result.plain_english.strip().endswith(".")
print(f"plain_english rules   : no '!', ends with '.' — OK")

print()
print("Phase 4 PRD verification: PASSED")
