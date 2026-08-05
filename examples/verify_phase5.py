"""Phase 5 PRD verification — Physical Robustness Simulator."""
import os, sys, tempfile, torch, torchvision.models as models
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.engine.loader import load_model
from kaal.attacks.fgsm import fgsm_attack
from kaal.attacks.physical import (
    test_physical_robustness,
    ALL_TRANSFORM_NAMES,
    list_transforms,
)

# Setup
tmp = tempfile.mkdtemp()
m = models.resnet18(weights=None)
m.eval()
torch.save(m, os.path.join(tmp, "r18.pt"))
kaal_model = load_model(os.path.join(tmp, "r18.pt"))

# Generate a strong adversarial example
torch.manual_seed(42)
tensor = torch.randn(3, 224, 224) * 0.5
fgsm_res = fgsm_attack(kaal_model, tensor, epsilon=0.5)
print(f"FGSM attack success: {fgsm_res.success} "
      f"(class {fgsm_res.original_class} → {fgsm_res.adversarial_class})")

# Run full physical robustness test
print(f"\nRunning {len(ALL_TRANSFORM_NAMES)} transforms...")
result = test_physical_robustness(
    kaal_model,
    fgsm_res.adversarial_tensor,
    fgsm_res.original_class,
)

print(f"\n{'─'*55}")
print(f"  Overall survival rate : {result.overall_survival_rate:.1%}")
print(f"  Physical threat rating: {result.physical_threat_rating}")
print(f"  Most robust transform : {result.most_robust_transform}")
print(f"  Least robust transform: {result.least_robust_transform}")
print(f"  Plain English         : {result.plain_english}")
print(f"{'─'*55}")

print(f"\nCategory breakdown:")
for cat, rate in sorted(result.category_summary.items()):
    bar = "█" * int(rate * 20)
    print(f"  {cat:<22} {rate:.0%}  {bar}")

print(f"\nAll {len(result.transforms_tested)} transforms tested:")
for name, tr in sorted(result.per_transform_results.items()):
    status = "✓" if tr.success_rate > 0 else "✗"
    print(f"  {status} {name:<22} {tr.success_rate:.0%}")

# Verify all 7 categories present
cats = list_transforms()
assert set(cats.keys()) == {
    "jpeg_compression", "gaussian_noise", "brightness",
    "contrast", "rotation", "scaling", "gaussian_blur",
}
print(f"\nAll 7 transform categories present — OK")

# Verify count
assert len(ALL_TRANSFORM_NAMES) == 26, f"Expected 26 transforms, got {len(ALL_TRANSFORM_NAMES)}"
print(f"Total transform variants: {len(ALL_TRANSFORM_NAMES)} — OK")

# Verify rating is valid
assert result.physical_threat_rating in {"Lab Only", "Limited", "Field Ready"}
print(f"Threat rating '{result.physical_threat_rating}' — valid — OK")

# Verify plain_english rules
assert "!" not in result.plain_english
assert result.plain_english.strip().endswith(".")
print(f"plain_english rules — no '!', ends with '.' — OK")

print("\nPhase 5 PRD verification: PASSED")
