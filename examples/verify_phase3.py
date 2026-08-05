"""Phase 3 PRD verification — PGD vs FGSM comparison."""
import torch, torchvision.models as models, tempfile, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.engine.loader import load_model
from kaal.attacks.fgsm import fgsm_attack
from kaal.attacks.pgd import pgd_attack

# Save ResNet50 (no pretrained for speed)
tmp = tempfile.mkdtemp()
m = models.resnet50(weights=None)
m.eval()
path = os.path.join(tmp, 'r50.pt')
torch.save(m, path)
kaal_model = load_model(path)

torch.manual_seed(42)
eps = 0.1
fgsm_wins, pgd_wins = 0, 0

print("Testing PGD vs FGSM on 5 images at eps=0.1, steps=10")
print(f"{'Image':<8} {'FGSM':>8} {'PGD':>8}")
print("-" * 26)
for i in range(5):
    t = torch.randn(3, 224, 224) * 0.5
    f = fgsm_attack(kaal_model, t, epsilon=eps)
    p = pgd_attack(kaal_model, t, epsilon=eps, steps=10)
    fgsm_wins += int(f.success)
    pgd_wins  += int(p.success)
    print(f"{i+1:<8} {str(f.success):>8} {str(p.success):>8}")

print("-" * 26)
print(f"{'Total':<8} {fgsm_wins:>8} {pgd_wins:>8}")
print()

assert pgd_wins >= fgsm_wins, f"PGD ({pgd_wins}) should be >= FGSM ({fgsm_wins})"

# Show a full PGDResult fields
t2 = torch.randn(3, 224, 224) * 0.5
r = pgd_attack(kaal_model, t2, epsilon=0.2, steps=10)
print("Sample PGDResult fields:")
print("  success           :", r.success)
print("  original_class    :", r.original_class)
print("  adversarial_class :", r.adversarial_class)
print("  steps_to_success  :", r.steps_to_success)
print("  len(conf_per_step):", len(r.confidence_per_step))
print("  conf_per_step     :", [round(v, 3) for v in r.confidence_per_step])
print("  epsilon_used      :", r.epsilon_used)
print("  alpha_used        :", r.alpha_used)
print("  plain_english     :", r.plain_english)
print()
print("Phase 3 PRD verification: PASSED")
