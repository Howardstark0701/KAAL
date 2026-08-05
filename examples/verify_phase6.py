"""Phase 6 PRD verification — Explainability Layer."""
import os, sys, tempfile, torch, torchvision.models as models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.engine.loader import load_model
from kaal.attacks.fgsm import fgsm_attack
from kaal.attacks.pgd import pgd_attack
from kaal.explainability.gradcam import generate_gradcam, generate_gradcam_comparison
from kaal.explainability.saliency import generate_saliency
from kaal.explainability.confidence import generate_collapse_curve

tmp = tempfile.mkdtemp()
m = models.resnet18(weights=None); m.eval()
torch.save(m, os.path.join(tmp, "r18.pt"))
model = load_model(os.path.join(tmp, "r18.pt"))

torch.manual_seed(5)
tensor = torch.randn(3, 224, 224) * 0.4
fgsm   = fgsm_attack(model, tensor, epsilon=0.4)
pgd    = pgd_attack(model, tensor, epsilon=0.3, steps=15)

print("-- GradCAM --")
gcam = generate_gradcam(model, tensor)
print("heatmap shape      :", gcam.heatmap_array.shape)
print("heatmap range      : [%.3f, %.3f]" % (gcam.heatmap_array.min(), gcam.heatmap_array.max()))
print("overlay size       :", gcam.overlay_pil.size, gcam.overlay_pil.mode)
print("target_class_used  :", gcam.target_class_used)
print("top_attention_region:", gcam.top_attention_region)

print("\n-- GradCAM Comparison --")
cmp = generate_gradcam_comparison(model, tensor, fgsm.adversarial_tensor)
print("side_by_side size  :", cmp.side_by_side_pil.size)
print("attention_shift    : %.4f" % cmp.attention_shift_score)
print("plain_english      :", cmp.plain_english)
assert "!" not in cmp.plain_english and cmp.plain_english.endswith(".")

print("\n-- Saliency --")
sal = generate_saliency(model, tensor)
print("saliency shape     :", sal.saliency_array.shape)
print("saliency range     : [%.3f, %.3f]" % (sal.saliency_array.min(), sal.saliency_array.max()))
print("grayscale PIL      :", sal.saliency_pil.size, sal.saliency_pil.mode)
print("overlay PIL        :", sal.overlay_pil.size, sal.overlay_pil.mode)
print("top_sensitive_pct  : %.4f" % sal.top_sensitive_pixels_pct)
assert sal.saliency_pil.mode == "L"
assert sal.overlay_pil.mode == "RGB"

print("\n-- Confidence Collapse Curve --")
curve_path = os.path.join(tmp, "collapse.png")
path = generate_collapse_curve(pgd, curve_path)
size_kb = os.path.getsize(path) / 1024
print("output path        :", path)
print("file size          : %.1f KB" % size_kb)
magic = open(path, "rb").read(4)
print("PNG magic bytes    :", magic == b"\x89PNG")
assert size_kb > 50
print("steps in chart     :", len(pgd.confidence_per_step))
print("steps_to_success   :", pgd.steps_to_success)

cmp.side_by_side_pil.save(os.path.join(tmp, "gradcam_comparison.png"))
sal.overlay_pil.save(os.path.join(tmp, "saliency_overlay.png"))
print("\nOutputs saved to   :", tmp)
print("\nPhase 6 PRD verification: PASSED")
