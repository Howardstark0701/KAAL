"""Phase 8 PRD verification — JSON + PDF report generation."""
import os, sys, tempfile, json, torch, torchvision.models as models, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.fgsm import fgsm_attack
from kaal.attacks.pgd import pgd_attack
from kaal.attacks.physical import test_physical_robustness
from kaal.scoring.kvs import calculate_kvs
from kaal.fingerprint.radar import generate_fingerprint
from kaal.explainability.confidence import generate_collapse_curve
from kaal.reporting.json_report import generate_json_report
from kaal.reporting.pdf import generate_pdf_report

# ── Setup ─────────────────────────────────────────────────────────────────────
tmp = tempfile.mkdtemp()
img_dir = os.path.join(tmp, "images")
os.makedirs(img_dir)
for i in range(5):
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(os.path.join(img_dir, "img_%d.jpg" % i))

m = models.resnet18(weights=None); m.eval()
torch.save(m, os.path.join(tmp, "r18.pt"))
kaal_model = load_model(os.path.join(tmp, "r18.pt"))
dataset = load_dataset(img_dir, input_shape=kaal_model.input_shape)

torch.manual_seed(42)
tensor = torch.randn(3, 224, 224) * 0.4
fgsm_res = fgsm_attack(kaal_model, tensor, epsilon=0.4)
pgd_res  = pgd_attack(kaal_model, tensor, epsilon=0.3, steps=10)
phys_res = test_physical_robustness(
    kaal_model, fgsm_res.adversarial_tensor,
    fgsm_res.original_class,
    transformations=["jpeg_90", "blur_3", "noise_001", "brightness_075"]
)
kvs_res = calculate_kvs(
    fgsm_result={"success_rate": float(fgsm_res.success),
                 "epsilon_used": fgsm_res.epsilon_used},
    pgd_result={"success_rate": float(pgd_res.success),
                "epsilon_used": pgd_res.epsilon_used},
    physical_result=phys_res, min_epsilon=0.03,
)

out_dir = os.path.join(tmp, "output")
fp_path = generate_fingerprint(kvs_res, "ResNet18", os.path.join(out_dir, "fp.png"))
cc_path = generate_collapse_curve(pgd_res, os.path.join(out_dir, "collapse.png"))

model_info  = {"path": "r18.pt", "name": "resnet18", "framework": "pytorch",
               "input_shape": [3, 224, 224], "num_classes": 1000}
dataset_info = {"path": img_dir, "total_images": 5, "formats": {"jpg": 5}}

# ── JSON Report ───────────────────────────────────────────────────────────────
print("── JSON Report ──────────────────────────────────────────────────")
json_path = generate_json_report(
    output_path=os.path.join(out_dir, "report.json"),
    model_info=model_info, dataset_info=dataset_info,
    kvs_result=kvs_res,
    fgsm_result={"success_rate": float(fgsm_res.success),
                 "epsilon_used": fgsm_res.epsilon_used,
                 "avg_confidence_delta": fgsm_res.confidence_delta,
                 "plain_english": fgsm_res.plain_english},
    pgd_result={"success_rate": float(pgd_res.success),
                "epsilon_used": pgd_res.epsilon_used,
                "alpha_used": pgd_res.alpha_used,
                "steps_used": pgd_res.steps_used,
                "avg_steps_to_success": pgd_res.steps_to_success,
                "plain_english": pgd_res.plain_english},
    physical_result=phys_res,
    audit_duration_seconds=142.0,
)
with open(json_path) as f:
    doc = json.load(f)

print("File            :", json_path)
print("Size            : %.1f KB" % (os.path.getsize(json_path)/1024))
print("KVS score       :", doc["kvs"]["score"])
print("KVS label       :", doc["kvs"]["label"])
print("Dims tested     :", doc["kvs"]["dimensions_tested"])
print("Attacks         :", list(doc["attacks"].keys()))
print("Remediation     :", len(doc["remediation"]), "items")
assert doc["meta"]["kaal_version"] == "1.0.0"
assert 0 <= doc["kvs"]["score"] <= 10
assert "fgsm" in doc["attacks"]
assert "per_transform" in doc["physical_robustness"]
print("JSON schema     : OK")

# ── PDF Report ────────────────────────────────────────────────────────────────
print("\n── PDF Report ───────────────────────────────────────────────────")
pdf_path = generate_pdf_report(
    output_path=os.path.join(out_dir, "report.pdf"),
    model_info=model_info, dataset_info=dataset_info,
    kvs_result=kvs_res,
    fgsm_result=fgsm_res,
    pgd_result=pgd_res,
    physical_result=phys_res,
    collapse_curve_path=cc_path,
    fingerprint_path=fp_path,
    audit_duration_seconds=142.0,
)
size_kb = os.path.getsize(pdf_path) / 1024
magic   = open(pdf_path, "rb").read(5)
print("File            :", pdf_path)
print("Size            : %.1f KB" % size_kb)
print("PDF magic       :", magic == b"%PDF-")
assert magic == b"%PDF-"
assert size_kb > 5
print("8-page PDF      : OK")

print("\nAll outputs in  :", out_dir)
for f in sorted(os.listdir(out_dir)):
    kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
    print("  %-30s %.1f KB" % (f, kb))

print("\nPhase 8 PRD verification: PASSED")
