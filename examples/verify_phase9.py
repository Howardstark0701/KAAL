"""Phase 9 PRD verification — full CLI end-to-end."""
import os, sys, tempfile, subprocess, json, torch
import torchvision.models as models
import numpy as np
from PIL import Image

tmp = tempfile.mkdtemp()
img_dir = os.path.join(tmp, "images")
os.makedirs(img_dir)
for i in range(5):
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(os.path.join(img_dir, "img_%d.jpg" % i))

m = models.resnet18(weights=None); m.eval()
model_path = os.path.join(tmp, "r18.pt")
torch.save(m, model_path)

out_dir = os.path.join(tmp, "kaal_output")

print("Running: kaal audit --model r18.pt --dataset ./images/ --attacks fgsm,pgd")
print("─" * 60)

result = subprocess.run(
    [
        sys.executable, "-m", "kaal.cli", "audit",
        "--model",   model_path,
        "--dataset", img_dir,
        "--attacks", "fgsm,pgd",
        "--epsilon", "0.3",
        "--steps",   "5",
        "--output",  out_dir,
        "--report",  "pdf,json",
        "--no-gradcam",
    ],
    capture_output=False,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

print("─" * 60)
print("Exit code:", result.returncode)
assert result.returncode == 0, "kaal audit exited with non-zero code"

# Verify outputs
outputs = os.listdir(out_dir)
print("Output files:", sorted(outputs))
assert "report.json" in outputs, "report.json missing"
assert "report.pdf"  in outputs, "report.pdf missing"

with open(os.path.join(out_dir, "report.json")) as f:
    doc = json.load(f)
assert 0 <= doc["kvs"]["score"] <= 10
assert "fgsm" in doc["attacks"]
assert "pgd"  in doc["attacks"]

print()
print("report.json  :", "%.1f KB" % (os.path.getsize(os.path.join(out_dir, "report.json"))/1024))
print("report.pdf   :", "%.1f KB" % (os.path.getsize(os.path.join(out_dir, "report.pdf"))/1024))
print("KVS score    :", doc["kvs"]["score"], "-", doc["kvs"]["label"])

# Compare command test
import shutil
shutil.copy(os.path.join(out_dir, "report.json"), os.path.join(tmp, "before.json"))
shutil.copy(os.path.join(out_dir, "report.json"), os.path.join(tmp, "after.json"))

print()
print("Running: kaal compare --before before.json --after after.json")
result2 = subprocess.run(
    [
        sys.executable, "-m", "kaal.cli", "compare",
        "--before", os.path.join(tmp, "before.json"),
        "--after",  os.path.join(tmp, "after.json"),
        "--output", os.path.join(tmp, "compare_out"),
    ],
    capture_output=False,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
assert result2.returncode == 0
assert os.path.exists(os.path.join(tmp, "compare_out", "comparison.json"))
print("comparison.json: OK")

print()
print("Phase 9 PRD verification: PASSED")
