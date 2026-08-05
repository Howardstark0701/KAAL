"""Phase 7 PRD verification — KVS Scoring + Fingerprint Radar Chart."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaal.scoring.kvs import (
    calculate_kvs, get_kvs_label, get_kvs_color, REMEDIATION_MAP
)
from kaal.fingerprint.radar import generate_fingerprint

tmp = tempfile.mkdtemp()

# ── Test 1: full formula with all 5 dims ─────────────────────────────────────
print("── KVS Formula (all 5 dims) ──────────────────────────────────────")
kvs = calculate_kvs(
    fgsm_result={"success_rate": 0.87, "epsilon_used": 0.03},
    pgd_result={"success_rate": 0.94, "epsilon_used": 0.03},
    physical_result=type("R", (), {"overall_survival_rate": 0.51})(),
    blackbox_result=type("B", (), {"query_efficiency": 0.48})(),
    min_epsilon=0.03,
)
print("Score           :", kvs.score)
print("Label           :", kvs.label)
print("Color           :", kvs.color)
print("Dims tested     :", kvs.dimensions_tested)
print("Dims skipped    :", kvs.dimensions_skipped)
print("Dim scores      :", {k: round(v, 2) for k, v in kvs.dimension_scores.items()})
print("Remediation     :", len(kvs.remediation), "items")
print("Plain English   :", kvs.plain_english)
assert 0.0 <= kvs.score <= 10.0
assert kvs.label == get_kvs_label(kvs.score)
assert kvs.color == get_kvs_color(kvs.score)
assert "!" not in kvs.plain_english
assert kvs.plain_english.endswith(".")
assert len(kvs.dimensions_tested) + len(kvs.dimensions_skipped) == 5

# ── Test 2: label boundaries ─────────────────────────────────────────────────
print("\n── Label boundaries ──────────────────────────────────────────────")
for score, expected in [(0, "Robust"), (2, "Robust"), (3, "Low Risk"),
                         (5, "Medium Risk"), (7, "High Risk"),
                         (9, "Critical"), (10, "Catastrophic")]:
    actual = get_kvs_label(score)
    mark = "OK" if actual == expected else "FAIL"
    print(f"  {score:4.1f} → {actual:<15} [{mark}]")
    assert actual == expected

# ── Test 3: color codes ───────────────────────────────────────────────────────
print("\n── Color codes ───────────────────────────────────────────────────")
for score, expected in [(1.0, "#4ADE80"), (3.0, "#A3E635"), (5.0, "#FACC15"),
                         (7.0, "#FB923C"), (9.0, "#CC0000")]:
    actual = get_kvs_color(score)
    mark = "OK" if actual == expected else "FAIL"
    print(f"  {score:.1f} → {actual}  [{mark}]")
    assert actual == expected

# ── Test 4: fingerprint chart ─────────────────────────────────────────────────
print("\n── Fingerprint radar chart ───────────────────────────────────────")
fp_path = os.path.join(tmp, "fingerprint.png")
path = generate_fingerprint(kvs, "ResNet50", fp_path)
size_kb = os.path.getsize(path) / 1024
magic = open(path, "rb").read(4)
print("Output path     :", path)
print("File size       : %.1f KB" % size_kb)
print("PNG magic bytes :", magic == b"\x89PNG")
assert magic == b"\x89PNG"
assert size_kb > 20

# ── Test 5: comparison chart ──────────────────────────────────────────────────
print("\n── Comparison mode ───────────────────────────────────────────────")
kvs2 = calculate_kvs(
    fgsm_result={"success_rate": 0.30, "epsilon_used": 0.10},
    pgd_result={"success_rate": 0.40, "epsilon_used": 0.10},
    physical_result=type("R", (), {"overall_survival_rate": 0.20})(),
    min_epsilon=0.10,
)
print("Model B score   :", kvs2.score, kvs2.label)
cmp_path = os.path.join(tmp, "compare.png")
path2 = generate_fingerprint(kvs, "ResNet50", cmp_path,
                              comparison_kvs=kvs2, comparison_name="MobileNetV2")
print("Comparison chart:", path2)
assert os.path.exists(path2)

# ── Test 6: remediation map ───────────────────────────────────────────────────
print("\n── Remediation map ───────────────────────────────────────────────")
print("Keys:", list(REMEDIATION_MAP.keys()))
assert len(REMEDIATION_MAP) == 5

print("\nOutputs saved to:", tmp)
print("\nPhase 7 PRD verification: PASSED")
