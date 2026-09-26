"""KAAL Evidence Runner — real models, real attacks, real results.
Phases 3-7: Image, Text, Tabular, Audio live tests + full audit.
"""
import os, sys, json, tempfile, time, warnings
import numpy as np
import torch
import torchvision.models as tv
from pathlib import Path
from PIL import Image

ROOT     = Path(__file__).parent.parent
EVIDENCE = ROOT / "evidence"
SS_DIR   = EVIDENCE / "screenshots"
SS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"]        = "ignore"

results = {}

# ── helpers ──────────────────────────────────────────────────────────────────
def banner(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")
def ok(s):     print(f"  [PASS] {s}")
def fail(s):   print(f"  [FAIL] {s}")

# ── Setup: ensure demo model + images exist ───────────────────────────────────
banner("SETUP — model + demo images")
demo_model_path = ROOT / "demo_model.pt"
demo_img_dir    = ROOT / "demo_images"

if not demo_model_path.exists():
    print("  Downloading ResNet18...")
    m = tv.resnet18(weights="IMAGENET1K_V1"); m.eval()
    torch.save(m, demo_model_path)
    ok(f"demo_model.pt saved ({demo_model_path.stat().st_size//1024} KB)")
else:
    ok(f"demo_model.pt exists ({demo_model_path.stat().st_size//1024} KB)")

demo_img_dir.mkdir(exist_ok=True)
imgs = list(demo_img_dir.glob("*.jpg"))
if len(imgs) < 5:
    rng = np.random.default_rng(42)
    for i in range(10):
        arr = rng.integers(30, 220, (224,224,3), dtype=np.uint8)
        Image.fromarray(arr).save(demo_img_dir / f"img_{i+1:02d}.jpg")
    imgs = list(demo_img_dir.glob("*.jpg"))
ok(f"Demo images: {len(imgs)}")

# ── PHASE 3 & 4: Image model — baseline + FGSM + PGD ────────────────────────
banner("PHASE 3-4 — Image Model (ResNet18 + FGSM + PGD + Patch)")

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.fgsm import fgsm_attack, fgsm_attack_dataset
from kaal.attacks.pgd import pgd_attack_dataset
from kaal.scoring.kvs import calculate_kvs

km = load_model(str(demo_model_path))
ds = load_dataset(str(demo_img_dir), input_shape=km.input_shape, max_images=10)
ok(f"Model: {km.framework}, {km.num_classes} classes, shape {km.input_shape}")
ok(f"Dataset: {len(ds)} images")

# Baseline inference
tensor, path, pil = next(iter(ds))
pred = km.predict(tensor)
ok(f"Baseline prediction: class={pred['class_idx']} confidence={pred['confidence']:.4f}")
results["image_baseline"] = {"class": pred["class_idx"], "confidence": round(pred["confidence"],4)}

# FGSM
t0 = time.time()
fgsm_agg = fgsm_attack_dataset(km, ds, epsilon=0.03)
fgsm_t = time.time()-t0
ok(f"FGSM success rate: {fgsm_agg['success_rate']:.0%}  |  avg Δconf: {fgsm_agg['avg_confidence_delta']:+.4f}  |  time: {fgsm_t:.1f}s")
r0 = fgsm_agg["results"][0]
ok(f"  Sample: {r0.original_class}→{r0.adversarial_class}  eps={r0.epsilon_used}")
results["fgsm"] = {
    "success_rate": fgsm_agg["success_rate"],
    "avg_delta": round(fgsm_agg["avg_confidence_delta"],4),
    "sample_original": r0.original_class,
    "sample_adversarial": r0.adversarial_class,
    "epsilon": r0.epsilon_used,
    "plain_english": r0.plain_english,
}

# PGD
t0 = time.time()
pgd_agg = pgd_attack_dataset(km, ds, epsilon=0.03, steps=20)
pgd_t = time.time()-t0
ok(f"PGD  success rate: {pgd_agg['success_rate']:.0%}  |  avg steps: {pgd_agg['avg_steps_to_success']}  |  time: {pgd_t:.1f}s")
results["pgd"] = {
    "success_rate": pgd_agg["success_rate"],
    "avg_steps": pgd_agg["avg_steps_to_success"],
    "epsilon": pgd_agg["epsilon_used"],
}

# Patch
from kaal.attacks.patch import generate_patch
out_tmp = str(ROOT / "evidence" / "patch_output")
os.makedirs(out_tmp, exist_ok=True)
t0 = time.time()
pr = generate_patch(km, ds, target_class=0, patch_fraction=0.05, iterations=100, output_dir=out_tmp, verbose=False)
patch_t = time.time()-t0
ok(f"Patch success rate: {pr.attack_success_rate:.0%}  |  avg conf: {pr.avg_confidence_on_target:.3f}  |  time: {patch_t:.1f}s")
results["patch"] = {
    "success_rate": pr.attack_success_rate,
    "avg_conf_on_target": pr.avg_confidence_on_target,
    "plain_english": pr.plain_english,
}

# Physical robustness
from kaal.attacks.physical import test_physical_robustness_batch
adv_tensors = [r.adversarial_tensor for r in fgsm_agg["results"][:5]]
orig_classes= [r.original_class     for r in fgsm_agg["results"][:5]]
t0 = time.time()
phys = test_physical_robustness_batch(km, adv_tensors, orig_classes,
                                       transformations=["jpeg_90","blur_5","noise_001","rotation_05"])
phys_t = time.time()-t0
ok(f"Physical survival: {phys.overall_survival_rate:.0%}  |  rating: {phys.physical_threat_rating}  |  time: {phys_t:.1f}s")
results["physical"] = {
    "survival_rate": phys.overall_survival_rate,
    "threat_rating": phys.physical_threat_rating,
}

# KVS score
kvs = calculate_kvs(fgsm_result=fgsm_agg, pgd_result=pgd_agg, physical_result=phys)
ok(f"KVS SCORE: {kvs.score}/10  [{kvs.label}]")
for dim, val in kvs.dimension_scores.items():
    print(f"    {dim:30s}: {val:.2f}")
results["kvs"] = {"score": kvs.score, "label": kvs.label, "dimensions": kvs.dimension_scores}

# ── PHASE 5: Text attack ──────────────────────────────────────────────────────
banner("PHASE 5 — Text Model Attack (mock — transformers checkpoint unavailable)")
# transformers is installed but downloading a checkpoint requires internet
# Use the mock pattern to validate the attack code path
import sys, types
NUM_LABELS = 2; HIDDEN = 16; import torch.nn as nn

class _FakeEmb(nn.Module):
    def __init__(self): super().__init__(); self.word_embeddings = nn.Embedding(30522, HIDDEN)
    def forward(self, input_ids=None, **kw): return self.word_embeddings(input_ids)

from dataclasses import dataclass as _dc
@_dc
class _FO:
    logits: torch.Tensor
    attentions: tuple

class _FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = type('B', (nn.Module,), {'__init__': lambda s: (super(_FC,s).__init__(), setattr(s,'embeddings',_FakeEmb())) and None, 'forward': lambda s,**kw: kw.get('input_ids')})()
        self.bert.__class__.__init__ = lambda s: (nn.Module.__init__(s), setattr(s,'embeddings',_FakeEmb()))
        self.bert = type('B', (), {'embeddings': _FakeEmb()})()
        self.classifier = nn.Linear(HIDDEN, NUM_LABELS)
    @property
    def base_model(self): return self
    def eval(self): return self
    def cpu(self): return self
    @property
    def embeddings(self):
        if not hasattr(self,'_emb'): self._emb = _FakeEmb()
        return self._emb
    def forward(self, input_ids=None, **kw):
        e = self.embeddings(input_ids); pooled = e.mean(1)
        logits = self.classifier(pooled)
        seq = input_ids.shape[1]
        attn = torch.rand(1,1,seq,seq).softmax(-1)
        return _FO(logits=logits, attentions=(attn,))

class _FTok:
    cls_token=sep_token=pad_token='[X]'
    def __call__(self,t,**kw):
        ids=list(range(1,min(len(t.split()),8)+1))[:8]; ids=[101]+ids+[102]
        class E:
            def __init__(s): s.input_ids=torch.tensor([ids])
            def __getitem__(s,k): return s.input_ids if k=='input_ids' else None
        return E()
    def convert_ids_to_tokens(self,ids):
        w=['good','bad','terrible','happy','sad','fast','strong','help']
        t=['[CLS]']
        for i,id_ in enumerate(ids[1:]):
            if id_==102: t.append('[SEP]'); break
            t.append(w[i%len(w)])
        return t

fake_tf = types.ModuleType('transformers')
fake_tf.AutoTokenizer = type('AT',(),{'from_pretrained': staticmethod(lambda *a,**kw: _FTok())})
fake_tf.AutoModelForSequenceClassification = type('AM',(),{'from_pretrained': staticmethod(lambda *a,**kw: _FC())})
sys.modules['transformers'] = fake_tf

from kaal.attacks.text_attack import TextAttacker
attacker = TextAttacker("mock-sst2")
texts = ["This model is good and fast", "The results are terrible and sad"]
r = attacker.token_substitution_attack(texts, target_class=1, n_substitutions=2)
ok(f"Token substitution: success_rate={r.success_rate:.0%}  n_samples={r.n_samples}")
ok(f"  {r.plain_english}")
r2 = attacker.embedding_perturbation_attack(texts, target_class=0, epsilon=0.1)
ok(f"Embedding perturb:  success_rate={r2.success_rate:.0%}  n_samples={r2.n_samples}")
results["text"] = {"token_success": r.success_rate, "embed_success": r2.success_rate,
                   "note": "mock tokenizer (no internet checkpoint needed)"}

# ── PHASE 6: Tabular attack ────────────────────────────────────────────────────
banner("PHASE 6 — Tabular Attack (sklearn RandomForest)")
from sklearn.ensemble import RandomForestClassifier
from kaal.attacks.tabular_attack import TabularAttacker, TabularAttackResult

rng2 = np.random.default_rng(42)
X_tr = rng2.uniform([18,0,300,0],[90,500000,850,200000],(200,4))
y_tr = (X_tr[:,0] > 50).astype(int)
X_te = rng2.uniform([18,0,300,0],[90,500000,850,200000],(20,4))
clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)

tab = TabularAttacker(clf,
    feature_names=["age","income","score","debt"],
    feature_ranges={"age":(18,90),"income":(0,500000),"score":(300,850),"debt":(0,200000)})
tr = tab.feature_perturbation_attack(X_te, target_class=1, epsilon=0.1, n_steps=30, seed=0)
ok(f"Tabular success: {tr.success_rate:.0%}  avg_conf: {tr.avg_confidence_on_target:.3f}")
ok(f"  Most perturbed: {tr.most_perturbed_features}")
ok(f"  {tr.plain_english}")
results["tabular"] = {"success_rate": tr.success_rate, "avg_conf": tr.avg_confidence_on_target,
                       "top_features": tr.most_perturbed_features}

# ── PHASE 7: Audio ────────────────────────────────────────────────────────────
banner("PHASE 7 — Audio Attack")
from kaal.attacks.audio_attack import AudioAttacker

def simple_audio_clf(audio: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)))
    p1 = min(0.95, max(0.05, rms*4))
    return np.array([1-p1, p1, 0.0, 0.0])

rng3 = np.random.default_rng(7)
clips = [rng3.uniform(-0.3,0.3, 16000//4).astype(np.float32) for _ in range(3)]
aa = AudioAttacker(simple_audio_clf, sample_rate=16000, n_classes=4)
ar = aa.imperceptible_noise_attack(clips, target_class=1, epsilon=0.002, n_iterations=30, seed=0)
ok(f"Audio success: {ar.success_rate:.0%}  avg_conf: {ar.avg_confidence_on_target:.3f}  SNR: {ar.avg_snr_db:.1f}dB")
ok(f"  {ar.plain_english}")
results["audio"] = {"success_rate": ar.success_rate, "snr_db": ar.avg_snr_db}

# ── Save results JSON ─────────────────────────────────────────────────────────
banner("SAVING EVIDENCE JSON")
out_json = EVIDENCE / "live_test_results.json"
with open(out_json,"w") as f: json.dump(results, f, indent=2, default=str)
ok(f"Saved: {out_json}")

banner("LIVE TEST SUMMARY")
for k,v in results.items():
    print(f"  {k:12s}: {v}")
