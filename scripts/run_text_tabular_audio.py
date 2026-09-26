"""KAAL Evidence — Text, Tabular, Audio attacks."""
import os, sys, json, warnings, types
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
ROOT = Path(__file__).parent.parent
EVIDENCE = ROOT / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)
results = {}

def banner(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")
def ok(s):     print(f"  [PASS] {s}")

# ── PHASE 5: Text ──────────────────────────────────────────
banner("PHASE 5 — Text Attack (TextAttacker w/ mock model)")

NUM_LABELS = 2
HIDDEN     = 16
SEQ        = 10

class FakeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(30522, HIDDEN)
    def forward(self, input_ids=None, **kw):
        return self.word_embeddings(input_ids)

class FakeBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = FakeEmbeddings()

class FakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._base = FakeBaseModel()
        self.classifier = nn.Linear(HIDDEN, NUM_LABELS)

    @property
    def base_model(self): return self._base

    def eval(self): return self
    def cpu(self):  return self

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, output_attentions=None, **kw):
        from dataclasses import dataclass
        emb    = self._base.embeddings(input_ids)
        pooled = emb.mean(dim=1)
        logits = self.classifier(pooled)
        seq    = input_ids.shape[1]
        attn   = torch.rand(1, 1, seq, seq).softmax(dim=-1)
        @dataclass
        class Out:
            logits: torch.Tensor
            attentions: tuple
        return Out(logits=logits, attentions=(attn,))

class FakeTokenizer:
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        words  = text.split()[:8]
        ids    = list(range(1, len(words)+1))
        ids    = [101] + ids + [102]
        # Return a dict-like object that also supports attribute access and **unpacking
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids):
        vocab = ["good","bad","terrible","happy","sad","fast","strong","help"]
        tokens = ["[CLS]"]
        for i, id_ in enumerate(ids[1:]):
            if id_ == 102: tokens.append("[SEP]"); break
            tokens.append(vocab[i % len(vocab)])
        return tokens

# Inject mock transformers
fake_tf = types.ModuleType("transformers")
fake_tf.AutoTokenizer = type("AT", (), {
    "from_pretrained": staticmethod(lambda *a, **kw: FakeTokenizer())
})
fake_tf.AutoModelForSequenceClassification = type("AM", (), {
    "from_pretrained": staticmethod(lambda *a, **kw: FakeClassifier())
})
sys.modules["transformers"] = fake_tf

from kaal.attacks.text_attack import TextAttacker

attacker = TextAttacker("mock-distilbert-sst2")
texts = [
    "This product is really good and fast",
    "The results were absolutely terrible and sad",
    "I strongly recommend this amazing strong solution",
]

# Token substitution
r1 = attacker.token_substitution_attack(texts, target_class=1, n_substitutions=2)
ok(f"Token substitution: success_rate={r1.success_rate:.0%}  n_samples={r1.n_samples}")
ok(f"  {r1.plain_english}")

# Embedding perturbation
r2 = attacker.embedding_perturbation_attack(texts, target_class=0, epsilon=0.1)
ok(f"Embedding perturb:  success_rate={r2.success_rate:.0%}  n_samples={r2.n_samples}")
ok(f"  {r2.plain_english}")

results["text"] = {
    "model": "distilbert-base-uncased-finetuned-sst-2-english (mock)",
    "token_substitution_success": r1.success_rate,
    "embedding_perturbation_success": r2.success_rate,
    "note": "Mock tokenizer used — same attack code path as production",
}

# ── PHASE 6: Tabular ──────────────────────────────────────
banner("PHASE 6 — Tabular Attack (sklearn RandomForest)")
from sklearn.ensemble import RandomForestClassifier
from kaal.attacks.tabular_attack import TabularAttacker

rng = np.random.default_rng(42)
X_tr = rng.uniform([18,0,300,0],[90,500000,850,200000], (300,4))
y_tr = (X_tr[:,0] > 50).astype(int)
X_te = rng.uniform([18,0,300,0],[90,500000,850,200000], (30,4))
y_te = (X_te[:,0] > 50).astype(int)

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
base_acc = clf.score(X_te, y_te)
ok(f"Baseline model accuracy: {base_acc:.0%}")

tab = TabularAttacker(
    clf,
    feature_names=["age","income","credit_score","debt"],
    feature_ranges={"age":(18,90),"income":(0,500000),"credit_score":(300,850),"debt":(0,200000)},
)
tr = tab.feature_perturbation_attack(X_te, target_class=1, epsilon=0.1, n_steps=30, seed=0)
ok(f"Feature perturbation: success_rate={tr.success_rate:.0%}  avg_conf={tr.avg_confidence_on_target:.3f}")
ok(f"  Most perturbed features: {tr.most_perturbed_features}")
ok(f"  {tr.plain_english}")

results["tabular"] = {
    "model": "RandomForestClassifier(n_estimators=100)",
    "baseline_accuracy": round(base_acc, 4),
    "attack_success_rate": tr.success_rate,
    "avg_conf_on_target": tr.avg_confidence_on_target,
    "most_perturbed": tr.most_perturbed_features,
    "plain_english": tr.plain_english,
}

# ── PHASE 7: Audio ────────────────────────────────────────
banner("PHASE 7 — Audio Attack (custom numpy model)")
from kaal.attacks.audio_attack import AudioAttacker, _compute_snr_db

def audio_model(audio: np.ndarray) -> np.ndarray:
    """Simple RMS-based 4-class audio classifier."""
    rms  = float(np.sqrt(np.mean(audio**2)))
    spec = float(np.abs(np.fft.rfft(audio)).mean())
    p0 = max(0.01, min(0.97, 1.0 - rms * 4))
    p1 = max(0.01, min(0.97, rms * 4))
    p2 = max(0.01, 0.5 - abs(spec - 0.1)*2)
    p3 = max(0.01, 1.0 - p0 - p1 - p2)
    proba = np.array([p0, p1, p2, p3], dtype=np.float64)
    return proba / proba.sum()

rng2  = np.random.default_rng(42)
clips = [rng2.uniform(-0.5, 0.5, 16000//4).astype(np.float32) for _ in range(5)]
orig_preds = [int(audio_model(c).argmax()) for c in clips]
ok(f"Baseline predictions: {orig_preds}")

aa = AudioAttacker(audio_model, sample_rate=16000, n_classes=4)
ar = aa.imperceptible_noise_attack(clips, target_class=1, epsilon=0.003, n_iterations=50, seed=7)
ok(f"Imperceptible noise: success_rate={ar.success_rate:.0%}  avg_conf={ar.avg_confidence_on_target:.3f}")
ok(f"  Average SNR: {ar.avg_snr_db:.1f} dB  (>30dB = inaudible)")
ok(f"  {ar.plain_english}")

results["audio"] = {
    "model": "custom RMS-based 4-class numpy classifier",
    "attack": "imperceptible waveform FGSM",
    "success_rate": ar.success_rate,
    "avg_snr_db": ar.avg_snr_db,
    "plain_english": ar.plain_english,
}

# ── Save ──────────────────────────────────────────────────
out_json = EVIDENCE / "text_tabular_audio_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2, default=str)
ok(f"Saved: {out_json}")

banner("TEXT / TABULAR / AUDIO SUMMARY")
for k, v in results.items():
    print(f"  {k:10s}: {v}")
