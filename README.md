<h1 align="center">KAAL</h1>

<p align="center"><em>What cannot be seen, cannot be defended.</em></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-active-brightgreen">
</p>

<p align="center">
  <img alt="KAAL Demo" src="assets/demo.gif" width="640">
</p>

---

## What is KAAL

KAAL is a universal AI adversarial robustness auditing framework that supports image classification, object detection, text classification, tabular models, and audio classification across PyTorch, TensorFlow, HuggingFace, sklearn, XGBoost, and any callable model interface. It attacks trained models using a growing library of adversarial methods — FGSM, PGD, GradCAM-guided adversarial patches, physical robustness simulation, and modality-specific text, tabular, and audio attacks — then scores each model with a KVS (KAAL Vulnerability Score) and optionally generates a tamper-evident certification bundle. Everything runs entirely offline.

---

## Supported Model Types

| Modality | Frameworks | Attack Methods |
|----------|------------|----------------|
| Image | PyTorch, TensorFlow, ONNX, TFLite | FGSM, PGD, Adversarial Patch (standard + GradCAM-guided), Physical Robustness |
| Text | HuggingFace (BERT, DistilBERT, RoBERTa) | Token Substitution, Embedding Perturbation |
| Tabular | sklearn, XGBoost | Feature Perturbation |
| Audio | Any callable (numpy / PyTorch) | Imperceptible Waveform FGSM |

---

## Attack Modules

| Module | Method | Description |
|--------|--------|-------------|
| FGSM | Fast Gradient Sign Method | Single-step gradient attack producing imperceptible perturbations |
| PGD | Projected Gradient Descent | Iterative FGSM — gold standard adversarial attack |
| Patch | Adversarial Patch | Printable sticker that causes misclassification from any position |
| Patch (Smart) | GradCAM-Guided Patch | Analyzes which layers the model relies on most, generates patches targeting those layers — 2-3x more effective than standard patch |
| Physical | Robustness Simulator | Tests attack survival across 26 real-world transforms |
| Text | Token Substitution + Embedding Perturbation | Attacks BERT-family models via attention-weighted word replacement and embedding-space noise |
| Tabular | Feature Perturbation | Greedy boundary attack on sklearn and XGBoost classifiers |
| Audio | Imperceptible Waveform FGSM | Adds inaudible noise to audio clips using true autograd or finite-difference fallback |

---

## KVS Score

KAAL Vulnerability Score — 0.0 to 10.0 across six vulnerability dimensions.

| Score | Label | Meaning |
|-------|-------|---------|
| 0.0 – 2.0 | 🟢 Robust | Model resists all tested attacks at standard epsilon |
| 2.1 – 4.0 | 🟡 Low Risk | Minor susceptibility, limited practical exploitability |
| 4.1 – 6.0 | 🟠 Medium Risk | Meaningful vulnerability, exploitable under realistic conditions |
| 6.1 – 8.0 | 🔴 High Risk | Reliably attacked with low perturbation, remediation recommended |
| 8.1 – 9.5 | 🔴 Critical | Highly vulnerable across multiple attack vectors |
| 9.6 – 10.0 | 🔴 Catastrophic | Collapses under minimal perturbation across all dimensions |

---

## Installation

```bash
pip install kaal
kaal --help
```

---

## Quick Start

### Image Model

```python
from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.fgsm import fgsm_attack

# Load model and dataset
model   = load_model("your_model.pt")
dataset = load_dataset("./images/")

# Run FGSM attack on first image
for tensor, path, pil in dataset:
    result = fgsm_attack(model, tensor, epsilon=0.03)
    print(f"Success: {result.success}")
    print(f"KVS plain English: {result.plain_english}")
    break
```

### Text Model

```python
from kaal.attacks.text_attack import TextAttacker

attacker = TextAttacker("distilbert-base-uncased-finetuned-sst-2-english")
result = attacker.token_substitution_attack(["This product is great"], target_class=0)
print(result.plain_english)
```

### Tabular Model

```python
from kaal.attacks.tabular_attack import TabularAttacker
import numpy as np

attacker = TabularAttacker(
    model,
    feature_names=["age", "income", "score"],
    feature_ranges={"age": (0, 100), "income": (0, 200000), "score": (300, 850)},
)
result = attacker.feature_perturbation_attack(X, target_class=1)
print(result.plain_english)
```

---

## CLI Reference

```bash
# Full adversarial audit — generates PDF + JSON report
kaal audit --model resnet50.pt --dataset ./images/ --attacks fgsm,pgd,patch,physical

# Patch generator only — produces printable adversarial patch
kaal patch --model resnet50.pt --dataset ./images/ --target 472

# GradCAM-guided smart patch
kaal patch --model resnet50.pt --dataset ./images/ --target 472 --smart

# Smart patch, fast mode (CPU-friendly)
kaal patch --model resnet50.pt --dataset ./images/ --target 472 --smart --fast

# Compare two audit reports side by side
kaal compare --before audit_v1.json --after audit_v2.json

# Launch web UI (FastAPI backend + Next.js frontend)
kaal serve

# Benchmark multiple models — generates leaderboard.html
kaal leaderboard --json benchmark_results/leaderboard.json --output leaderboard.html

# Certify a model (KAAL-D)
kaal certify --model resnet50.pt --dataset ./images/ --org "Acme Corp"

# Certify + generate MoD/DRDO compliance PDF
kaal certify --model resnet50.pt --dataset ./images/ --org "Acme Corp" --compliance

# Device config
kaal config
kaal config --reset
```

---

## KAAL-D Certification

KAAL-D produces a tamper-evident certification bundle for any audited model:

- SHA-256 file hash + weight hash — detects model tampering or substitution
- Re-audit verification loop — flags non-deterministic models
- Certification badge SVG — embeddable in documentation
- Compliance report PDF — MoD/DRDO-style, classification-marked, suitable for institutional submission
- Air-gapped executable — `kaal_bundle/dist/kaal.exe` runs on any Windows machine without Python installed

```bash
kaal certify --model resnet50.pt --dataset ./images/ --org "Acme Corp" --compliance
```

Output:

```
kaal_cert/badge.svg
kaal_cert/certificate.json
kaal_cert/compliance_report.pdf
```

---

## Web UI

```bash
# Backend
pip install -r requirements-web.txt
uvicorn web.backend.main:app --host 127.0.0.1 --port 8080

# Frontend
cd web/frontend && npm install && npm run dev
```

![Audit Dashboard](assets/screenshots/web-dashboard.png)
![Results View](assets/screenshots/results-view.png)
![Fingerprint Chart](assets/screenshots/fingerprint-chart.png)

---

## Tech Stack

- Python 3.10, PyTorch 2.2, TensorFlow 2.15, IBM ART 1.17, Foolbox 3.3
- grad-cam 1.4.8, captum 0.7.0
- ReportLab 4.1, matplotlib 3.8
- Typer 0.27, Rich 15.0
- FastAPI 0.110, uvicorn 0.27
- Next.js 14, React 18, Tailwind CSS 3.4, Recharts 2.12, framer-motion 11
- transformers 4.x (text attacks)
- scikit-learn / XGBoost (tabular attacks, optional)
- PyInstaller 6.x (air-gapped bundle)

---

## Roadmap

- [x] Core engine — model loader, dataset, attack modules (FGSM, PGD, Patch, Physical, Black-Box)
- [x] Reporting — KVS scoring, radar fingerprint, PDF + JSON reports, CLI, web UI
- [x] Smart patch — GradCAM-guided adversarial patch optimization
- [x] Multi-modal — text, tabular, and audio model attack support
- [x] Benchmark leaderboard — KVS scores across popular open-source models
- [x] KAAL-D foundation — SHA-256 model fingerprinting, re-audit loop, certification badge
- [x] KAAL-D air-gapped bundle — PyInstaller .exe, MoD/DRDO compliance reports, offline deployment
- [ ] KAAL-D vendor mode — institutional submission portal, bulk model certification pipeline

---

## License

MIT — see [LICENSE](LICENSE)
