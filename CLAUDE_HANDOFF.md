# KAAL — Handoff Notes for Claude

This file is a self-contained onboarding doc for a future Claude session doing **bug-fixing and improvements** on KAAL. Read it fully before touching code. It points to the two work-summary docs, maps the architecture, lists known open items, and records the environment quirks that will otherwise bite you.

---

## 1. What KAAL is

KAAL ("काल", Sanskrit for time/death/the inevitable) is a **universal AI adversarial-robustness auditing tool** — MIT-licensed, fully offline. It attacks trained models (image / text / tabular / audio) with FGSM, PGD, adversarial patches (standard + GradCAM-guided), a physical-robustness simulator, and modality-specific attacks; scores each model with a **KVS** (KAAL Vulnerability Score, 0–10 across six dimensions); and can emit a tamper-evident **certification bundle**. Supports PyTorch, TensorFlow, ONNX, TFLite, HuggingFace, and callable interfaces.

## 2. Repo & environment facts

| Item | Value |
|---|---|
| Repo root | `C:\Users\patha\KAAL` |
| Python | `C:\Users\patha\KAAL\.venv\Scripts\python.exe` (venv; Python 3.10+) |
| Branch | `main` (tracking `origin/main`) |
| Remote | `https://github.com/Howardstark0701/KAAL.git` |
| Version | `1.0.0` (`kaal/__init__.py`) |
| Install | `setup.py` reads `requirements.txt` into `install_requires` **at install time** |
| Key commits | `5888006` v1.0.0 · `530a86b` findings C1–C5/H1–H5/M1–M8/L1–L8 · `8a80ec0` dep cleanup + torch>=2.4 |

## 3. Environment quirks (read this twice)

- **Windows cp1252 console.** When running Python, prefix: `PYTHONIOENCODING=utf-8`. When opening files in code, always pass `encoding="utf-8"` (e.g. `open(path, "r", encoding="utf-8")`).
- **Editable-install staleness.** `pip install -e .` snapshots `requirements.txt` into the `kaal` package metadata at install time. After any requirements change, refresh with `pip install -e . --no-deps` or pip will keep warning about stale pins. Verify with `pip show kaal` and `pip check`.
- **Installed stack (verified working):** torch 2.13.0+cpu, torchvision 0.28.0, numpy 1.26.4, tensorflow 2.15.0, scikit-learn 1.1.3 (transitive via grad-cam, unpinned — do not re-add a pin), grad-cam 1.4.8, setuptools 84.0.0.
- **Test suite takes ~4.5 minutes** (331 tests, `weights_only` warnings are expected — see §7 H4). Use `run_in_background` + read the output file rather than blocking sleeps.

## 4. Running KAAL

CLI commands (`python -m kaal.cli`): **audit** · **serve** · **patch** · **compare** · **certify** · **leaderboard** · **config**. There is **no `kaal benchmark` command** — benchmark is the Python API `kaal.benchmark.runner.run_benchmark()`; `kaal leaderboard` only renders HTML from a benchmark JSON.

```bash
cd C:/Users/patha/KAAL
# CLI audit
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kaal.cli audit --model demo_model.pt --dataset demo_images --help
# Full test suite (slow)
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m pytest -q
# Web UI (FastAPI backend + Next.js frontend)
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kaal.cli serve
# PyInstaller bundle (see kaal_bundle/kaal.spec)
bash kaal_bundle/build.sh   # or kaal_bundle\build.bat on Windows
```

## 5. Architecture / code map

```
kaal/
  cli.py                      # Typer CLI — audit, serve, patch, compare, certify, leaderboard, config
  config.py                   # device / config management
  engine/
    loader.py                 # model loading (torch/tf/onnx/tflite/keras/hf/callable) + weights_only guards
    dataset.py                # dataset loading, resize via model input_shape, dynamic-shape errors
    utils.py                  # resolve_input_shape() shared helper, misc utilities
  attacks/
    fgsm.py  pgd.py           # image white-box attacks (PGD reports first-success tensor)
    patch.py  patch_smart.py  # adversarial patch (standard + GradCAM-guided)
    physical.py               # robustness across 26 transforms
    blackbox.py               # NES black-box (untargeted update = x − α·sign(ĝ))
    text_attack.py tabular_attack.py audio_attack.py   # modality-specific, reached_target flag
    gradcam.py                # attack-side gradcam
  defence/
    certification.py          # certify_model() + _run_audit()
    fingerprint.py            # model fingerprinting
    compliance_report.py
  benchmark/
    runner.py                 # run_benchmark() — Python API, not CLI
    leaderboard_page.py
  scoring/kvs.py              # calculate_kvs() — six-dimension KVS score.
                              #   Dim3a reads the measured epsilon_robustness_curve()
                              #   Dim5 reads black-box success_rate + query cost
                              #   (never the attack's internal query_efficiency)
  explainability/             # gradcam, saliency, confidence
  reporting/                  # pdf, json_report, html_report
  fingerprint/radar.py        # radar chart data
web/
  backend/main.py             # FastAPI app, lifespan
  backend/jobs/store.py       # job/file store (thread-locked, cleanup_expired)
  backend/routes/             # audit.py, patch.py, report.py
  backend/ws/progress.py      # websocket progress
  frontend/                   # Next.js + Tailwind; pages: index, audit, patch, compare, results
tests/                        # pytest — conftest builds session-scoped ResNet18 + image fixtures
kaal_bundle/kaal.spec         # PyInstaller spec (excludes TF/ONNX/web/tests; keep lean)
scripts/                      # run_leaderboard.py, make_demo_gif.py, take_screenshots.js
```

## 6. Work summaries (read before starting)

- **`FIXES.md`** (repo root) — condensed summary of the full hardening pass.
- **`C:\Users\patha\BUG_FINDINGS.md`** — the detailed findings log (C1–C5, H1–H5, M1–M8, L1–L8, items 4–6). **It lives OUTSIDE the git repo** (`C:\Users\patha\`, not `C:\Users\patha\KAAL\`) and is not tracked — it will not be present in a fresh clone; ask the user for it if you need the deep detail.

**Read `FIXES.md` to the end.** A later verification pass (findings F1–F9)
found that two of the fixes below had not actually worked — Dim3a was still a
constant, and black-box never reached the reports — and turned up seven more
defects. Treat the "26 findings closed" claim as superseded by that section.

Two standing traps that pass tests but are wrong:

- **A dimension that cannot vary is not a measurement.** Dim3a was reported
  fixed twice while still returning 10.0 in every reachable case, because its
  unit tests fed hand-written values that no real attack produces. When you
  touch a scoring dimension, run it against a fragile *and* a robust model and
  confirm the outputs differ.
- **A score must be substantiated by the report.** F2 shipped a KVS naming a
  dimension the report could not evidence. If `dimensions_tested` names it,
  `attacks.<name>` must be populated.

**Do not re-fix anything already fixed.** All 26 original findings are closed. The heavyweight ones: NES update direction (C1), black-box end-to-end dispatch + KVS Dim5 (C2), empirical Dim3a/3b (C3), fail-closed attack validation (C4), patch try/except in web (C5), `weights_only=True` torch.load with warn-and-fallback (H4), dynamic-shape `--input-size` handling (H5), job-store leak + lock (M2), `reached_target` on text/tabular/audio (M5), audio autograd double-softmax (M6), PGD first-success tensor (M7), `--input-size` on certify + `input_shape` on `run_benchmark()` (item 6).

## 7. Known open items & honest caveats

- **BUG_FINDINGS.md is not in the repo** — a fresh clone won't have it. Handoff gap; user is aware.
- **`MODEL_ROUTING.md`, `"MODEL_ROUTING - Copy.md"`, `kaal_cert_test/`, `leaderboard.html`, `demo_images/`, `scripts/take_screenshots.js` are untracked** — check with the user before committing; likely user-owned scratch/artifacts.
- **`scikit-learn==1.1.3` is transitive** via `grad-cam` (unpinned). Harmless, no version constraints. Don't "fix" it.
- **H4 verified caveat (version-independent):** on every torch version, `weights_only=True` refuses arbitrary custom classes (`torch.save(model)` full-module files) → the warn-and-fallback still fires. state_dict files load via the safe path. torch ≥2.4 only adds opt-in `torch.serialization.add_safe_globals()`. The `UserWarning` from `_load_pytorch` in tests is expected and safe.
- **No `kaal benchmark` CLI command exists** — only the `run_benchmark()` API + `kaal leaderboard` HTML renderer. Adding a real subcommand is a design decision (larger scope), not a bug.
- **Some tests download/build a ResNet18** in `conftest.py` (session-scoped, no pretrained weights) — needs network for `torchvision` model registration only; keep fixtures offline-safe if adding tests.

## 8. Working conventions (user's rules — do not violate)

- **Exact-scope edits.** Change only what the instruction describes. Do **not** refactor adjacent code, rename public symbols, or alter the **public API shape** (dataclass fields, function signatures, CLI flags) unless explicitly told.
- **Verify before claiming.** Functional tests over guesses. Run the suite; report failures with output, don't hand-wave. For Windows-Python, use `PYTHONIOENCODING=utf-8`.
- **Grep-gate for dependency removals.** Before removing a dependency, grep the whole codebase (excluding requirements.txt, *.spec, *.md) for its imports; if any real import exists, stop and report.
- **Commit messages** end with `Co-Authored-By: Claude <noreply@anthropic.com>`. Commit + push to `main` only when the user asks.
- **Report commit hashes** when the user asks to commit/push.

## 9. Suggested next areas (bugs/improvements, in rough priority)

1. **`kaal benchmark` CLI gap** — decide and implement (or explicitly reject) a real CLI command; README currently over-promises.
2. **Web frontend state** — untracked `web/frontend/.env.local.example` exists; verify upload/results flows against the hardened backend (M1–M4 touched those paths).
3. **FIXES.md / BUG_FINDINGS.md consistency** — BUG_FINDINGS.md is outside the repo; decide whether to bring a copy into the repo for future sessions.
4. **Test speed** — 331 tests / 4.5 min; the session-scoped ResNet18 rebuild per run could be cached or made opt-in.
5. **PyInstaller bundle** — verify `build.bat` still produces a working exe on the current torch stack (spec excludes changed).
6. **`requirements-web.txt`** — confirm it stays in sync with `web/backend` imports.
