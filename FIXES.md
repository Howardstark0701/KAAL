# KAAL — Fixes Summary

A summary of the bug-fix and hardening pass on the KAAL adversarial-robustness auditing tool.
Full findings detail lives in `C:\Users\patha\BUG_FINDINGS.md` (outside the git repo).

**Session scope:** attack-engine correctness, CLI/web/certification/benchmark hardening, dependency cleanup, and a torch major-version upgrade. **All 26 findings resolved** (Critical5/5 · High5/5 · Medium8/8 · Low8/8).

---

## Critical (C1–C5)

| # | Fix |
|---|---|
| C1 | Untargeted NES black-box update direction corrected to `x − α·sign(ĝ)` — the attack can now actually succeed |
| C2 | Black-box NES attack dispatched end-to-end (CLI, web pipeline, certification, benchmark); `blackbox_result` fed to `calculate_kvs` so KVS Dim5 computes |
| C3 | Constant Dim3 "perturbation threshold" replaced with **Dim3a empirical robustness** + **Dim3b adversarial overconfidence**; `min_epsilon` removed; weights always renormalised. **Superseded — see F1 below: the Dim3a introduced here was still a constant.** |
| C4 | Attack names validated fail-closed (`_validate_attacks`) in certification / benchmark / web — unknown or empty attacks error instead of certifying "Robust" |
| C5 | Web pipeline wraps `generate_patch()` in try/except — one failing attack degrades to a warning instead of failing the whole job |

## High (H1–H5)

| # | Fix |
|---|---|
| H1 | `generate_patch()` raises `NotImplementedError` on non-PyTorch frameworks |
| H2 | Patch training now seeds Python `random` as well as `torch` — reproducible runs |
| H3 | Device-safe gradient target (`image_tensor.device`), single-image batch assert, `.cpu()` before `.numpy()` |
| H4 | Both `torch.load` calls use `weights_only=True` with a warn-and-fallback — untrusted model files no longer silently execute arbitrary code |
| H5 | `_extract_hw` None-guards every dimension and raises a clear `--input-size` error for dynamic-shape ONNX/TFLite models |

## Medium (M1–M8)

| # | Fix |
|---|---|
| M1 | Web upload filenames sanitized via `Path(name).name` — no path traversal |
| M2 | Job/file store no longer leaks: `cleanup_expired()` purges the `_files` registry + on-disk temp dirs; `update()`/`cleanup_expired()` guarded by a `threading.Lock` |
| M3 | Malformed/absent `Content-Length` → HTTP400 / HTTP411 (no more 500, no more 500 MB bypass) |
| M4 | Internal paths/tracebacks no longer exposed to API clients (generic error + server-side logging) |
| M5 | `reached_target: bool` added to text / tabular / audio attack result dataclasses (True iff `adv_class == target_class`) |
| M6 | Audio autograd no longer softmaxes twice; torch path differentiates `−log P(target)` against the model output |
| M7 | PGD reports the **first-success** tensor, so `steps_to_success` and `adversarial_confidence` describe the same example |
| M8 | `is_supported_model()` accepts `.keras` |

## Low (L1–L8)

Dead imports (L1), dead code (L2, L3), deprecated `@app.on_event("startup")` → lifespan (L4), unseeded RNGs (L5), cosmetic progress bar documented (L6), misleading `serve` ImportError handling (L7), fast-fail GPU availability check (L8).

---

## Follow-up items (4–6)

| Item | Fix |
|---|---|
| 4 | `torch==2.2.0` → `torch>=2.4`, `torchvision==0.17.0` → `torchvision>=0.19.0` — no conflicting torch upper bounds in the graph |
| 5 | `patch_result=` passed at all four `calculate_kvs()` call sites so KVS Dim3b consumes patch confidence data |
| 6 | `--input-size` added to `kaal certify` (same spec/parse/override as `kaal audit`) threaded through `certify_model`/`_run_audit`; `run_benchmark(input_shape=...)` override; shared `resolve_input_shape()` helper |

## Dependency changes

- **Removed** from `requirements.txt`: `adversarial-robustness-toolbox`, `foolbox`, `scikit-learn` (none imported by KAAL code — grep-verified; sklearn appears only in a docstring example and was already `excludes`-listed in the PyInstaller spec).
- **Removed** dead `sklearn` / `scikit_learn` excludes from `kaal_bundle/kaal.spec`.
- **Upgraded** to **torch 2.13.0+cpu / torchvision 0.28.0** — resolves the ART↔sklearn pin conflict; verified clean install and metadata (`pip check` → "No broken requirements found").

## Verification

- **331/331 tests pass** on torch 2.13.0+cpu (faster than on 2.2.0).
- KAAL core + web modules import cleanly on the new torch.
- H4 note corrected: the `weights_only=True` fallback for full `nn.Module` files is **version-independent** (torch never allows custom classes by default; ≥2.4 only adds opt-in `add_safe_globals`), verified on 2.13.0.

## Commits (on `main`, pushed to `origin/main`)

- `530a86b` — fix: close out audit findings C1–C5, H1–H5, M1–M8, L1–L8 + follow-ups
- `8a80ec0` — chore: remove unused deps (ART, foolbox, sklearn), bump torch>=2.4

## Open notes

- There is **no `kaal benchmark` CLI command** — benchmark is the `run_benchmark()` Python API; the README's `kaal leaderboard` only renders HTML from JSON. Its input-shape override is an API parameter, not a CLI flag.
- `scikit-learn`1.1.3 remains installed transitively via `grad-cam` (unpinned) — harmless, no version constraints on it now.

---

# Second pass — findings F1–F9

A verification pass over the fixes above found that two of them had not
actually worked, and turned up seven more defects. All nine are now closed.

## Scoring correctness

| # | Severity | Fix |
|---|---|---|
| F1 | High | **Dim3a was still a constant 10.0.** The C3 replacement scored `min(mean_L∞/ε, 1.0) × 10`, but FGSM builds its perturbation as `ε·sign(∇)`, so `amax` over ~150k pixels returns exactly ε by construction and the ratio is always 1.0. The zero-success branch also returned 10.0, so *both* branches were 10.0. Replaced with a real measurement: `epsilon_robustness_curve()` sweeps FGSM across an n/255 ladder and finds the smallest budget that misclassifies ≥50% of the sample. Verified: undefended ResNet → 10.0, wide-margin model → 0.0. Also closes open item R9 (epsilon robustness curve). |
| F9 | High | **Dim5 reported maximum vulnerability for models that resisted black-box attacks.** It read the attack's internal `query_efficiency` — the fraction of NES steps that reduced the loss — which sits near 1.0 on a smooth descent that never flips a single label. Observed: 0/10 attacks succeeded, Dim5 scored 10.0. Now `success_rate × (0.5 + 0.5 × query_cheapness) × 10`, which is 0.0 when nothing was broken. Demo-model KVS corrected 9.1 → 7.7. |

## Reporting integrity

| # | Severity | Fix |
|---|---|---|
| F2 | High | **Black-box was scored into the KVS but absent from the JSON and PDF reports** — `report.json` listed `blackbox_efficiency` in `dimensions_tested` while `attacks.blackbox` was `null`. Two layers: `blackbox_result=` was never passed to either report generator, and `_build_attacks_section()` read single-result fields (`.success`, `.queries_used`) that the dataset aggregate does not carry. The report now handles both shapes, the PDF gained a `blackbox_result` parameter and a summary-table row, and the `SimpleNamespace` wrappers at all four dispatch sites were removed in favour of passing the aggregate dict directly — that re-wrapping was the root cause of the field loss. The measured epsilon curve is emitted as a new `epsilon_robustness` section. |

## Web API

| # | Severity | Fix |
|---|---|---|
| F3 | Medium | `POST /api/audit/start` returned **200 for unknown attack names**, because `_validate_attacks()` ran inside the background task rather than the handler. The doomed job consumed one of the two per-IP rate-limit slots. Validation moved into the handler → 400 before any job is created. |
| F4 | Medium | Model-upload failures returned the loader's raw message, which **embedded the absolute save path** (server username, temp-directory layout). The generic handler was sanitised by M4; this explicit `HTTPException` was not. Now logs server-side and returns only the client's own filename plus the exception class. |

## Frontend

| # | Severity | Fix |
|---|---|---|
| F5 | Medium | The radar had **no `PolarRadiusAxis` domain**, so recharts auto-scaled the outer ring to a "nice" number above the data max (measured: a 10.0 axis reached 0.8333 of the ring = 10/12). Fingerprints rescaled per model and could not be compared by shape or area — the entire point of a fingerprint. Pinned to `[0, 10]`; verified `fractionOfOuter == score/10` on every axis. |
| F6 | Low | **Untested dimensions were plotted at radius 0**, which on a vulnerability scale reads as "perfectly robust", contradicting the "not tested" label directly below the chart. Skipped dimensions now render as a break in the polygon. |
| F7 | Low | The compare page asked for `xxxxxxxx-xxxx-…` **UUID-shaped job IDs**; since SB2 they are 43-char URL-safe tokens that the backend 400s if given a UUID. Placeholder corrected, and the in-memory expiry noted. |
| F8 | Low | The landing page claimed **sklearn and XGBoost support** the loader does not have (it accepts `.pt/.pth/.h5/.keras/.onnx/.tflite`), and its "Attack Modules" list omitted Black-Box, which is first-class and offered on the audit page. Copy corrected, Black-Box module added. |

## Verification

- **341 tests pass** (was 331) — the 10 new ones are regression guards, including an explicit assertion that Dim3a is *not* constant and that Dim5 is 0.0 when no black-box attack succeeded.
- Suite runs in ~250s, faster than the 351s baseline.
- `tsc --noEmit` and `next build` both clean.
- End-to-end verified through the CLI, the REST API, and the browser UI.

## Note on scores changing

F1 and F9 both corrected dimensions that were pinned high, so **KVS scores from
before this pass are not comparable to scores after it**. The demo model moved
from 9.1 (Critical) to 7.7 (High Risk). Any stored audit predating this change
should be re-run before being compared or certified against.
