"""Adversarial attacks on audio classification models.

kaal/attacks/audio_attack.py

Target model interface
----------------------
Any callable that accepts a 1D numpy array of shape (n_samples,) at
`sample_rate` Hz and returns class probabilities as a numpy array of
shape (n_classes,).

Examples of compatible models:
    - A fine-tuned Wav2Vec2 wrapped to accept raw waveforms
    - An MFCC-based sklearn pipeline wrapped in a callable
    - Any custom audio classifier with the described interface

Attack: Imperceptible Noise via Sign Gradient (audio FGSM)
-----------------------------------------------------------
Iterative version of FGSM applied directly to the waveform.
Each iteration:
    1. Convert current waveform to a differentiable torch tensor.
    2. Wrap model_callable in a thin torch.autograd.Function so
       we can compute a surrogate gradient w.r.t. the input.
       Because most audio models are not differentiable in PyTorch
       (e.g. sklearn, ONNX, custom numpy pipelines), we use a
       finite-difference gradient estimate when the model is not
       a torch.nn.Module — and a true backward pass when it is.
    3. Loss = -log P(target_class)
    4. Update: audio += epsilon * sign(gradient)
    5. Clamp to [-1.0, 1.0].

SNR computation
---------------
SNR_dB = 10 * log10(signal_power / noise_power)
    signal_power = mean(original^2)
    noise_power  = mean((perturbed - original)^2)
Higher SNR = more imperceptible perturbation.
A perturbation is typically inaudible above ~30 dB SNR.

Only dependencies: torch, numpy.
No librosa, no torchaudio, no scipy required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AudioAttackResult
# ---------------------------------------------------------------------------

@dataclass
class AudioAttackResult:
    """Result of an audio adversarial attack."""

    success_rate: float
    """Fraction of clips where the attack caused a class change (0–1)."""

    avg_confidence_on_target: float
    """Average P(target_class) across all clips after attack (0–1)."""

    avg_snr_db: float
    """Average SNR of the perturbation in dB. Higher = more imperceptible.
    Typical values: >30 dB = inaudible, 20–30 dB = barely noticeable."""

    n_samples: int
    """Number of audio clips processed."""

    plain_english: str
    """One factual sentence describing the outcome. No drama."""


# ---------------------------------------------------------------------------
# AudioAttacker
# ---------------------------------------------------------------------------

class AudioAttacker:
    """Adversarial attacker for audio classification models.

    Accepts any callable with signature:
        model_callable(audio: np.ndarray) -> np.ndarray
        audio shape:  (n_audio_samples,)   float32/float64, range [-1, 1]
        output shape: (n_classes,)          probabilities summing to ~1

    Usage:
        def my_model(audio: np.ndarray) -> np.ndarray:
            # ... feature extraction + classify ...
            return probabilities   # shape (n_classes,)

        attacker = AudioAttacker(my_model, sample_rate=16000, n_classes=10)
        result   = attacker.imperceptible_noise_attack(
            audio_arrays=[clip1, clip2],
            target_class=3,
            epsilon=0.002,
            n_iterations=100,
        )
        print(result.avg_snr_db, result.success_rate)
    """

    # Number of finite-difference probes per gradient estimate
    # (used when model is not a torch.nn.Module)
    _FD_N_PROBES: int = 8

    def __init__(
        self,
        model_callable: Callable[[np.ndarray], np.ndarray],
        sample_rate: int = 16000,
        n_classes: int = 10,
    ) -> None:
        """
        Args:
            model_callable: Callable mapping (n_audio_samples,) → (n_classes,).
            sample_rate:    Audio sample rate in Hz. Default 16000.
            n_classes:      Number of output classes. Default 10.

        Raises:
            TypeError: model_callable is not callable.
            ValueError: sample_rate or n_classes out of valid range.
        """
        if not callable(model_callable):
            raise TypeError(
                f"model_callable must be callable, got {type(model_callable).__name__}."
            )
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be >= 1, got {sample_rate}.")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}.")

        self._model      = model_callable
        self._sr         = sample_rate
        self._n_classes  = n_classes

        # Detect whether the wrapped model is a torch.nn.Module so we can
        # use true autograd instead of finite differences.
        self._is_torch_module = isinstance(model_callable, torch.nn.Module)

    # ------------------------------------------------------------------
    # imperceptible_noise_attack
    # ------------------------------------------------------------------

    def imperceptible_noise_attack(
        self,
        audio_arrays: list[np.ndarray],
        target_class: int,
        epsilon: float = 0.002,
        n_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> AudioAttackResult:
        """Add imperceptible L-inf bounded noise to audio clips.

        Implements iterative FGSM (PGD-style, no epsilon-ball projection,
        just clamp to valid audio range) directly on the waveform.

        For non-differentiable models (sklearn, custom numpy pipelines),
        the gradient is estimated via antithetic finite differences —
        the same technique used in NES black-box attacks.

        Args:
            audio_arrays:  List of 1D numpy arrays, each shape (n_audio_samples,).
                           Values should be in [-1.0, 1.0].
            target_class:  Class index to steer toward.
            epsilon:       Per-step L-inf noise magnitude per iteration.
                           Default 0.002 (~0.2% of full audio range per step).
            n_iterations:  Number of gradient ascent steps. Default 100.
            seed:          Optional random seed (affects FD probe directions).

        Returns:
            AudioAttackResult with success rate, avg confidence, avg SNR,
            and a plain-English summary.

        Raises:
            ValueError: target_class out of range, or empty audio_arrays.
        """
        if not (0 <= target_class < self._n_classes):
            raise ValueError(
                f"target_class={target_class} out of range for model "
                f"with {self._n_classes} classes."
            )
        if not audio_arrays:
            raise ValueError("audio_arrays must not be empty.")

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        successes     = 0
        total_conf    = 0.0
        total_snr_db  = 0.0
        n             = len(audio_arrays)

        for audio in audio_arrays:
            audio_np = np.asarray(audio, dtype=np.float64).flatten()

            # ── Original prediction ───────────────────────────────────────
            orig_proba = self._call_model(audio_np)
            orig_class = int(orig_proba.argmax())

            # ── Iterative perturbation ────────────────────────────────────
            x = audio_np.copy()

            for _ in range(n_iterations):
                grad = self._estimate_gradient(x, target_class, rng)
                # Ascend toward target class: minimize -log P(target) = maximize P(target)
                x = x + epsilon * np.sign(grad)
                # Clamp to valid audio range
                x = np.clip(x, -1.0, 1.0)

            # ── Final evaluation ──────────────────────────────────────────
            final_proba = self._call_model(x)
            final_class = int(final_proba.argmax())

            if final_class != orig_class:
                successes += 1
            total_conf   += float(final_proba[target_class])

            # ── SNR ───────────────────────────────────────────────────────
            total_snr_db += _compute_snr_db(audio_np, x)

        success_rate = successes / n
        avg_conf     = total_conf / n
        avg_snr_db   = total_snr_db / n

        return AudioAttackResult(
            success_rate=round(float(success_rate), 4),
            avg_confidence_on_target=round(float(avg_conf), 4),
            avg_snr_db=round(float(avg_snr_db), 2),
            n_samples=n,
            plain_english=_build_plain_english(
                success_rate, avg_conf, avg_snr_db,
                target_class, n, epsilon, n_iterations,
            ),
        )

    # ------------------------------------------------------------------
    # Gradient estimation
    # ------------------------------------------------------------------

    def _estimate_gradient(
        self,
        x: np.ndarray,
        target_class: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Estimate ∂log P(target_class)/∂x.

        For torch.nn.Module models: use true autograd.
        For all others: antithetic finite differences (NES estimator).

        Returns a numpy array of the same shape as x.
        """
        if self._is_torch_module:
            return self._autograd_gradient(x, target_class)
        return self._fd_gradient(x, target_class, rng)

    def _autograd_gradient(
        self,
        x: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """True gradient via PyTorch autograd (torch.nn.Module models only)."""
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_(True)

        # model is a torch.nn.Module — call it directly
        logits = self._model(x_t)                        # (1, n_classes) or (n_classes,)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=1)
        loss  = -torch.log(probs[0, target_class] + 1e-12)
        loss.backward()

        grad = x_t.grad.detach().squeeze(0).numpy().astype(np.float64)
        return grad

    def _fd_gradient(
        self,
        x: np.ndarray,
        target_class: int,
        rng: np.random.Generator,
        sigma: float = 1e-3,
    ) -> np.ndarray:
        """Antithetic finite-difference gradient estimate (NES, black-box).

        Uses self._FD_N_PROBES antithetic pairs:
            grad ≈ (1 / (2 * n * sigma)) * Σ [f(x+σu) - f(x-σu)] * u
        where f(x) = log P(target_class | x).

        Cost: 2 * _FD_N_PROBES model calls per iteration.
        """
        grad = np.zeros_like(x, dtype=np.float64)
        n    = self._FD_N_PROBES

        for _ in range(n):
            u        = rng.standard_normal(x.shape)

            x_plus   = np.clip(x + sigma * u, -1.0, 1.0)
            x_minus  = np.clip(x - sigma * u, -1.0, 1.0)

            p_plus   = self._call_model(x_plus)
            p_minus  = self._call_model(x_minus)

            f_plus   = float(np.log(p_plus[target_class]  + 1e-12))
            f_minus  = float(np.log(p_minus[target_class] + 1e-12))

            grad    += (f_plus - f_minus) * u

        grad /= 2.0 * n * sigma
        return grad

    # ------------------------------------------------------------------
    # Model call helper
    # ------------------------------------------------------------------

    def _call_model(self, audio: np.ndarray) -> np.ndarray:
        """Call model_callable and return a (n_classes,) probability array."""
        if self._is_torch_module:
            with torch.no_grad():
                x_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                out = self._model(x_t)
                if out.dim() > 1:
                    out = out.squeeze(0)
                proba = F.softmax(out, dim=0).numpy().astype(np.float64)
        else:
            raw   = self._model(audio.astype(np.float32))
            proba = np.asarray(raw, dtype=np.float64).flatten()

        # Normalise to valid probability distribution (guard against model quirks)
        proba = np.clip(proba, 0.0, 1.0)
        total = proba.sum()
        if total > 1e-8:
            proba /= total
        else:
            proba = np.ones(self._n_classes, dtype=np.float64) / self._n_classes

        return proba


# ---------------------------------------------------------------------------
# SNR helper
# ---------------------------------------------------------------------------

def _compute_snr_db(original: np.ndarray, perturbed: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB.

    SNR_dB = 10 * log10(signal_power / noise_power)

    Args:
        original:  Original audio waveform (1D float array).
        perturbed: Perturbed audio waveform (same shape).

    Returns:
        SNR in dB. Returns +inf if noise power is zero (no perturbation),
        and -inf if signal power is zero (silent input).
    """
    signal_power = float(np.mean(original ** 2))
    noise_power  = float(np.mean((perturbed - original) ** 2))

    if noise_power < 1e-20:
        return float("inf")     # no perturbation at all
    if signal_power < 1e-20:
        return float("-inf")    # silent input — SNR undefined

    return 10.0 * math.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# plain_english
# ---------------------------------------------------------------------------

def _build_plain_english(
    success_rate: float,
    avg_conf: float,
    avg_snr_db: float,
    target_class: int,
    n: int,
    epsilon: float,
    n_iterations: int,
) -> str:
    """One factual sentence. No drama, no exclamation marks."""
    snr_str = (
        f"{avg_snr_db:.1f} dB SNR" if math.isfinite(avg_snr_db)
        else "infinite SNR (no perturbation)"
    )
    imperceptible = "imperceptible" if avg_snr_db > 30 else "audible"
    return (
        f"Imperceptible audio noise attack (epsilon={epsilon}, {n_iterations} iterations) "
        f"on {n} audio clip{'s' if n != 1 else ''} achieved {success_rate:.0%} success rate "
        f"against class {target_class} with average confidence {avg_conf:.2f}; "
        f"perturbation averaged {snr_str} ({imperceptible} threshold is ~30 dB)."
    )
