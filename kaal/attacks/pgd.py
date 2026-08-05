"""PGD Attack — Projected Gradient Descent.

Spec 10.2 — Phase 3, Kiro Prompt 3.1.

Math:
    x₀ = x + uniform_noise(−ε, +ε)        ← random start within epsilon ball

    For step t = 1 to T:
      xₜ = xₜ₋₁ + α × sign(∇ₓ J(θ, xₜ₋₁, y))
      xₜ = clip(xₜ, x−ε, x+ε)            ← project back to epsilon ball
      xₜ = clip(xₜ, valid_range)          ← keep valid normalized pixel range

    Where:
        α = step size  (default: ε / 10)
        T = number of steps

PGD is strictly stronger than FGSM:
    - Multiple iterative steps instead of one
    - Projects back to the epsilon ball after every step
    - Random restart capability for escaping local optima
    - Records confidence at every step for collapse curve (Spec 11.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from kaal.engine.loader import KaalModel
from kaal.engine.utils import tensor_to_pil


# ---------------------------------------------------------------------------
# Normalized pixel value bounds
# ---------------------------------------------------------------------------
# ImageNet normalization maps [0, 1] pixels to roughly [-2.1, 2.6].
# We clamp to slightly wider [-3, 3] to be safe across all channels.
_PIXEL_MIN = -3.0
_PIXEL_MAX =  3.0


# ---------------------------------------------------------------------------
# Result dataclass — Spec 10.2
# ---------------------------------------------------------------------------

@dataclass
class PGDResult:
    """Complete result of a single PGD attack run."""

    success: bool
    """True if the attack changed the model's top-1 prediction."""

    original_class: int
    """Model's predicted class index before the attack."""

    original_confidence: float
    """Model's confidence on original_class before the attack (0.0–1.0)."""

    adversarial_class: int
    """Model's predicted class index after the attack."""

    adversarial_confidence: float
    """Model's confidence on adversarial_class after the attack (0.0–1.0)."""

    confidence_delta: float
    """Drop in confidence on original class: original_confidence − post_confidence_on_original."""

    steps_to_success: int
    """Step index (1-based) at which the first misclassification occurred.
    -1 if the attack never caused misclassification."""

    confidence_per_step: list[float]
    """Model confidence on the original class at every PGD step.
    Index 0 = after step 1, index N-1 = after step N.
    Used to generate the confidence collapse curve (Spec 11.3)."""

    epsilon_used: float
    """Epsilon value used."""

    alpha_used: float
    """Step size (alpha) used."""

    steps_used: int
    """Total number of PGD steps executed."""

    restarts_used: int
    """Number of random restarts used."""

    adversarial_tensor: torch.Tensor
    """The best adversarial image tensor found (same shape as input, normalized)."""

    adversarial_pil: Image.Image
    """The best adversarial image as a PIL Image (denormalized, for saving/display)."""

    plain_english: str
    """One factual sentence describing what happened. No drama, no exclamation marks."""


# ---------------------------------------------------------------------------
# pgd_attack() — main entry point
# ---------------------------------------------------------------------------

def pgd_attack(
    model: KaalModel,
    image_tensor: torch.Tensor,
    epsilon: float = 0.03,
    alpha: Optional[float] = None,
    steps: int = 40,
    restarts: int = 1,
    targeted: bool = False,
    target_class: Optional[int] = None,
) -> PGDResult:
    """Run a PGD attack on one image.

    Args:
        model:        KaalModel from kaal.engine.loader.
        image_tensor: Normalized torch.Tensor, shape (C, H, W) or (1, C, H, W).
        epsilon:      Maximum perturbation magnitude (L∞ norm). Default 0.03.
        alpha:        Step size per iteration. Default: epsilon / 10.
        steps:        Number of PGD iterations. Default 40.
        restarts:     Number of random restarts. Best result across all restarts
                      is returned. Default 1 (no restart).
        targeted:     If False (default), untargeted — maximize loss on true class.
                      If True, targeted — minimize loss on target_class.
        target_class: Required when targeted=True.

    Returns:
        PGDResult dataclass. confidence_per_step records model confidence on
        the original class at every step — used for the collapse curve chart.

    Raises:
        ValueError: targeted=True without target_class, or invalid parameters.
        NotImplementedError: Model framework does not support gradients.

    Example:
        result = pgd_attack(model, tensor, epsilon=0.03, steps=40)
        print(f"Success: {result.success}, steps: {result.steps_to_success}")
        print(result.plain_english)
    """
    # --- Input validation ----------------------------------------------------
    if targeted and target_class is None:
        raise ValueError(
            "targeted=True requires target_class to be specified.\n"
            "→ Pass the integer class index you want the model to predict."
        )

    epsilon = float(epsilon)
    if not (0.0 < epsilon <= 1.0):
        raise ValueError(
            f"epsilon must be in range (0.0, 1.0], got {epsilon}.\n"
            "→ Typical values: 0.01 (subtle), 0.03 (standard), 0.1 (strong)."
        )

    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")

    restarts = int(restarts)
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}.")

    # Default alpha = epsilon / 10  (standard PGD convention)
    if alpha is None:
        alpha = epsilon / 10.0
    alpha = float(alpha)

    # Squeeze batch dim
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    # --- Get original prediction --------------------------------------------
    original_pred = model.predict(image_tensor)
    original_class = original_pred["class_idx"]
    original_confidence = original_pred["confidence"]
    attack_class = target_class if targeted else original_class

    # --- Run restarts, keep best result ------------------------------------
    best_result: Optional[_PGDRunResult] = None

    for restart_idx in range(restarts):
        run = _pgd_single_run(
            model=model,
            image_tensor=image_tensor,
            original_class=original_class,
            attack_class=attack_class,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            targeted=targeted,
            seed_offset=restart_idx,
        )

        # Keep this run if:
        # 1. It's the first run, OR
        # 2. This run succeeded and previous didn't, OR
        # 3. Both succeeded but this one succeeded earlier, OR
        # 4. Both failed but this one reduced confidence more
        if best_result is None:
            best_result = run
        elif run.success and not best_result.success:
            best_result = run
        elif run.success and best_result.success:
            if run.steps_to_success < best_result.steps_to_success:
                best_result = run
        elif not run.success and not best_result.success:
            # Both failed — prefer the one with lowest final confidence on original
            if run.confidence_per_step[-1] < best_result.confidence_per_step[-1]:
                best_result = run

    # Shouldn't be None at this point (restarts >= 1), but guard anyway
    assert best_result is not None

    # --- Build final result -------------------------------------------------
    adv_pred = model.predict(best_result.adversarial_tensor)
    adversarial_class = adv_pred["class_idx"]
    adversarial_confidence = adv_pred["confidence"]
    post_conf_on_original = adv_pred["all_confidences"][original_class]
    confidence_delta = original_confidence - post_conf_on_original

    success = best_result.success

    plain_english = _build_plain_english(
        success=success,
        targeted=targeted,
        original_class=original_class,
        adversarial_class=adversarial_class,
        original_confidence=original_confidence,
        adversarial_confidence=adversarial_confidence,
        steps_to_success=best_result.steps_to_success,
        steps_used=steps,
        epsilon=epsilon,
        alpha=alpha,
    )

    return PGDResult(
        success=success,
        original_class=original_class,
        original_confidence=original_confidence,
        adversarial_class=adversarial_class,
        adversarial_confidence=adversarial_confidence,
        confidence_delta=confidence_delta,
        steps_to_success=best_result.steps_to_success,
        confidence_per_step=best_result.confidence_per_step,
        epsilon_used=epsilon,
        alpha_used=alpha,
        steps_used=steps,
        restarts_used=restarts,
        adversarial_tensor=best_result.adversarial_tensor.detach(),
        adversarial_pil=tensor_to_pil(best_result.adversarial_tensor),
        plain_english=plain_english,
    )


# ---------------------------------------------------------------------------
# Internal: single PGD run (one restart)
# ---------------------------------------------------------------------------

@dataclass
class _PGDRunResult:
    """Internal result for one PGD restart."""
    success: bool
    steps_to_success: int       # 1-based step index, -1 if never succeeded
    confidence_per_step: list[float]
    adversarial_tensor: torch.Tensor


def _pgd_single_run(
    model: KaalModel,
    image_tensor: torch.Tensor,
    original_class: int,
    attack_class: int,
    epsilon: float,
    alpha: float,
    steps: int,
    targeted: bool,
    seed_offset: int = 0,
) -> _PGDRunResult:
    """Execute one PGD run with a fresh random start.

    Args:
        seed_offset: Added to ensure different noise per restart.
    """
    # Random start: uniform noise in [-ε, +ε] added to original
    torch.manual_seed(seed_offset * 137 + 42)
    noise = torch.empty_like(image_tensor).uniform_(-epsilon, epsilon)
    x = (image_tensor + noise).clamp(_PIXEL_MIN, _PIXEL_MAX)

    confidence_per_step: list[float] = []
    steps_to_success = -1
    best_adversarial = x.clone()

    for step in range(1, steps + 1):
        # Compute gradient at current x
        grad = model.gradient(x, attack_class)

        # PGD step: move in sign(grad) direction
        if targeted:
            # Targeted: move against gradient (increase logit for target)
            x = x - alpha * grad.sign()
        else:
            # Untargeted: move with gradient (increase loss on true class)
            x = x + alpha * grad.sign()

        # Project back to epsilon ball around original image
        x = torch.max(torch.min(x, image_tensor + epsilon), image_tensor - epsilon)

        # Keep in valid normalized pixel range
        x = x.clamp(_PIXEL_MIN, _PIXEL_MAX)

        # Detach to avoid accumulating computation graph across steps
        x = x.detach()

        # Evaluate current x — record confidence on original class
        pred = model.predict(x)
        conf_on_original = pred["all_confidences"][original_class]
        confidence_per_step.append(float(conf_on_original))

        # Check for first success
        current_class = pred["class_idx"]
        if targeted:
            step_success = (current_class == attack_class)
        else:
            step_success = (current_class != original_class)

        if step_success:
            if steps_to_success == -1:
                steps_to_success = step   # record first success step (1-based)
            best_adversarial = x.clone()  # keep updating — later steps may be stronger

    # If never succeeded, store the final x as best adversarial attempt
    if steps_to_success == -1:
        best_adversarial = x.clone()

    final_success = (steps_to_success != -1)

    return _PGDRunResult(
        success=final_success,
        steps_to_success=steps_to_success,
        confidence_per_step=confidence_per_step,
        adversarial_tensor=best_adversarial,
    )


# ---------------------------------------------------------------------------
# Batch helper — run PGD over a full dataset
# ---------------------------------------------------------------------------

def pgd_attack_dataset(
    model: KaalModel,
    dataset,
    epsilon: float = 0.03,
    alpha: Optional[float] = None,
    steps: int = 40,
    restarts: int = 1,
    targeted: bool = False,
    target_class: Optional[int] = None,
    max_images: Optional[int] = None,
) -> dict:
    """Run PGD over all images in a KaalDataset and return aggregate results.

    Args:
        model:        KaalModel instance.
        dataset:      KaalDataset from load_dataset().
        epsilon:      Perturbation strength.
        alpha:        Step size. Defaults to epsilon/10.
        steps:        PGD iterations.
        restarts:     Random restarts per image.
        targeted:     Targeted mode flag.
        target_class: Required if targeted=True.
        max_images:   Optional cap.

    Returns:
        dict with keys:
            "results":              list[PGDResult]
            "success_rate":         float
            "avg_confidence_delta": float
            "avg_steps_to_success": float  (-1 if no successes)
            "epsilon_used":         float
            "alpha_used":           float
            "steps_used":           int
            "total_images":         int
            "successful_attacks":   int
    """
    results: list[PGDResult] = []
    count = 0

    for tensor, path, pil in dataset:
        if max_images is not None and count >= max_images:
            break
        result = pgd_attack(
            model, tensor,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            restarts=restarts,
            targeted=targeted,
            target_class=target_class,
        )
        results.append(result)
        count += 1

    if not results:
        effective_alpha = alpha if alpha is not None else epsilon / 10.0
        return {
            "results": [],
            "success_rate": 0.0,
            "avg_confidence_delta": 0.0,
            "avg_steps_to_success": -1.0,
            "epsilon_used": epsilon,
            "alpha_used": effective_alpha,
            "steps_used": steps,
            "total_images": 0,
            "successful_attacks": 0,
        }

    successful = [r for r in results if r.success]
    success_rate = len(successful) / len(results)
    avg_delta = sum(r.confidence_delta for r in results) / len(results)
    avg_steps = (
        sum(r.steps_to_success for r in successful) / len(successful)
        if successful else -1.0
    )

    return {
        "results": results,
        "success_rate": round(success_rate, 4),
        "avg_confidence_delta": round(avg_delta, 4),
        "avg_steps_to_success": round(avg_steps, 2),
        "epsilon_used": results[0].epsilon_used,
        "alpha_used": results[0].alpha_used,
        "steps_used": steps,
        "total_images": len(results),
        "successful_attacks": len(successful),
    }


# ---------------------------------------------------------------------------
# plain_english builder
# ---------------------------------------------------------------------------

def _build_plain_english(
    success: bool,
    targeted: bool,
    original_class: int,
    adversarial_class: int,
    original_confidence: float,
    adversarial_confidence: float,
    steps_to_success: int,
    steps_used: int,
    epsilon: float,
    alpha: float,
) -> str:
    """One factual sentence. No exclamation marks, no threat language.

    Always references epsilon as "perturbation magnitude ε={value}".
    """
    eps_str = f"ε={epsilon:.4f}".rstrip("0").rstrip(".")
    alpha_str = f"α={alpha:.5f}".rstrip("0").rstrip(".")

    if targeted:
        if success:
            return (
                f"Targeted PGD at perturbation magnitude {eps_str}, {alpha_str}, "
                f"{steps_to_success} steps steered model prediction from class "
                f"{original_class} to target class {adversarial_class} with "
                f"confidence {adversarial_confidence:.2f}."
            )
        else:
            return (
                f"Targeted PGD at perturbation magnitude {eps_str}, {alpha_str}, "
                f"{steps_used} steps did not achieve target class {adversarial_class}; "
                f"model retained prediction class {original_class} with confidence "
                f"{adversarial_confidence:.2f}."
            )
    else:
        if success:
            return (
                f"PGD at perturbation magnitude {eps_str}, {alpha_str} caused "
                f"misclassification from class {original_class} to class "
                f"{adversarial_class} at step {steps_to_success} of {steps_used}; "
                f"original confidence {original_confidence:.2f} reduced to "
                f"{adversarial_confidence:.2f}."
            )
        else:
            return (
                f"PGD at perturbation magnitude {eps_str}, {alpha_str} over "
                f"{steps_used} steps did not cause misclassification; model retained "
                f"class {original_class} with confidence {adversarial_confidence:.2f}."
            )
