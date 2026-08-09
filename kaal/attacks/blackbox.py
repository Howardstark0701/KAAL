"""Black-Box Attack — NES Gradient Estimation.

Spec 10.4 — Phase 5.

NES (Natural Evolution Strategies) is a score-based black-box attack.
It estimates the gradient using only model output probabilities (logits/scores),
with no access to model weights, architecture, or internal gradients.

Algorithm:
    For each step t = 1 to T:
        1. Sample n antithetic Gaussian probe vectors:
               u_i ~ N(0, I),  shape (C, H, W)
           Antithetic pairs: evaluate both +u and −u per probe.

        2. Estimate gradient:
               ĝ = (1 / (2·n·σ)) × Σᵢ [ f(x + σ·uᵢ) − f(x − σ·uᵢ) ] × uᵢ
           where f(·) = model confidence on the target (attack) class.

        3. Apply gradient step (ascent on attack class loss):
               x ← x − α × sign(ĝ)          ← untargeted: decrease confidence on true class
               x ← x + α × sign(ĝ)          ← targeted: increase confidence on target class

        4. Project back to L∞ epsilon ball:
               x ← clip(x, x_orig − ε, x_orig + ε)
               x ← clip(x, PIXEL_MIN, PIXEL_MAX)

    Each step costs 2·n model queries (antithetic pairs).
    Total budget: max_queries queries across all steps.

Query Efficiency:
    query_efficiency = successful_steps / total_steps_run
    Reflects how efficiently the attack used its query budget.
    This is the Dim 5 input to the KVS scorer.

References:
    Wierstra et al. (2014) — Natural Evolution Strategies
    Ilyas et al. (2018) — Black-Box Adversarial Attacks with Limited Queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

from kaal.engine.loader import KaalModel
from kaal.engine.utils import tensor_to_pil


# ---------------------------------------------------------------------------
# Normalized pixel bounds — same as FGSM / PGD
# ---------------------------------------------------------------------------
_PIXEL_MIN = -3.0
_PIXEL_MAX  =  3.0


# ---------------------------------------------------------------------------
# BlackBoxResult — Spec 10.4
# ---------------------------------------------------------------------------

@dataclass
class BlackBoxResult:
    """Complete result of a single NES black-box attack run."""

    success: bool
    """True if the attack caused misclassification (untargeted) or reached
    the target class (targeted)."""

    original_class: int
    """Model's predicted class index before the attack."""

    original_confidence: float
    """Model's confidence on original_class before the attack (0.0–1.0)."""

    adversarial_class: int
    """Model's predicted class index after the attack."""

    adversarial_confidence: float
    """Model's confidence on adversarial_class after the attack (0.0–1.0)."""

    queries_used: int
    """Total number of model queries consumed by this attack run."""

    max_queries: int
    """The query budget this attack was allowed."""

    query_efficiency: float
    """Fraction of query budget steps that produced a confidence reduction.
    Range: 0.0–1.0. Used as Dimension 5 input in KVS scoring."""

    epsilon_used: float
    """L∞ perturbation bound used."""

    sigma_used: float
    """NES probe noise standard deviation used."""

    alpha_used: float
    """Step size used for each gradient update."""

    steps_used: int
    """Number of NES gradient steps actually executed."""

    steps_to_success: int
    """Step (1-based) at which first misclassification occurred.
    -1 if the attack never succeeded."""

    adversarial_tensor: torch.Tensor
    """The best adversarial image tensor found (same shape as input, normalized)."""

    adversarial_pil: Image.Image
    """The best adversarial image as a PIL Image (denormalized, for saving/display)."""

    plain_english: str
    """One factual sentence describing the outcome. No drama, no exclamation marks."""


# ---------------------------------------------------------------------------
# blackbox_attack() — main entry point
# ---------------------------------------------------------------------------

def blackbox_attack(
    model: KaalModel,
    image_tensor: torch.Tensor,
    epsilon: float = 0.05,
    alpha: Optional[float] = None,
    sigma: float = 0.01,
    n_samples: int = 20,
    max_queries: int = 1000,
    targeted: bool = False,
    target_class: Optional[int] = None,
    seed: Optional[int] = None,
) -> BlackBoxResult:
    """Run a single NES black-box attack on one image.

    This attack uses only the model's output probabilities — no gradients,
    no architecture access. It is suitable for auditing black-box APIs.

    Args:
        model:        KaalModel from kaal.engine.loader.
        image_tensor: Normalized torch.Tensor, shape (C, H, W) or (1, C, H, W).
        epsilon:      L∞ perturbation bound. Default 0.05 (larger than white-box
                      attacks, as black-box attacks are less precise).
        alpha:        Step size per NES update. Default: epsilon / 10.
        sigma:        Probe noise std dev for NES gradient estimation. Default 0.01.
        n_samples:    Antithetic probe pairs per NES step. Default 20 (= 40 queries/step).
                      More probes → better gradient estimate, more queries used.
        max_queries:  Total query budget. Attack stops when budget is exhausted.
                      Default 1000.
        targeted:     If False (default), untargeted — cause any misclassification.
                      If True, targeted — steer prediction to target_class.
        target_class: Required when targeted=True.
        seed:         Optional RNG seed for reproducible NES probes.
                      np.random and torch are both seeded when provided.

    Returns:
        BlackBoxResult dataclass. query_efficiency feeds into KVS Dim 5.

    Raises:
        ValueError: targeted=True without target_class, or invalid parameters.

    Example:
        result = blackbox_attack(model, tensor, epsilon=0.05, max_queries=500)
        if result.success:
            print(f"Succeeded in {result.queries_used} queries")
            print(result.plain_english)
    """
    # --- Input validation ---------------------------------------------------
    if targeted and target_class is None:
        raise ValueError(
            "targeted=True requires target_class to be specified.\n"
            "→ Pass the integer class index you want the model to predict."
        )

    epsilon = float(epsilon)
    if not (0.0 < epsilon <= 1.0):
        raise ValueError(
            f"epsilon must be in range (0.0, 1.0], got {epsilon}.\n"
            "→ Typical values: 0.03 (subtle), 0.05 (standard), 0.1 (strong)."
        )

    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    max_queries = int(max_queries)
    if max_queries < 2:
        raise ValueError(f"max_queries must be >= 2, got {max_queries}.")

    if alpha is None:
        alpha = epsilon / 10.0
    alpha = float(alpha)

    # Squeeze batch dim
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    # --- Get original prediction --------------------------------------------
    original_pred    = model.predict(image_tensor)
    original_class   = original_pred["class_idx"]
    original_conf    = original_pred["confidence"]
    attack_class     = target_class if targeted else original_class
    queries_used     = 1   # for the initial predict call

    # --- RNG seeding (reproducibility) --------------------------------------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # --- Run NES attack loop ------------------------------------------------
    x              = image_tensor.clone()
    best_adv       = x.clone()
    steps_to_success = -1
    steps_used     = 0
    successful_steps = 0  # steps where confidence on original class decreased

    # Track confidence on the attack class at start (for targeted), or
    # confidence on original class (for untargeted) for measuring progress
    prev_conf_on_original = original_conf

    # Each step uses 2 * n_samples queries (antithetic pairs)
    queries_per_step = 2 * n_samples

    while queries_used + queries_per_step <= max_queries:
        steps_used += 1
        grad_estimate = _nes_gradient(
            model=model,
            x=x,
            attack_class=attack_class,
            sigma=sigma,
            n_samples=n_samples,
        )
        queries_used += queries_per_step

        # Gradient step on attack class confidence:
        # Untargeted: attack_class == original_class → decreasing its
        #             confidence means we MOVE AGAINST the gradient
        # Targeted:   attack_class == target_class  → we want to INCREASE
        #             confidence on target → move WITH the gradient
        if targeted:
            # Move toward target_class: increase confidence on target
            x = x + alpha * grad_estimate.sign()
        else:
            # Move away from original_class: decrease confidence on original
            x = x - alpha * grad_estimate.sign()

        # Project back to epsilon ball around original image
        x = torch.max(
            torch.min(x, image_tensor + epsilon),
            image_tensor - epsilon,
        )
        # Clamp to valid normalized pixel range
        x = x.clamp(_PIXEL_MIN, _PIXEL_MAX).detach()

        # Evaluate current x
        queries_used += 1
        pred = model.predict(x)
        current_class = pred["class_idx"]
        conf_on_original = pred["all_confidences"][original_class]

        # Track whether this step moved us in the right direction
        if conf_on_original < prev_conf_on_original:
            successful_steps += 1
        prev_conf_on_original = conf_on_original

        # Check for success
        if targeted:
            step_success = (current_class == attack_class)
        else:
            step_success = (current_class != original_class)

        if step_success:
            if steps_to_success == -1:
                steps_to_success = steps_used
            best_adv = x.clone()

        # Early exit if targeted and already succeeded (save query budget)
        if targeted and step_success:
            break

    # If we never found a misclassification, best_adv is the latest x
    if steps_to_success == -1:
        best_adv = x.clone()

    # --- Build final metrics ------------------------------------------------
    final_pred          = model.predict(best_adv)
    queries_used        += 1
    adversarial_class   = final_pred["class_idx"]
    adversarial_conf    = final_pred["confidence"]

    success = (steps_to_success != -1)

    # Query efficiency = fraction of NES steps that made useful progress
    query_efficiency = (
        round(successful_steps / steps_used, 4) if steps_used > 0 else 0.0
    )

    plain_english = _build_plain_english(
        success=success,
        targeted=targeted,
        original_class=original_class,
        adversarial_class=adversarial_class,
        original_conf=original_conf,
        adversarial_conf=adversarial_conf,
        queries_used=queries_used,
        max_queries=max_queries,
        epsilon=epsilon,
    )

    return BlackBoxResult(
        success=success,
        original_class=original_class,
        original_confidence=original_conf,
        adversarial_class=adversarial_class,
        adversarial_confidence=adversarial_conf,
        queries_used=queries_used,
        max_queries=max_queries,
        query_efficiency=query_efficiency,
        epsilon_used=epsilon,
        sigma_used=sigma,
        alpha_used=alpha,
        steps_used=steps_used,
        steps_to_success=steps_to_success,
        adversarial_tensor=best_adv.detach(),
        adversarial_pil=tensor_to_pil(best_adv),
        plain_english=plain_english,
    )


# ---------------------------------------------------------------------------
# Batch variant — run across a full dataset
# ---------------------------------------------------------------------------

def blackbox_attack_dataset(
    model: KaalModel,
    dataset,
    epsilon: float = 0.05,
    alpha: Optional[float] = None,
    sigma: float = 0.01,
    n_samples: int = 20,
    max_queries: int = 1000,
    targeted: bool = False,
    target_class: Optional[int] = None,
    max_images: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """Run NES black-box attack over all images in a KaalDataset.

    Args:
        model:        KaalModel instance.
        dataset:      KaalDataset from load_dataset().
        epsilon:      L∞ perturbation bound.
        alpha:        Step size. Defaults to epsilon / 10.
        sigma:        NES probe noise std dev.
        n_samples:    Antithetic probe pairs per step.
        max_queries:  Query budget per image.
        targeted:     Targeted mode flag.
        target_class: Required if targeted=True.
        max_images:   Optional cap on images to process.
        seed:         Optional RNG seed forwarded to each blackbox_attack call.

    Returns:
        dict with keys:
            "results":               list[BlackBoxResult]
            "success_rate":          float
            "avg_queries_used":      float
            "avg_query_efficiency":  float
            "epsilon_used":          float
            "total_images":          int
            "successful_attacks":    int
    """
    results: list[BlackBoxResult] = []
    count = 0

    for tensor, path, pil in dataset:
        if max_images is not None and count >= max_images:
            break
        result = blackbox_attack(
            model, tensor,
            epsilon=epsilon,
            alpha=alpha,
            sigma=sigma,
            n_samples=n_samples,
            max_queries=max_queries,
            targeted=targeted,
            target_class=target_class,
            seed=seed,
        )
        results.append(result)
        count += 1

    if not results:
        return {
            "results": [],
            "success_rate": 0.0,
            "avg_queries_used": 0.0,
            "avg_query_efficiency": 0.0,
            "epsilon_used": epsilon,
            "total_images": 0,
            "successful_attacks": 0,
        }

    successful = [r for r in results if r.success]
    return {
        "results": results,
        "success_rate": round(len(successful) / len(results), 4),
        "avg_queries_used": round(
            sum(r.queries_used for r in results) / len(results), 2
        ),
        "avg_query_efficiency": round(
            sum(r.query_efficiency for r in results) / len(results), 4
        ),
        "epsilon_used": results[0].epsilon_used,
        "total_images": len(results),
        "successful_attacks": len(successful),
    }


# ---------------------------------------------------------------------------
# NES gradient estimator — core of the black-box attack
# ---------------------------------------------------------------------------

def _nes_gradient(
    model: KaalModel,
    x: torch.Tensor,
    attack_class: int,
    sigma: float,
    n_samples: int,
) -> torch.Tensor:
    """Estimate gradient of model confidence on attack_class using NES.

    Uses antithetic sampling: each probe u is evaluated as both +u and −u,
    halving variance for the same query budget.

    Args:
        model:        KaalModel — only .predict() is called (black-box).
        x:            Current adversarial image tensor (C, H, W).
        attack_class: Class whose confidence we are differentiating w.r.t.
        sigma:        Probe noise standard deviation.
        n_samples:    Number of antithetic probe pairs.

    Returns:
        Estimated gradient tensor, same shape as x.

    Queries made: 2 * n_samples (two model calls per antithetic pair).
    """
    grad = torch.zeros_like(x)
    shape = x.shape

    for _ in range(n_samples):
        # Draw a random direction in input space
        u = torch.randn(shape)

        # Query model at x + σu and x − σu
        x_plus  = (x + sigma * u).clamp(_PIXEL_MIN, _PIXEL_MAX)
        x_minus = (x - sigma * u).clamp(_PIXEL_MIN, _PIXEL_MAX)

        pred_plus  = model.predict(x_plus)
        pred_minus = model.predict(x_minus)

        # Confidence on the attack class for each probe direction
        f_plus  = pred_plus["all_confidences"][attack_class]
        f_minus = pred_minus["all_confidences"][attack_class]

        # Antithetic NES gradient contribution
        grad += (f_plus - f_minus) * u

    # Normalise: divide by (2 * n_samples * sigma)
    grad = grad / (2.0 * n_samples * sigma)
    return grad


# ---------------------------------------------------------------------------
# plain_english builder — matches Spec 10.4 style
# ---------------------------------------------------------------------------

def _build_plain_english(
    success: bool,
    targeted: bool,
    original_class: int,
    adversarial_class: int,
    original_conf: float,
    adversarial_conf: float,
    queries_used: int,
    max_queries: int,
    epsilon: float,
) -> str:
    """One factual sentence. No exclamation marks, no threat language."""
    eps_str = f"ε={epsilon:.4f}".rstrip("0").rstrip(".")

    if targeted:
        if success:
            return (
                f"Targeted NES black-box attack at perturbation magnitude {eps_str} "
                f"steered model prediction from class {original_class} to target class "
                f"{adversarial_class} using {queries_used} of {max_queries} queries."
            )
        else:
            return (
                f"Targeted NES black-box attack at perturbation magnitude {eps_str} "
                f"did not achieve target class {adversarial_class} within "
                f"{queries_used} queries; model retained class {original_class} "
                f"with confidence {adversarial_conf:.2f}."
            )
    else:
        if success:
            return (
                f"NES black-box attack at perturbation magnitude {eps_str} caused "
                f"misclassification from class {original_class} to class "
                f"{adversarial_class} using {queries_used} of {max_queries} queries; "
                f"original confidence {original_conf:.2f} reduced to {adversarial_conf:.2f}."
            )
        else:
            return (
                f"NES black-box attack at perturbation magnitude {eps_str} did not "
                f"cause misclassification within {queries_used} queries; model "
                f"retained class {original_class} with confidence {adversarial_conf:.2f}."
            )
