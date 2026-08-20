"""FGSM Attack — Fast Gradient Sign Method.

Spec 10.1 — Phase 2, Kiro Prompt 2.1.

Math:
    x_adv = x + ε × sign(∇ₓ J(θ, x, y))

    Where:
        x     = original image tensor
        ε     = perturbation strength (epsilon)
        ∇ₓ J  = gradient of loss w.r.t. input pixels
        sign() = returns +1 or -1 for each element
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from kaal.engine.loader import KaalModel
from kaal.engine.utils import tensor_to_pil, denormalize


# ---------------------------------------------------------------------------
# Result dataclass — Spec 10.1
# ---------------------------------------------------------------------------

@dataclass
class FGSMResult:
    """Complete result of a single FGSM attack run."""

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
    """Drop in confidence on the original class: original_confidence − post_attack_confidence_on_original."""

    epsilon_used: float
    """The epsilon value used for this attack."""

    adversarial_tensor: torch.Tensor
    """The perturbed image tensor (same shape as input, normalized)."""

    adversarial_pil: Image.Image
    """The perturbed image as a PIL Image (denormalized, for saving/display)."""

    perturbation_tensor: torch.Tensor
    """The noise added to the original image: adversarial_tensor − original_tensor."""

    plain_english: str
    """One factual sentence describing what happened. No drama, no exclamation marks."""


# ---------------------------------------------------------------------------
# fgsm_attack() — main entry point
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: KaalModel,
    image_tensor: torch.Tensor,
    epsilon: float = 0.03,
    targeted: bool = False,
    target_class: Optional[int] = None,
) -> FGSMResult:
    """Run a single FGSM attack on one image.

    Args:
        model:        KaalModel from kaal.engine.loader.
        image_tensor: Normalized torch.Tensor, shape (C, H, W).
                      Must be the ImageNet-normalized tensor from load_dataset().
        epsilon:      Perturbation strength. Typical range: 0.001–0.1.
                      0.03 = imperceptible threshold for most images.
        targeted:     If False (default), maximises loss on the true prediction
                      (untargeted — causes any misclassification).
                      If True, minimises loss on target_class
                      (targeted — steers prediction to a specific class).
        target_class: Required when targeted=True. Class index to steer toward.

    Returns:
        FGSMResult dataclass with all attack details.

    Raises:
        ValueError: If targeted=True and target_class is None.
        NotImplementedError: If model framework doesn't support gradients.

    Example:
        result = fgsm_attack(model, tensor, epsilon=0.03)
        if result.success:
            result.adversarial_pil.save("adversarial.png")
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

    # Ensure no batch dimension
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    # --- Step 1: get original prediction ------------------------------------
    original_pred = model.predict(image_tensor)
    original_class = original_pred["class_idx"]
    original_confidence = original_pred["confidence"]

    # Determine the class to attack
    attack_class = target_class if targeted else original_class

    # --- Step 2: compute gradient -------------------------------------------
    grad = model.gradient(image_tensor, attack_class)

    # --- Step 3: apply FGSM perturbation ------------------------------------
    # Untargeted: step in direction of gradient (increases loss on true class)
    # Targeted:   step against gradient (decreases loss on target, increases logit)
    sign = grad.sign()
    if targeted:
        # Move against gradient → increases logit for target_class
        perturbation = -epsilon * sign
    else:
        # Move with gradient → increases loss on original class
        perturbation = epsilon * sign

    adversarial_tensor = (image_tensor + perturbation).clamp(-3.0, 3.0)
    # Clamp to ~[-3, 3] which covers the normalized ImageNet range well.
    # Real pixel range [0, 1] in normalized space sits around [-2.1, 2.6].

    # --- Step 4: evaluate adversarial image ---------------------------------
    adversarial_pred = model.predict(adversarial_tensor)
    adversarial_class = adversarial_pred["class_idx"]
    adversarial_confidence = adversarial_pred["confidence"]

    # Confidence delta = drop in model's confidence on original class
    post_confidence_on_original = adversarial_pred["all_confidences"][original_class]
    confidence_delta = original_confidence - post_confidence_on_original

    # Determine success
    if targeted:
        success = (adversarial_class == target_class)
    else:
        success = (adversarial_class != original_class)

    # --- Step 5: build PIL outputs ------------------------------------------
    adversarial_pil = tensor_to_pil(adversarial_tensor)
    perturbation_tensor = adversarial_tensor - image_tensor

    # --- Step 6: generate plain_english -------------------------------------
    plain_english = _build_plain_english(
        success=success,
        targeted=targeted,
        original_class=original_class,
        adversarial_class=adversarial_class,
        original_confidence=original_confidence,
        adversarial_confidence=adversarial_confidence,
        epsilon=epsilon,
    )

    return FGSMResult(
        success=success,
        original_class=original_class,
        original_confidence=original_confidence,
        adversarial_class=adversarial_class,
        adversarial_confidence=adversarial_confidence,
        confidence_delta=confidence_delta,
        epsilon_used=epsilon,
        adversarial_tensor=adversarial_tensor.detach(),
        adversarial_pil=adversarial_pil,
        perturbation_tensor=perturbation_tensor.detach(),
        plain_english=plain_english,
    )


# ---------------------------------------------------------------------------
# Batch helper — run FGSM over a full dataset and return aggregate stats
# ---------------------------------------------------------------------------

def fgsm_attack_dataset(
    model: KaalModel,
    dataset,
    epsilon: float = 0.03,
    targeted: bool = False,
    target_class: Optional[int] = None,
    max_images: Optional[int] = None,
) -> dict:
    """Run FGSM over all images in a KaalDataset and return aggregate results.

    Args:
        model:       KaalModel instance.
        dataset:     KaalDataset from load_dataset().
        epsilon:     Perturbation strength.
        targeted:    Whether to use targeted attack.
        target_class: Required if targeted=True.
        max_images:  Optional cap on number of images to process.

    Returns:
        dict with keys:
            "results":             list[FGSMResult]
            "success_rate":        float  (0.0–1.0)
            "avg_confidence_delta": float
            "epsilon_used":        float
            "total_images":        int
            "successful_attacks":  int
    """
    results: list[FGSMResult] = []
    count = 0

    for tensor, path, pil in dataset:
        if max_images is not None and count >= max_images:
            break
        result = fgsm_attack(
            model, tensor,
            epsilon=epsilon,
            targeted=targeted,
            target_class=target_class,
        )
        results.append(result)
        count += 1

    if not results:
        return {
            "results": [],
            "success_rate": 0.0,
            "avg_confidence_delta": 0.0,
            "epsilon_used": epsilon,
            "total_images": 0,
            "successful_attacks": 0,
        }

    successful = sum(1 for r in results if r.success)
    success_rate = successful / len(results)
    avg_delta = sum(r.confidence_delta for r in results) / len(results)

    return {
        "results": results,
        "success_rate": round(success_rate, 4),
        "avg_confidence_delta": round(avg_delta, 4),
        "epsilon_used": epsilon,
        "total_images": len(results),
        "successful_attacks": successful,
    }


# ---------------------------------------------------------------------------
# Epsilon robustness curve — the measured perturbation threshold
# ---------------------------------------------------------------------------

# Ladder in the units the adversarial-robustness literature reports: eighths of
# a step through n/255. Geometric rather than linear because robustness
# thresholds are scale-free — the meaningful gap is 2/255 vs 8/255, not
# 0.10 vs 0.11. Aligning to /255 also keeps every rung representable in an
# 8-bit image; note that the first rung (0.5/255) sits *below* one quantisation
# step, so a model broken there is broken by a perturbation that cannot survive
# being saved as a PNG.
DEFAULT_EPSILON_LADDER: tuple[float, ...] = (
    0.5 / 255,    # 0.00196 — sub-quantisation
    1.0 / 255,    # 0.00392
    2.0 / 255,    # 0.00784
    4.0 / 255,    # 0.01569
    8.0 / 255,    # 0.03137 — the de facto standard ImageNet L∞ budget
    16.0 / 255,   # 0.06275
    32.0 / 255,   # 0.12549
)

# Fraction of the dataset that must be misclassified for an epsilon to count as
# the model's breaking point. 0.5 = "this perturbation fools the model more
# often than not".
_EPSILON_TARGET_RATE = 0.5

# Curve measurement is O(len(ladder) x images), so it samples rather than
# sweeping the whole dataset. 20 images at 8 epsilons = 160 FGSM calls.
DEFAULT_CURVE_MAX_IMAGES = 20


def epsilon_robustness_curve(
    model: KaalModel,
    dataset,
    ladder: Optional[tuple] = None,
    max_images: Optional[int] = None,
) -> dict:
    """Measure FGSM success rate across a ladder of epsilon values.

    This is the *measured* perturbation threshold — the smallest L∞ budget at
    which the model misclassifies at least half the sample. Unlike a fixed-
    epsilon attack (whose perturbation always saturates to exactly epsilon by
    construction, carrying no information about the model), this discriminates
    between a model that breaks at 0.002 and one that holds to 0.064.

    Args:
        model:      KaalModel instance.
        dataset:    KaalDataset from load_dataset().
        ladder:     Ascending epsilon values to test. Defaults to
                    DEFAULT_EPSILON_LADDER.
        max_images: Cap on images sampled per epsilon. Defaults to
                    DEFAULT_CURVE_MAX_IMAGES.

    Returns:
        dict with keys:
            "epsilons":       list[float]  — the rungs actually evaluated
            "success_rates":  list[float]  — misclassification rate at each rung
            "epsilon_50":     float | None — smallest epsilon reaching the
                              target rate; None if the model never breaks
            "target_rate":    float
            "images_sampled": int

    Example:
        curve = epsilon_robustness_curve(model, dataset)
        print(curve["epsilon_50"])   # e.g. 0.004
    """
    rungs = tuple(ladder) if ladder else DEFAULT_EPSILON_LADDER
    cap = max_images if max_images is not None else DEFAULT_CURVE_MAX_IMAGES

    tensors = []
    for tensor, _path, _pil in dataset:
        if len(tensors) >= cap:
            break
        tensors.append(tensor)

    if not tensors:
        return {
            "epsilons": [], "success_rates": [], "epsilon_50": None,
            "target_rate": _EPSILON_TARGET_RATE, "images_sampled": 0,
        }

    epsilons: list[float] = []
    rates: list[float] = []
    epsilon_50: Optional[float] = None

    for eps in rungs:
        successes = 0
        for tensor in tensors:
            if fgsm_attack(model, tensor, epsilon=eps).success:
                successes += 1
        rate = successes / len(tensors)
        epsilons.append(float(eps))
        rates.append(round(rate, 4))

        if epsilon_50 is None and rate >= _EPSILON_TARGET_RATE:
            epsilon_50 = float(eps)
            # Success rate is monotonic in epsilon for practical purposes, and
            # the remaining rungs cost real time. Stop once the threshold and
            # the shape of the curve up to it are known.
            break

    return {
        "epsilons": epsilons,
        "success_rates": rates,
        "epsilon_50": epsilon_50,
        "target_rate": _EPSILON_TARGET_RATE,
        "images_sampled": len(tensors),
    }


# ---------------------------------------------------------------------------
# plain_english builder — Spec 10.1 rules
# ---------------------------------------------------------------------------

def _build_plain_english(
    success: bool,
    targeted: bool,
    original_class: int,
    adversarial_class: int,
    original_confidence: float,
    adversarial_confidence: float,
    epsilon: float,
) -> str:
    """Build a one-sentence factual description of the attack outcome.

    Rules from Spec 10.1:
        - Maximum one sentence
        - No exclamation marks
        - No threat language
        - State what happened factually
        - Always refer to epsilon as "perturbation magnitude ε={value}"
    """
    eps_str = f"ε={epsilon:.4f}".rstrip("0").rstrip(".")

    if targeted:
        if success:
            return (
                f"Targeted FGSM at perturbation magnitude {eps_str} steered model "
                f"prediction from class {original_class} to target class {adversarial_class} "
                f"with confidence {adversarial_confidence:.2f}."
            )
        else:
            return (
                f"Targeted FGSM at perturbation magnitude {eps_str} did not achieve "
                f"target class {adversarial_class}; model retained prediction "
                f"class {original_class} with confidence {adversarial_confidence:.2f}."
            )
    else:
        if success:
            return (
                f"Model prediction changed from class {original_class} to class "
                f"{adversarial_class} under perturbation magnitude {eps_str}; "
                f"original confidence {original_confidence:.2f} reduced to {adversarial_confidence:.2f}."
            )
        else:
            return (
                f"Model prediction held at class {original_class} with confidence "
                f"{adversarial_confidence:.2f} under perturbation magnitude {eps_str}; "
                f"attack did not cause misclassification."
            )
