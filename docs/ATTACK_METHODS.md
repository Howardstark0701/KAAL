# KAAL — Attack Methods Reference

## FGSM — Fast Gradient Sign Method

Single-step gradient-based attack. Fastest attack — good baseline.

```
x_adv = x + ε × sign(∇ₓ J(θ, x, y))
```

**Parameters:** epsilon (perturbation strength)  
**Spec:** Section 10.1

---

## PGD — Projected Gradient Descent

Iterative FGSM with random restart. Gold standard adversarial attack.

```
xₜ = clip(xₜ₋₁ + α × sign(∇ₓ J(θ, xₜ₋₁, y)), x−ε, x+ε)
```

**Parameters:** epsilon, alpha (step size), steps, restarts  
**Spec:** Section 10.2

---

## Adversarial Patch

Gradient ascent trained patch that causes misclassification when placed anywhere in frame.
Outputs a printable PDF for physical deployment testing.

**Parameters:** target_class, patch_fraction, iterations  
**Spec:** Section 10.3

---

## Black-Box Attack (NES)

Query-only attack using Natural Evolution Strategy gradient estimation.
No access to model internals required.

**Parameters:** epsilon, max_queries, samples_per_step, sigma  
**Spec:** Section 10.4

---

## Physical Robustness Simulator

Tests whether adversarial examples survive real-world image transformations.
7 transform types: JPEG compression, Gaussian noise, brightness, contrast, rotation, scaling, blur.

**Output:** Lab Only / Limited / Field Ready rating  
**Spec:** Section 10.5
