"""Adversarial attacks on tabular classifiers (sklearn, XGBoost, LightGBM).

kaal/attacks/tabular_attack.py

Works with any classifier that exposes a .predict_proba(X) method returning
an (n_samples, n_classes) probability array — sklearn RandomForest,
GradientBoosting, LogisticRegression, SVM (with probability=True),
XGBClassifier, LGBMClassifier, CatBoostClassifier, etc.

Attack method: gradient-free boundary walking.
    No gradients required. Uses random perturbations with greedy acceptance
    (keep a step if and only if P(target_class) increases). This is a
    zeroth-order attack that works on any black-box scorer.

Only stdlib + numpy required. sklearn/xgboost imports are guarded and
only needed at runtime when you pass such a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# TabularAttackResult
# ---------------------------------------------------------------------------

@dataclass
class TabularAttackResult:
    """Result of a tabular adversarial attack."""

    success_rate: float
    """Fraction of samples where the attack caused a class change (0–1)."""

    avg_confidence_on_target: float
    """Average P(target_class) across all samples after attack (0–1)."""

    n_samples: int
    """Number of input rows processed."""

    reached_target: bool
    """True if any row's adversarial prediction equals target_class.

    Distinct from success_rate (which counts any class change) — a "successful"
    attack may have landed on a wrong class other than the requested target.
    """

    most_perturbed_features: list[str]
    """Top-3 feature names by mean absolute perturbation across all samples."""

    plain_english: str
    """One factual sentence describing the outcome. No drama."""


# ---------------------------------------------------------------------------
# TabularAttacker
# ---------------------------------------------------------------------------

class TabularAttacker:
    """Gradient-free adversarial attacker for tabular classifiers.

    Works with any sklearn-compatible model that exposes .predict_proba().

    Usage:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from kaal.attacks.tabular_attack import TabularAttacker

        model = RandomForestClassifier().fit(X_train, y_train)

        attacker = TabularAttacker(
            model=model,
            feature_names=["age", "income", "score", "debt"],
            feature_ranges={
                "age":    (18.0, 90.0),
                "income": (0.0, 500_000.0),
                "score":  (300.0, 850.0),
                "debt":   (0.0, 200_000.0),
            },
        )
        result = attacker.feature_perturbation_attack(
            X=X_test[:50],
            target_class=1,
            epsilon=0.1,
            n_steps=20,
        )
        print(result.success_rate, result.most_perturbed_features)
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        feature_ranges: dict[str, tuple[float, float]],
    ) -> None:
        """
        Args:
            model:          Any classifier with a .predict_proba(X) method.
                            X shape: (n_samples, n_features).
                            Returns: (n_samples, n_classes) float array.
            feature_names:  Ordered list of feature column names.
                            Length must match X.shape[1] at attack time.
            feature_ranges: Dict mapping feature_name → (min_val, max_val).
                            Features not listed here are unclamped (no bounds
                            applied to their perturbations).

        Raises:
            TypeError:  model does not have a .predict_proba() method.
            ValueError: feature_names is empty.
        """
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError(
                f"Model {type(model).__name__} does not have a .predict_proba() method.\n"
                "→ For SVMs, set probability=True when constructing the classifier.\n"
                "→ For XGBoost, use XGBClassifier (not XGBRegressor).\n"
                "→ For plain decision trees, wrap in sklearn.calibration.CalibratedClassifierCV."
            )
        if not feature_names:
            raise ValueError("feature_names must not be empty.")

        self._model         = model
        self._feature_names = list(feature_names)
        self._feature_ranges = dict(feature_ranges)

        # Pre-build per-feature (min, max) arrays for fast vectorised clamping.
        # Features without a declared range get (-inf, +inf).
        n = len(feature_names)
        self._mins = np.full(n, -np.inf, dtype=np.float64)
        self._maxs = np.full(n,  np.inf, dtype=np.float64)
        for i, name in enumerate(feature_names):
            if name in feature_ranges:
                lo, hi = feature_ranges[name]
                self._mins[i] = lo
                self._maxs[i] = hi

    # ------------------------------------------------------------------
    # feature_perturbation_attack
    # ------------------------------------------------------------------

    def feature_perturbation_attack(
        self,
        X: np.ndarray,
        target_class: int,
        epsilon: float = 0.1,
        n_steps: int = 20,
        seed: Optional[int] = None,
    ) -> TabularAttackResult:
        """Gradient-free boundary attack on tabular data.

        For each row in X:
            1. Record the original prediction and P(target_class).
            2. Repeat for n_steps:
               a. Sample a random perturbation delta ~ Uniform(-epsilon, +epsilon)
                  scaled by the feature's allowed range (so epsilon=0.1 means
                  10% of the feature's declared range per step).
               b. Apply: x_candidate = x_current + delta
               c. Clamp to feature_ranges.
               d. If P(target_class | x_candidate) > P(target_class | x_current),
                  accept x_candidate (greedy hill-climb toward target class).
            3. Record final prediction and total perturbation applied.

        Args:
            X:            Input array, shape (n_samples, n_features).
                          Must match len(feature_names) in column count.
            target_class: Class index to drive the model toward.
            epsilon:      Per-step perturbation as a fraction of each feature's
                          declared range. Default 0.1 (10% per step).
                          For features without a declared range, epsilon is
                          applied as an absolute value (epsilon * 1.0).
            n_steps:      Number of greedy perturbation steps per sample.
            seed:         Optional random seed for reproducibility.

        Returns:
            TabularAttackResult with success rate, avg confidence, and top
            3 most-perturbed features.

        Raises:
            ValueError: X has wrong number of features, or target_class out of range.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_cols = X.shape
        if n_cols != len(self._feature_names):
            raise ValueError(
                f"X has {n_cols} columns but feature_names has "
                f"{len(self._feature_names)} entries."
            )

        # Validate target_class against model output shape
        proba_check = self._predict_proba(X[:1])
        n_classes = proba_check.shape[1]
        if not (0 <= target_class < n_classes):
            raise ValueError(
                f"target_class={target_class} out of range for model "
                f"with {n_classes} classes."
            )

        rng = np.random.default_rng(seed)

        # Per-feature perturbation scale: epsilon × feature_range_width
        # For unbounded features, scale = epsilon (absolute).
        scales = np.where(
            np.isfinite(self._maxs - self._mins),
            epsilon * (self._maxs - self._mins),
            epsilon,
        )  # (n_features,)

        successes         = 0
        total_conf        = 0.0
        reached_target    = False
        total_perturbation = np.zeros(n_cols, dtype=np.float64)  # |x_adv - x_orig| summed

        for row_idx in range(n_samples):
            x_orig = X[row_idx].copy()
            x_curr = x_orig.copy()

            orig_proba    = self._predict_proba(x_curr.reshape(1, -1))[0]
            orig_class    = int(orig_proba.argmax())
            curr_conf     = float(orig_proba[target_class])

            for _ in range(n_steps):
                # Random perturbation within [-scale, +scale] per feature
                delta       = rng.uniform(-1.0, 1.0, size=n_cols) * scales
                x_candidate = np.clip(x_curr + delta, self._mins, self._maxs)

                cand_proba = self._predict_proba(x_candidate.reshape(1, -1))[0]
                cand_conf  = float(cand_proba[target_class])

                # Greedy accept: keep if P(target_class) improved
                if cand_conf > curr_conf:
                    x_curr    = x_candidate
                    curr_conf = cand_conf

            # Final evaluation
            final_proba = self._predict_proba(x_curr.reshape(1, -1))[0]
            final_class = int(final_proba.argmax())

            if final_class != orig_class:
                successes += 1
            if final_class == target_class:
                reached_target = True
            total_conf        += float(final_proba[target_class])
            total_perturbation += np.abs(x_curr - x_orig)

        success_rate = successes / n_samples if n_samples > 0 else 0.0
        avg_conf     = total_conf / n_samples if n_samples > 0 else 0.0

        # Top-3 features by mean absolute perturbation
        mean_perturbation = total_perturbation / max(n_samples, 1)
        top3_indices      = np.argsort(mean_perturbation)[::-1][:3]
        most_perturbed    = [self._feature_names[i] for i in top3_indices
                             if mean_perturbation[i] > 0]

        return TabularAttackResult(
            success_rate=round(float(success_rate), 4),
            avg_confidence_on_target=round(float(avg_conf), 4),
            n_samples=n_samples,
            reached_target=reached_target,
            most_perturbed_features=most_perturbed,
            plain_english=_build_plain_english(
                success_rate, avg_conf, target_class, n_samples,
                epsilon, n_steps, most_perturbed,
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Call model.predict_proba() and return (n_samples, n_classes) array."""
        proba = self._model.predict_proba(X)
        return np.asarray(proba, dtype=np.float64)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _build_plain_english(
    success_rate: float,
    avg_conf: float,
    target_class: int,
    n: int,
    epsilon: float,
    n_steps: int,
    top_features: list[str],
) -> str:
    """One factual sentence. No drama, no exclamation marks."""
    feat_str = (
        f"most perturbed features: {', '.join(top_features)}"
        if top_features else "no features perturbed"
    )
    return (
        f"Gradient-free tabular attack (epsilon={epsilon}, {n_steps} steps) "
        f"on {n} sample{'s' if n != 1 else ''} achieved {success_rate:.0%} "
        f"success rate against class {target_class} with average confidence "
        f"{avg_conf:.2f}; {feat_str}."
    )
