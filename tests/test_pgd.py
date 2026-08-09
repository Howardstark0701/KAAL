"""Tests for kaal/attacks/pgd.py — Phase 3 verification."""

from __future__ import annotations

import pytest
import torch
import torchvision.models as models
import numpy as np
from PIL import Image

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.pgd import PGDResult, pgd_attack, pgd_attack_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kaal_model(tmp_path_factory):
    model = models.resnet18(weights=None)
    model.eval()
    path = str(tmp_path_factory.mktemp("models") / "resnet18.pt")
    torch.save(model, path)
    return load_model(path)


@pytest.fixture(scope="module")
def random_tensor():
    torch.manual_seed(99)
    return torch.randn(3, 224, 224) * 0.5


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("images")
    for i in range(5):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(d / f"img_{i}.jpg"))
    return str(d)


# ---------------------------------------------------------------------------
# PGDResult structure tests
# ---------------------------------------------------------------------------

class TestPGDResultStructure:
    """Every field in PGDResult must exist with correct type."""

    def test_result_is_dataclass(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert isinstance(r, PGDResult)

    def test_success_is_bool(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert isinstance(r.success, bool)

    def test_class_indices_are_int(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert isinstance(r.original_class, int)
        assert isinstance(r.adversarial_class, int)

    def test_confidences_in_range(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert 0.0 <= r.original_confidence <= 1.0
        assert 0.0 <= r.adversarial_confidence <= 1.0

    def test_confidence_delta_is_float(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert isinstance(r.confidence_delta, float)

    def test_steps_to_success_is_int(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert isinstance(r.steps_to_success, int)

    def test_steps_to_success_is_minus_one_when_failed(self, kaal_model, random_tensor):
        """At epsilon=0 the attack cannot succeed — steps_to_success must be -1."""
        # Use tiny epsilon that forces failure
        r = pgd_attack(kaal_model, random_tensor, epsilon=1e-9, steps=3)
        if not r.success:
            assert r.steps_to_success == -1

    def test_confidence_per_step_length_matches_steps(self, kaal_model, random_tensor):
        steps = 7
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=steps)
        assert len(r.confidence_per_step) == steps

    def test_confidence_per_step_values_in_range(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        for v in r.confidence_per_step:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_epsilon_and_alpha_recorded(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, alpha=0.005, steps=5)
        assert r.epsilon_used == 0.05
        assert abs(r.alpha_used - 0.005) < 1e-9

    def test_default_alpha_is_epsilon_over_10(self, kaal_model, random_tensor):
        eps = 0.04
        r = pgd_attack(kaal_model, random_tensor, epsilon=eps, steps=5)
        assert abs(r.alpha_used - eps / 10.0) < 1e-9

    def test_steps_used_recorded(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=8)
        assert r.steps_used == 8

    def test_restarts_used_recorded(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5, restarts=2)
        assert r.restarts_used == 2

    def test_adversarial_tensor_shape(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert r.adversarial_tensor.shape == random_tensor.shape

    def test_adversarial_pil_is_rgb_image(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert isinstance(r.adversarial_pil, Image.Image)
        assert r.adversarial_pil.mode == "RGB"

    def test_plain_english_is_single_sentence(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        text = r.plain_english
        assert isinstance(text, str)
        assert len(text) > 0
        assert "!" not in text
        assert text.strip().endswith(".")

    def test_plain_english_mentions_epsilon(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert "ε=" in r.plain_english


# ---------------------------------------------------------------------------
# PGD behaviour tests
# ---------------------------------------------------------------------------

class TestPGDBehaviour:
    """PGD must obey the epsilon ball constraint and improve over FGSM."""

    def test_adversarial_differs_from_original(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        diff = (r.adversarial_tensor - random_tensor).abs().max().item()
        assert diff > 0.0

    def test_perturbation_bounded_by_epsilon(self, kaal_model, random_tensor):
        eps = 0.05
        r = pgd_attack(kaal_model, random_tensor, epsilon=eps, steps=10)
        max_perturb = (r.adversarial_tensor - random_tensor).abs().max().item()
        assert max_perturb <= eps + 1e-4, (
            f"Max perturbation {max_perturb:.6f} exceeds epsilon {eps}"
        )

    def test_success_consistent_with_class_change(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        expected = (r.adversarial_class != r.original_class)
        assert r.success == expected

    def test_steps_to_success_within_steps_range(self, kaal_model, random_tensor):
        steps = 10
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.3, steps=steps)
        if r.success:
            assert 1 <= r.steps_to_success <= steps

    def test_confidence_per_step_generally_decreasing(self, kaal_model, random_tensor):
        """For a successful attack, the last confidence should be lower than the first."""
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.3, steps=20)
        if r.success and len(r.confidence_per_step) >= 2:
            # Final confidence on original class should be lower than initial
            assert r.confidence_per_step[-1] <= r.confidence_per_step[0] + 0.1

    def test_pgd_succeeds_where_fgsm_fails(self, kaal_model):
        """PGD should achieve >= FGSM success rate at same epsilon on 10 images."""
        from kaal.attacks.fgsm import fgsm_attack
        torch.manual_seed(0)
        eps = 0.05
        pgd_wins, fgsm_wins, both_fail = 0, 0, 0

        for _ in range(10):
            t = torch.randn(3, 224, 224) * 0.5
            f_res = fgsm_attack(kaal_model, t, epsilon=eps)
            p_res = pgd_attack(kaal_model, t, epsilon=eps, steps=20)
            if p_res.success and not f_res.success:
                pgd_wins += 1
            elif f_res.success and not p_res.success:
                fgsm_wins += 1
            elif not f_res.success and not p_res.success:
                both_fail += 1

        # PGD should never lose to FGSM:
        # PGD wins ≥ FGSM wins (PGD is strictly stronger or equal)
        assert pgd_wins >= fgsm_wins, (
            f"FGSM beat PGD on {fgsm_wins} images where PGD failed — "
            "this indicates a PGD implementation bug"
        )

    def test_more_steps_not_worse(self, kaal_model, random_tensor):
        """40-step PGD should succeed at least as often as 5-step PGD."""
        torch.manual_seed(1)
        eps = 0.1
        success_5  = sum(
            pgd_attack(kaal_model, torch.randn(3, 224, 224) * 0.5, epsilon=eps, steps=5).success
            for _ in range(8)
        )
        success_40 = sum(
            pgd_attack(kaal_model, torch.randn(3, 224, 224) * 0.5, epsilon=eps, steps=40).success
            for _ in range(8)
        )
        assert success_40 >= success_5

    def test_batch_input_accepted(self, kaal_model, random_tensor):
        batched = random_tensor.unsqueeze(0)  # (1, C, H, W)
        r = pgd_attack(kaal_model, batched, epsilon=0.05, steps=5)
        assert r.adversarial_tensor.dim() == 3


# ---------------------------------------------------------------------------
# Random restarts tests
# ---------------------------------------------------------------------------

class TestPGDRestarts:
    def test_single_restart_default(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5)
        assert r.restarts_used == 1

    def test_multiple_restarts_recorded(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.05, steps=5, restarts=3)
        assert r.restarts_used == 3

    def test_multiple_restarts_produce_valid_result(self, kaal_model, random_tensor):
        r = pgd_attack(kaal_model, random_tensor, epsilon=0.1, steps=10, restarts=3)
        assert isinstance(r, PGDResult)
        assert r.adversarial_tensor.shape == random_tensor.shape

    def test_restarts_improve_or_equal_success(self, kaal_model):
        """More restarts should not reduce success rate."""
        torch.manual_seed(5)
        eps = 0.08
        n = 8

        success_1  = sum(
            pgd_attack(kaal_model, torch.randn(3, 224, 224) * 0.5,
                       epsilon=eps, steps=10, restarts=1).success
            for _ in range(n)
        )
        success_3  = sum(
            pgd_attack(kaal_model, torch.randn(3, 224, 224) * 0.5,
                       epsilon=eps, steps=10, restarts=3).success
            for _ in range(n)
        )
        assert success_3 >= success_1 - 1  # allow ±1 for randomness


# ---------------------------------------------------------------------------
# Targeted attack tests
# ---------------------------------------------------------------------------

class TestPGDTargeted:
    def test_targeted_requires_target_class(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="target_class"):
            pgd_attack(kaal_model, random_tensor, targeted=True, target_class=None)

    def test_targeted_success_means_correct_class(self, kaal_model, random_tensor):
        target = 7
        r = pgd_attack(
            kaal_model, random_tensor,
            epsilon=0.5, steps=5,   # fast: 5 steps only
            targeted=True, target_class=target,
        )
        if r.success:
            assert r.adversarial_class == target

    def test_targeted_plain_english_mentions_target(self, kaal_model, random_tensor):
        r = pgd_attack(
            kaal_model, random_tensor,
            epsilon=0.3, steps=3,   # fast: 3 steps only
            targeted=True, target_class=55,
        )
        assert "55" in r.plain_english or "target" in r.plain_english.lower()


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestPGDValidation:
    def test_invalid_epsilon_zero(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="epsilon"):
            pgd_attack(kaal_model, random_tensor, epsilon=0.0, steps=5)

    def test_invalid_epsilon_over_one(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="epsilon"):
            pgd_attack(kaal_model, random_tensor, epsilon=1.5, steps=5)

    def test_invalid_steps_zero(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="steps"):
            pgd_attack(kaal_model, random_tensor, epsilon=0.03, steps=0)

    def test_invalid_restarts_zero(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="restarts"):
            pgd_attack(kaal_model, random_tensor, epsilon=0.03, steps=5, restarts=0)

    def test_invalid_alpha_zero(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="alpha"):
            pgd_attack(kaal_model, random_tensor, epsilon=0.03, steps=5, alpha=0.0)

    def test_invalid_alpha_negative(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="alpha"):
            pgd_attack(kaal_model, random_tensor, epsilon=0.03, steps=5, alpha=-0.01)


# ---------------------------------------------------------------------------
# Dataset-level tests
# ---------------------------------------------------------------------------

class TestPGDDataset:
    # Use max_images=2 and steps=2 throughout — keeps tests fast on CPU
    _EPS   = 0.05
    _STEPS = 2
    _MAX   = 2

    def test_returns_required_keys(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = pgd_attack_dataset(kaal_model, dataset, epsilon=self._EPS,
                                 steps=self._STEPS, max_images=self._MAX)
        required = {
            "results", "success_rate", "avg_confidence_delta",
            "avg_steps_to_success", "epsilon_used", "alpha_used",
            "steps_used", "total_images", "successful_attacks",
        }
        assert required == set(agg.keys())

    def test_total_images_correct(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = pgd_attack_dataset(kaal_model, dataset, epsilon=self._EPS,
                                 steps=self._STEPS, max_images=self._MAX)
        assert agg["total_images"] == self._MAX

    def test_success_rate_in_range(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = pgd_attack_dataset(kaal_model, dataset, epsilon=self._EPS,
                                 steps=self._STEPS, max_images=self._MAX)
        assert 0.0 <= agg["success_rate"] <= 1.0

    def test_max_images_cap(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = pgd_attack_dataset(kaal_model, dataset, epsilon=self._EPS,
                                 steps=self._STEPS, max_images=2)
        assert agg["total_images"] == 2

    def test_results_are_pgd_results(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = pgd_attack_dataset(kaal_model, dataset, epsilon=self._EPS,
                                 steps=self._STEPS, max_images=self._MAX)
        for r in agg["results"]:
            assert isinstance(r, PGDResult)

    def test_confidence_per_step_all_populated(self, kaal_model, image_dir):
        steps = 3
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = pgd_attack_dataset(kaal_model, dataset, epsilon=self._EPS,
                                 steps=steps, max_images=self._MAX)
        for r in agg["results"]:
            assert len(r.confidence_per_step) == steps
