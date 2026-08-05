"""Tests for kaal/attacks/fgsm.py — Phase 2 verification."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torchvision.models as models
from PIL import Image
import numpy as np

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.fgsm import FGSMResult, fgsm_attack, fgsm_attack_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kaal_model(tmp_path_factory):
    """Small ResNet18 (no pretrained weights) saved as .pt for fast tests."""
    import torchvision.models as m
    model = m.resnet18(weights=None)
    model.eval()
    path = str(tmp_path_factory.mktemp("models") / "resnet18.pt")
    torch.save(model, path)
    return load_model(path)


@pytest.fixture(scope="module")
def random_tensor():
    """Random normalized-range tensor (3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(3, 224, 224) * 0.5


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    """Temp directory with 5 random JPEG images."""
    d = tmp_path_factory.mktemp("images")
    for i in range(5):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(d / f"img_{i}.jpg"))
    return str(d)


# ---------------------------------------------------------------------------
# FGSMResult structure tests
# ---------------------------------------------------------------------------

class TestFGSMResultStructure:
    """FGSMResult must contain all required fields with correct types."""

    def test_result_is_dataclass(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert isinstance(result, FGSMResult)

    def test_success_is_bool(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert isinstance(result.success, bool)

    def test_class_indices_are_int(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert isinstance(result.original_class, int)
        assert isinstance(result.adversarial_class, int)

    def test_confidences_are_float_in_range(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert isinstance(result.original_confidence, float)
        assert isinstance(result.adversarial_confidence, float)
        assert 0.0 <= result.original_confidence <= 1.0
        assert 0.0 <= result.adversarial_confidence <= 1.0

    def test_epsilon_recorded_correctly(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.05)
        assert result.epsilon_used == 0.05

    def test_adversarial_tensor_same_shape(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert result.adversarial_tensor.shape == random_tensor.shape

    def test_perturbation_tensor_same_shape(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert result.perturbation_tensor.shape == random_tensor.shape

    def test_adversarial_pil_is_image(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        assert isinstance(result.adversarial_pil, Image.Image)
        assert result.adversarial_pil.mode == "RGB"

    def test_plain_english_is_one_sentence(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        text = result.plain_english
        assert isinstance(text, str)
        assert len(text) > 0
        assert "!" not in text, "plain_english must not contain exclamation marks"
        # Should end with a period
        assert text.strip().endswith(".")

    def test_confidence_delta_sign(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        # Delta = original_confidence - post_attack_confidence_on_original
        # Can be positive (confidence dropped) or negative (unlikely but possible)
        assert isinstance(result.confidence_delta, float)


# ---------------------------------------------------------------------------
# Attack behaviour tests
# ---------------------------------------------------------------------------

class TestFGSMBehaviour:
    """FGSM must perturb the image and modify model predictions."""

    def test_adversarial_differs_from_original(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        diff = (result.adversarial_tensor - random_tensor).abs().max().item()
        assert diff > 0.0, "Adversarial tensor must differ from input"

    def test_perturbation_bounded_by_epsilon(self, kaal_model, random_tensor):
        eps = 0.05
        result = fgsm_attack(kaal_model, random_tensor, epsilon=eps)
        max_perturb = result.perturbation_tensor.abs().max().item()
        # Allow tiny floating-point tolerance
        assert max_perturb <= eps + 1e-5, (
            f"Max perturbation {max_perturb:.6f} exceeds epsilon {eps}"
        )

    def test_perturbation_tensor_matches_difference(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        computed = (result.adversarial_tensor - random_tensor).abs().max().item()
        stored   = result.perturbation_tensor.abs().max().item()
        assert abs(computed - stored) < 1e-5

    def test_success_matches_class_change(self, kaal_model, random_tensor):
        result = fgsm_attack(kaal_model, random_tensor, epsilon=0.03)
        expected_success = (result.adversarial_class != result.original_class)
        assert result.success == expected_success

    def test_strong_epsilon_increases_success(self, kaal_model):
        """Higher epsilon should make attack easier — test with 10 random images."""
        torch.manual_seed(0)
        successes_low  = 0
        successes_high = 0
        for _ in range(10):
            t = torch.randn(3, 224, 224) * 0.5
            successes_low  += fgsm_attack(kaal_model, t, epsilon=0.001).success
            successes_high += fgsm_attack(kaal_model, t, epsilon=0.3).success
        # High epsilon should succeed at least as often as low epsilon
        assert successes_high >= successes_low

    def test_batch_input_squeezed(self, kaal_model, random_tensor):
        """Attack should handle (1, C, H, W) input by squeezing batch dim."""
        batched = random_tensor.unsqueeze(0)
        result = fgsm_attack(kaal_model, batched, epsilon=0.03)
        assert result.adversarial_tensor.dim() == 3


# ---------------------------------------------------------------------------
# Targeted attack tests
# ---------------------------------------------------------------------------

class TestFGSMTargeted:
    def test_targeted_requires_target_class(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="target_class"):
            fgsm_attack(kaal_model, random_tensor, targeted=True, target_class=None)

    def test_targeted_success_definition(self, kaal_model, random_tensor):
        """Targeted attack succeeds only if adversarial_class == target_class."""
        target = 42
        result = fgsm_attack(
            kaal_model, random_tensor,
            epsilon=0.5,
            targeted=True,
            target_class=target,
        )
        if result.success:
            assert result.adversarial_class == target

    def test_targeted_plain_english_mentions_target(self, kaal_model, random_tensor):
        result = fgsm_attack(
            kaal_model, random_tensor,
            epsilon=0.3,
            targeted=True,
            target_class=99,
        )
        assert "99" in result.plain_english or "target" in result.plain_english.lower()


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestFGSMValidation:
    def test_invalid_epsilon_zero(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="epsilon"):
            fgsm_attack(kaal_model, random_tensor, epsilon=0.0)

    def test_invalid_epsilon_over_one(self, kaal_model, random_tensor):
        with pytest.raises(ValueError, match="epsilon"):
            fgsm_attack(kaal_model, random_tensor, epsilon=1.5)


# ---------------------------------------------------------------------------
# Dataset-level attack tests
# ---------------------------------------------------------------------------

class TestFGSMDataset:
    def test_attack_dataset_returns_correct_keys(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = fgsm_attack_dataset(kaal_model, dataset, epsilon=0.05)
        required_keys = {
            "results", "success_rate", "avg_confidence_delta",
            "epsilon_used", "total_images", "successful_attacks",
        }
        assert required_keys == set(agg.keys())

    def test_attack_dataset_total_images(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = fgsm_attack_dataset(kaal_model, dataset, epsilon=0.05)
        assert agg["total_images"] == len(dataset)

    def test_attack_dataset_success_rate_range(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = fgsm_attack_dataset(kaal_model, dataset, epsilon=0.05)
        assert 0.0 <= agg["success_rate"] <= 1.0

    def test_attack_dataset_max_images(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = fgsm_attack_dataset(kaal_model, dataset, epsilon=0.05, max_images=2)
        assert agg["total_images"] == 2

    def test_attack_dataset_results_are_fgsm_results(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        agg = fgsm_attack_dataset(kaal_model, dataset, epsilon=0.05)
        for r in agg["results"]:
            assert isinstance(r, FGSMResult)
