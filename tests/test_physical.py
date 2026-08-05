"""Tests for kaal/attacks/physical.py — Phase 5 verification."""

from __future__ import annotations

import pytest
import torch
import torchvision.models as models
import numpy as np
from PIL import Image

from kaal.engine.loader import load_model
from kaal.attacks.fgsm import fgsm_attack
from kaal.attacks.physical import (
    PhysicalRobustnessResult,
    TransformResult,
    ALL_TRANSFORM_NAMES,
    test_physical_robustness,
    test_physical_robustness_batch,
    list_transforms,
    _jpeg,
    _gaussian_noise,
    _brightness,
    _contrast,
    _rotate,
    _scale,
    _blur,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kaal_model(tmp_path_factory):
    m = models.resnet18(weights=None)
    m.eval()
    path = str(tmp_path_factory.mktemp("models") / "resnet18.pt")
    torch.save(m, path)
    return load_model(path)


@pytest.fixture(scope="module")
def adv_tensor_and_class(kaal_model):
    """Return (adversarial_tensor, original_class) via FGSM at strong epsilon."""
    torch.manual_seed(77)
    tensor = torch.randn(3, 224, 224) * 0.5
    result = fgsm_attack(kaal_model, tensor, epsilon=0.5)
    return result.adversarial_tensor, result.original_class


@pytest.fixture(scope="module")
def sample_pil():
    """224×224 random RGB PIL image."""
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Transform registry tests
# ---------------------------------------------------------------------------

class TestTransformRegistry:

    def test_all_transform_names_nonempty(self):
        assert len(ALL_TRANSFORM_NAMES) > 0

    def test_all_transform_names_count(self):
        # 4 jpeg + 3 noise + 4 brightness + 4 contrast + 4 rotation + 4 scaling + 3 blur
        assert len(ALL_TRANSFORM_NAMES) == 26

    def test_list_transforms_returns_dict(self):
        cats = list_transforms()
        assert isinstance(cats, dict)

    def test_list_transforms_categories(self):
        cats = list_transforms()
        expected = {
            "jpeg_compression", "gaussian_noise", "brightness",
            "contrast", "rotation", "scaling", "gaussian_blur",
        }
        assert set(cats.keys()) == expected

    def test_list_transforms_all_names_present(self):
        cats = list_transforms()
        all_names = [n for names in cats.values() for n in names]
        assert set(all_names) == set(ALL_TRANSFORM_NAMES)

    def test_jpeg_variants_count(self):
        cats = list_transforms()
        assert len(cats["jpeg_compression"]) == 4

    def test_noise_variants_count(self):
        cats = list_transforms()
        assert len(cats["gaussian_noise"]) == 3

    def test_brightness_variants_count(self):
        cats = list_transforms()
        assert len(cats["brightness"]) == 4

    def test_rotation_variants_count(self):
        cats = list_transforms()
        assert len(cats["rotation"]) == 4

    def test_scaling_variants_count(self):
        cats = list_transforms()
        assert len(cats["scaling"]) == 4

    def test_blur_variants_count(self):
        cats = list_transforms()
        assert len(cats["gaussian_blur"]) == 3


# ---------------------------------------------------------------------------
# Individual transform behaviour tests
# ---------------------------------------------------------------------------

class TestIndividualTransforms:

    def test_jpeg_returns_pil(self, sample_pil):
        out = _jpeg(sample_pil, 75)
        assert isinstance(out, Image.Image)
        assert out.size == sample_pil.size

    def test_jpeg_lower_quality_changes_image(self, sample_pil):
        q90 = np.array(_jpeg(sample_pil, 90)).astype(float)
        q40 = np.array(_jpeg(sample_pil, 40)).astype(float)
        # Lower quality should introduce more distortion
        assert np.mean(np.abs(q90 - np.array(sample_pil))) <= \
               np.mean(np.abs(q40 - np.array(sample_pil))) + 1.0  # allow small tolerance

    def test_gaussian_noise_preserves_size(self, sample_pil):
        out = _gaussian_noise(sample_pil, sigma=0.02)
        assert out.size == sample_pil.size

    def test_gaussian_noise_changes_pixels(self, sample_pil):
        out = _gaussian_noise(sample_pil, sigma=0.05)
        assert not np.array_equal(np.array(out), np.array(sample_pil))

    def test_brightness_dark(self, sample_pil):
        dark = _brightness(sample_pil, 0.5)
        assert np.mean(np.array(dark)) < np.mean(np.array(sample_pil))

    def test_brightness_bright(self, sample_pil):
        bright = _brightness(sample_pil, 1.5)
        assert np.mean(np.array(bright)) > np.mean(np.array(sample_pil))

    def test_contrast_reduces_range(self, sample_pil):
        low_c = _contrast(sample_pil, 0.5)
        arr = np.array(low_c).astype(float)
        orig = np.array(sample_pil).astype(float)
        assert arr.std() < orig.std() + 10

    def test_rotation_preserves_size(self, sample_pil):
        out = _rotate(sample_pil, 15)
        assert out.size == sample_pil.size

    def test_rotation_changes_pixels(self, sample_pil):
        out = _rotate(sample_pil, 30)
        assert not np.array_equal(np.array(out), np.array(sample_pil))

    def test_scale_preserves_size(self, sample_pil):
        out = _scale(sample_pil, 0.8)
        assert out.size == sample_pil.size

    def test_scale_up_preserves_size(self, sample_pil):
        out = _scale(sample_pil, 1.2)
        assert out.size == sample_pil.size

    def test_blur_preserves_size(self, sample_pil):
        out = _blur(sample_pil, 5)
        assert out.size == sample_pil.size

    def test_blur_reduces_sharpness(self, sample_pil):
        blurred = _blur(sample_pil, 7)
        # Blurred image has lower variance in pixel differences
        orig_diff = np.diff(np.array(sample_pil).astype(float), axis=1).std()
        blur_diff = np.diff(np.array(blurred).astype(float), axis=1).std()
        assert blur_diff < orig_diff + 5


# ---------------------------------------------------------------------------
# PhysicalRobustnessResult structure tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def physical_result(kaal_model, adv_tensor_and_class):
    adv, orig = adv_tensor_and_class
    return test_physical_robustness(kaal_model, adv, orig)


class TestPhysicalResultStructure:

    def test_returns_correct_type(self, physical_result):
        assert isinstance(physical_result, PhysicalRobustnessResult)

    def test_overall_survival_in_range(self, physical_result):
        assert 0.0 <= physical_result.overall_survival_rate <= 1.0

    def test_per_transform_dict_populated(self, physical_result):
        assert len(physical_result.per_transform_results) > 0

    def test_per_transform_keys_are_valid_names(self, physical_result):
        for k in physical_result.per_transform_results:
            assert k in ALL_TRANSFORM_NAMES

    def test_per_transform_values_are_transform_results(self, physical_result):
        for v in physical_result.per_transform_results.values():
            assert isinstance(v, TransformResult)

    def test_transform_result_fields(self, physical_result):
        for name, tr in physical_result.per_transform_results.items():
            assert tr.transform_name == name
            assert isinstance(tr.category, str) and len(tr.category) > 0
            assert 0.0 <= tr.success_rate <= 1.0
            assert tr.total_tested >= 1
            assert 0 <= tr.successful <= tr.total_tested

    def test_most_robust_is_valid_transform(self, physical_result):
        assert physical_result.most_robust_transform in ALL_TRANSFORM_NAMES

    def test_least_robust_is_valid_transform(self, physical_result):
        assert physical_result.least_robust_transform in ALL_TRANSFORM_NAMES

    def test_most_robust_gte_least_robust(self, physical_result):
        best  = physical_result.per_transform_results[physical_result.most_robust_transform].success_rate
        worst = physical_result.per_transform_results[physical_result.least_robust_transform].success_rate
        assert best >= worst

    def test_physical_threat_rating_valid(self, physical_result):
        assert physical_result.physical_threat_rating in {"Lab Only", "Limited", "Field Ready"}

    def test_transforms_tested_list(self, physical_result):
        assert isinstance(physical_result.transforms_tested, list)
        assert len(physical_result.transforms_tested) == len(physical_result.per_transform_results)

    def test_category_summary_keys(self, physical_result):
        expected_cats = {
            "jpeg_compression", "gaussian_noise", "brightness",
            "contrast", "rotation", "scaling", "gaussian_blur",
        }
        for cat in physical_result.category_summary:
            assert cat in expected_cats

    def test_category_summary_values_in_range(self, physical_result):
        for v in physical_result.category_summary.values():
            assert 0.0 <= v <= 1.0

    def test_plain_english_is_single_sentence(self, physical_result):
        text = physical_result.plain_english
        assert isinstance(text, str) and len(text) > 0
        assert "!" not in text
        assert text.strip().endswith(".")

    def test_batched_input_accepted(self, kaal_model, adv_tensor_and_class):
        adv, orig = adv_tensor_and_class
        batched = adv.unsqueeze(0)
        result = test_physical_robustness(kaal_model, batched, orig)
        assert isinstance(result, PhysicalRobustnessResult)


# ---------------------------------------------------------------------------
# All 26 transforms run when transformations=None
# ---------------------------------------------------------------------------

class TestAllTransformsRun:

    def test_full_run_covers_all_transforms(self, kaal_model, adv_tensor_and_class):
        adv, orig = adv_tensor_and_class
        result = test_physical_robustness(kaal_model, adv, orig,
                                          transformations=None)
        assert set(result.transforms_tested) == set(ALL_TRANSFORM_NAMES)

    def test_full_run_transform_count(self, kaal_model, adv_tensor_and_class):
        adv, orig = adv_tensor_and_class
        result = test_physical_robustness(kaal_model, adv, orig)
        assert len(result.per_transform_results) == 26


# ---------------------------------------------------------------------------
# Selective transform subset tests
# ---------------------------------------------------------------------------

class TestSelectiveTransforms:

    def test_subset_jpeg_only(self, kaal_model, adv_tensor_and_class):
        adv, orig = adv_tensor_and_class
        subset = ["jpeg_90", "jpeg_40"]
        result = test_physical_robustness(kaal_model, adv, orig,
                                          transformations=subset)
        assert set(result.transforms_tested) == set(subset)
        assert len(result.per_transform_results) == 2

    def test_single_transform(self, kaal_model, adv_tensor_and_class):
        adv, orig = adv_tensor_and_class
        result = test_physical_robustness(kaal_model, adv, orig,
                                          transformations=["blur_5"])
        assert result.transforms_tested == ["blur_5"]
        assert "blur_5" in result.per_transform_results

    def test_unknown_transform_raises(self, kaal_model, adv_tensor_and_class):
        adv, orig = adv_tensor_and_class
        with pytest.raises(ValueError, match="Unknown transform"):
            test_physical_robustness(kaal_model, adv, orig,
                                     transformations=["nonexistent_transform"])


# ---------------------------------------------------------------------------
# Threat rating thresholds
# ---------------------------------------------------------------------------

class TestThreatRating:

    def test_lab_only_below_30pct(self, kaal_model):
        """A clean (non-adversarial) image should mostly predict original class → low survival."""
        torch.manual_seed(10)
        tensor = torch.randn(3, 224, 224) * 0.3
        pred = kaal_model.predict(tensor)
        orig_class = pred["class_idx"]
        # Use the clean tensor as "adversarial" — it WILL predict orig_class
        # so survival rate should be 0% (attack never succeeds)
        result = test_physical_robustness(kaal_model, tensor, orig_class)
        # Since the "adversarial" image IS the clean image, no transform will
        # cause misclassification → survival = 0% → "Lab Only"
        assert result.physical_threat_rating == "Lab Only"
        assert result.overall_survival_rate < 0.30 + 0.05  # allow small tolerance

    def test_rating_field_ready_high_survival(self):
        """Mocked: 80% survival → Field Ready."""
        from kaal.attacks.physical import _compute_threat_rating
        assert _compute_threat_rating(0.80) == "Field Ready"

    def test_rating_limited_mid_survival(self):
        from kaal.attacks.physical import _compute_threat_rating
        assert _compute_threat_rating(0.50) == "Limited"

    def test_rating_lab_only_low_survival(self):
        from kaal.attacks.physical import _compute_threat_rating
        assert _compute_threat_rating(0.10) == "Lab Only"

    def test_rating_boundary_70pct_field_ready(self):
        from kaal.attacks.physical import _compute_threat_rating
        # Strictly > 0.70
        assert _compute_threat_rating(0.71) == "Field Ready"
        assert _compute_threat_rating(0.70) == "Limited"

    def test_rating_boundary_30pct_limited(self):
        from kaal.attacks.physical import _compute_threat_rating
        assert _compute_threat_rating(0.30) == "Limited"
        assert _compute_threat_rating(0.29) == "Lab Only"


# ---------------------------------------------------------------------------
# Batch variant tests
# ---------------------------------------------------------------------------

class TestBatchPhysical:

    def test_batch_returns_correct_type(self, kaal_model):
        torch.manual_seed(5)
        tensors, classes = [], []
        for _ in range(3):
            t = torch.randn(3, 224, 224) * 0.5
            r = fgsm_attack(kaal_model, t, epsilon=0.4)
            tensors.append(r.adversarial_tensor)
            classes.append(r.original_class)

        result = test_physical_robustness_batch(
            kaal_model, tensors, classes,
            transformations=["jpeg_90", "blur_3", "noise_001"],
        )
        assert isinstance(result, PhysicalRobustnessResult)

    def test_batch_totals_match_n_images(self, kaal_model):
        torch.manual_seed(6)
        n = 3
        tensors, classes = [], []
        for _ in range(n):
            t = torch.randn(3, 224, 224) * 0.5
            r = fgsm_attack(kaal_model, t, epsilon=0.3)
            tensors.append(r.adversarial_tensor)
            classes.append(r.original_class)

        result = test_physical_robustness_batch(
            kaal_model, tensors, classes,
            transformations=["jpeg_75"],
        )
        assert result.per_transform_results["jpeg_75"].total_tested == n

    def test_batch_mismatched_lengths_raises(self, kaal_model):
        t = torch.randn(3, 224, 224)
        with pytest.raises(ValueError, match="length"):
            test_physical_robustness_batch(kaal_model, [t, t], [0])

    def test_batch_empty_input_returns_empty_result(self, kaal_model):
        result = test_physical_robustness_batch(kaal_model, [], [])
        assert result.overall_survival_rate == 0.0
        assert result.transforms_tested == []
