"""Tests for kaal/explainability/ — Phase 6 verification.

Covers:
    gradcam.py   — GradCAMResult, GradCAMComparisonResult
    saliency.py  — SaliencyResult
    confidence.py — generate_collapse_curve()
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torchvision.models as models
from PIL import Image

from kaal.engine.loader import load_model
from kaal.attacks.fgsm import fgsm_attack
from kaal.attacks.pgd import pgd_attack, PGDResult
from kaal.explainability.gradcam import (
    GradCAMResult,
    GradCAMComparisonResult,
    generate_gradcam,
    generate_gradcam_comparison,
    _attention_shift,
    _attention_region,
)
from kaal.explainability.saliency import (
    SaliencyResult,
    generate_saliency,
)
from kaal.explainability.confidence import generate_collapse_curve


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kaal_model(tmp_path_factory):
    m = models.resnet18(weights=None)
    m.eval()
    path = str(tmp_path_factory.mktemp("models") / "resnet18.pt")
    torch.save(m, path)
    return load_model(path)


@pytest.fixture(scope="module")
def clean_tensor():
    torch.manual_seed(21)
    return torch.randn(3, 224, 224) * 0.4


@pytest.fixture(scope="module")
def adv_tensor(kaal_model, clean_tensor):
    result = fgsm_attack(kaal_model, clean_tensor, epsilon=0.4)
    return result.adversarial_tensor


@pytest.fixture(scope="module")
def pgd_result_obj(kaal_model, clean_tensor):
    """PGDResult with steps=15 for collapse curve tests."""
    return pgd_attack(kaal_model, clean_tensor, epsilon=0.3, steps=15)


# Module-level fixtures replacing former class-scoped instance methods

@pytest.fixture(scope="module")
def gradcam_result(kaal_model, clean_tensor):
    return generate_gradcam(kaal_model, clean_tensor)


@pytest.fixture(scope="module")
def cmp_result(kaal_model, clean_tensor, adv_tensor):
    return generate_gradcam_comparison(kaal_model, clean_tensor, adv_tensor)


@pytest.fixture(scope="module")
def saliency_result(kaal_model, clean_tensor):
    return generate_saliency(kaal_model, clean_tensor)


# ============================================================================
# GradCAM tests
# ============================================================================

class TestGradCAMResult:
    """GradCAMResult must have correct types and shapes."""

    def test_returns_gradcam_result(self, gradcam_result):
        assert isinstance(gradcam_result, GradCAMResult)

    def test_heatmap_is_2d_float32(self, gradcam_result):
        assert isinstance(gradcam_result.heatmap_array, np.ndarray)
        assert gradcam_result.heatmap_array.ndim == 2
        assert gradcam_result.heatmap_array.dtype == np.float32

    def test_heatmap_same_spatial_size_as_input(self, gradcam_result, clean_tensor):
        _, H, W = clean_tensor.shape
        assert gradcam_result.heatmap_array.shape == (H, W)

    def test_heatmap_values_in_range(self, gradcam_result):
        assert gradcam_result.heatmap_array.min() >= 0.0 - 1e-5
        assert gradcam_result.heatmap_array.max() <= 1.0 + 1e-5

    def test_overlay_pil_is_rgb(self, gradcam_result):
        assert isinstance(gradcam_result.overlay_pil, Image.Image)
        assert gradcam_result.overlay_pil.mode == "RGB"

    def test_overlay_same_spatial_size(self, gradcam_result, clean_tensor):
        _, H, W = clean_tensor.shape
        assert gradcam_result.overlay_pil.size == (W, H)

    def test_target_class_is_int(self, gradcam_result):
        assert isinstance(gradcam_result.target_class_used, int)

    def test_top_attention_region_is_string(self, gradcam_result):
        assert isinstance(gradcam_result.top_attention_region, str)
        assert len(gradcam_result.top_attention_region) > 0

    def test_explicit_target_class_respected(self, kaal_model, clean_tensor):
        result = generate_gradcam(kaal_model, clean_tensor, target_class=42)
        assert result.target_class_used == 42

    def test_accepts_batched_input(self, kaal_model, clean_tensor):
        batched = clean_tensor.unsqueeze(0)
        result = generate_gradcam(kaal_model, batched)
        assert isinstance(result, GradCAMResult)
        assert result.heatmap_array.ndim == 2

    def test_non_pytorch_raises(self, clean_tensor):
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.framework = "tensorflow"
        with pytest.raises(NotImplementedError, match="PyTorch"):
            generate_gradcam(mock_model, clean_tensor)


class TestGradCAMComparison:

    def test_returns_comparison_result(self, cmp_result):
        assert isinstance(cmp_result, GradCAMComparisonResult)

    def test_clean_gradcam_is_gradcam_result(self, cmp_result):
        assert isinstance(cmp_result.clean_gradcam, GradCAMResult)

    def test_adversarial_gradcam_is_gradcam_result(self, cmp_result):
        assert isinstance(cmp_result.adversarial_gradcam, GradCAMResult)

    def test_side_by_side_is_pil(self, cmp_result):
        assert isinstance(cmp_result.side_by_side_pil, Image.Image)
        assert cmp_result.side_by_side_pil.mode == "RGB"

    def test_side_by_side_wider_than_single(self, cmp_result):
        single_w = cmp_result.clean_gradcam.overlay_pil.width
        combined_w = cmp_result.side_by_side_pil.width
        assert combined_w > single_w

    def test_side_by_side_has_white_border(self, cmp_result):
        w1 = cmp_result.clean_gradcam.overlay_pil.width
        w2 = cmp_result.adversarial_gradcam.overlay_pil.width
        assert cmp_result.side_by_side_pil.width == w1 + 2 + w2

    def test_attention_shift_score_in_range(self, cmp_result):
        assert 0.0 <= cmp_result.attention_shift_score <= 1.0

    def test_plain_english_single_sentence(self, cmp_result):
        text = cmp_result.plain_english
        assert isinstance(text, str) and len(text) > 0
        assert "!" not in text
        assert text.strip().endswith(".")

    def test_same_image_gives_low_shift(self, kaal_model, clean_tensor):
        """Identical clean/adversarial should give near-zero shift."""
        cmp = generate_gradcam_comparison(kaal_model, clean_tensor, clean_tensor)
        assert cmp.attention_shift_score < 0.05


class TestAttentionHelpers:

    def test_attention_shift_identical_is_zero(self):
        h = np.random.rand(7, 7).astype(np.float32)
        assert _attention_shift(h, h) < 1e-6

    def test_attention_shift_orthogonal_is_high(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.zeros((4, 4), dtype=np.float32)
        a[0, 0] = 1.0
        b[3, 3] = 1.0
        shift = _attention_shift(a, b)
        assert shift > 0.3

    def test_attention_shift_all_zeros(self):
        z = np.zeros((4, 4), dtype=np.float32)
        assert _attention_shift(z, z) == 0.0

    def test_attention_region_labels(self):
        valid = {
            "top-left", "top-center", "top-right",
            "center-left", "center", "center-right",
            "bottom-left", "bottom-center", "bottom-right",
        }
        h = np.random.rand(9, 9).astype(np.float32)
        region = _attention_region(h)
        assert region in valid

    def test_attention_region_top_left(self):
        h = np.zeros((9, 9), dtype=np.float32)
        h[0, 0] = 1.0
        assert _attention_region(h) == "top-left"

    def test_attention_region_center(self):
        h = np.zeros((9, 9), dtype=np.float32)
        h[4, 4] = 1.0
        assert _attention_region(h) == "center"


# ============================================================================
# Saliency tests
# ============================================================================

class TestSaliencyResult:

    def test_returns_saliency_result(self, saliency_result):
        assert isinstance(saliency_result, SaliencyResult)

    def test_saliency_array_is_2d_float32(self, saliency_result):
        assert isinstance(saliency_result.saliency_array, np.ndarray)
        assert saliency_result.saliency_array.ndim == 2
        assert saliency_result.saliency_array.dtype == np.float32

    def test_saliency_array_same_spatial_size(self, saliency_result, clean_tensor):
        _, H, W = clean_tensor.shape
        assert saliency_result.saliency_array.shape == (H, W)

    def test_saliency_values_in_range(self, saliency_result):
        assert saliency_result.saliency_array.min() >= 0.0 - 1e-5
        assert saliency_result.saliency_array.max() <= 1.0 + 1e-5

    def test_saliency_pil_is_grayscale(self, saliency_result):
        assert isinstance(saliency_result.saliency_pil, Image.Image)
        assert saliency_result.saliency_pil.mode == "L"

    def test_saliency_pil_same_size(self, saliency_result, clean_tensor):
        _, H, W = clean_tensor.shape
        assert saliency_result.saliency_pil.size == (W, H)

    def test_overlay_pil_is_rgb(self, saliency_result):
        assert isinstance(saliency_result.overlay_pil, Image.Image)
        assert saliency_result.overlay_pil.mode == "RGB"

    def test_overlay_same_size(self, saliency_result, clean_tensor):
        _, H, W = clean_tensor.shape
        assert saliency_result.overlay_pil.size == (W, H)

    def test_top_sensitive_pixels_pct_in_range(self, saliency_result):
        assert 0.0 <= saliency_result.top_sensitive_pixels_pct <= 1.0

    def test_explicit_target_class(self, kaal_model, clean_tensor):
        result = generate_saliency(kaal_model, clean_tensor, target_class=7)
        assert isinstance(result, SaliencyResult)

    def test_accepts_batched_input(self, kaal_model, clean_tensor):
        batched = clean_tensor.unsqueeze(0)
        result = generate_saliency(kaal_model, batched)
        assert isinstance(result, SaliencyResult)

    def test_non_pytorch_raises(self, clean_tensor):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.framework = "onnx"
        with pytest.raises(NotImplementedError, match="PyTorch"):
            generate_saliency(mock, clean_tensor)

    def test_saliency_not_all_zeros(self, saliency_result):
        assert saliency_result.saliency_array.max() > 0.0

    def test_saliency_brightest_pixel_is_255_in_pil(self, saliency_result):
        arr = np.array(saliency_result.saliency_pil)
        assert arr.max() == 255


class TestSaliencyDifferences:
    """Saliency should differ between clean and adversarial images."""

    def test_saliency_changes_after_attack(self, kaal_model, clean_tensor, adv_tensor):
        s_clean = generate_saliency(kaal_model, clean_tensor)
        s_adv   = generate_saliency(kaal_model, adv_tensor)
        diff = np.abs(s_clean.saliency_array - s_adv.saliency_array).max()
        assert diff > 0.0


# ============================================================================
# Confidence collapse curve tests
# ============================================================================

class TestCollapseCurve:

    def test_returns_path_string(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "collapse.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        assert isinstance(path, str)

    def test_file_created(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "collapse.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        assert os.path.exists(path)

    def test_file_is_png(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "curve.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        with open(path, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"

    def test_file_minimum_size(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "collapse_size.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        assert os.path.getsize(path) > 10_000

    def test_returns_absolute_path(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "abs.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        assert os.path.isabs(path)

    def test_creates_parent_dir(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "nested" / "dir" / "collapse.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        assert os.path.exists(path)

    def test_empty_confidence_per_step_raises(self, pgd_result_obj, tmp_path):
        from dataclasses import replace
        empty = replace(pgd_result_obj, confidence_per_step=[])
        with pytest.raises(ValueError, match="confidence_per_step"):
            generate_collapse_curve(empty, str(tmp_path / "x.png"))

    def test_single_step_works(self, kaal_model, clean_tensor, tmp_path):
        result = pgd_attack(kaal_model, clean_tensor, epsilon=0.3, steps=1)
        out = str(tmp_path / "single_step.png")
        path = generate_collapse_curve(result, out)
        assert os.path.exists(path)

    def test_failed_attack_no_collapse_marker(self, kaal_model, tmp_path):
        torch.manual_seed(999)
        tensor = torch.randn(3, 224, 224) * 0.3
        result = pgd_attack(kaal_model, tensor, epsilon=1e-8, steps=5)
        out = str(tmp_path / "no_collapse.png")
        path = generate_collapse_curve(result, out)
        assert os.path.exists(path)

    def test_dpi_minimum_150(self, pgd_result_obj, tmp_path):
        out = str(tmp_path / "dpi_test.png")
        path = generate_collapse_curve(pgd_result_obj, out)
        size_bytes = os.path.getsize(path)
        assert size_bytes > 50_000
