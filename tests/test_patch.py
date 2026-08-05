"""Tests for kaal/attacks/patch.py — Phase 4 verification."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torchvision.models as models
import numpy as np
from PIL import Image

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.attacks.patch import (
    PatchResult,
    apply_patch,
    generate_patch,
    patch_to_printable,
)


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
def image_dir(tmp_path_factory):
    """4 small images — enough to train a patch in minimal iterations."""
    d = tmp_path_factory.mktemp("images")
    for i in range(4):
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(d / f"img_{i}.jpg"))
    return str(d)


@pytest.fixture(scope="module")
def tiny_patch_tensor():
    """Small 10×10 normalized patch tensor for geometry tests."""
    torch.manual_seed(0)
    return torch.randn(3, 10, 10) * 0.1


@pytest.fixture(scope="module")
def sample_image_tensor():
    """224×224 normalized image tensor."""
    torch.manual_seed(1)
    return torch.randn(3, 224, 224) * 0.3


# ---------------------------------------------------------------------------
# apply_patch() tests
# ---------------------------------------------------------------------------

class TestApplyPatch:

    def test_returns_tensor_same_shape(self, sample_image_tensor, tiny_patch_tensor):
        result = apply_patch(sample_image_tensor, tiny_patch_tensor, x=10, y=10)
        assert result.shape == sample_image_tensor.shape

    def test_patch_region_modified(self, sample_image_tensor, tiny_patch_tensor):
        result = apply_patch(sample_image_tensor, tiny_patch_tensor, x=5, y=5)
        ph, pw = tiny_patch_tensor.shape[1], tiny_patch_tensor.shape[2]
        # The patched region should equal the patch
        patched_region = result[:, 5:5+ph, 5:5+pw]
        assert torch.allclose(patched_region, tiny_patch_tensor[:, :ph, :pw])

    def test_unpatched_region_unchanged(self, sample_image_tensor, tiny_patch_tensor):
        x, y = 50, 50
        result = apply_patch(sample_image_tensor, tiny_patch_tensor, x=x, y=y)
        ph, pw = tiny_patch_tensor.shape[1], tiny_patch_tensor.shape[2]
        # Region before patch should be untouched
        assert torch.allclose(result[:, :y, :], sample_image_tensor[:, :y, :])

    def test_does_not_modify_original(self, sample_image_tensor, tiny_patch_tensor):
        original_copy = sample_image_tensor.clone()
        apply_patch(sample_image_tensor, tiny_patch_tensor, x=0, y=0)
        assert torch.allclose(sample_image_tensor, original_copy)

    def test_accepts_batched_image(self, sample_image_tensor, tiny_patch_tensor):
        # apply_patch accepts (1, C, H, W) and returns the same shape back
        batched = sample_image_tensor.unsqueeze(0)  # (1, C, H, W)
        result = apply_patch(batched, tiny_patch_tensor, x=0, y=0)
        # Batch dim preserved — output shape matches input shape
        assert result.shape == batched.shape

    def test_patch_at_origin(self, sample_image_tensor, tiny_patch_tensor):
        result = apply_patch(sample_image_tensor, tiny_patch_tensor, x=0, y=0)
        ph, pw = tiny_patch_tensor.shape[1], tiny_patch_tensor.shape[2]
        assert torch.allclose(result[:, :ph, :pw], tiny_patch_tensor)

    def test_patch_clipped_at_boundary(self, sample_image_tensor, tiny_patch_tensor):
        """Patch placed at image boundary should not raise — clips gracefully."""
        _, img_h, img_w = sample_image_tensor.shape
        result = apply_patch(sample_image_tensor, tiny_patch_tensor,
                             x=img_w - 3, y=img_h - 3)
        assert result.shape == sample_image_tensor.shape

    def test_patch_fully_out_of_bounds_returns_unchanged(
            self, sample_image_tensor, tiny_patch_tensor):
        _, img_h, img_w = sample_image_tensor.shape
        result = apply_patch(sample_image_tensor, tiny_patch_tensor,
                             x=img_w + 10, y=img_h + 10)
        assert torch.allclose(result, sample_image_tensor)

    def test_patch_with_different_sizes(self, sample_image_tensor):
        # Large patch covering most of image
        large_patch = torch.ones(3, 200, 200)
        result = apply_patch(sample_image_tensor, large_patch, x=0, y=0)
        assert result.shape == sample_image_tensor.shape
        # Top-left 200×200 region should all be 1.0
        assert torch.allclose(result[:, :200, :200], large_patch[:, :200, :200])


# ---------------------------------------------------------------------------
# PatchResult structure tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patch_result(kaal_model, image_dir, tmp_path_factory):
    """Train a tiny patch (5 iterations) for fast structure testing."""
    dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
    out = str(tmp_path_factory.mktemp("patch_out"))
    return generate_patch(
        kaal_model, dataset,
        target_class=42,
        patch_fraction=0.05,
        iterations=5,
        output_dir=out,
        verbose=False,
    )


class TestPatchResultStructure:
    """PatchResult must contain all required fields with correct types."""

    def test_result_is_patchresult(self, patch_result):
        assert isinstance(patch_result, PatchResult)

    def test_patch_tensor_is_3d(self, patch_result):
        assert patch_result.patch_tensor.dim() == 3
        assert patch_result.patch_tensor.shape[0] == 3  # 3 channels

    def test_patch_pil_is_rgb(self, patch_result):
        assert isinstance(patch_result.patch_pil, Image.Image)
        assert patch_result.patch_pil.mode == "RGB"

    def test_success_rate_in_range(self, patch_result):
        assert 0.0 <= patch_result.attack_success_rate <= 1.0

    def test_avg_confidence_in_range(self, patch_result):
        assert 0.0 <= patch_result.avg_confidence_on_target <= 1.0

    def test_target_class_recorded(self, patch_result):
        assert patch_result.target_class == 42

    def test_patch_fraction_recorded(self, patch_result):
        assert patch_result.patch_fraction_used == 0.05

    def test_iterations_recorded(self, patch_result):
        assert patch_result.iterations_used == 5

    def test_plain_english_is_one_sentence(self, patch_result):
        text = patch_result.plain_english
        assert isinstance(text, str)
        assert len(text) > 0
        assert "!" not in text
        assert text.strip().endswith(".")

    def test_plain_english_mentions_target_class(self, patch_result):
        assert "42" in patch_result.plain_english

    def test_plain_english_mentions_iterations(self, patch_result):
        assert "5" in patch_result.plain_english

    def test_pdf_path_exists_when_output_dir_given(self, patch_result):
        assert patch_result.patch_printable_pdf_path != ""
        assert os.path.exists(patch_result.patch_printable_pdf_path)
        assert patch_result.patch_printable_pdf_path.endswith(".pdf")

    def test_pdf_path_empty_when_no_output_dir(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        result = generate_patch(
            kaal_model, dataset,
            target_class=10,
            patch_fraction=0.05,
            iterations=3,
            output_dir=None,
            verbose=False,
        )
        assert result.patch_printable_pdf_path == ""


# ---------------------------------------------------------------------------
# patch_fraction → patch size tests
# ---------------------------------------------------------------------------

class TestPatchSize:

    def test_larger_fraction_gives_larger_patch(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)

        r_small = generate_patch(kaal_model, dataset, target_class=0,
                                 patch_fraction=0.02, iterations=2, verbose=False)
        r_large = generate_patch(kaal_model, dataset, target_class=0,
                                 patch_fraction=0.15, iterations=2, verbose=False)

        small_area = r_small.patch_tensor.shape[1] * r_small.patch_tensor.shape[2]
        large_area = r_large.patch_tensor.shape[1] * r_large.patch_tensor.shape[2]
        assert large_area > small_area

    def test_patch_tensor_pixel_range(self, kaal_model, image_dir):
        """Patch should stay within valid normalized pixel range after training."""
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        r = generate_patch(kaal_model, dataset, target_class=0,
                           patch_fraction=0.05, iterations=5, verbose=False)
        # Denormalized patch should be in [0, 1]
        from kaal.engine.utils import denormalize
        raw = denormalize(r.patch_tensor)
        assert raw.min().item() >= -1e-4
        assert raw.max().item() <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestPatchValidation:

    def test_invalid_patch_fraction_zero(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        with pytest.raises(ValueError, match="patch_fraction"):
            generate_patch(kaal_model, dataset, target_class=0,
                           patch_fraction=0.0, iterations=5, verbose=False)

    def test_invalid_patch_fraction_over_half(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        with pytest.raises(ValueError, match="patch_fraction"):
            generate_patch(kaal_model, dataset, target_class=0,
                           patch_fraction=0.9, iterations=5, verbose=False)

    def test_invalid_iterations_zero(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        with pytest.raises(ValueError, match="iterations"):
            generate_patch(kaal_model, dataset, target_class=0,
                           patch_fraction=0.05, iterations=0, verbose=False)

    def test_invalid_learning_rate(self, kaal_model, image_dir):
        dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
        with pytest.raises(ValueError, match="learning_rate"):
            generate_patch(kaal_model, dataset, target_class=0,
                           patch_fraction=0.05, iterations=5,
                           learning_rate=-0.01, verbose=False)


# ---------------------------------------------------------------------------
# patch_to_printable() tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_patch(kaal_model, image_dir):
    dataset = load_dataset(image_dir, input_shape=kaal_model.input_shape)
    result = generate_patch(kaal_model, dataset, target_class=5,
                            patch_fraction=0.05, iterations=3,
                            output_dir=None, verbose=False)
    return result.patch_tensor


class TestPatchToPrintable:

    def test_returns_path_string(self, trained_patch, tmp_path):
        out = str(tmp_path / "test_patch.pdf")
        path = patch_to_printable(trained_patch, size_cm=10.0, output_path=out)
        assert isinstance(path, str)

    def test_pdf_file_created(self, trained_patch, tmp_path):
        out = str(tmp_path / "patch.pdf")
        path = patch_to_printable(trained_patch, size_cm=10.0, output_path=out)
        assert os.path.exists(path)

    def test_pdf_is_valid_pdf(self, trained_patch, tmp_path):
        """Check file starts with the PDF magic bytes %PDF-."""
        out = str(tmp_path / "patch_valid.pdf")
        path = patch_to_printable(trained_patch, size_cm=10.0, output_path=out)
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-", f"File does not start with PDF header: {header}"

    def test_pdf_nonzero_size(self, trained_patch, tmp_path):
        out = str(tmp_path / "patch_size.pdf")
        path = patch_to_printable(trained_patch, size_cm=10.0, output_path=out)
        assert os.path.getsize(path) > 1000  # at least 1KB

    def test_different_sizes_produce_different_files(self, trained_patch, tmp_path):
        out_a = str(tmp_path / "patch_10.pdf")
        out_b = str(tmp_path / "patch_20.pdf")
        patch_to_printable(trained_patch, size_cm=10.0, output_path=out_a)
        patch_to_printable(trained_patch, size_cm=20.0, output_path=out_b)
        # Larger physical size → larger file (more image data)
        assert os.path.getsize(out_b) >= os.path.getsize(out_a)

    def test_invalid_size_cm_zero(self, trained_patch, tmp_path):
        with pytest.raises(ValueError, match="size_cm"):
            patch_to_printable(trained_patch, size_cm=0.0,
                               output_path=str(tmp_path / "x.pdf"))

    def test_invalid_dpi_too_low(self, trained_patch, tmp_path):
        with pytest.raises(ValueError, match="dpi"):
            patch_to_printable(trained_patch, size_cm=10.0, dpi=10,
                               output_path=str(tmp_path / "x.pdf"))

    def test_default_output_path(self, trained_patch, tmp_path):
        """When output_path=None, should save to ./patch_print.pdf."""
        import os
        orig_dir = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            path = patch_to_printable(trained_patch, size_cm=10.0, output_path=None)
            assert os.path.exists(path)
            assert path.endswith("patch_print.pdf")
        finally:
            os.chdir(orig_dir)
