"""Shared pytest fixtures for KAAL test suite."""

import pytest
import torch
import torchvision.models as models
from PIL import Image
import numpy as np


@pytest.fixture(scope="session")
def demo_model_path(tmp_path_factory):
    """Download and save a small ResNet18 as a demo model for tests."""
    model = models.resnet18(weights=None)  # no pretrained weights for speed
    # Minimal init — just test structure
    path = tmp_path_factory.mktemp("models") / "demo_model.pt"
    torch.save(model, str(path))
    return str(path)


@pytest.fixture(scope="session")
def demo_image():
    """Return a random 224x224 RGB PIL Image for testing."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture(scope="session")
def demo_image_dir(tmp_path_factory, demo_image):
    """Create a temp directory with 5 demo images."""
    img_dir = tmp_path_factory.mktemp("images")
    for i in range(5):
        demo_image.save(str(img_dir / f"test_{i}.jpg"))
    return str(img_dir)
