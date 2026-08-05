"""Tiny trained demo model for KAAL examples.

Downloads a pretrained ResNet50 and saves it as demo_model.pt
for use in quick_start.py and other example scripts.
"""

import os
import torch
import torchvision.models as models


def get_demo_model(output_path: str = "./demo_model.pt") -> str:
    """Download ResNet50 pretrained and save to output_path.

    Returns the path to the saved model file.
    Skips download if file already exists.
    """
    if os.path.exists(output_path):
        print(f"Demo model already exists at {output_path}")
        return output_path

    print("Downloading ResNet50 pretrained weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    torch.save(model, output_path)
    print(f"Demo model saved to {output_path}")
    return output_path


if __name__ == "__main__":
    get_demo_model()
