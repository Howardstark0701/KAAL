"""GradCAM extractor — kaal/attacks/gradcam.py

A standalone, hook-based GradCAM implementation for any PyTorch nn.Module
loaded via KaalModel. No third-party grad-cam library required.

Algorithm (Selvaraju et al., 2017):
    1. Forward pass — capture activations from the last Conv2d layer.
    2. Backward pass on the target class score — capture gradients at
       that same layer.
    3. Global-average-pool the gradients across spatial dims → channel weights.
    4. Weighted sum of activations → raw CAM.
    5. ReLU → keep only positive contributions.
    6. Upsample to input (H, W) and normalise to [0, 1].

Public API
----------
GradCAMExtractor(model: KaalModel)
    .compute(image_tensor, target_class) -> np.ndarray  shape (H, W)
    .get_target_layer_name()             -> str

extract_saliency(model, image_tensor, target_class)
    -> tuple[np.ndarray, str]           (saliency_map_2d, layer_name)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaal.engine.loader import KaalModel


class GradCAMExtractor:
    """Hook-based GradCAM extractor for PyTorch models.

    Usage:
        extractor = GradCAMExtractor(kaal_model)
        saliency  = extractor.compute(image_tensor, target_class=42)
        layer     = extractor.get_target_layer_name()

    The hooks are registered fresh on every call to compute() and removed
    immediately after — no persistent state between calls.
    """

    def __init__(self, model: KaalModel) -> None:
        """
        Args:
            model: KaalModel loaded via kaal.engine.loader.load_model().
                   Must be a PyTorch model (framework == "pytorch").

        Raises:
            NotImplementedError: If model.framework is not "pytorch".
        """
        if model.framework != "pytorch":
            raise NotImplementedError(
                f"GradCAMExtractor requires a PyTorch model, "
                f"got '{model.framework}'.\n"
                "→ Convert your model to PyTorch (.pt / .pth) to use GradCAM.\n"
                "→ TensorFlow, ONNX, and TFLite models are not supported."
            )

        self._kaal_model = model
        self._raw_model  = model.model          # the underlying nn.Module
        self._layer_name, self._target_layer = self._find_last_conv()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute a GradCAM saliency map for one image.

        Args:
            image_tensor: Normalized torch.Tensor, shape (C, H, W) or
                          (1, C, H, W). Must be the ImageNet-normalised
                          tensor produced by load_dataset().
            target_class: Integer class index to compute GradCAM for.

        Returns:
            2D numpy array of shape (H, W) with dtype float32.
            Values are in [0, 1] — 1 = highest attention.

        Note:
            Hooks are registered at the start of this method and removed
            before it returns, even if an exception is raised.
        """
        # Squeeze batch dim
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        _, H, W = image_tensor.shape

        self._raw_model.eval()

        # Storage for hook outputs
        activations: list[torch.Tensor] = []
        gradients:   list[torch.Tensor] = []

        # Hook functions
        def _fwd_hook(module: nn.Module, inp, out: torch.Tensor) -> None:
            activations.append(out.detach().clone())

        def _bwd_hook(module: nn.Module, grad_in, grad_out) -> None:
            gradients.append(grad_out[0].detach().clone())

        # Register — capture handles so we can remove them in finally block
        h_fwd = self._target_layer.register_forward_hook(_fwd_hook)
        h_bwd = self._target_layer.register_full_backward_hook(_bwd_hook)

        try:
            # Forward pass
            inp    = image_tensor.unsqueeze(0).requires_grad_(True)
            logits = self._raw_model(inp)

            # Zero gradients from any previous pass
            self._raw_model.zero_grad()

            # Backward on the target class score only
            score = logits[0, target_class]
            score.backward()

        finally:
            # Always remove hooks — even on exception
            h_fwd.remove()
            h_bwd.remove()

        # Build CAM from captured tensors
        if not activations or not gradients:
            # Should never happen for a standard CNN, but guard defensively
            return np.zeros((H, W), dtype=np.float32)

        act  = activations[0].squeeze(0)    # (C, h, w)
        grad = gradients[0].squeeze(0)      # (C, h, w)

        # Global-average-pool gradients → per-channel importance weights
        weights = grad.mean(dim=(1, 2))     # (C,)

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * act).sum(dim=0)  # (h, w)

        # ReLU: only keep regions where increasing the target class score
        # positively correlated with activation
        cam = F.relu(cam)

        # Upsample to original input spatial size
        cam_4d  = cam.unsqueeze(0).unsqueeze(0)                 # (1,1,h,w)
        cam_up  = F.interpolate(
            cam_4d, size=(H, W), mode="bilinear", align_corners=False
        )
        cam_np  = cam_up.squeeze().cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max - c_min > 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            # Uniform activation — return zeros (model has no spatial preference)
            cam_np = np.zeros((H, W), dtype=np.float32)

        return cam_np

    def get_target_layer_name(self) -> str:
        """Return the dotted name of the Conv2d layer being hooked.

        Example return values:
            "layer4.1.conv2"       (ResNet)
            "features.28"          (VGG / MobileNet)
            "model.22"             (custom model)
        """
        return self._layer_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_last_conv(self) -> tuple[str, nn.Module]:
        """Walk all named modules and return (name, module) of the last Conv2d.

        Raises:
            RuntimeError: No Conv2d layer found in the model.
        """
        last_name: Optional[str]        = None
        last_module: Optional[nn.Module] = None

        for name, module in self._raw_model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name   = name
                last_module = module

        if last_module is None:
            raise RuntimeError(
                "No Conv2d layer found in this model.\n"
                "→ GradCAM requires a model with at least one convolutional layer.\n"
                "→ Fully-connected / transformer-only models are not supported."
            )

        return last_name, last_module


# ---------------------------------------------------------------------------
# Standalone convenience function
# ---------------------------------------------------------------------------

def extract_saliency(
    model: KaalModel,
    image_tensor: torch.Tensor,
    target_class: int,
) -> tuple[np.ndarray, str]:
    """Compute a GradCAM saliency map and return it with the layer name.

    Convenience wrapper around GradCAMExtractor for one-shot use.

    Args:
        model:        KaalModel (PyTorch only).
        image_tensor: Normalized tensor (C, H, W) or (1, C, H, W).
        target_class: Class index to compute GradCAM for.

    Returns:
        (saliency_map, layer_name) where:
            saliency_map — float32 numpy array of shape (H, W), values in [0, 1]
            layer_name   — dotted string name of the hooked Conv2d layer

    Raises:
        NotImplementedError: Model is not PyTorch.
        RuntimeError:        Model has no Conv2d layers.

    Example:
        saliency, layer = extract_saliency(model, tensor, target_class=42)
        print(f"Hooked layer: {layer}")
        print(f"Peak attention at pixel: {saliency.argmax()}")
    """
    extractor   = GradCAMExtractor(model)
    saliency    = extractor.compute(image_tensor, target_class)
    layer_name  = extractor.get_target_layer_name()
    return saliency, layer_name
