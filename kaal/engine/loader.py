"""Model loader — Spec 9.1 (Phase 1, Kiro Prompt 1.2).

Accepts any supported model file and returns a unified KaalModel
interface regardless of framework.

Supported formats:
    .h5 / .keras   — Keras / TensorFlow SavedModel
    .pt / .pth     — PyTorch
    .onnx          — ONNX
    .tflite        — TensorFlow Lite
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# KaalModel — unified interface
# ---------------------------------------------------------------------------

class KaalModel:
    """Unified model interface returned by load_model().

    Wraps any supported framework and exposes a consistent API
    for predictions and gradient computation.
    """

    def __init__(
        self,
        framework: str,
        model_obj,
        input_shape: tuple,
        num_classes: int,
        class_names: Optional[list[str]] = None,
    ):
        self._framework = framework
        self._model = model_obj
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._class_names = class_names or [str(i) for i in range(num_classes)]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def framework(self) -> str:
        """Framework identifier: 'pytorch' | 'tensorflow' | 'onnx' | 'tflite'."""
        return self._framework

    @property
    def input_shape(self) -> tuple:
        """Model input shape.

        PyTorch:     (C, H, W)  e.g. (3, 224, 224)
        TensorFlow:  (H, W, C)  e.g. (224, 224, 3)
        """
        return self._input_shape

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def model(self):
        """Raw underlying model object."""
        return self._model

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(self, image_tensor: torch.Tensor) -> dict:
        """Run inference and return structured prediction dict.

        Args:
            image_tensor: Normalized torch.Tensor.
                          Shape: (C, H, W) or (1, C, H, W).

        Returns:
            dict with keys:
                "class_idx":       int
                "class_name":      str
                "confidence":      float  (0.0 – 1.0)
                "all_confidences": list[float]
        """
        if self._framework == "pytorch":
            return self._predict_pytorch(image_tensor)
        elif self._framework == "tensorflow":
            return self._predict_tensorflow(image_tensor)
        elif self._framework == "onnx":
            return self._predict_onnx(image_tensor)
        elif self._framework == "tflite":
            return self._predict_tflite(image_tensor)
        else:
            raise RuntimeError(f"Unknown framework: {self._framework}")

    def _predict_pytorch(self, image_tensor: torch.Tensor) -> dict:
        import torch.nn.functional as F

        self._model.eval()
        with torch.no_grad():
            inp = _ensure_batch(image_tensor)
            logits = self._model(inp)
            probs = F.softmax(logits, dim=1).squeeze(0)

        all_conf = probs.cpu().tolist()
        class_idx = int(probs.argmax().item())
        return {
            "class_idx": class_idx,
            "class_name": self._class_names[class_idx],
            "confidence": float(all_conf[class_idx]),
            "all_confidences": all_conf,
        }

    def _predict_tensorflow(self, image_tensor: torch.Tensor) -> dict:
        import tensorflow as tf

        # Convert torch tensor → numpy → TF tensor
        np_img = _torch_to_numpy(image_tensor, channel_last=True)
        tf_input = tf.convert_to_tensor(np_img[np.newaxis], dtype=tf.float32)
        logits = self._model(tf_input, training=False)
        probs = tf.nn.softmax(logits).numpy().squeeze()

        all_conf = probs.tolist()
        class_idx = int(np.argmax(probs))
        return {
            "class_idx": class_idx,
            "class_name": self._class_names[class_idx],
            "confidence": float(all_conf[class_idx]),
            "all_confidences": all_conf,
        }

    def _predict_onnx(self, image_tensor: torch.Tensor) -> dict:
        import onnxruntime as ort
        import torch.nn.functional as F

        np_img = _ensure_batch(image_tensor).numpy().astype(np.float32)
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: np_img})
        logits = torch.tensor(outputs[0])
        probs = F.softmax(logits, dim=1).squeeze(0)

        all_conf = probs.tolist()
        class_idx = int(probs.argmax().item())
        return {
            "class_idx": class_idx,
            "class_name": self._class_names[class_idx],
            "confidence": float(all_conf[class_idx]),
            "all_confidences": all_conf,
        }

    def _predict_tflite(self, image_tensor: torch.Tensor) -> dict:
        import tensorflow as tf
        import torch.nn.functional as F

        interpreter = self._model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        np_img = _torch_to_numpy(image_tensor, channel_last=True)
        np_img = np_img[np.newaxis].astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], np_img)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]["index"])

        probs_t = torch.tensor(logits)
        probs = F.softmax(probs_t, dim=1).squeeze(0)

        all_conf = probs.tolist()
        class_idx = int(probs.argmax().item())
        return {
            "class_idx": class_idx,
            "class_name": self._class_names[class_idx],
            "confidence": float(all_conf[class_idx]),
            "all_confidences": all_conf,
        }

    # ------------------------------------------------------------------
    # gradient()
    # ------------------------------------------------------------------

    def gradient(
        self, image_tensor: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        """Compute gradient of loss w.r.t. input pixels.

        Args:
            image_tensor: Normalized torch.Tensor, shape (C, H, W) or (1, C, H, W).
            target_class: Class index to compute gradient for.

        Returns:
            torch.Tensor with same shape as image_tensor (without batch dim).
        """
        if self._framework == "pytorch":
            return self._gradient_pytorch(image_tensor, target_class)
        elif self._framework == "tensorflow":
            return self._gradient_tensorflow(image_tensor, target_class)
        else:
            raise NotImplementedError(
                f"gradient() not supported for framework '{self._framework}'. "
                "Use a PyTorch or TensorFlow model for gradient-based attacks."
            )

    def _gradient_pytorch(
        self, image_tensor: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        import torch.nn.functional as F

        self._model.eval()
        inp = _ensure_batch(image_tensor).clone().requires_grad_(True)
        logits = self._model(inp)
        loss = F.cross_entropy(logits, torch.tensor([target_class]))
        self._model.zero_grad()
        loss.backward()
        grad = inp.grad.data.squeeze(0)  # remove batch dim
        return grad

    def _gradient_tensorflow(
        self, image_tensor: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        import tensorflow as tf

        np_img = _torch_to_numpy(image_tensor, channel_last=True)
        tf_input = tf.Variable(
            tf.convert_to_tensor(np_img[np.newaxis], dtype=tf.float32)
        )

        with tf.GradientTape() as tape:
            logits = self._model(tf_input, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                [target_class], logits
            )

        grad_np = tape.gradient(loss, tf_input).numpy().squeeze(0)
        # Convert channel-last → channel-first for consistency
        grad_np = np.transpose(grad_np, (2, 0, 1))
        return torch.tensor(grad_np)


# ---------------------------------------------------------------------------
# load_model() — public entry point
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    class_names: Optional[list[str]] = None,
    num_classes: Optional[int] = None,
) -> KaalModel:
    """Load a model from file and return a unified KaalModel object.

    Args:
        model_path:   Path to model file.
                      Supported: .h5, .keras, .pt, .pth, .onnx, .tflite
        class_names:  Optional list of class name strings.
                      If None, classes are named by integer index.
        num_classes:  Optional override for number of output classes.

    Returns:
        KaalModel instance with .predict(), .gradient(), .framework,
        and .input_shape attributes.

    Raises:
        FileNotFoundError: Model file does not exist.
        ValueError:        File extension not supported.
        RuntimeError:      Model failed to load (corrupted or incompatible).
    """
    path = Path(model_path)

    # --- Validate path -------------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            "→ Check the path is correct and the file exists."
        )

    ext = path.suffix.lower()
    supported = {".h5", ".keras", ".pt", ".pth", ".onnx", ".tflite"}
    if ext not in supported:
        raise ValueError(
            f"Unsupported model format: '{ext}'\n"
            f"→ Supported formats: {', '.join(sorted(supported))}\n"
            "→ Convert your model to one of these formats first."
        )

    # --- Route to framework loader -------------------------------------------
    try:
        if ext in {".pt", ".pth"}:
            return _load_pytorch(str(path), class_names, num_classes)
        elif ext in {".h5", ".keras"}:
            return _load_tensorflow(str(path), class_names, num_classes)
        elif ext == ".onnx":
            return _load_onnx(str(path), class_names, num_classes)
        elif ext == ".tflite":
            return _load_tflite(str(path), class_names, num_classes)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{model_path}': {exc}\n"
            "→ Verify the file is not corrupted.\n"
            "→ Ensure the model was saved in a compatible format.\n"
            f"→ Framework expected: {_ext_to_framework(ext)}"
        ) from exc


# ---------------------------------------------------------------------------
# Framework-specific loaders
# ---------------------------------------------------------------------------

def _load_pytorch(
    path: str,
    class_names: Optional[list[str]],
    num_classes: Optional[int],
) -> KaalModel:
    """Load PyTorch .pt / .pth model."""
    try:
        # Try loading as full model first (torch.save(model, path))
        model = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        raise RuntimeError(
            f"Could not load PyTorch model from '{path}'.\n"
            "→ Ensure the file was saved with torch.save(model, path) "
            "or torch.save(model.state_dict(), path).\n"
            "→ If saving state_dict, you must provide the model architecture separately."
        )

    if isinstance(model, dict):
        raise RuntimeError(
            f"'{path}' appears to contain a state_dict, not a full model.\n"
            "→ KAAL requires a complete model object saved with torch.save(model, path).\n"
            "→ Load your architecture and call model.load_state_dict(torch.load(path)) first,\n"
            "  then save the full model: torch.save(model, 'model_full.pt')"
        )

    if not isinstance(model, nn.Module):
        raise RuntimeError(
            f"'{path}' does not contain a PyTorch nn.Module.\n"
            "→ KAAL expects a torch.nn.Module saved with torch.save(model, path)."
        )

    model.eval()

    input_shape, n_classes = _infer_pytorch_shape(model)
    if num_classes is not None:
        n_classes = num_classes
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    return KaalModel(
        framework="pytorch",
        model_obj=model,
        input_shape=input_shape,
        num_classes=n_classes,
        class_names=class_names,
    )


def _load_tensorflow(
    path: str,
    class_names: Optional[list[str]],
    num_classes: Optional[int],
) -> KaalModel:
    """Load TensorFlow / Keras .h5 / .keras model."""
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError(
            "TensorFlow is not installed.\n"
            "→ Install it with: pip install tensorflow==2.15.0"
        )

    try:
        model = tf.keras.models.load_model(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TensorFlow/Keras model from '{path}': {exc}\n"
            "→ Ensure the model was saved with model.save('path.h5') or model.save('path.keras')."
        )

    input_shape, n_classes = _infer_tf_shape(model)
    if num_classes is not None:
        n_classes = num_classes
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    return KaalModel(
        framework="tensorflow",
        model_obj=model,
        input_shape=input_shape,
        num_classes=n_classes,
        class_names=class_names,
    )


def _load_onnx(
    path: str,
    class_names: Optional[list[str]],
    num_classes: Optional[int],
) -> KaalModel:
    """Load ONNX model via onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime is not installed.\n"
            "→ Install it with: pip install onnxruntime==1.17.0"
        )

    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ONNX model from '{path}': {exc}\n"
            "→ Verify the ONNX file is valid (use onnx.checker.check_model)."
        )

    input_info = session.get_inputs()[0]
    shape = tuple(input_info.shape[1:])  # strip batch dim: [1, C, H, W] → (C, H, W)

    output_info = session.get_outputs()[0]
    n_classes = num_classes or (output_info.shape[-1] if output_info.shape else 1000)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    return KaalModel(
        framework="onnx",
        model_obj=session,
        input_shape=shape,
        num_classes=n_classes,
        class_names=class_names,
    )


def _load_tflite(
    path: str,
    class_names: Optional[list[str]],
    num_classes: Optional[int],
) -> KaalModel:
    """Load TensorFlow Lite model."""
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError(
            "TensorFlow is not installed.\n"
            "→ Install it with: pip install tensorflow==2.15.0"
        )

    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TFLite model from '{path}': {exc}\n"
            "→ Verify the .tflite file is valid and not corrupted."
        )

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # TFLite input: [1, H, W, C] channel-last — convert to (C, H, W)
    shape_hwc = tuple(input_details[0]["shape"][1:])  # (H, W, C)
    shape_chw = (shape_hwc[2], shape_hwc[0], shape_hwc[1])  # (C, H, W)

    n_classes = num_classes or int(output_details[0]["shape"][-1])

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    return KaalModel(
        framework="tflite",
        model_obj=interpreter,
        input_shape=shape_chw,
        num_classes=n_classes,
        class_names=class_names,
    )


# ---------------------------------------------------------------------------
# Shape inference helpers
# ---------------------------------------------------------------------------

def _infer_pytorch_shape(model: nn.Module) -> tuple[tuple, int]:
    """Infer (input_shape, num_classes) from a PyTorch model.

    Tries a dummy forward pass with standard sizes.
    Falls back to (3, 224, 224) and 1000 classes if inference fails.
    """
    model.eval()
    for h, w in [(224, 224), (299, 299), (384, 384), (512, 512)]:
        try:
            dummy = torch.zeros(1, 3, h, w)
            with torch.no_grad():
                out = model(dummy)
            n_classes = out.shape[-1]
            return (3, h, w), n_classes
        except Exception:
            continue

    # Fallback
    return (3, 224, 224), 1000


def _infer_tf_shape(model) -> tuple[tuple, int]:
    """Infer (input_shape, num_classes) from a TF/Keras model.

    Returns shape in TF convention: (H, W, C).
    """
    input_shape = model.input_shape  # e.g. (None, 224, 224, 3)
    output_shape = model.output_shape  # e.g. (None, 1000)

    h = int(input_shape[1]) if input_shape[1] else 224
    w = int(input_shape[2]) if input_shape[2] else 224
    c = int(input_shape[3]) if input_shape[3] else 3

    n_classes = int(output_shape[-1]) if output_shape[-1] else 1000
    return (h, w, c), n_classes


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has a batch dimension: (C,H,W) → (1,C,H,W)."""
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def _torch_to_numpy(tensor: torch.Tensor, channel_last: bool = False) -> np.ndarray:
    """Convert torch Tensor (C,H,W) to numpy array.

    Args:
        channel_last: If True, output is (H,W,C). Otherwise (C,H,W).
    """
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = t.numpy()
    if channel_last and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _ext_to_framework(ext: str) -> str:
    return {
        ".pt": "pytorch", ".pth": "pytorch",
        ".h5": "tensorflow", ".keras": "tensorflow",
        ".onnx": "onnx",
        ".tflite": "tflite",
    }.get(ext, "unknown")
