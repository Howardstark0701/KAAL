"""Model fingerprinting — kaal/defence/fingerprint.py

Provides tamper-detection for saved model files via two independent hashes:

1. file_hash   — SHA-256 of the raw bytes on disk.
                 Detects any change to the file (weights, metadata, format).

2. weight_hash — SHA-256 of the concatenated raw bytes of every parameter
                 tensor, in named_parameters() order.
                 Detects weight-only changes even if the file container
                 differs (e.g. the same weights re-saved in a different format).

Both hashes together give high confidence that a model file is authentic
and has not been silently modified, poisoned, or swapped.

Only stdlib dependencies: hashlib, json, os, pathlib, datetime.
torch is import-guarded — if not installed, weight_hash is set to
"N/A (torch not available)" and the fingerprint still works for
file-level integrity checking.

Usage:
    from kaal.defence.fingerprint import fingerprint_model, verify_fingerprint

    # Create and save a fingerprint
    fp = fingerprint_model("resnet50.pt")
    fp.to_json("resnet50_fingerprint.json")

    # Later: verify the file hasn't changed
    stored = ModelFingerprint.from_json("resnet50_fingerprint.json")
    result = verify_fingerprint("resnet50.pt", stored)
    if not result.overall_match:
        print("WARNING:", result.discrepancy)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ModelFingerprint
# ---------------------------------------------------------------------------

@dataclass
class ModelFingerprint:
    """Cryptographic fingerprint of a model file.

    Attributes:
        file_hash:        SHA-256 hex digest of the raw file bytes.
        weight_hash:      SHA-256 hex digest of the concatenated parameter
                          tensor bytes (in named_parameters() order).
                          "N/A (torch not available)" if torch is absent.
        file_size_bytes:  File size in bytes at fingerprint creation time.
        modified_at:      ISO 8601 UTC timestamp of the file's last modification.
        model_path:       Absolute path to the model file.
        generated_at:     ISO 8601 UTC timestamp of when this fingerprint
                          was generated.
    """

    file_hash:        str
    weight_hash:      str
    file_size_bytes:  int
    modified_at:      str
    model_path:       str
    generated_at:     str

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, output_path: Optional[str] = None) -> str:
        """Serialize this fingerprint to a JSON string and optionally save it.

        Args:
            output_path: If provided, write the JSON to this file path.
                         Parent directories are created if needed.

        Returns:
            The JSON string (regardless of whether it was also saved).
        """
        data = asdict(self)
        js   = json.dumps(data, indent=2)

        if output_path is not None:
            dest = Path(output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(js, encoding="utf-8")

        return js

    @classmethod
    def from_json(cls, source: str) -> "ModelFingerprint":
        """Deserialize a ModelFingerprint from a JSON string or file path.

        Args:
            source: Either a JSON string (starts with '{') or a path to a
                    .json file.

        Returns:
            ModelFingerprint instance.

        Raises:
            FileNotFoundError: source is a path that does not exist.
            ValueError:        source is malformed JSON or missing required fields.
        """
        if source.strip().startswith("{"):
            raw = source
        else:
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(
                    f"Fingerprint file not found: '{source}'"
                )
            raw = p.read_text(encoding="utf-8")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in fingerprint: {e}") from e

        required = {
            "file_hash", "weight_hash", "file_size_bytes",
            "modified_at", "model_path", "generated_at",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Fingerprint JSON is missing required fields: {sorted(missing)}"
            )

        return cls(
            file_hash=str(data["file_hash"]),
            weight_hash=str(data["weight_hash"]),
            file_size_bytes=int(data["file_size_bytes"]),
            modified_at=str(data["modified_at"]),
            model_path=str(data["model_path"]),
            generated_at=str(data["generated_at"]),
        )


# ---------------------------------------------------------------------------
# FingerprintVerification
# ---------------------------------------------------------------------------

@dataclass
class FingerprintVerification:
    """Result of verifying a model file against a stored fingerprint."""

    file_hash_match:   bool
    """True if the current file's SHA-256 matches the stored file_hash."""

    weight_hash_match: bool
    """True if the current weight hash matches the stored weight_hash.
    Always True when weight_hash is 'N/A (torch not available)' in either
    the stored or the freshly-computed fingerprint (cannot verify)."""

    overall_match: bool
    """True only when both file_hash_match and weight_hash_match are True."""

    discrepancy: Optional[str]
    """Human-readable explanation of any mismatch. None if everything matches."""


# ---------------------------------------------------------------------------
# fingerprint_model()
# ---------------------------------------------------------------------------

def fingerprint_model(model_path: str) -> ModelFingerprint:
    """Compute the cryptographic fingerprint of a model file.

    Args:
        model_path: Path to the model file (.pt, .pth, .h5, .onnx, etc.).

    Returns:
        ModelFingerprint with file hash, weight hash, size, and timestamps.

    Raises:
        FileNotFoundError: model_path does not exist.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            "→ Check the path is correct and the file exists."
        )

    # ── File-level hash ───────────────────────────────────────────────────
    file_hash = _sha256_file(path)

    # ── File metadata ─────────────────────────────────────────────────────
    stat         = path.stat()
    file_size    = int(stat.st_size)
    modified_at  = _iso_from_mtime(stat.st_mtime)

    # ── Weight-level hash ─────────────────────────────────────────────────
    weight_hash = _compute_weight_hash(path)

    return ModelFingerprint(
        file_hash=file_hash,
        weight_hash=weight_hash,
        file_size_bytes=file_size,
        modified_at=modified_at,
        model_path=str(path),
        generated_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# verify_fingerprint()
# ---------------------------------------------------------------------------

def verify_fingerprint(
    model_path: str,
    fingerprint: ModelFingerprint,
) -> FingerprintVerification:
    """Verify a model file against a stored ModelFingerprint.

    Recomputes both hashes and compares them to the stored values.

    Args:
        model_path:   Path to the model file to verify.
        fingerprint:  Previously generated ModelFingerprint to compare against.

    Returns:
        FingerprintVerification with per-hash match flags and a discrepancy
        message if anything mismatches.

    Raises:
        FileNotFoundError: model_path does not exist.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found for verification: '{model_path}'"
        )

    # Recompute fresh fingerprint
    fresh = fingerprint_model(str(path))

    # ── File hash comparison ──────────────────────────────────────────────
    file_match = fresh.file_hash == fingerprint.file_hash

    # ── Weight hash comparison ────────────────────────────────────────────
    _NA = "N/A (torch not available)"
    if fingerprint.weight_hash == _NA or fresh.weight_hash == _NA:
        # Cannot verify weight hash — treat as unverified (not a failure)
        weight_match = True
        weight_note  = " (weight hash not verified — torch unavailable)"
    else:
        weight_match = fresh.weight_hash == fingerprint.weight_hash
        weight_note  = ""

    overall = file_match and weight_match

    # ── Build discrepancy message ─────────────────────────────────────────
    discrepancy: Optional[str] = None
    if not overall:
        parts: list[str] = []
        if not file_match:
            parts.append(
                f"FILE HASH MISMATCH — stored: {fingerprint.file_hash[:16]}…  "
                f"current: {fresh.file_hash[:16]}… "
                f"(file size stored={fingerprint.file_size_bytes} B, "
                f"current={fresh.file_size_bytes} B)"
            )
        if not weight_match:
            parts.append(
                f"WEIGHT HASH MISMATCH — stored: {fingerprint.weight_hash[:16]}…  "
                f"current: {fresh.weight_hash[:16]}… "
                f"(model weights may have been modified)"
            )
        discrepancy = "; ".join(parts) + weight_note
    elif weight_note:
        # Overall match but weight hash was skipped — note it
        discrepancy = weight_note.strip()

    return FingerprintVerification(
        file_hash_match=file_match,
        weight_hash_match=weight_match,
        overall_match=overall,
        discrepancy=discrepancy if discrepancy else None,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file by reading it in chunks (memory efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_weight_hash(path: Path) -> str:
    """Extract model weights and return SHA-256 of the concatenated bytes.

    Handles PyTorch (.pt / .pth) and attempts to load other formats via
    torch.load with weights_only=False as a fallback.

    Returns "N/A (torch not available)" if torch is not installed.
    Returns "N/A (weight extraction failed: <reason>)" on any other error.
    """
    try:
        import torch
    except ImportError:
        return "N/A (torch not available)"

    try:
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            import warnings
            warnings.warn(
                f"Could not load {path} with weights_only=True (may contain custom objects). "
                "Falling back to weights_only=False — only load model files you trust.",
                UserWarning,
                stacklevel=2,
            )
            state = torch.load(str(path), map_location="cpu", weights_only=False)
        obj = state
    except Exception as e:
        return f"N/A (weight extraction failed: {e})"

    # Collect all parameter tensors
    h       = hashlib.sha256()
    found   = False

    # Case 1: full nn.Module
    if hasattr(obj, "parameters"):
        try:
            for name, param in obj.named_parameters():
                # Hash the raw bytes of each parameter in deterministic order
                data = param.detach().cpu().to(torch.float32).numpy().tobytes()
                h.update(data)
                found = True
        except Exception:
            pass

    # Case 2: state_dict (OrderedDict / dict of tensors)
    if not found and isinstance(obj, dict):
        for key in sorted(obj.keys()):
            val = obj[key]
            if hasattr(val, "numpy"):
                try:
                    data = val.detach().cpu().to(torch.float32).numpy().tobytes()
                    h.update(key.encode("utf-8"))   # include key for ordering
                    h.update(data)
                    found = True
                except Exception:
                    pass

    if not found:
        return "N/A (no extractable weights found)"

    return h.hexdigest()


def _iso_from_mtime(mtime: float) -> str:
    """Convert a POSIX mtime float to ISO 8601 UTC string."""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
