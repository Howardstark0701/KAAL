"""KAAL device configuration — first-run setup and per-command device resolution.

Config file: ~/.kaal/config.json
Format:      {"device": "cpu"} | {"device": "gpu"}

Public API
----------
get_device()                 → str          first-run prompt if needed
resolve_device(override)     → str          what every command calls
get_defaults(device)         → dict         adjusted tuning defaults
reset_config()               → None         delete config and re-prompt

Only stdlib + Rich (already a dependency); torch is imported lazily, only for
the CUDA check on --device gpu.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CONFIG_DIR  = Path.home() / ".kaal"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_VALID_DEVICES = {"cpu", "gpu"}

console = Console()

# ---------------------------------------------------------------------------
# Defaults per device
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict] = {
    "cpu": {
        "iterations":      100,
        "pgd_steps":       20,
        "patch_positions": 3,
        "fast":            True,
    },
    "gpu": {
        "iterations":      500,
        "pgd_steps":       40,
        "patch_positions": 8,
        "fast":            False,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return the configured device, running first-time setup if needed.

    Returns:
        "cpu" or "gpu"
    """
    if _CONFIG_FILE.exists():
        return _load_config()
    return _run_first_time_setup()


def resolve_device(cli_override: Optional[str]) -> str:
    """Resolve the final device for a command.

    Priority: CLI flag > saved config > first-time prompt.

    Args:
        cli_override: Value passed via --device, or None if not supplied.

    Returns:
        "cpu" or "gpu"

    Raises:
        SystemExit: If cli_override is not "cpu" or "gpu".
    """
    if cli_override is not None:
        normalized = cli_override.strip().lower()
        if normalized not in _VALID_DEVICES:
            console.print(
                f"[red]Error:[/red] --device must be 'cpu' or 'gpu', got '{cli_override}'"
            )
            raise SystemExit(1)
        if normalized == "gpu":
            _ensure_cuda_available()
        return normalized
    return get_device()


def _ensure_cuda_available() -> None:
    """Fail fast with a clear message if 'gpu' was requested but CUDA is absent."""
    try:
        import torch
    except ImportError:
        console.print(
            "[red]Error:[/red] --device gpu requires PyTorch, which is not installed."
        )
        raise SystemExit(1)
    if not torch.cuda.is_available():
        console.print(
            "[red]Error:[/red] --device gpu requested, but CUDA is not available "
            "on this machine."
        )
        console.print("  Install a CUDA-enabled PyTorch build, or run with --device cpu.")
        raise SystemExit(1)


def get_defaults(device: str) -> dict:
    """Return tuning defaults adjusted for the given device.

    Args:
        device: "cpu" or "gpu"

    Returns:
        dict with keys: iterations, pgd_steps, patch_positions, fast
    """
    return dict(_DEFAULTS.get(device, _DEFAULTS["cpu"]))


def reset_config() -> None:
    """Delete ~/.kaal/config.json and re-run the first-time setup prompt."""
    if _CONFIG_FILE.exists():
        _CONFIG_FILE.unlink()
        console.print("[dim]Config reset.[/dim]")
    _run_first_time_setup()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config() -> str:
    """Read and return device from config file. Falls back to 'cpu' on error."""
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        device = str(data.get("device", "cpu")).lower()
        return device if device in _VALID_DEVICES else "cpu"
    except (json.JSONDecodeError, OSError):
        return "cpu"


def _save_config(device: str) -> None:
    """Write device choice to ~/.kaal/config.json."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"device": device}, f)


def _run_first_time_setup() -> str:
    """Display the first-time setup prompt and save the result.

    Returns:
        "cpu" or "gpu"
    """
    # Draw the setup panel
    console.print()
    console.print(Panel.fit(
        "[bold white]KAAL — First-time Setup[/bold white]\n\n"
        "  Select your compute device:\n\n"
        "  [bold cyan]\\[1][/bold cyan]  CPU  [dim](laptop, no GPU)[/dim]\n"
        "  [bold cyan]\\[2][/bold cyan]  GPU  [dim](CUDA-enabled machine)[/dim]",
        border_style="dim white",
        padding=(0, 2),
    ))
    console.print()

    device = _prompt_device_choice()
    _save_config(device)

    label = "CPU" if device == "cpu" else "GPU (CUDA)"
    console.print(
        f"  [green]✓[/green] Saved: [bold]{label}[/bold]  "
        f"[dim]— override anytime with --device cpu/gpu[/dim]"
    )
    console.print()
    return device


def _prompt_device_choice() -> str:
    """Loop until the user enters 1 or 2. Returns 'cpu' or 'gpu'."""
    while True:
        try:
            raw = console.input("  [bold]Enter choice (1 or 2):[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive environment — default to CPU silently
            console.print("\n  [dim]Non-interactive mode detected. Defaulting to CPU.[/dim]")
            return "cpu"

        if raw == "1":
            return "cpu"
        if raw == "2":
            return "gpu"

        console.print(
            f"  [red]Invalid choice '[/red]{raw}[red]'.[/red]  "
            "[dim]Please enter 1 for CPU or 2 for GPU.[/dim]"
        )
