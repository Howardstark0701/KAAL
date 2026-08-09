"""KAAL Benchmark Runner — kaal/benchmark/runner.py

Runs a full KAAL audit on each model in a list and collects structured
results in a BenchmarkEntry dataclass. Results are:

    - Printed live to console as a Rich table (one row added per model)
    - Saved / merged into <output_dir>/leaderboard.json
    - Returned as a list sorted by kvs_score descending (most vulnerable first)

Usage:
    from kaal.benchmark.runner import run_benchmark

    entries = run_benchmark(
        model_specs=[
            ("./models/resnet50.pt",   "ResNet-50"),
            ("./models/mobilenet.pt",  "MobileNet-V2"),
        ],
        dataset_dir="./images/",
        attacks=["fgsm", "pgd"],
        max_images=20,
    )
    for e in entries:
        print(e.model_name, e.kvs_score, e.kvs_label)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box

from kaal.engine.loader import load_model
from kaal.engine.dataset import load_dataset
from kaal.engine.utils import resolve_input_shape
from kaal.scoring.kvs import calculate_kvs


# ---------------------------------------------------------------------------
# Attack-name validation
# ---------------------------------------------------------------------------

_VALID_ATTACKS = {"fgsm", "pgd", "patch", "blackbox", "physical"}


def _validate_attacks(attacks) -> set[str]:
    """Validate attack names against the supported set.

    Raises:
        ValueError: If any name is unknown, or the set is empty.
    """
    attack_set = {a.lower().strip() for a in attacks if a.strip()}
    unknown = attack_set - _VALID_ATTACKS
    if unknown:
        raise ValueError(
            f"Unknown attack name(s): {', '.join(sorted(unknown))}. "
            f"Supported attacks: {', '.join(sorted(_VALID_ATTACKS))}."
        )
    if not attack_set:
        raise ValueError(
            "No attacks specified. "
            f"Supported attacks: {', '.join(sorted(_VALID_ATTACKS))}."
        )
    return attack_set

console = Console()

# ---------------------------------------------------------------------------
# BenchmarkEntry
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    """Results for a single model audit in the benchmark run."""

    model_name: str
    """Human-readable display name passed in model_specs."""

    model_path: str
    """Absolute path to the model file."""

    kvs_score: float
    """Overall KVS vulnerability score, 0.0–10.0."""

    kvs_label: str
    """Risk label: Robust / Low Risk / Medium Risk / High Risk / Critical / Catastrophic."""

    fgsm_success_rate: Optional[float]
    """FGSM attack success rate (0–1). None if FGSM not in attacks list."""

    pgd_success_rate: Optional[float]
    """PGD attack success rate (0–1). None if PGD not in attacks list."""

    patch_success_rate: Optional[float]
    """Patch attack success rate (0–1). None if patch not in attacks list."""

    num_classes: int
    """Number of output classes detected from the model."""

    input_shape: tuple
    """Model input shape, e.g. (3, 224, 224)."""

    audit_timestamp: str
    """ISO 8601 UTC timestamp of when this audit completed."""

    blackbox_success_rate: Optional[float] = None
    """Black-box NES attack success rate (0–1). None if blackbox not run."""


# ---------------------------------------------------------------------------
# run_benchmark()
# ---------------------------------------------------------------------------

def run_benchmark(
    model_specs: list[tuple[str, str]],
    dataset_dir: str,
    attacks: list[str] = None,
    output_dir: str = "./benchmark_results/",
    max_images: int = 50,
    input_shape: Optional[tuple] = None,
) -> list[BenchmarkEntry]:
    """Run a full KAAL audit on each model and return sorted results.

    Args:
        model_specs: List of (model_path, display_name) tuples.
        dataset_dir: Path to directory of test images.
        attacks:     List of attack names to run. Default: ["fgsm", "pgd", "patch"].
                     Supported: "fgsm", "pgd", "patch", "physical".
        output_dir:  Directory for leaderboard.json and per-model output.
        max_images:  Maximum images per model (keeps runtime reasonable).
        input_shape: Optional explicit (H, W) or (C, H, W) override for the
                     dataset, for dynamic-shape ONNX/TFLite models whose own
                     input_shape has None spatial dims. Defaults to each
                     model's own input shape.

    Returns:
        List of BenchmarkEntry sorted by kvs_score descending
        (most vulnerable model first).
    """
    if attacks is None:
        attacks = ["fgsm", "pgd", "patch"]

    attack_set = _validate_attacks(attacks)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Build the live Rich table ────────────────────────────────────────────
    table = _make_table()
    entries: list[BenchmarkEntry] = []

    console.print()
    console.print("[bold white]KAAL Benchmark[/bold white]")
    console.print(f"  Models:   {len(model_specs)}")
    console.print(f"  Attacks:  {', '.join(sorted(attack_set)).upper()}")
    console.print(f"  Images:   up to {max_images} per model")
    console.print(f"  Output:   {output_path.resolve()}")
    console.print()

    for idx, (model_path, model_name) in enumerate(model_specs, start=1):
        console.print(
            f"[dim][{idx}/{len(model_specs)}][/dim] "
            f"Auditing [bold]{model_name}[/bold]  "
            f"[dim]({model_path})[/dim]"
        )

        entry = _audit_one(
            model_path=model_path,
            model_name=model_name,
            dataset_dir=dataset_dir,
            attack_set=attack_set,
            max_images=max_images,
            model_output_dir=str(output_path / _safe_name(model_name)),
            input_shape=input_shape,
        )
        entries.append(entry)

        # ── Print updated table (re-render sorted so far) ────────────────────
        sorted_so_far = sorted(entries, key=lambda e: e.kvs_score, reverse=True)
        live_table = _make_table()
        _refresh_table(live_table, sorted_so_far)
        console.print(live_table)
        console.print()

    # ── Final sort ───────────────────────────────────────────────────────────
    entries.sort(key=lambda e: e.kvs_score, reverse=True)

    # ── Save / merge leaderboard.json ────────────────────────────────────────
    lb_path = output_path / "leaderboard.json"
    _save_leaderboard(entries, lb_path)
    console.print(f"[dim]Leaderboard saved → {lb_path.resolve()}[/dim]")
    console.print()

    return entries


# ---------------------------------------------------------------------------
# Single-model audit
# ---------------------------------------------------------------------------

def _audit_one(
    model_path: str,
    model_name: str,
    dataset_dir: str,
    attack_set: set[str],
    max_images: int,
    model_output_dir: str,
    input_shape: Optional[tuple] = None,
) -> BenchmarkEntry:
    """Run attacks on one model and return a BenchmarkEntry."""

    # Load model
    try:
        km = load_model(model_path)
    except Exception as exc:
        console.print(f"  [red]Failed to load model:[/red] {exc}")
        return _error_entry(model_name, model_path)

    # Load dataset (capped)
    try:
        ds = load_dataset(
            dataset_dir,
            input_shape=resolve_input_shape(km.input_shape, input_shape),
            max_images=max_images,
        )
    except Exception as exc:
        console.print(f"  [red]Failed to load dataset:[/red] {exc}")
        return _error_entry(model_name, model_path)

    fgsm_agg   = None
    pgd_agg    = None
    patch_result = None
    blackbox_result = None
    blackbox_rate   = None
    adv_tensors: list = []
    orig_classes: list = []

    # ── FGSM ─────────────────────────────────────────────────────────────────
    if "fgsm" in attack_set:
        try:
            from kaal.attacks.fgsm import fgsm_attack_dataset
            with console.status("  FGSM..."):
                fgsm_agg = fgsm_attack_dataset(km, ds, epsilon=0.03)
            for r in fgsm_agg["results"]:
                adv_tensors.append(r.adversarial_tensor)
                orig_classes.append(r.original_class)
            console.print(
                f"  [dim]FGSM:[/dim] {fgsm_agg['success_rate']:.0%} success"
            )
        except Exception as exc:
            console.print(f"  [yellow]FGSM skipped:[/yellow] {exc}")

    # ── PGD ──────────────────────────────────────────────────────────────────
    if "pgd" in attack_set:
        try:
            from kaal.attacks.pgd import pgd_attack_dataset
            with console.status("  PGD..."):
                pgd_agg = pgd_attack_dataset(km, ds, epsilon=0.03, steps=20)
            if not adv_tensors:
                for r in pgd_agg["results"]:
                    adv_tensors.append(r.adversarial_tensor)
                    orig_classes.append(r.original_class)
            console.print(
                f"  [dim]PGD:[/dim]  {pgd_agg['success_rate']:.0%} success"
            )
        except Exception as exc:
            console.print(f"  [yellow]PGD skipped:[/yellow] {exc}")

    # ── Patch ─────────────────────────────────────────────────────────────────
    if "patch" in attack_set:
        try:
            from kaal.attacks.patch import generate_patch
            Path(model_output_dir).mkdir(parents=True, exist_ok=True)
            with console.status("  Patch..."):
                patch_result = generate_patch(
                    km, ds,
                    target_class=0,
                    patch_fraction=0.05,
                    iterations=100,          # capped for benchmark speed
                    output_dir=model_output_dir,
                    verbose=False,
                )
            console.print(
                f"  [dim]Patch:[/dim] {patch_result.attack_success_rate:.0%} success"
            )
        except Exception as exc:
            console.print(f"  [yellow]Patch skipped:[/yellow] {exc}")

    # ── Black-box (NES) ────────────────────────────────────────────────────────
    if "blackbox" in attack_set:
        try:
            from kaal.attacks.blackbox import blackbox_attack_dataset
            with console.status("  Black-box..."):
                bb_agg = blackbox_attack_dataset(km, ds, epsilon=0.03,
                                                 max_images=max_images)
            from types import SimpleNamespace
            # KVS Dim 5 reads `.query_efficiency`, so expose the dataset
            # aggregate as an object rather than the raw dict.
            blackbox_result = SimpleNamespace(
                query_efficiency=bb_agg["avg_query_efficiency"],
                success_rate=bb_agg["success_rate"],
            )
            blackbox_rate = bb_agg["success_rate"]
            console.print(
                f"  [dim]Black-box:[/dim] {bb_agg['success_rate']:.0%} success"
            )
        except Exception as exc:
            console.print(f"  [yellow]Black-box skipped:[/yellow] {exc}")

    # ── Physical (optional) ───────────────────────────────────────────────────
    phys_result = None
    if "physical" in attack_set and adv_tensors:
        try:
            from kaal.attacks.physical import test_physical_robustness_batch
            n = min(len(adv_tensors), 10)
            with console.status("  Physical..."):
                phys_result = test_physical_robustness_batch(
                    km, adv_tensors[:n], orig_classes[:n]
                )
            console.print(
                f"  [dim]Physical:[/dim] {phys_result.overall_survival_rate:.0%} survival"
            )
        except Exception as exc:
            console.print(f"  [yellow]Physical skipped:[/yellow] {exc}")

    # ── KVS score ─────────────────────────────────────────────────────────────
    kvs = calculate_kvs(
        fgsm_result=fgsm_agg,
        pgd_result=pgd_agg,
        patch_result=patch_result,
        physical_result=phys_result,
        blackbox_result=blackbox_result,
    )

    return BenchmarkEntry(
        model_name=model_name,
        model_path=str(Path(model_path).resolve()),
        kvs_score=kvs.score,
        kvs_label=kvs.label,
        fgsm_success_rate=fgsm_agg["success_rate"] if fgsm_agg else None,
        pgd_success_rate=pgd_agg["success_rate"] if pgd_agg else None,
        patch_success_rate=patch_result.attack_success_rate if patch_result else None,
        blackbox_success_rate=blackbox_rate,
        num_classes=km.num_classes,
        input_shape=tuple(km.input_shape),
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Rich table helpers
# ---------------------------------------------------------------------------

def _make_table() -> Table:
    """Build the benchmark Rich table with headers."""
    t = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold dim white",
        show_lines=False,
        padding=(0, 1),
    )
    t.add_column("Rank",  justify="right",  style="dim white",  no_wrap=True, min_width=4)
    t.add_column("Model", justify="left",   style="bold white", no_wrap=False, min_width=20)
    t.add_column("KVS",   justify="right",  style="white",      no_wrap=True, min_width=5)
    t.add_column("Label", justify="left",   style="white",      no_wrap=True, min_width=14)
    t.add_column("FGSM",  justify="right",  style="dim white",  no_wrap=True, min_width=6)
    t.add_column("PGD",   justify="right",  style="dim white",  no_wrap=True, min_width=6)
    t.add_column("Patch", justify="right",  style="dim white",  no_wrap=True, min_width=6)
    t.add_column("BB",    justify="right",  style="dim white",  no_wrap=True, min_width=6)
    return t


def _refresh_table(table: Table, entries: list[BenchmarkEntry]) -> None:
    """Populate a fresh table with sorted entries."""
    for rank, e in enumerate(entries, start=1):
        kvs_color = _kvs_color(e.kvs_score)
        table.add_row(
            str(rank),
            e.model_name,
            f"[{kvs_color}]{e.kvs_score:.1f}[/{kvs_color}]",
            f"[{kvs_color}]{e.kvs_label}[/{kvs_color}]",
            _pct(e.fgsm_success_rate),
            _pct(e.pgd_success_rate),
            _pct(e.patch_success_rate),
            _pct(e.blackbox_success_rate),
        )


def _pct(value: Optional[float]) -> str:
    return f"{value:.0%}" if value is not None else "[dim]—[/dim]"


def _kvs_color(score: float) -> str:
    if score <= 2.0:  return "green"
    if score <= 4.0:  return "yellow_green"
    if score <= 6.0:  return "yellow"
    if score <= 8.0:  return "dark_orange"
    return "bold red"


# ---------------------------------------------------------------------------
# Leaderboard JSON persistence
# ---------------------------------------------------------------------------

def _save_leaderboard(entries: list[BenchmarkEntry], path: Path) -> None:
    """Merge new entries into leaderboard.json (newest entry per model wins)."""
    existing: dict[str, dict] = {}

    # Load existing data if file is present
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            # raw is a list of dicts keyed by model_name
            for item in raw:
                existing[item["model_name"]] = item
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt file — start fresh

    # Merge: newest run wins for each model_name
    for e in entries:
        d = asdict(e)
        d["input_shape"] = list(e.input_shape)   # ensure JSON-serialisable
        existing[e.model_name] = d

    # Write sorted by kvs_score descending
    merged = sorted(existing.values(), key=lambda x: x["kvs_score"], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Convert a display name to a safe directory name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _error_entry(model_name: str, model_path: str) -> BenchmarkEntry:
    """Return a zeroed BenchmarkEntry for a model that failed to load."""
    return BenchmarkEntry(
        model_name=model_name,
        model_path=str(Path(model_path).resolve()),
        kvs_score=0.0,
        kvs_label="Error",
        fgsm_success_rate=None,
        pgd_success_rate=None,
        patch_success_rate=None,
        num_classes=0,
        input_shape=(0,),
        audit_timestamp=datetime.now(timezone.utc).isoformat(),
    )
