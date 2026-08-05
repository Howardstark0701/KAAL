"""KAAL CLI — Spec 15.1 — Phase 9, Kiro Prompt 9.1.

Framework: Typer + Rich
Install:   pip install kaal
Entry:     kaal --help

Commands:
    kaal audit    — full adversarial vulnerability audit
    kaal serve    — launch web UI
    kaal patch    — generate adversarial patch only
    kaal compare  — compare two audit JSON reports
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import box
from rich.text import Text

# ── App & console ─────────────────────────────────────────────────────────────
app = typer.Typer(
    name="kaal",
    help="KAAL — Adversarial Robustness Auditing Tool\n\n"
         '"What cannot be seen, cannot be defended."',
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()

# ── KAAL colour palette for Rich ─────────────────────────────────────────────
_RED    = "bold red"
_MUTED  = "dim white"
_MONO   = "bold white"
_GREEN  = "green"


# =============================================================================
# kaal audit
# =============================================================================

@app.command()
def audit(
    model: str = typer.Option(..., "--model",   help="Path to model file (.h5/.pt/.onnx/.tflite)"),
    dataset: str = typer.Option(..., "--dataset", help="Path to image directory"),
    attacks: str = typer.Option("fgsm,pgd,patch,physical", "--attacks",
                                help="Comma-separated: fgsm,pgd,patch,blackbox,physical"),
    epsilon: float = typer.Option(0.03,  "--epsilon", help="Perturbation strength"),
    steps:   int   = typer.Option(40,    "--steps",   help="PGD steps"),
    output:  str   = typer.Option("./kaal_output/", "--output", help="Output directory"),
    report:  str   = typer.Option("pdf,json",        "--report", help="Report formats: pdf,json,html,all"),
    no_gradcam: bool = typer.Option(False, "--no-gradcam", help="Skip GradCAM (faster)"),
    quiet:      bool = typer.Option(False, "--quiet",      help="Suppress progress, show final result only"),
):
    """Run a full adversarial vulnerability audit."""
    start_time = time.time()

    if not quiet:
        _print_banner()

    # ── Validate inputs ───────────────────────────────────────────────────────
    model_path   = Path(model)
    dataset_path = Path(dataset)

    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model file not found: {model}")
        raise typer.Exit(1)
    if not dataset_path.exists() or not dataset_path.is_dir():
        console.print(f"[red]Error:[/red] Dataset directory not found: {dataset}")
        raise typer.Exit(1)

    attack_list = [a.strip().lower() for a in attacks.split(",") if a.strip()]
    valid_attacks = {"fgsm", "pgd", "patch", "blackbox", "physical"}
    bad = [a for a in attack_list if a not in valid_attacks]
    if bad:
        console.print(f"[red]Error:[/red] Unknown attack(s): {', '.join(bad)}")
        raise typer.Exit(1)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + dataset ──────────────────────────────────────────────────
    from kaal.engine.loader import load_model
    from kaal.engine.dataset import load_dataset

    if not quiet:
        console.print(f"[dim]Model:[/dim]   [bold]{model_path.name}[/bold]")
        console.print(f"[dim]Dataset:[/dim] [bold]{dataset_path}[/bold]")
        console.print(f"[dim]Attacks:[/dim] {', '.join(a.upper() for a in attack_list)}")
        console.print(f"[dim]Output:[/dim]  [bold]{output_dir}[/bold]")
        console.print()
        console.print("─" * 46)
        console.print()

    try:
        kaal_model = load_model(str(model_path))
    except Exception as e:
        console.print(f"[red]Failed to load model:[/red] {e}")
        raise typer.Exit(1)

    try:
        kaal_dataset = load_dataset(str(dataset_path),
                                    input_shape=kaal_model.input_shape)
    except Exception as e:
        console.print(f"[red]Failed to load dataset:[/red] {e}")
        raise typer.Exit(1)

    if not quiet:
        console.print(
            f"[dim]Model:[/dim]    {model_path.name}  "
            f"([dim]{kaal_model.framework}[/dim], {kaal_model.num_classes} classes)"
        )
        console.print(
            f"[dim]Dataset:[/dim]  {dataset_path}    "
            f"([dim]{len(kaal_dataset)} images found[/dim])"
        )
        console.print()

    # ── Attack results storage ────────────────────────────────────────────────
    fgsm_agg      = None
    pgd_agg       = None
    patch_result  = None
    phys_result   = None
    gradcam_cmp   = None
    collapse_path = None
    adv_tensors   = []
    orig_classes  = []

    total_steps = len(attack_list) + 1  # +1 for reports step
    step_num = 0

    # ── FGSM ──────────────────────────────────────────────────────────────────
    if "fgsm" in attack_list:
        step_num += 1
        _print_step(quiet, step_num, total_steps, "Running FGSM attack...",
                    f"ε = {epsilon} | {len(kaal_dataset)} images")

        from kaal.attacks.fgsm import fgsm_attack_dataset
        fgsm_agg = _run_with_progress(
            quiet,
            lambda: fgsm_attack_dataset(kaal_model, kaal_dataset, epsilon=epsilon),
            len(kaal_dataset),
        )
        for r in fgsm_agg["results"]:
            adv_tensors.append(r.adversarial_tensor)
            orig_classes.append(r.original_class)

        if not quiet:
            console.print(
                f"      [green]Success rate: {fgsm_agg['success_rate']:.0%}[/green]  |  "
                f"Avg Δ confidence: {fgsm_agg['avg_confidence_delta']:+.2f}"
            )
            console.print()

    # ── PGD ───────────────────────────────────────────────────────────────────
    if "pgd" in attack_list:
        step_num += 1
        _print_step(quiet, step_num, total_steps, "Running PGD attack...",
                    f"ε = {epsilon} | α = {epsilon/10:.4f} | {steps} steps | "
                    f"{len(kaal_dataset)} images")

        from kaal.attacks.pgd import pgd_attack_dataset
        pgd_agg = _run_with_progress(
            quiet,
            lambda: pgd_attack_dataset(kaal_model, kaal_dataset,
                                       epsilon=epsilon, steps=steps),
            len(kaal_dataset),
        )
        if not adv_tensors:
            for r in pgd_agg["results"]:
                adv_tensors.append(r.adversarial_tensor)
                orig_classes.append(r.original_class)

        if not quiet:
            avg_steps = pgd_agg.get("avg_steps_to_success", -1)
            console.print(
                f"      [green]Success rate: {pgd_agg['success_rate']:.0%}[/green]  |  "
                f"Avg steps to success: {avg_steps}"
            )
            console.print()

    # ── Patch ─────────────────────────────────────────────────────────────────
    if "patch" in attack_list:
        step_num += 1
        target_cls = 0
        _print_step(quiet, step_num, total_steps, "Generating adversarial patch...",
                    f"Target class: {target_cls} | 500 iterations")

        from kaal.attacks.patch import generate_patch
        patch_result = _run_with_progress(
            quiet,
            lambda: generate_patch(
                kaal_model, kaal_dataset,
                target_class=target_cls,
                patch_fraction=0.05,
                iterations=500,
                output_dir=str(output_dir),
                verbose=False,
            ),
            500,
        )
        if not quiet:
            console.print(
                f"      [green]Patch success rate: {patch_result.attack_success_rate:.0%}[/green]"
            )
            console.print()

    # ── Physical ──────────────────────────────────────────────────────────────
    if "physical" in attack_list and adv_tensors:
        step_num += 1
        _print_step(quiet, step_num, total_steps, "Running physical robustness tests...",
                    "7 transforms | per adversarial image")

        from kaal.attacks.physical import test_physical_robustness_batch
        phys_result = _run_with_progress(
            quiet,
            lambda: test_physical_robustness_batch(
                kaal_model, adv_tensors[:min(len(adv_tensors), 20)],
                orig_classes[:min(len(orig_classes), 20)],
            ),
            len(adv_tensors[:20]),
        )
        if not quiet:
            console.print(
                f"      [green]Survival rate: {phys_result.overall_survival_rate:.0%}[/green]  |  "
                f"Rating: {phys_result.physical_threat_rating}"
            )
            console.print()

    # ── Explainability + Reports ──────────────────────────────────────────────
    step_num += 1
    _print_step(quiet, step_num, total_steps,
                "Generating GradCAM, fingerprint, reports...", "")

    # GradCAM (skip if --no-gradcam or no FGSM results)
    if not no_gradcam and fgsm_agg and fgsm_agg["results"]:
        try:
            from kaal.explainability.gradcam import generate_gradcam_comparison
            first = fgsm_agg["results"][0]
            # Reconstruct clean tensor from adversarial - perturbation
            clean_t = first.adversarial_tensor - first.perturbation_tensor
            gradcam_cmp = generate_gradcam_comparison(
                kaal_model, clean_t, first.adversarial_tensor
            )
            _save_gradcam(gradcam_cmp, output_dir)
        except Exception:
            pass  # GradCAM is best-effort

    # Confidence collapse curve (from PGD)
    if pgd_agg and pgd_agg["results"]:
        try:
            from kaal.explainability.confidence import generate_collapse_curve
            collapse_path = str(output_dir / "collapse_curve.png")
            generate_collapse_curve(pgd_agg["results"][0], collapse_path)
        except Exception:
            collapse_path = None

    # KVS score
    from kaal.scoring.kvs import calculate_kvs
    kvs_result = calculate_kvs(
        fgsm_result=fgsm_agg,
        pgd_result=pgd_agg,
        physical_result=phys_result,
        min_epsilon=epsilon,
    )

    # Fingerprint
    fingerprint_path = None
    try:
        from kaal.fingerprint.radar import generate_fingerprint
        fingerprint_path = str(output_dir / "fingerprint.png")
        generate_fingerprint(kvs_result, model_path.stem, fingerprint_path)
    except Exception:
        fingerprint_path = None

    # Model + dataset info dicts
    model_info = {
        "path": str(model_path),
        "name": model_path.stem,
        "framework": kaal_model.framework,
        "input_shape": list(kaal_model.input_shape),
        "num_classes": kaal_model.num_classes,
    }
    dataset_info = {
        "path": str(dataset_path),
        "total_images": len(kaal_dataset),
        "formats": kaal_dataset.format_counts,
    }

    duration = time.time() - start_time

    # JSON report
    report_formats = [r.strip().lower() for r in report.split(",")]
    if "all" in report_formats:
        report_formats = ["pdf", "json", "html"]

    if "json" in report_formats:
        from kaal.reporting.json_report import generate_json_report
        generate_json_report(
            output_path=str(output_dir / "report.json"),
            model_info=model_info,
            dataset_info=dataset_info,
            kvs_result=kvs_result,
            fgsm_result=fgsm_agg,
            pgd_result=pgd_agg,
            patch_result=patch_result,
            physical_result=phys_result,
            audit_duration_seconds=duration,
        )

    # PDF report
    if "pdf" in report_formats:
        from kaal.reporting.pdf import generate_pdf_report
        generate_pdf_report(
            output_path=str(output_dir / "report.pdf"),
            model_info=model_info,
            dataset_info=dataset_info,
            kvs_result=kvs_result,
            fgsm_result=fgsm_agg["results"][0] if fgsm_agg and fgsm_agg["results"] else None,
            pgd_result=pgd_agg["results"][0] if pgd_agg and pgd_agg["results"] else None,
            patch_result=patch_result,
            physical_result=phys_result,
            gradcam_comparison=gradcam_cmp,
            collapse_curve_path=collapse_path,
            fingerprint_path=fingerprint_path,
            audit_duration_seconds=duration,
        )

    if not quiet:
        console.print("[dim]      ████████████████████████[/dim] [green]100%[/green]")
        console.print()

    # ── Final KVS display ─────────────────────────────────────────────────────
    _print_kvs_display(kvs_result, quiet)

    # ── Output file tree ──────────────────────────────────────────────────────
    if not quiet:
        _print_output_tree(output_dir)

    # ── Duration ─────────────────────────────────────────────────────────────
    m, s = divmod(int(duration), 60)
    dur_str = f"{m}m {s:02d}s" if m else f"{s}s"
    console.print()
    console.print(f"  [dim]Audit complete. Duration: {dur_str}[/dim]")


# =============================================================================
# kaal serve
# =============================================================================

@app.command()
def serve(
    port: int = typer.Option(8080,        "--port", help="Port number"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host address"),
):
    """Launch the KAAL web UI (FastAPI + Next.js)."""
    _print_banner()
    console.print(f"  Starting KAAL web server on [bold]http://{host}:{port}[/bold]")
    console.print(f"  [dim]Frontend:[/dim] http://localhost:3000  (start separately with: cd web/frontend && npm run dev)")
    console.print()
    try:
        import uvicorn
        from web.backend.main import app as fastapi_app
        uvicorn.run(fastapi_app, host=host, port=port)
    except ImportError:
        console.print("[red]Error:[/red] Web dependencies not installed.")
        console.print("  Run: [bold]pip install -r requirements-web.txt[/bold]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error starting server:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# kaal patch
# =============================================================================

@app.command()
def patch(
    model:    str   = typer.Option(...,   "--model",    help="Path to model file"),
    dataset:  str   = typer.Option(...,   "--dataset",  help="Path to image directory"),
    target:   int   = typer.Option(...,   "--target",   help="Target class index"),
    size:     float = typer.Option(0.05,  "--size",     help="Patch size as image fraction"),
    print_cm: float = typer.Option(15.0,  "--print-cm", help="Physical print size in cm"),
    output:   str   = typer.Option("./kaal_output/", "--output", help="Output directory"),
):
    """Generate an adversarial patch only."""
    _print_banner()

    model_path   = Path(model)
    dataset_path = Path(dataset)

    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model not found: {model}")
        raise typer.Exit(1)
    if not dataset_path.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {dataset}")
        raise typer.Exit(1)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from kaal.engine.loader import load_model
    from kaal.engine.dataset import load_dataset
    from kaal.attacks.patch import generate_patch

    console.print(f"  [dim]Model:[/dim]        {model_path.name}")
    console.print(f"  [dim]Target class:[/dim] {target}")
    console.print(f"  [dim]Patch size:[/dim]   {size*100:.0f}% of image area")
    console.print()

    kaal_model   = load_model(str(model_path))
    kaal_dataset = load_dataset(str(dataset_path), input_shape=kaal_model.input_shape)

    with console.status("[bold red]Training adversarial patch...[/bold red]"):
        result = generate_patch(
            kaal_model, kaal_dataset,
            target_class=target,
            patch_fraction=size,
            iterations=500,
            output_dir=str(output_dir),
            print_size_cm=print_cm,
            verbose=False,
        )

    console.print(f"  [green]Success rate:[/green] {result.attack_success_rate:.0%}")
    console.print(f"  [dim]Avg confidence on target:[/dim] {result.avg_confidence_on_target:.3f}")
    console.print(f"  [dim]Print size:[/dim] {print_cm:.0f} cm × {print_cm:.0f} cm")
    console.print()
    console.print(f"  Patch PNG:       [bold]{output_dir}/patch.png[/bold]")
    console.print(f"  Printable PDF:   [bold]{result.patch_printable_pdf_path}[/bold]")


# =============================================================================
# kaal compare
# =============================================================================

@app.command()
def compare(
    before: str = typer.Option(..., "--before", help="Path to first audit JSON"),
    after:  str = typer.Option(..., "--after",  help="Path to second audit JSON"),
    output: str = typer.Option("./kaal_output/compare/", "--output",
                               help="Output directory for comparison report"),
):
    """Compare two audit JSON reports side by side."""
    _print_banner()

    before_path = Path(before)
    after_path  = Path(after)

    if not before_path.exists():
        console.print(f"[red]Error:[/red] File not found: {before}")
        raise typer.Exit(1)
    if not after_path.exists():
        console.print(f"[red]Error:[/red] File not found: {after}")
        raise typer.Exit(1)

    with open(before_path) as f:
        doc_a = json.load(f)
    with open(after_path) as f:
        doc_b = json.load(f)

    kvs_a = doc_a.get("kvs", {})
    kvs_b = doc_b.get("kvs", {})
    score_a = kvs_a.get("score", 0.0)
    score_b = kvs_b.get("score", 0.0)
    delta   = score_b - score_a

    console.print(f"  [dim]Before:[/dim] {before_path.name}  KVS {score_a:.1f} [{kvs_a.get('label', '?')}]")
    console.print(f"  [dim]After:[/dim]  {after_path.name}   KVS {score_b:.1f} [{kvs_b.get('label', '?')}]")
    console.print()

    arrow = "↓" if delta < 0 else "↑" if delta > 0 else "→"
    color = "green" if delta < 0 else "red" if delta > 0 else "dim"
    console.print(
        f"  Overall KVS delta: [{color}]{arrow} {abs(delta):.1f}  "
        f"({'improvement' if delta < 0 else 'regression' if delta > 0 else 'no change'})[/{color}]"
    )
    console.print()

    # Dimension-level comparison table
    dims_a = kvs_a.get("dimension_scores", {})
    dims_b = kvs_b.get("dimension_scores", {})
    all_dims = sorted(set(list(dims_a.keys()) + list(dims_b.keys())))

    table = Table(box=box.SIMPLE, show_header=True, header_style="dim white")
    table.add_column("Dimension",     style="dim white", min_width=26)
    table.add_column("Before",        style="white",     justify="right", min_width=8)
    table.add_column("After",         style="white",     justify="right", min_width=8)
    table.add_column("Change",        justify="right",   min_width=10)

    for dim in all_dims:
        va = dims_a.get(dim, 0.0)
        vb = dims_b.get(dim, 0.0)
        d  = vb - va
        chg_text = Text(
            f"{'+' if d > 0 else ''}{d:.1f}",
            style="red" if d > 0 else "green" if d < 0 else "dim white",
        )
        table.add_row(
            dim.replace("_", " ").title(),
            f"{va:.1f}",
            f"{vb:.1f}",
            chg_text,
        )
    console.print(table)

    # Save comparison JSON
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "before": {"file": str(before_path), "kvs": kvs_a},
        "after":  {"file": str(after_path),  "kvs": kvs_b},
        "delta":  {"overall": round(delta, 4),
                   "dimensions": {d: round(dims_b.get(d, 0) - dims_a.get(d, 0), 4)
                                   for d in all_dims}},
    }
    cmp_json = output_dir / "comparison.json"
    with open(cmp_json, "w") as f:
        json.dump(comparison, f, indent=2)
    console.print(f"  Comparison saved to: [bold]{cmp_json}[/bold]")


# =============================================================================
# Helper functions
# =============================================================================

def _print_banner():
    """Print the KAAL header box (Spec 15.1 exact format)."""
    console.print()
    console.print("╔══════════════════════════════════════════════╗", style="dim white")
    console.print("║                    [bold red]KAAL v1.0.0[/bold red]               ║")
    console.print("║     Adversarial Robustness Auditing Tool     ║", style="dim white")
    console.print("╚══════════════════════════════════════════════╝", style="dim white")
    console.print()


def _print_step(quiet: bool, step: int, total: int, title: str, detail: str):
    """Print a numbered pipeline step header."""
    if quiet:
        return
    console.print(f"[bold white][{step}/{total}][/bold white] {title}")
    if detail:
        console.print(f"      [dim]{detail}[/dim]")


def _run_with_progress(quiet: bool, fn, n_items: int):
    """Run fn() with a Rich progress bar, return its result."""
    if quiet:
        return fn()

    with Progress(
        TextColumn("      "),
        BarColumn(bar_width=24, style="red", complete_style="bold red"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=n_items)
        # We can't hook into the fn iteration, so just run it and advance on done
        result = fn()
        progress.update(task, completed=n_items)

    return result


def _print_kvs_display(kvs_result, quiet: bool):
    """Print the KVS score summary exactly as per Spec 15.1."""
    from kaal.scoring.kvs import get_kvs_label, get_kvs_color

    score = kvs_result.score if kvs_result else 0.0
    label = kvs_result.label if kvs_result else "N/A"

    # Score colour mapping to Rich colour names
    def _rich_color(s):
        if s <= 2.0:  return "green"
        if s <= 4.0:  return "yellow_green"
        if s <= 6.0:  return "yellow"
        if s <= 8.0:  return "dark_orange"
        return "bold red"

    color = _rich_color(score)
    filled = int(score / 10 * 20)
    bar = "█" * filled + "░" * (20 - filled)

    console.print("─" * 46, style="dim white")
    console.print()
    console.print(
        f"  [bold]KVS SCORE: [{color}]{score:.1f}[/{color}]"
        f"[/bold]  [{color}]{bar}[/{color}]  "
        f"[bold]{label.upper()}[/bold]"
    )
    console.print()

    if kvs_result:
        for dim, val in kvs_result.dimension_scores.items():
            dim_filled = int(val / 10 * 23)
            dim_bar = "█" * dim_filled + "░" * (23 - dim_filled)
            short = dim.replace("_susceptibility", "").replace("_", " ").title()
            c = _rich_color(val)
            console.print(
                f"  [dim]{short:<12}[/dim]  [{c}]{val:4.1f}[/{c}]  "
                f"[{c}]{dim_bar}[/{c}]"
            )
    console.print()
    console.print("─" * 46, style="dim white")


def _print_output_tree(output_dir: Path):
    """Print the file tree of generated outputs."""
    console.print()
    console.print(f"  [dim]Reports saved to:[/dim] [bold]{output_dir}/[/bold]")
    entries = sorted(output_dir.rglob("*"))
    files   = [e for e in entries if e.is_file()]
    dirs    = {e for e in entries if e.is_dir()}

    shown_dirs: set = set()
    for f in files:
        rel = f.relative_to(output_dir)
        parts = rel.parts
        if len(parts) > 1 and parts[0] not in shown_dirs:
            console.print(f"  [dim]├── {parts[0]}/[/dim]")
            shown_dirs.add(parts[0])
        elif len(parts) == 1:
            console.print(f"  [dim]├── {f.name}[/dim]")


def _save_gradcam(gradcam_cmp, output_dir: Path):
    """Save GradCAM comparison image to output dir."""
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    sbs = getattr(gradcam_cmp, "side_by_side_pil", None)
    if sbs is not None:
        sbs.save(str(gradcam_dir / "gradcam_comparison.png"))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    app()
