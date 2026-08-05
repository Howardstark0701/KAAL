"""Confidence collapse curve generator — Spec 11.3 — Phase 6, Kiro Prompt 6.3.

For PGD attacks specifically — plots model confidence on the correct class
vs attack iteration. Generates a chart showing the exact moment confidence
collapses, saved as a PNG.

Chart style (exact KAAL dark theme):
    Figure background : #0A0A0A
    Axes background   : #111111
    Line color        : #CC0000, linewidth 2
    Grid              : #1F1F1F, dashed, alpha 0.5
    Text              : #F2F2F2
    Collapse marker   : vertical dashed line at steps_to_success in #888888
    X-axis label      : "PGD Iteration"
    Y-axis label      : "Model Confidence (Original Class)"
    Title             : "Confidence Collapse Curve"
    DPI               : 150 minimum
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyparsing")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import os
from typing import Optional

from kaal.engine.utils import ensure_dir


# ---------------------------------------------------------------------------
# generate_collapse_curve()
# ---------------------------------------------------------------------------

def generate_collapse_curve(
    pgd_result,
    output_path: str,
) -> str:
    """Generate and save a confidence collapse curve chart from a PGDResult.

    Args:
        pgd_result:  A PGDResult object. Uses confidence_per_step and
                     steps_to_success fields.
        output_path: Full path to save the PNG file (e.g. './output/collapse.png').
                     Parent directory is created automatically.

    Returns:
        Absolute path to the saved PNG file.

    Raises:
        ValueError: pgd_result has no confidence_per_step data.
        ImportError: matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for all platforms
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        raise ImportError(
            "matplotlib is not installed.\n"
            "→ Install it with: pip install matplotlib==3.8.3"
        )

    if not pgd_result.confidence_per_step:
        raise ValueError(
            "pgd_result.confidence_per_step is empty.\n"
            "→ Run pgd_attack() with steps >= 1 to collect confidence data."
        )

    # Resolve output path
    output_path = os.path.abspath(output_path)
    ensure_dir(os.path.dirname(output_path))

    steps          = list(range(1, len(pgd_result.confidence_per_step) + 1))
    confidences    = pgd_result.confidence_per_step
    steps_to_succ  = pgd_result.steps_to_success   # -1 if attack failed

    # ── KAAL colour constants ────────────────────────────────────────────────
    BG          = "#0A0A0A"
    SURFACE     = "#111111"
    ACCENT      = "#CC0000"
    TEXT        = "#F2F2F2"
    MUTED       = "#888888"
    GRID        = "#1F1F1F"

    # ── Figure setup ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # ── Confidence line ──────────────────────────────────────────────────────
    ax.plot(
        steps, confidences,
        color=ACCENT,
        linewidth=2,
        solid_capstyle="round",
        label="Confidence (original class)",
        zorder=3,
    )

    # Shaded area under curve
    ax.fill_between(steps, confidences, alpha=0.12, color=ACCENT, zorder=2)

    # ── Collapse marker ──────────────────────────────────────────────────────
    if steps_to_succ != -1:
        ax.axvline(
            x=steps_to_succ,
            color=MUTED,
            linestyle="--",
            linewidth=1.2,
            label=f"First misclassification (step {steps_to_succ})",
            zorder=4,
        )
        # Annotate the collapse point
        collapse_conf = confidences[steps_to_succ - 1]
        ax.annotate(
            f"step {steps_to_succ}\n{collapse_conf:.2f}",
            xy=(steps_to_succ, collapse_conf),
            xytext=(steps_to_succ + max(1, len(steps) * 0.05),
                    collapse_conf + 0.08),
            color=MUTED,
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
        )

    # ── Starting confidence dot ──────────────────────────────────────────────
    ax.scatter([steps[0]], [confidences[0]],
               color=TEXT, s=30, zorder=5, label=f"Start: {confidences[0]:.2f}")

    # ── Grid ─────────────────────────────────────────────────────────────────
    ax.grid(
        True,
        color=GRID,
        linestyle="--",
        linewidth=0.6,
        alpha=0.5,
        zorder=1,
    )
    ax.set_axisbelow(True)

    # ── Axis limits and labels ───────────────────────────────────────────────
    x_min = 1
    x_max = max(steps) if len(steps) > 1 else 2   # avoid degenerate xlim on single-step
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.02, 1.05)

    ax.set_xlabel("PGD Iteration",
                  color=TEXT, fontsize=11, labelpad=8)
    ax.set_ylabel("Model Confidence (Original Class)",
                  color=TEXT, fontsize=11, labelpad=8)
    ax.set_title("Confidence Collapse Curve",
                 color=TEXT, fontsize=13, fontweight="bold", pad=14)

    # ── Tick styling ─────────────────────────────────────────────────────────
    ax.tick_params(colors=TEXT, which="both", length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.setp(ax.get_xticklabels(), color=TEXT, fontsize=9)
    plt.setp(ax.get_yticklabels(), color=TEXT, fontsize=9)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc="upper right",
        facecolor=BG,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        framealpha=0.85,
    )

    # ── Subtitle with attack params ──────────────────────────────────────────
    subtitle_parts = [
        f"ε={pgd_result.epsilon_used:.4f}".rstrip("0").rstrip("."),
        f"α={pgd_result.alpha_used:.5f}".rstrip("0").rstrip("."),
        f"{pgd_result.steps_used} steps",
    ]
    if pgd_result.restarts_used > 1:
        subtitle_parts.append(f"{pgd_result.restarts_used} restarts")

    fig.text(
        0.5, 0.01,
        "  |  ".join(subtitle_parts),
        ha="center", va="bottom",
        color=MUTED, fontsize=8,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)

    return output_path
