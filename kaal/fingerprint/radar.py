"""Vulnerability fingerprint radar chart generator.

Spec 13.1 — Phase 7, Kiro Prompt 7.2.

Generates a hexagonal radar/spider chart showing model vulnerability
across all six KVS dimensions. Every model gets a unique visual shape.

Chart specifications (exact KAAL dark theme):
    Type             : Radar / Spider (hexagon — 6 axes)
    Background       : #0A0A0A
    Fill             : #CC0000 at 30% opacity (alpha=0.3)
    Border           : #CC0000, linewidth 2
    Dots             : #CC0000, radius 4
    Grid circles     : #1F1F1F, dashed
    Axis labels      : #F2F2F2, monospace
    Scale per axis   : 0 (center/robust) to 10 (outer/vulnerable)

Axes order (clockwise from top):
    1. FGSM Susceptibility        (top)
    2. PGD Susceptibility         (top-right)
    3. Physical Survivability     (bottom-right)
    4. Black-Box Efficiency       (bottom-left)
    5. Empirical Robustness       (top-left)
    6. Adversarial Overconfidence (left)

Comparison mode:
    Primary model  : #CC0000 fill + border
    Comparison     : #F2F2F2 border only, 20% opacity, dashed
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyparsing")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

import math
import os
from typing import Optional

import numpy as np

from kaal.engine.utils import ensure_dir


# ---------------------------------------------------------------------------
# Axis definitions — order matches Spec 13.1
# ---------------------------------------------------------------------------

_AXES: list[str] = [
    "fgsm_susceptibility",
    "pgd_susceptibility",
    "physical_survivability",
    "blackbox_efficiency",
    "empirical_robustness",
    "adversarial_overconfidence",
]

_AXIS_LABELS: list[str] = [
    "FGSM",
    "PGD",
    "Physical",
    "Black-box",
    "Empirical",
    "Overconf",
]

# KAAL colour constants
_BG         = "#0A0A0A"
_SURFACE    = "#111111"
_ACCENT     = "#CC0000"
_TEXT       = "#F2F2F2"
_MUTED      = "#888888"
_GRID       = "#1F1F1F"


# ---------------------------------------------------------------------------
# generate_fingerprint() — main entry point
# ---------------------------------------------------------------------------

def generate_fingerprint(
    kvs_result,
    model_name: str,
    output_path: str,
    comparison_kvs=None,
    comparison_name: Optional[str] = None,
    dpi: int = 150,
) -> str:
    """Generate and save a vulnerability fingerprint radar chart.

    Args:
        kvs_result:      KVSResult from calculate_kvs(). Uses dimension_scores.
        model_name:      Model name shown as chart title.
        output_path:     Full path to save the PNG (parent dir created if needed).
        comparison_kvs:  Optional second KVSResult to overlay.
                         Shown as white dashed outline, no fill.
        comparison_name: Name label for the comparison model.
        dpi:             Output DPI. Default 150 (minimum per Spec 13.1).

    Returns:
        Absolute path to the saved PNG file.

    Raises:
        ImportError: matplotlib is not installed.
        ValueError:  No dimension scores available in kvs_result.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        raise ImportError(
            "matplotlib is not installed.\n"
            "→ Install it with: pip install matplotlib==3.8.3"
        )

    output_path = os.path.abspath(output_path)
    ensure_dir(os.path.dirname(output_path))

    # ── Extract scores for primary model ─────────────────────────────────────
    primary_scores = _extract_scores(kvs_result)

    if all(v == 0.0 for v in primary_scores):
        # All zeros is valid (highly robust model) — proceed normally
        pass

    # ── Extract scores for comparison model (optional) ───────────────────────
    comparison_scores: Optional[list[float]] = None
    if comparison_kvs is not None:
        comparison_scores = _extract_scores(comparison_kvs)

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 7), facecolor=_BG, dpi=dpi)
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(_SURFACE)

    # Number of axes
    N     = len(_AXES)
    angles = _make_angles(N)    # N+1 angles (last == first to close polygon)

    # ── Grid styling ─────────────────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])     # we draw our own labels below

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(
        ["2", "4", "6", "8", "10"],
        color=_MUTED, fontsize=7,
    )
    ax.set_ylim(0, 10)

    # Grid lines
    ax.grid(color=_GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_color(_GRID)

    # ── Draw primary model polygon ───────────────────────────────────────────
    _draw_radar(
        ax, angles, primary_scores,
        fill_color=_ACCENT,
        fill_alpha=0.30,
        line_color=_ACCENT,
        line_width=2.0,
        dot_color=_ACCENT,
        dot_size=40,
        linestyle="-",
        label=model_name,
        zorder=3,
    )

    # ── Draw comparison model polygon ────────────────────────────────────────
    if comparison_scores is not None:
        _draw_radar(
            ax, angles, comparison_scores,
            fill_color=_TEXT,
            fill_alpha=0.08,
            line_color=_TEXT,
            line_width=1.5,
            dot_color=_TEXT,
            dot_size=20,
            linestyle="--",
            label=comparison_name or "Comparison",
            zorder=2,
        )

    # ── Axis labels (outside the chart) ──────────────────────────────────────
    label_radius = 11.5
    for i, (angle, label) in enumerate(zip(angles[:-1], _AXIS_LABELS)):
        ha = _label_ha(angle)
        va = _label_va(angle)
        score_val = primary_scores[i]
        ax.text(
            angle, label_radius,
            f"{label}\n{score_val:.1f}",
            ha=ha, va=va,
            color=_TEXT,
            fontsize=9,
            fontfamily="monospace",
            fontweight="bold",
        )

    # ── Title ────────────────────────────────────────────────────────────────
    kvs_score = getattr(kvs_result, "score", None)
    kvs_label = getattr(kvs_result, "label", "")
    kvs_color = getattr(kvs_result, "color", _ACCENT)

    title_text = model_name
    if kvs_score is not None:
        title_text = f"{model_name}   KVS {kvs_score:.1f}  [{kvs_label}]"

    fig.text(
        0.5, 0.97,
        title_text,
        ha="center", va="top",
        color=_TEXT,
        fontsize=11,
        fontfamily="monospace",
        fontweight="bold",
    )

    # KVS score coloured dot beside title
    if kvs_score is not None:
        fig.patches.append(
            mpatches.FancyBboxPatch(
                (0.5 - 0.01, 0.94), 0.02, 0.02,
                boxstyle="circle,pad=0.0",
                facecolor=kvs_color,
                transform=fig.transFigure,
                zorder=10,
            )
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    if comparison_scores is not None:
        handles = [
            mpatches.Patch(facecolor=_ACCENT, edgecolor=_ACCENT,
                           alpha=0.5, label=model_name),
            mpatches.Patch(facecolor=_TEXT, edgecolor=_TEXT,
                           alpha=0.3, label=comparison_name or "Comparison"),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            facecolor=_BG,
            edgecolor=_GRID,
            labelcolor=_TEXT,
            fontsize=8,
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(pad=2.0)
    plt.savefig(
        output_path, dpi=dpi,
        bbox_inches="tight",
        facecolor=_BG,
        edgecolor="none",
    )
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_angles(n: int) -> np.ndarray:
    """Return n+1 angles (radians) evenly spaced starting from top (π/2)."""
    # Start at top (90°) going clockwise
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    # Close the polygon
    return np.concatenate([angles, [angles[0]]])


def _draw_radar(
    ax,
    angles: np.ndarray,
    scores: list[float],
    fill_color: str,
    fill_alpha: float,
    line_color: str,
    line_width: float,
    dot_color: str,
    dot_size: float,
    linestyle: str,
    label: str,
    zorder: int,
) -> None:
    """Draw one radar polygon on the polar axes."""
    # Close the data polygon
    values = list(scores) + [scores[0]]

    ax.plot(
        angles, values,
        color=line_color,
        linewidth=line_width,
        linestyle=linestyle,
        zorder=zorder,
        label=label,
        solid_capstyle="round",
    )
    ax.fill(
        angles, values,
        color=fill_color,
        alpha=fill_alpha,
        zorder=zorder - 1,
    )
    # Vertex dots
    ax.scatter(
        angles[:-1], scores,
        color=dot_color,
        s=dot_size,
        zorder=zorder + 1,
    )


def _extract_scores(kvs_result) -> list[float]:
    """Extract the six axis scores from a KVSResult in _AXES order.

    Missing dimensions default to 0.0.
    """
    dim_scores: dict = getattr(kvs_result, "dimension_scores", {}) or {}
    return [float(dim_scores.get(dim, 0.0)) for dim in _AXES]


def _label_ha(angle: float) -> str:
    """Horizontal alignment for a polar axis label at given angle (radians)."""
    x = math.cos(angle)
    if x > 0.1:
        return "left"
    elif x < -0.1:
        return "right"
    return "center"


def _label_va(angle: float) -> str:
    """Vertical alignment for a polar axis label at given angle (radians)."""
    y = math.sin(angle)
    if y > 0.1:
        return "bottom"
    elif y < -0.1:
        return "top"
    return "center"
