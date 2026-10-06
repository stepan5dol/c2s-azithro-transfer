"""
plot_style.py — shared publication style for figures/figures_beautiful/*.

One place to fix fonts, dpi, and the color vocabulary so every "beautiful"
figure in this folder reads as one consistent system instead of N one-off
matplotlib defaults. Import and call apply() once at the top of each
*_beautiful.py script, before creating any figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

GT_COLOR = "#4C72B0"
PRED_COLOR = "#DD8452"
HO_COLOR = "#8C8C8C"

CT_COLORS = {
    "gCap":     "#2CA02C",
    "aCap":     "#D62728",
    "Pericyte": "#1F77B4",
    "VEC":      "#9467BD",
}

COND_COLORS = {
    "HO":  HO_COLOR,
    "AZI": "#17BECF",
}

COND_COLORS_HUMAN = {
    "Acute":   HO_COLOR,
    "BPD":     "#17BECF",
    "BPD-PH":  "#E45756",
}

SIG_COLOR = "#333333"

DPI = 300

FS_SUPTITLE = 12
FS_TITLE = 11
FS_LABEL = 10
FS_TICK = 9.5
FS_LEGEND = 9
FS_ANNOT = 9

PANEL_SIZE = (4.2, 5.0)


def apply() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def out_dir(fig_dir: Path, tag: str) -> Path:
    """figures/figures_beautiful/<tag>/ , one subfolder per sampling temperature."""
    d = fig_dir / "figures_beautiful" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


SIG_ALPHA = 0.05


def sig_label(p: float) -> str:
    """The printed p-value. Stars and "ns" are deliberately not used: a reader
    comparing two arms of a figure otherwise compares star counts, which are a
    function of n as much as of effect size -- and the two arms here differ in
    n by an order of magnitude (see the power curve in autophagy_panel/reports/
    rat_topk_detectability_gcap_up_power_curve.csv). The number is printed
    always; significance is carried by weight, not by a separate token."""
    if p < 1e-3:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


sig_stars = sig_label


def sig_bracket(ax, x1: float, x2: float, y: float, h: float, p: float, color: str = SIG_COLOR,
                 fontsize: float = 9) -> None:
    """Draw one horizontal significance bracket between x1 and x2 at height y
    (bracket arms rise by h), labelled with the p-value itself -- bold when
    p < SIG_ALPHA, normal weight otherwise."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color=color)
    ax.text((x1 + x2) / 2, y + h, sig_label(p), ha="center", va="bottom",
            fontsize=fontsize, color=color,
            fontweight="bold" if p < SIG_ALPHA else "normal")
