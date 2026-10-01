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

# Semantic roles, reused across every figure that needs them.
GT_COLOR = "#4C72B0"      # real / ground-truth cells
PRED_COLOR = "#DD8452"    # model-generated cells
HO_COLOR = "#8C8C8C"      # disease / hyperoxia baseline (gray = "before")

CT_COLORS = {
    "gCap":     "#2CA02C",  # green
    "aCap":     "#D62728",  # red
    "Pericyte": "#1F77B4",  # blue
    "VEC":      "#9467BD",  # purple
}

COND_COLORS = {
    "HO":  HO_COLOR,   # disease / hyperoxia -- same gray used for "disease" everywhere else
    "AZI": "#17BECF",  # treated / azithromycin -- cyan, distinct from CT_COLORS and GT/PRED
}

COND_COLORS_HUMAN = {
    "Acute":   HO_COLOR,   # acute preterm injury -- gray, same "disease/baseline" role as rat HO
    "BPD":     "#17BECF",  # chronic outcome 1 -- cyan, same slot as rat AZI
    "BPD-PH":  "#E45756",  # chronic outcome 2 (distinct subtype, not a progression from BPD) -- red
}

SIG_COLOR = "#333333"

DPI = 300

# Panel typography. Figures that are meant to be assembled into one main figure
# must not each carry their own font sizes -- a panel scaled to fit then arrives
# with text a different size from its neighbours. Both delta cores read these,
# so the sizes are identical across species and can be changed in one place.
FS_SUPTITLE = 12    # figure title
FS_TITLE = 11       # panel title: cell type and n
FS_LABEL = 10       # axis label
FS_TICK = 9.5       # tick labels
FS_LEGEND = 9       # legend entries
FS_ANNOT = 9        # arm labels above the violins

# One violin panel of a single cell type, in inches. Fixed so the three panels
# of the main figure arrive at the same scale.
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


# Older name, kept so existing callers keep working; same p-value output.
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
