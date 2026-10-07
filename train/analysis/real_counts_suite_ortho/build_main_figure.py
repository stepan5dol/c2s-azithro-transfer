#!/usr/bin/env python3
"""
Assembles the main figure from three finished panels and writes its caption.

    A   human, term-born control vs BPD 7 months, gCap
    B   rat, hyperoxia vs hyperoxia + azithromycin, gCap
    C   cameraPR screen of the azithromycin prediction, four cell types

Panels are placed at their native size (pixels / plot_style.DPI), which keeps
the font sizes of plot_style.

    python build_main_figure.py -> figures_main/main_figure.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ABS_DIR = HERE.parent
sys.path.insert(0, str(ABS_DIR))
import plot_style as PS

PANEL_A = HERE / "human_panel_variants/figures/human_delta_term_born_to_bpd_7mo_all_rescued_gcap.png"
PANEL_B = HERE / "rat_panel_variants/figures/rat_delta_hyperoxia_to_azithromycin_all_rescued_gcap.png"
PANEL_C = HERE / "deck_figures/camerapr_panelC_dotplot.png"

OUT_DIR = HERE / "figures_main"
FIG_NO = "Fig N"

MARGIN = 0.32
GAP_X = 0.10
GAP_Y = 0.14

CAPTION = (
    f"{FIG_NO}.  Measured and predicted single-cell perturbation deltas, each taken relative to "
    "the same baseline cell, and the pathway screen of the azithromycin prediction.  "
    "(A) Human, term-born control versus bronchopulmonary dysplasia at 7 months postnatal.  "
    "(B) Rat, hyperoxia versus hyperoxia plus azithromycin.  Both panels show general capillary "
    "endothelial cells; the other three cell types are in the supplementary figures.  Each violin "
    "is the distribution over cells of the mean delta across one gene group, and p values are "
    "two-sided Mann-Whitney against background within a block.  "
    "(C) Four panels out of a cameraPR (limma) screen over all 8421 terms of five Enrichr "
    "libraries, genes split by the direction the disease moved them.  The two inflammatory panels "
    "are the strongest signal the screen returns; the two autophagy panels are shown because the "
    "rat azithromycin experiment pointed at autophagy before this screen was run, and "
    "Macroautophagy is a 114-gene subset of the 128-gene Autophagy, so those two rows are one "
    "signal drawn twice.  Black outline marks FDR < 0.05, opacity is -log10 FDR, and dot area is "
    "the number of genes in that half, printed above each dot."
)


def native_size(path: Path) -> tuple[float, float]:
    """Panel size in inches at plot_style.DPI."""
    if not path.exists():
        raise SystemExit(f"panel missing: {path}")
    h, w = mpimg.imread(path).shape[:2]
    return w / PS.DPI, h / PS.DPI


def main():
    OUT_DIR.mkdir(exist_ok=True)

    (aw, ah), (bw, bh), (cw, ch) = (native_size(p) for p in (PANEL_A, PANEL_B, PANEL_C))
    row1_w = aw + GAP_X + bw
    content_w = max(row1_w, cw)
    fig_w = content_w + 2 * MARGIN
    fig_h = 2 * MARGIN + max(ah, bh) + GAP_Y + ch

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    def put(path, x, y, w, h, letter):
        ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
        ax.imshow(mpimg.imread(path), aspect="auto")
        ax.set_axis_off()
        fig.text((x - 0.24) / fig_w, (y + h) / fig_h, letter, ha="left", va="top",
                 fontsize=15, fontweight="bold", color="#222222")

    row1_y = MARGIN + ch + GAP_Y
    x0 = MARGIN + (content_w - row1_w) / 2
    put(PANEL_A, x0, row1_y, aw, ah, "A")
    put(PANEL_B, x0 + aw + GAP_X, row1_y, bw, bh, "B")
    put(PANEL_C, MARGIN + (content_w - cw) / 2, MARGIN, cw, ch, "C")

    (OUT_DIR / "main_figure_caption.txt").write_text(CAPTION + "\n")
    print(f"-> {OUT_DIR / 'main_figure_caption.txt'}")

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"main_figure.{ext}"
        fig.savefig(out, dpi=PS.DPI, facecolor="white")
        print(f"-> {out}  ({fig_w:.2f} x {fig_h:.2f} in)")
    plt.close(fig)


if __name__ == "__main__":
    main()
