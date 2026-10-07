#!/usr/bin/env python3
"""
Panel C of the main figure: four terms of the cameraPR screen
(human_panel_variants/limma_sets.py, azi arm), per cell type and per half of
the term (genes lowered or raised by disease).

Terms: Hallmark Inflammatory Response and Interferon Gamma Response; Reactome
Autophagy and Macroautophagy (Macroautophagy is a 114-gene subset of the
128-gene Autophagy set).

Colour: toward or away from the control; black outline: cameraPR FDR < 0.05;
opacity: -log10 FDR; dot area and printed number: genes in the half.

    python build_panelC_dotplot.py [--cohort BPD7mo|BPDPH7mo|Acute26]
        -> deck_figures/camerapr_panelC_dotplot.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "deck_figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from deck_figures import GRID, INK, INK2, MUTED, SEQ_HI, style
import plot_style as PS

REPORTS = HERE / "human_panel_variants" / "autophagy_program" / "reports"

COHORTS = {
    "BPD7mo":   ("bpd7mo_limma_sets.csv",   "Human, term-born versus BPD at 7 months"),
    "BPDPH7mo": ("bpdph7mo_limma_sets.csv", "Human, term-born versus BPD-PH at 7 months"),
    "Acute26":  ("acute26_limma_sets.csv",  "Human, gestational week 24 versus acute injury, week 26"),
}

CTS = ["gCap", "aCap", "Pericyte", "VEC"]
POS, NEG = SEQ_HI, "#e34948"

TERMS = [
    ("Inflammatory Response", "MSigDB_Hallmark_2020__Inflammatory Response"),
    ("Interferon Gamma Response", "MSigDB_Hallmark_2020__Interferon Gamma Response"),
    ("Autophagy", "Reactome_2022__Autophagy R-HSA-9612973"),
    ("Macroautophagy", "Reactome_2022__Macroautophagy R-HSA-1632852"),
]


def load(sets_csv: Path):
    c = pd.read_csv(sets_csv)
    c = c[c.arm == "azi"].copy()
    c["half"] = c.term.str.extract(r"@@(dn|up)")
    c["base"] = c.term.str.replace(r" @@(dn|up)", "", regex=True)
    c = c[c.half.notna()]
    return c.set_index(["base", "cell_type", "half"])


def dotplot(cam: pd.DataFrame, cohort: str, subtitle: str) -> None:
    PS.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, half, want, title in (
            (axes[0], "dn", "Up", "lowered by disease  ·  should rise"),
            (axes[1], "up", "Down", "raised by disease  ·  should fall")):
        style(ax)
        for yi, (label, term) in enumerate(TERMS):
            for xi, ct in enumerate(CTS):
                key = (term, ct, half)
                if key not in cam.index:
                    ax.text(xi, yi, "·", ha="center", va="center", color=MUTED, fontsize=11)
                    continue
                row = cam.loc[key]
                n, q = int(row.NGenes), float(row.camerapr_fdr)
                right = row.camerapr_direction == want
                v = -np.log10(max(q, 1e-12))
                ax.scatter([xi], [yi], s=26 + 7.0 * n, zorder=3,
                           color=(POS if right else NEG),
                           alpha=min(0.28 + 0.24 * v, 1.0),
                           edgecolor=INK if q < 0.05 else "none", linewidth=1.4)
                ax.text(xi, yi - 0.34, f"{n}", ha="center", va="center",
                        fontsize=6.8, color=MUTED, zorder=4)
        ax.set_xticks(range(len(CTS)))
        ax.set_xticklabels(CTS, fontsize=PS.FS_TICK, color=INK)
        ax.set_xlim(-0.6, len(CTS) - 0.4)
        ax.set_ylim(len(TERMS) - 0.45, -0.55)
        ax.set_yticks(range(len(TERMS)))
        ax.set_yticklabels([lab for lab, _ in TERMS], fontsize=PS.FS_TICK, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=PS.FS_TITLE, color=INK, loc="left", pad=10)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc=POS, mec="none", ms=9,
               label="moves back toward the control"),
        Line2D([], [], marker="o", ls="none", mfc=NEG, mec="none", ms=9,
               label="moves further from it"),
        Line2D([], [], marker="o", ls="none", mfc=MUTED, mec=INK, ms=9, mew=1.4,
               label="black outline = cameraPR FDR < 0.05"),
        Line2D([], [], marker="o", ls="none", mfc=MUTED, mec="none", ms=6,
               label="opacity = −log10 FDR,  area = genes in the half (printed)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False,
        fontsize=PS.FS_LEGEND, labelcolor=INK2, handletextpad=0.5, columnspacing=1.8)

    fig.patch.set_facecolor("white")
    for a in axes:
        a.set_facecolor("white")
    fig.suptitle(subtitle, fontsize=PS.FS_SUPTITLE, y=1.02)
    stem = "camerapr_panelC_dotplot" + ("" if cohort == "BPD7mo" else f"_{cohort.lower()}")
    out = OUT / f"{stem}.png"
    fig.savefig(out, dpi=PS.DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}")


def report(cam: pd.DataFrame, cohort: str) -> None:
    rows = []
    for label, term in TERMS:
        for half, want in (("up", "Down"), ("dn", "Up")):
            for ct in CTS:
                key = (term, ct, half)
                if key not in cam.index:
                    continue
                r = cam.loc[key]
                rows.append({
                    "panel": label,
                    "half": "raised by disease" if half == "up" else "lowered by disease",
                    "cell_type": ct, "n_genes": int(r.NGenes),
                    "direction": r.camerapr_direction,
                    "fdr": float(r.camerapr_fdr),
                    "significant": bool(r.camerapr_fdr < 0.05 and r.camerapr_direction == want),
                })
    d = pd.DataFrame(rows)
    stem = "camerapr_panelC_dotplot" + ("" if cohort == "BPD7mo" else f"_{cohort.lower()}")
    d.to_csv(OUT / f"{stem}.csv", index=False)
    for half, g in d.groupby("half"):
        print(f"\n=== {half}")
        print(g.pivot_table(index="panel", columns="cell_type", values="fdr").round(4).to_string())
        print("  significant cell types per panel:",
              g[g.significant].groupby("panel").cell_type.count().to_dict())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="BPD7mo", choices=sorted(COHORTS))
    cohort = ap.parse_args().cohort
    csv_name, subtitle = COHORTS[cohort]
    print(f"[cohort] {cohort} <- {csv_name}")
    cam = load(REPORTS / csv_name)
    dotplot(cam, cohort, subtitle)
    report(cam, cohort)


if __name__ == "__main__":
    main()
