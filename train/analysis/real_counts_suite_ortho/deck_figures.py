#!/usr/bin/env python3
"""
Figures for the autophagy analysis of the human BPD7mo arm:

    autophagy_terms.png   autophagy terms lowered by BPD7mo and the predicted
                          azithromycin shift of each half of the term
    funnel.png            autophagy genes remaining after each filter step
    ceiling.png           model fold enrichment against the real-real ceiling
    recovery_ratio.png    recovery_ratio per autophagy gene lowered by disease

Also defines the colours and axis style used by build_panelC_dotplot.py.

    python deck_figures.py -> deck_figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "deck_figures"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent))
from plot_style import GT_COLOR, PRED_COLOR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
ACCENT_FIG = "#b03a2e"
RAMP = ["#c8d6e8", "#7b9fd0", GT_COLOR]

SEQ_HI, SEQ_LO = "#2a78d6", "#86b6ef"
DIV_POS, DIV_NEG, DIV_MID = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "recovery", ["#0d366b", DIV_POS, "#9ec5f4", DIV_MID, "#f4aeae", DIV_NEG, "#8f2626"][::-1])

CTS = ["gCap", "aCap", "Pericyte", "VEC"]
SHORTC = {"BPDPH7mo": "BPD-PH", "BPD7mo": "BPD7mo", "Acute26": "Acute26"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> deck_figures/{name}")


def rd(rel):
    return pd.read_csv(HERE / rel)


LIB_TAG = {"GO_Biological_Process_2023": "GO", "KEGG_2021_Human": "KEGG",
           "Reactome_2022": "Reactome", "WikiPathways_2024_Human": "WikiPathways"}


def short(term: str, lib: str | None = None) -> str:
    """Term name without the accession, with the library tag when lib is given."""
    import re
    t = re.sub(r"\s*R-HSA-?\d*\s*$", "", term)
    t = re.sub(r"\s*\(GO:\d+\)\s*$", "", t)
    t = re.sub(r"\s*WP\d+\s*$", "", t)
    return f"{t.strip()}  [{LIB_TAG.get(lib, lib)}]" if lib else t.strip()


def _term_halves(terms, condition="BPD7mo"):
    """Per term and cell type, the predicted azithromycin shift of the two
    halves of the term separately:

        recovery = log2FC_AZI - log2FC_disease, both against the Term control
        down     = members with log2FC_disease < -0.25, expected recovery > 0
        up       = members with log2FC_disease > +0.25, expected recovery < 0

    p: one-sample Wilcoxon signed-rank test, NaN below 6 members.
    """
    import json
    from scipy.stats import wilcoxon

    blob = json.loads((HERE / "autophagy_panel/reports/autophagy_terms_found.json").read_text())
    members = {}
    for lib_terms in blob.values():
        if isinstance(lib_terms, dict):
            for t, genes in lib_terms.items():
                members[t] = set(genes)

    d = rd("human_panel_variants/reports/human_rescue_genomewide.csv")
    d = d[(d.condition == condition) & d.passes_min_pct_disease & d.passes_min_pct_azi].copy()
    d["rec"] = d.avg_log2FC_azi_top800 - d.avg_log2FC_disease_top800

    rows = []
    for t in terms:
        for ct in CTS:
            g = d[(d.cell_type == ct) & d.gene.isin(members.get(t, set()))]
            row = dict(term=t, cell_type=ct, n_elig=len(g),
                       n_flat=int((g.avg_log2FC_disease_top800.abs() <= 0.25).sum()))
            for half, sub in (("down", g[g.avg_log2FC_disease_top800 < -0.25]),
                              ("up", g[g.avg_log2FC_disease_top800 > 0.25])):
                v = sub.rec.dropna()
                row[f"n_{half}"] = len(v)
                row[f"med_{half}"] = v.median() if len(v) else float("nan")
                row[f"p_{half}"] = wilcoxon(v).pvalue if len(v) >= 6 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def autophagy_terms():
    """Left: number of cell types in which BPD7mo lowers each autophagy term
    (GSEA NES). Right: for the terms lowered in all four, the median predicted
    shift of each half; colour is signed by the expected direction of the half,
    the printed number is the raw median.
    """
    c = rd("human_panel_variants/autophagy_program/reports/bpd7mo_arms_concordance.csv")
    a = c[c.is_autophagy_term & (c.condition == "BPD7mo")].dropna(
        subset=["NES_disease", "NES_recovery"])

    rows = []
    for t, d in a.groupby("term_name"):
        if len(d) < 4:
            continue
        rows.append(dict(
            term=t, lib=d.library.iloc[0],
            falls=int((d.NES_disease < 0).sum()),
            falls_sig=int(((d.NES_disease < 0) & (d.q_disease < 0.25)).sum()),
            med_dis=d.NES_disease.median(),
            qlo=d.loc[d.NES_disease < 0, "q_disease"].min() if (d.NES_disease < 0).any() else float("nan"),
            qhi=d.loc[d.NES_disease < 0, "q_disease"].max() if (d.NES_disease < 0).any() else float("nan")))
    r = pd.DataFrame(rows)
    n_never = int((r.falls == 0).sum())
    r = r[r.falls > 0].sort_values(["falls", "falls_sig", "med_dis"],
                                   ascending=[False, False, True]).reset_index(drop=True)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16.0, 4.7),
                                   gridspec_kw={"width_ratios": [1.0, 1.45], "wspace": 0.60})

    style(axA)
    y = np.arange(len(r))
    axA.barh(y, r.falls, height=0.60, color=SEQ_LO, zorder=2)
    axA.barh(y, r.falls_sig, height=0.60, color=SEQ_HI, zorder=3)
    axA.set_yticks(y)
    axA.set_yticklabels([short(t, l) for t, l in zip(r.term, r.lib)], fontsize=9, color=INK)
    axA.set_xlim(0, 4.9)
    axA.set_xticks([0, 1, 2, 3, 4])
    axA.set_ylim(len(r) - 0.45, -0.75)
    axA.set_xlabel("cell types (of 4) in which BPD7mo lowers the term NET\n"
                   "(GSEA over all its members at once — every term holds both directions)",
                   fontsize=9.5, color=INK2, linespacing=1.5)
    axA.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for yi, (f, lo, hi) in enumerate(zip(r.falls, r.qlo, r.qhi)):
        q = f"q {lo:.3f}" if abs(hi - lo) < 5e-4 else f"q {lo:.3f}–{hi:.2f}"
        axA.text(f + 0.10, yi, q, va="center", fontsize=8, color=INK2)
    axA.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=SEQ_HI, ec="none", label="q < 0.25"),
        plt.Rectangle((0, 0), 1, 1, fc=SEQ_LO, ec="none", label="q ≥ 0.25")],
        loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2,
        handlelength=1.1, handletextpad=0.5)
    axA.set_title("A   Which autophagy terms BPD7mo lowers", fontsize=10.5,
                  color=INK, loc="left", pad=10, fontweight="bold")
    axA.text(0.0, -0.335, f"{n_never} further autophagy-named terms fall in no cell type "
                          f"and are not drawn", transform=axA.transAxes,
             fontsize=8, color=MUTED, va="top")

    core = r[(r.falls == 4) & (r.falls_sig >= 3)].sort_values("med_dis")
    h = _term_halves(list(core.term)).set_index(["term", "cell_type"])
    terms = list(core.term)
    axB.set_facecolor(SURFACE)
    for sp in axB.spines.values():
        sp.set_visible(False)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_xlim(-0.5, 8.5); axB.set_ylim(len(terms) - 0.5, -1.9)

    VMAX = 1.25
    for i, t in enumerate(terms):
        for j, ct in enumerate(CTS):
            row = h.loc[(t, ct)]
            for k, (half, sgn) in enumerate((("down", +1.0), ("up", -1.0))):
                x = j * 2 + k
                med, p, n = row[f"med_{half}"], row[f"p_{half}"], int(row[f"n_{half}"])
                if n == 0:
                    axB.add_patch(plt.Rectangle((x - 0.46, i - 0.42), 0.92, 0.84,
                                                fc=SURFACE, ec=GRID, lw=1.0, zorder=2))
                    axB.text(x, i, "—", ha="center", va="center", fontsize=9, color=MUTED, zorder=3)
                    continue
                sig = np.isfinite(p) and p < 0.05
                fc = DIVERGING((sgn * med) / (2 * VMAX) + 0.5)
                axB.add_patch(plt.Rectangle((x - 0.46, i - 0.42), 0.92, 0.84, fc=fc,
                                            ec=INK if sig else SURFACE,
                                            lw=1.6 if sig else 1.0, zorder=2))
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                tc = "#ffffff" if lum < 0.55 else INK
                axB.text(x, i - 0.10, f"{med:+.2f}", ha="center", va="center", fontsize=9.5,
                         color=tc, fontweight="bold" if sig else "normal", zorder=3)
                axB.text(x, i + 0.24, f"n={n}", ha="center", va="center", fontsize=7,
                         color=tc, alpha=0.85, zorder=3)

    for i, t in enumerate(terms):
        axB.text(-0.65, i, short(t, None), ha="right", va="center", fontsize=9.5, color=INK)
    for j, ct in enumerate(CTS):
        axB.text(j * 2 + 0.5, -1.32, ct, ha="center", va="center", fontsize=10,
                 color=INK, fontweight="bold")
        axB.plot([j * 2 - 0.46, j * 2 + 1.46], [-1.02, -1.02], color=GRID, lw=1.2,
                 clip_on=False, zorder=1)
        for k, lab, colr in ((0, "lowered\nshould rise", SEQ_HI),
                             (1, "raised\nshould fall", DIV_NEG)):
            axB.text(j * 2 + k, -0.76, lab, ha="center", va="center", fontsize=7.6,
                     color=colr, linespacing=1.3)

    axB.set_title("B   Where the AZI prediction puts each HALF of the five terms lowered everywhere",
                  fontsize=10.5, color=INK, loc="left", pad=52, fontweight="bold")
    axB.text(0.0, 1.020, "every term is split into the members BPD7mo LOWERED (log2FC < −0.25) and "
                         "the members it RAISED (> +0.25)\nthe cell is their median "
                         "log2FC(AZI vs disease) and how many members;  bold + outline = "
                         "signed-rank p < 0.05",
             transform=axB.transAxes, fontsize=8.5, color=INK2, va="bottom", linespacing=1.6)

    pb = axB.get_position()
    cax = fig.add_axes([pb.x0 + 0.10, pb.y0 - 0.055, 0.16, 0.030])
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto", cmap=DIVERGING)
    cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_color(GRID)
    cax.text(-0.02, 0.5, "wrong way", transform=cax.transAxes, ha="right", va="center",
             fontsize=8.5, color=DIV_NEG)
    cax.text(1.02, 0.5, "back toward the control", transform=cax.transAxes, ha="left",
             va="center", fontsize=8.5, color=SEQ_HI)
    cax.text(0.5, -0.55, "shading is signed by the direction that would be correct "
                         "for that half — the number is the raw median",
             transform=cax.transAxes, ha="center", va="top", fontsize=7.8, color=MUTED)

    save(fig, "autophagy_terms.png")
    h.reset_index().to_csv(OUT / "autophagy_terms_halves.csv", index=False)


def funnel():
    """Autophagy genes lowered by BPD7mo after each filter step, per cell type."""
    import json
    blob = json.loads((HERE / "autophagy_panel/reports/autophagy_terms_found.json").read_text())
    universe = set()
    for terms in blob.values():
        if isinstance(terms, dict):
            for genes in terms.values():
                universe |= set(genes)

    d = rd("human_panel_variants/reports/human_rescue_genomewide.csv")
    d = d[d.condition == "BPD7mo"]
    steps, labels = [], ["WORKBOOK, step 1   disease_candidate\nmin.pct ≥ 0.1  and  |log2FC_disease| > 0.25",
                         "WORKBOOK, step 2   + rescued\nalso |log2FC_AZI| > 0.25  and  recovery_ratio < 0.9",
                         "FIGURE gate   + p-gate\np_adj_disease < 0.05  and  p_adj_AZI ≥ 0.05"]
    for ct in CTS:
        g = d[(d.cell_type == ct) & d.gene.isin(universe)]
        g = g[g.passes_min_pct_disease & g.passes_min_pct_azi]
        a = g[g.avg_log2FC_disease_top800 < -0.25]
        b = a[a.rescued]
        c = b[(b.p_val_adj_disease_top800 < 0.05)
              & (b.p_val_adj_azi_top800.isna() | (b.p_val_adj_azi_top800 >= 0.05))]
        steps.append([len(a), len(b), len(c)])

    fig, ax = plt.subplots(figsize=(11.0, 3.9))
    style(ax)
    x = np.arange(len(CTS))
    w = 0.26
    for i, (lab, colr) in enumerate(zip(labels, [RAMP[0], RAMP[1], GT_COLOR])):
        v = [st[i] for st in steps]
        ax.bar(x + (i - 1) * (w + 0.02), v, w, color=colr, edgecolor=SURFACE,
               linewidth=1.4, zorder=3, label=lab)
        for xi, vi in enumerate(v):
            ax.text(xi + (i - 1) * (w + 0.02), vi + 1.5, str(vi), ha="center",
                    fontsize=9, color=INK, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(CTS, fontsize=10, color=INK)
    ax.set_ylabel("autophagy genes BPD7mo lowered", fontsize=9.5, color=INK2)
    ax.set_ylim(0, max(st[0] for st in steps) * 1.22)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.30), handletextpad=0.5, columnspacing=2.2)
    save(fig, "funnel.png")


def ceiling():
    m = rd("human/reports/synthetic_down_matches_real_down_bpd7mo.csv").set_index("cell_type")
    c = rd("human/reports/real_matches_real_down_bpd7mo.csv").set_index("cell_type")
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    style(ax)
    x = np.arange(len(CTS))
    w = 0.34
    real = [c.loc[ct, "fold"] for ct in CTS]
    mod = [m.loc[ct, "fold"] for ct in CTS]
    ax.bar(x - w / 2 - 0.012, real, w, color=GT_COLOR, edgecolor=SURFACE, linewidth=1.4,
           label="real BPD7mo cells (ceiling)", zorder=3)
    ax.bar(x + w / 2 + 0.012, mod, w, color=PRED_COLOR, edgecolor=SURFACE, linewidth=1.4,
           label="model prediction", zorder=3)
    for xi, (rv, mv) in enumerate(zip(real, mod)):
        ax.text(xi - w / 2, rv + 0.03, f"{rv:.2f}", ha="center", fontsize=9, color=INK2)
        ax.text(xi + w / 2, mv + 0.03, f"{mv:.2f}", ha="center", fontsize=9, color=INK2)
        ax.text(xi, 0.16, f"{100 * mv / rv:.0f}% of ceiling", ha="center", va="center",
                fontsize=9.5, color=INK, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec="none"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ct}\n{int(m.loc[ct, 'n_pred_cells'])} model / "
                        f"{int(c.loc[ct, 'n_disease_cells'])} real cells" for ct in CTS],
                       fontsize=9, color=INK)
    ax.set_ylabel("fold enrichment of falling genes\nover an expression-matched null",
                  fontsize=9.5, color=INK2)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(3.48, 1.02, "no effect", fontsize=8.5, color=MUTED, va="bottom", ha="right")
    ax.set_ylim(0, 2.1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left", ncol=2)
    save(fig, "ceiling.png")


def recovery_ratio():
    """recovery_ratio = |log2FC_AZI| / |log2FC_disease|, both against the Term
    control, per autophagy gene lowered by disease; 0.9, the rat threshold, is
    drawn as a reference line. Genes absent from all predictions are counted
    ("silent") and not plotted.
    """
    import json
    blob = json.loads((HERE / "autophagy_panel/reports/autophagy_terms_found.json").read_text())
    groups = pd.read_csv(HERE / "human_panel_variants/autophagy_program/reports/term_membership.csv"
                         ).set_index("term")["group"].to_dict()
    tg = {}
    for terms in blob.values():
        if isinstance(terms, dict):
            for t, g in terms.items():
                tg[t] = set(g)
    core = set().union(*[g for t, g in tg.items() if groups[t] == "core"])

    d = pd.read_csv(HERE / "human_panel_variants/reports/human_rescue_genomewide.csv")
    d = d[d.gene.isin(core) & d.passes_min_pct_disease & d.passes_min_pct_azi]
    d = d[d.avg_log2FC_disease_top800 < -0.25]

    conds = ["BPD7mo", "Acute26", "BPDPH7mo"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, cond in zip(axes, conds):
        style(ax)
        ax.set_yscale("log")
        labels = []
        for i, ct in enumerate(CTS):
            g = d[(d.condition == cond) & (d.cell_type == ct)]
            named = g[g.pct_azi_pred > 0]
            rr = named.recovery_ratio.dropna().clip(0.04, 9.0)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(rr)), rr, s=17, color=GT_COLOR,
                       alpha=0.55, linewidth=0, zorder=3)
            med = rr.median()
            ax.plot([i - 0.30, i + 0.30], [med, med], color=ACCENT_FIG, lw=2.6,
                    zorder=5, solid_capstyle="round")
            ax.text(i, 11.0, f"{med:.2f}", ha="center", fontsize=9.5, color=ACCENT_FIG,
                    fontweight="bold")
            n_sil = int((g.pct_azi_pred == 0).sum())
            if n_sil:
                ax.text(i, 0.043, f"{n_sil} silent", ha="center", va="center",
                        fontsize=7.5, color=MUTED, zorder=6)
            labels.append(f"{ct}\nn={len(rr)}")
        ax.axhline(1.0, color=INK2, lw=1.1, zorder=2)
        ax.axhline(0.9, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xticks(range(len(CTS)))
        ax.set_xticklabels(labels, fontsize=8.5, color=INK)
        ax.set_xlim(-0.6, len(CTS) - 0.4)
        ax.set_ylim(0.035, 16)
        ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4", "8"])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_title(cond, fontsize=11.5, color=INK, loc="left", pad=24, fontweight="bold")
    axes[0].set_ylabel("recovery_ratio = |log2FC_AZI| / |log2FC_disease|\n"
                       "below 1 = closer to the healthy control", fontsize=9.5, color=INK2)
    axes[2].text(len(CTS) - 0.45, 1.06, "no improvement", fontsize=8, color=INK2,
                 va="bottom", ha="right")
    axes[2].text(len(CTS) - 0.45, 0.85, "0.9  rat gate", fontsize=8, color=MUTED,
                 va="top", ha="right")
    fig.text(0.5, 1.02, "Autophagy genes disease lowered: union of the 25 core terms, min.pct on both "
                        "arms, log2FC vs Term below -0.25 - the model takes no part in selecting them",
             ha="center", fontsize=9, color=INK2)
    save(fig, "recovery_ratio.png")


if __name__ == "__main__":
    autophagy_terms()
    funnel()
    ceiling()
    recovery_ratio()
