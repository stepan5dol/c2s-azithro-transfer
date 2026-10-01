#!/usr/bin/env python3
"""
venn_rat_human_model.py — three-way overlap, one panel per cell type.

  RAT RESCUED      genes of the rat rescue workbooks (real RA / HO / HO+AZI)
  BPD7mo CHANGED   disease_candidate in human_rescue_genomewide.csv
  MODEL RESCUED    rescued in the same table, i.e. disease_candidate AND
                   azi_candidate AND recovery_ratio < 0.9

Both human circles are the pipeline's own columns on the same contrast, so the
third set nests inside the second exactly -- every rescued gene is a disease
candidate by definition. Mixing contrasts here is easy and wrong: taking
"changed" as p_val_adj_disease_full < 0.05 with |avg_log2FC_disease_full| > 0.25
puts only 20-38% of the rescued genes inside it, and the model then appears to
rescue more genes than disease changed.

Rat and human are joined through ortho.py / ortholog_pairs.tsv: HCOP consensus
(nine predictors, a pair accepted at support >= 4) plus the RGD/HGNC curated list,
which passes regardless of votes. Each list is mapped in its own direction -- the
rat workbook rat->human, both human lists human->rat -- and only what both
directions confirm is kept, so an intersection of two circles is exactly
ortho.join of those lists. Symbols are normalised through HGNC and RGD before
joining and returned in the spelling each dataset uses.

Output: figures_venn/venn_rat_bpd7mo_model.png + .csv of every region.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib_venn import venn3, venn3_circles

BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
HERE = Path(__file__).parent
OUT_DIR = HERE / "figures_venn"
GENOMEWIDE = (HERE / "human_panel_variants" / "reports" / "human_rescue_genomewide.csv")
RAT_H5AD = BASE / "rat.ho.azi.integrated.h5ad"

sys.path.insert(0, str(HERE))
import ortho  # noqa: E402
sys.path.insert(0, str(HERE.parent))
import plot_style as PS  # noqa: E402
PS.apply()   # same font family as every other figure

CONDITION = "BPD7mo"
# No p-value or log2FC constants here on purpose: neither side is gated on p.
# The rat circle is the whole delivered workbook, the two human circles are the
# pipeline's own disease_candidate / rescued columns, and those are built from
# min.pct and |log2FC| only. Leftover PADJ / LFC constants were removed with the
# contrast they belonged to.

# rat workbook -> human cell type label in the genome-wide table
CELL_TYPES = {
    "gCap": "gcap rescue gene.xlsx",
    "aCap": "acap resuce gene.xlsx",
    "VEC": "Venous_rescue gene.xlsx",
    "Pericyte": "pericyte_rescue gene.xlsx",
}

# validated with dataviz/scripts/validate_palette.js --mode light: all checks pass
C_RAT, C_DIS, C_MODEL = "#0072B2", "#CC79A7", "#D55E00"
# Spelled out in the same language as the main figure: species first, then what
# was compared, then whether the set is measured or predicted. The old labels
# ("Rat rescued (real, 3 arms)", "BPD7mo changed (disease candidate)") named
# pipeline columns, which only a reader of this repo could decode.
LABELS = ("Rat, rescued by azithromycin\n(measured)",
          "Human, changed in BPD at 7 months\n(measured)",
          "Human, rescued by azithromycin\n(model prediction)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--padj", type=float, default=None,
                    help="also require the DISEASE arm to be significant at this "
                         "adjusted p on BOTH species: rat p_val_adj_HO and human "
                         "p_val_adj_disease_top800. Off by default, because neither "
                         "the rat workbooks nor the human pipeline gate on p.")
    ap.add_argument("--bare", action="store_true",
                    help="publication version: circles, counts, legend and title "
                         "only. The percentage lines, the joinable-both-ways line "
                         "and the printed gene lists are dropped from the canvas; "
                         "all of it stays in the CSVs written beside the figure.")
    args = ap.parse_args()
    padj = args.padj
    bare = args.bare
    suffix = ("" if padj is None else f"_padj{padj:g}".replace(".", "")) + ("_bare" if bare else "")
    OUT_DIR.mkdir(exist_ok=True)

    H = pd.read_csv(GENOMEWIDE)
    H = H[H["condition"] == CONDITION]
    axis = set(H["gene"].unique())
    if padj is not None:
        print(f"[gate] disease arm required significant at p_adj < {padj} on both "
              f"species; the same gate is applied to the rescued circle so it stays "
              f"nested inside the changed one")

    # THE UNIVERSE IS THE JOIN, not one species' gene list. Each list is mapped in
    # its own direction -- the rat workbook rat->human, both human lists
    # human->rat -- and only pairs both directions agree on are kept
    # (ortho.join). A circle can then only be missing a gene because the biology
    # says so, never because the gene had no counterpart to be compared against;
    # the un-joinable remainder is reported per panel instead of silently
    # inflating the areas.
    import anndata as ad
    rat_axis = set(ad.read_h5ad(RAT_H5AD, backed="r").var_names)
    pairs = ortho.load_pairs(human_axis=axis, rat_axis=rat_axis)
    prov = ortho.provenance()
    print(f"[orthology] {len(pairs)} pairs on both axes "
          f"({pairs['rat_symbol'].nunique()} rat / {pairs['human_symbol'].nunique()} human genes); "
          f"{int(pairs['rgd_curated'].sum())} RGD-curated, "
          f"median support {pairs['support_n'].median():.0f}")
    print(f"[orthology] HCOP + RGD, accepted at support>={ortho.MIN_SUPPORT} or curated; "
          f"table built {prov['built_at']}, {prov['n_pairs']} pairs total")

    # The tall canvas exists only to hold the text blocks under each panel. With
    # --bare there is nothing under the circles, so the figure is as tall as the
    # circles are and the legend/title band keeps its proportion.
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.0 if bare else 11.5))
    fig.subplots_adjust(wspace=0.20, top=0.86 if bare else 0.93,
                        bottom=0.03 if bare else 0.50)
    rows = []
    tri_rows = []

    for ax, (ct, xlsx) in zip(axes, CELL_TYPES.items()):
        wb = pd.read_excel(BASE / xlsx)
        if padj is not None:
            wb = wb[wb["p_val_adj_HO"] < padj]
        raw = {str(g) for g in wb["gene"]}
        h = H[H["cell_type"] == ct]
        if padj is not None:
            h = h[h["p_val_adj_disease_top800"] < padj]
        # BOTH human circles come from the pipeline's own columns, on the same
        # contrast. `rescued` is by definition disease_candidate AND azi_candidate
        # AND recovery_ratio < 0.9, so disease_candidate is the set it lives
        # inside and the nesting in the figure is exact: 517/517, 960/960,
        # 503/503, 979/979 of the rescued genes are disease candidates.
        # An earlier version used p_val_adj_disease_full < 0.05 and
        # |avg_log2FC_disease_full| > 0.25 instead -- a different contrast (full,
        # not top-800) with a p-gate the rescue definition does not use. Only
        # 20-38% of rescued genes fell inside it, so the model appeared to rescue
        # more genes than disease had changed.
        dis = set(h.loc[h["disease_candidate"] == True, "gene"])  # noqa: E712
        mod = set(h.loc[h["rescued"] == True, "gene"])  # noqa: E712

        # Each list is mapped in ITS OWN direction and the circles are the results:
        # the rat workbook goes rat->human, both human lists go human->rat. An
        # intersection of two circles is therefore exactly the two-way join of
        # those lists -- ortho.join is fwd & rev, and fwd / rev are these sets --
        # so the figure cannot drift from the join by construction.
        r2h, _ = ortho.rat_to_human(raw, human_axis=axis)
        h2r_dis, _ = ortho.human_to_rat(dis, rat_axis=rat_axis)
        h2r_mod, _ = ortho.human_to_rat(mod, rat_axis=rat_axis)
        rep_dis = ortho.join_report(raw, dis, human_axis=axis, rat_axis=rat_axis)
        rep_mod = ortho.join_report(raw, mod, human_axis=axis, rat_axis=rat_axis)

        A = {(r, h) for r, hs in r2h.items() for h in hs}
        B = {(r, h) for h, rs in h2r_dis.items() for r in rs}
        C = {(r, h) for h, rs in h2r_mod.items() for r in rs}
        key = lambda df: set(zip(df["rat_symbol"], df["human_symbol"]))
        assert A & B == key(ortho.join(raw, dis, human_axis=axis, rat_axis=rat_axis))
        assert A & C == key(ortho.join(raw, mod, human_axis=axis, rat_axis=rat_axis))

        # set labels are identical across panels -> one shared legend instead
        v = venn3([A, B, C], set_labels=("", "", ""), ax=ax,
                  set_colors=(C_RAT, C_DIS, C_MODEL), alpha=0.55)
        venn3_circles([A, B, C], ax=ax, linewidth=0.8, color="#555555")
        for t in (v.subset_labels or []):
            if t:
                t.set_fontsize(10)

        pct = lambda n, d: f"{100 * n / d:.0f}%" if d else "—"
        ax.set_title(ct, fontsize=13, pad=10)

        # the point of the figure is WHICH genes, so the triple overlap is named
        tri = sorted({h for _, h in (A & B & C)})
        tri_rows += [{"cell_type": ct, "human_symbol": g} for g in tri]

        if not bare:
            # stacked, not one line: a single line is wider than the panel and collides
            ax.text(0.5, -0.06,
                    f"of the rat list — {pct(len(A & B), len(A))} also changed in {CONDITION}\n"
                    f"{pct(len(A & C), len(A))} also rescued by the model\n"
                    f"{pct(len(A & B & C), len(A))} in all three",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, color="#333333", linespacing=1.6)
            ax.text(0.5, -0.28,
                    f"joinable both ways — rat {rep_dis['rat_mapped']}/{rep_dis['rat_in']}, "
                    f"{CONDITION} {rep_dis['human_mapped']}/{rep_dis['human_in']}, "
                    f"model {rep_mod['human_mapped']}/{rep_mod['human_in']}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="#888888")

            block = "\n".join(", ".join(tri[i:i + 3]) for i in range(0, len(tri), 3))
            ax.text(0.5, -0.40, f"in all three ({len(tri)})",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9.5, color="#333333", fontweight="bold")
            ax.text(0.5, -0.47, block, transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.6, color="#333333", linespacing=1.45, family="DejaVu Sans")

        rows.append({
            "cell_type": ct,
            "rat_symbols": rep_dis["rat_in"],
            "rat_joinable": rep_dis["rat_mapped"],
            "disease_symbols": rep_dis["human_in"],
            "disease_joinable": rep_dis["human_mapped"],
            "model_symbols": rep_mod["human_in"],
            "model_joinable": rep_mod["human_mapped"],
            "pairs_rat": len(A), "pairs_disease": len(B), "pairs_model": len(C),
            "rat_only": len(A - B - C), "disease_only": len(B - A - C),
            "model_only": len(C - A - B),
            "rat_disease": len(A & B - C), "rat_model": len(A & C - B),
            "disease_model": len(B & C - A), "all_three": len(A & B & C),
            "pct_rat_in_disease": round(100 * len(A & B) / len(A), 1),
            "pct_rat_in_model": round(100 * len(A & C) / len(A), 1),
            "pct_rat_in_both": round(100 * len(A & B & C) / len(A), 1),
        })

    handles = [Patch(facecolor=c, alpha=0.55, edgecolor="#555555", label=l)
               for c, l in zip((C_RAT, C_DIS, C_MODEL),
                               (lbl.replace("\n", " ") for lbl in LABELS))]
    # fig-fraction offsets: the same fraction is fewer inches on the short canvas,
    # so the legend has to sit lower on it to clear the title.
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.925 if bare else 0.965),
               ncol=3, frameon=False, fontsize=10.5, handlelength=1.6,
               columnspacing=3.0)
    fig.suptitle(
        "Overlap of the rat rescued genes, the genes changed in human bronchopulmonary "
        "dysplasia at 7 months, and the genes the model predicts azithromycin rescues"
        + ("" if padj is None else f"   —   disease arm p_adj < {padj:g} on both species"),
        fontsize=13, y=1.02 if bare else 1.00)
    png = OUT_DIR / f"venn_rat_{CONDITION.lower()}_model{suffix}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # the gene names leave the canvas with --bare, so they are always written out
    pd.DataFrame(tri_rows).to_csv(
        OUT_DIR / f"venn_rat_{CONDITION.lower()}_model{suffix}_triple_genes.csv",
        index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"venn_rat_{CONDITION.lower()}_model{suffix}.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n-> {png}")


if __name__ == "__main__":
    main()
