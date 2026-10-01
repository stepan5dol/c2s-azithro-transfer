#!/usr/bin/env python3
"""
a8_equivalence_beautiful_species.py — copy of a8_equivalence_beautiful.py,
parametrized to run the SAME A8 equivalence test (pred-gt cross distance vs
gt-gt real-cell-variability envelope, Q90 threshold) on both species:

  - rat: common.py (C), rat.ho.azi.integrated.h5ad calibration, HO/AZI, 4 cell
    types x 2 conditions = 8 groups pooled -- identical to the original script.
  - human: common_human.py (CH), calibration fit on the real BPD-PH counts
    csv (not the rat h5ad), Acute/BPD/BPD-PH, 4 cell types x 3 conditions =
    12 groups pooled. Uses the SAME construction as rat: for each condition,
    gt = the real next-disease-stage cell, pred = the model's forward
    prediction from the same real baseline (this is NOT the AZI
    counterfactual file -- there is no real ground truth for that one).

Output: figures/figures_beautiful/<tag>/equivalence_to_real_variability.png
(rat, same path as the original) and
figures_human/figures_beautiful/<tag>/equivalence_to_real_variability.png
(human, new).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C
import common_human as CH
import plot_style as PS
import rank_expr_model as rem


def run_temperature(tag, jsonl_path, model, loader, ct_order, cond_order, min_n, fig_dir, species_label):
    out_dir = PS.out_dir(fig_dir, tag)
    groups = loader(jsonl_path)

    all_gt_gt, all_cross = [], []
    n_pairs_used = 0
    for ct in ct_order:
        for cond in cond_order:
            group = groups.get((ct, cond), [])
            if len(group) < max(min_n, 3):
                continue
            _, gt_X, pred_X = C.reconstruct_group(group, model)
            dom = C.filtered_gene_domain(gt_X, pred_X)
            gt_f, pred_f = gt_X[:, dom], pred_X[:, dom]
            all_gt_gt.append(C.pairwise_dist(gt_f))
            all_cross.append(C.cdist(pred_f, gt_f).flatten())
            n_pairs_used += 1

    if not all_gt_gt:
        print(f"[{species_label}/{tag}] no groups with n>=min_n -- skipped")
        return

    gt_gt = np.concatenate(all_gt_gt)
    cross = np.concatenate(all_cross)

    q90 = np.percentile(gt_gt, 90)
    frac_cross_below_q90 = float((cross <= q90).mean())

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for arr, label, color in ((gt_gt, "Real – Real (cell-to-cell variability)", PS.GT_COLOR),
                               (cross, "Model – Real", PS.PRED_COLOR)):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, label=label, lw=2.2)

    ax.axvline(q90, color=PS.GT_COLOR, lw=1.2, ls="--", alpha=0.8)
    ax.fill_betweenx([0, 1], 0, q90, color=PS.GT_COLOR, alpha=0.07,
                     label=f"{100 * frac_cross_below_q90:.0f}% of model cells within envelope")

    ax.set_xlabel("Pairwise distance (calibrated expression space)")
    ax.set_ylabel("Cumulative fraction of pairs")
    ax.set_title(f"Equivalence to real cell-to-cell variability -- {species_label} ({tag})", pad=12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=9.5)

    fig_path = out_dir / "equivalence_to_real_variability.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{species_label}/{tag}] groups pooled={n_pairs_used}  "
          f"n_gt_gt_pairs={len(gt_gt)}  n_cross_pairs={len(cross)}  "
          f"Q90(real-real)={q90:.2f}  frac_cross_within_envelope={frac_cross_below_q90:.3f}")
    print(f"-> {fig_path}")
    return {"species": species_label, "tag": tag, "q90": q90, "frac_within": frac_cross_below_q90,
            "n_gt_gt": len(gt_gt), "n_cross": len(cross), "n_groups": n_pairs_used}


def main():
    PS.apply()
    summary = []

    print("=" * 100)
    print("RAT")
    print("=" * 100)
    rat_model = rem.fit(C.H5AD)
    print(f"[calibration] rat slope={rat_model.slope:.5f} intercept={rat_model.intercept:.4f} r2={rat_model.r2:.4f}")
    # strict T=1.0 only (project convention, see feedback_strict_t1_only) -- not the t0.8 axis
    r = run_temperature("t1.0", C.RUNS["t1.0"], rat_model, C.load_cells, C.CT_ORDER, C.COND_ORDER,
                         C.MIN_N, C.FIG_DIR, "rat")
    if r:
        summary.append(r)

    print()
    print("=" * 100)
    print("HUMAN")
    print("=" * 100)
    human_model = rem.fit_from_csv(CH.COUNTS_CSV)
    print(f"[calibration] human r2={human_model.r2:.4f}")
    r = run_temperature("t1.0", CH.RUNS["t1.0"], human_model, CH.load_cells, CH.CT_ORDER, CH.COND_ORDER,
                         CH.MIN_N, CH.FIG_DIR, "human")
    if r:
        summary.append(r)

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for r in summary:
        print(f"  {r['species']:6s} {r['tag']:5s}  groups={r['n_groups']:3d}  "
              f"Q90(real-real)={r['q90']:6.2f}  frac(model within envelope)={100*r['frac_within']:5.1f}%")


if __name__ == "__main__":
    main()
