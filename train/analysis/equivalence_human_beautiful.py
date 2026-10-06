#!/usr/bin/env python3
"""
equivalence_human_beautiful.py — human counterpart to a8_equivalence_beautiful.py.
Same statistic: pooled ECDF of real-real pairwise distance (cell-to-cell
variability envelope) vs. model-real cross distance, in calibrated expression
space, over the three human disease-trajectory conditions (Acute/BPD/BPD-PH).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common_human as CH
import plot_style as PS
import rank_expr_model as rem


def run_temperature(tag: str, jsonl_path, model: rem.RankExprModel) -> None:
    out_dir = PS.out_dir(CH.FIG_DIR, tag)
    groups = CH.load_cells(jsonl_path)

    all_real_real, all_cross = [], []
    for ct in CH.CT_ORDER:
        for cond in CH.COND_ORDER:
            group = groups.get((ct, cond), [])
            if len(group) < max(CH.MIN_N, 3):
                continue
            _, gt_X, pred_X = CH.reconstruct_group(group, model)
            dom = CH.filtered_gene_domain(gt_X, pred_X)
            gt_f, pred_f = gt_X[:, dom], pred_X[:, dom]
            all_real_real.append(CH.pairwise_dist(gt_f))
            all_cross.append(CH.cdist(pred_f, gt_f).flatten())

    if not all_real_real:
        return
    real_real = np.concatenate(all_real_real)
    cross = np.concatenate(all_cross)

    q90 = np.percentile(real_real, 90)
    frac_cross_below_q90 = float((cross <= q90).mean())

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for arr, label, color in ((real_real, "Real – Real (cell-to-cell variability)", PS.GT_COLOR),
                               (cross, "Model – Real", PS.PRED_COLOR)):
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=color, label=label, lw=2.2)

    ax.axvline(q90, color=PS.GT_COLOR, lw=1.2, ls="--", alpha=0.8)
    ax.fill_betweenx([0, 1], 0, q90, color=PS.GT_COLOR, alpha=0.07,
                     label=f"{100 * frac_cross_below_q90:.0f}% of model cells within envelope")

    ax.set_xlabel("Pairwise distance (calibrated expression space)")
    ax.set_ylabel("Cumulative fraction of pairs")
    ax.set_title(f"Equivalence to real cell-to-cell variability, human ({tag})", pad=12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=9.5)

    fig_path = out_dir / "equivalence_to_real_variability_human.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {fig_path}")


if __name__ == "__main__":
    PS.apply()
    model = rem.fit_from_csv(CH.COUNTS_CSV)
    print(f"[calibration] slope={model.slope:.5f} intercept={model.intercept:.4f} r2={model.r2:.4f}")
    for tag, path in CH.RUNS.items():
        run_temperature(tag, path, model)
