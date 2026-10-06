#!/usr/bin/env python3
"""
A8 equivalence, real-counts version, human. Same construction as
rat/a8_equivalence_real.py: is model-pred-to-real-cell distance within the
envelope of real-cell-to-real-cell variability?

Only the no-AZI disease-trajectory arms are buildable this way (Acute26,
BPD7mo, BPDPH7mo -- each has a real target population with real raw counts).
The AZI-counterfactual arm has NO real-cell envelope at all -- no human
patient in this cohort ever received azithromycin (see
pathway_module_analysis/human_rescue_gene_pipeline.py's docstring) -- so it
is not included here, not omitted by oversight.

gt_gt = pairwise real-real distances within the real disease population
(top-800-truncated). cross = model's no-AZI forward pred (reconstructed) vs
that same real population.

Only T=1.0 / test_inference_results3.jsonl (see real_counts_suite/README.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABS_DIR = Path(__file__).resolve().parents[2]  # train/analysis
PMA_DIR = ABS_DIR / "pathway_module_analysis"
sys.path.insert(0, str(ABS_DIR))
import common_human as CH  # noqa: E402
import plot_style as PS  # noqa: E402
import rank_expr_model as rem  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
import common_real as CR  # noqa: E402

sys.path.insert(0, str(PMA_DIR))
_saved_common = sys.modules.pop("common", None)
import real_counts as RC  # noqa: E402
if _saved_common is not None:
    sys.modules["common"] = _saved_common

CONDITIONS = {"Acute26": "Acute", "BPD7mo": "BPD", "BPDPH7mo": "BPD-PH"}
HERE = Path(__file__).parent


def main():
    model = rem.fit_from_csv(CH.COUNTS_CSV)
    print(f"[calibration] r2={model.r2:.4f}")
    groups = CH.load_cells(CH.RUNS["t1.0"])

    PS.apply()
    summary_rows = []
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(6.2 * len(CONDITIONS), 5.2), constrained_layout=True)

    for ax, (cond_csv, cond_model) in zip(axes, CONDITIONS.items()):
        print(f"[data] real {cond_csv} disease counts (top-800-truncated)...")
        disease_mats = RC.real_disease_matrices(cond_csv, model)

        all_gt_gt, all_cross = [], []
        n_groups = 0
        for ct in CH.CT_ORDER:
            recs = groups.get((ct, cond_model), [])
            if len(recs) < CH.MIN_N or ct not in disease_mats:
                continue
            real_pool_X = CR.truncate_topk(disease_mats[ct])
            pred_X = rem.reconstruct_batch([r["pred"] for r in recs], model, CR.K)

            dom = CH.filtered_gene_domain(real_pool_X, pred_X)
            gt_f, pred_f = real_pool_X[:, dom], pred_X[:, dom]
            if len(gt_f) < 3:
                continue
            all_gt_gt.append(CH.pairwise_dist(gt_f))
            all_cross.append(CH.cdist(pred_f, gt_f).flatten())
            n_groups += 1

        if not all_gt_gt:
            continue
        gt_gt = np.concatenate(all_gt_gt)
        cross = np.concatenate(all_cross)
        q90 = np.percentile(gt_gt, 90)
        frac_within = float((cross <= q90).mean())

        for arr, label, color in ((gt_gt, "Dataset – Dataset (cell-to-cell variability)", PS.GT_COLOR),
                                   (cross, "Model – Dataset", PS.PRED_COLOR)):
            xs = np.sort(arr)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=color, label=label, lw=2.2)
        ax.axvline(q90, color=PS.GT_COLOR, lw=1.2, ls="--", alpha=0.8)
        ax.fill_betweenx([0, 1], 0, q90, color=PS.GT_COLOR, alpha=0.07,
                         label=f"{100*frac_within:.0f}% of model cells within envelope")
        ax.set_xlabel("Pairwise distance (dataset counts, top-800-matched)")
        ax.set_ylabel("Cumulative fraction of pairs")
        ax.set_title(f"{cond_csv} (t1.0)", pad=12)
        ax.set_ylim(-0.02, 1.02); ax.set_xlim(left=0)
        ax.legend(loc="lower right", fontsize=8.5)

        print(f"[{cond_csv}] groups={n_groups}  n_gt_gt={len(gt_gt)}  n_cross={len(cross)}  "
              f"Q90(real-real)={q90:.2f}  frac_within={frac_within:.3f}")
        summary_rows.append({"condition": cond_csv, "n_groups": n_groups, "q90_real_real": q90,
                             "frac_model_within_envelope": frac_within})

    fig_path = HERE / "figures" / "a8_equivalence_real.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {fig_path}")

    rep = pd.DataFrame(summary_rows)
    rep_path = HERE / "reports" / "report_a8_equivalence_real.csv"
    rep.to_csv(rep_path, index=False)
    print(rep.to_string(index=False))
    print(f"-> {rep_path}")


if __name__ == "__main__":
    main()
