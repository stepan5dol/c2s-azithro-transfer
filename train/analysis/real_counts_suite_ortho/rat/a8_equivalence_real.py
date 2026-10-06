#!/usr/bin/env python3
"""
A8 equivalence, real-counts version: is model-pred-to-real-cell distance
within the envelope of real-cell-to-real-cell variability? Same TOST-like
ECDF construction as abs_analysis/a8_equivalence_beautiful_species.py, but
"gt" (real side) is now TRUE raw counts (via recovered barcodes,
recover_rat_test_barcodes.py), top-800-truncated to match pred's forced
K=800 budget -- not reconstructed from a rank list like the original.

For each condition (RA, HO, AZI): pool all real cells that appear as that
condition's src or tgt across both recovered transitions (RA->HO, HO->AZI),
deduplicated by barcode. gt_gt = pairwise real-real distances within that
condition's pool. cross = distance from model's pred (predicting that
condition from the other) to the same real pool.

Only T=1.0 / test_inference_results3.jsonl (see real_counts_suite/README.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABS_DIR = Path(__file__).resolve().parents[2]  # train/analysis
sys.path.insert(0, str(ABS_DIR))
import common as C  # noqa: E402
import plot_style as PS  # noqa: E402
import rank_expr_model as rem  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
import common_real as CR  # noqa: E402

HERE = Path(__file__).parent
CSV_PATH = HERE / "recovered_rat_test_pairs.csv"
TEST_INFERENCE_PATH = Path(__file__).resolve().parents[3] / "results/test_inference_results_t10.jsonl"

CT_RAW_TO_CT = {"general capillary endothelial cell (endothelial)": "gCap",
                "aerocyte capillary endothelial cell (endothelial)": "aCap",
                "pericyte (mural)": "Pericyte",
                "pulmonary venous endothelial cell (endothelial)": "VEC"}


def load_real_counts_by_barcode(barcodes: list[str], gene_names: list[str]) -> dict[str, np.ndarray]:
    print("  [real] reading rat.ho.azi.integrated.h5ad raw counts...")
    adata = anndata.read_h5ad(C.H5AD)
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    assert list(adata.var_names) == gene_names
    bc_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
    needed = [bc for bc in barcodes if bc in bc_idx]
    rows = [bc_idx[bc] for bc in needed]
    Xs = X[rows]
    if sp.issparse(Xs):
        Xs = Xs.toarray()
    Xs = np.asarray(Xs, dtype=np.float64)
    lib = Xs.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    Xn = np.log1p(Xs / lib * 1e4).astype(np.float32)
    del adata
    return {bc: Xn[i] for i, bc in enumerate(needed)}


def main():
    model = rem.fit(C.H5AD)
    print(f"[calibration] slope={model.slope:.5f} intercept={model.intercept:.4f} r2={model.r2:.4f}")

    df = pd.read_csv(CSV_PATH)
    assert df["verified"].all()
    df["ct"] = df["cell_type"].map(CT_RAW_TO_CT)
    assert df["ct"].notna().all()

    all_bc = list(pd.unique(df[["src_barcode", "tgt_barcode"]].values.ravel()))
    real_by_bc = load_real_counts_by_barcode(all_bc, model.gene_names)

    pred_by_idx = {}
    with open(TEST_INFERENCE_PATH) as f:
        for line in f:
            r = json.loads(line)
            pred_by_idx[r["idx"]] = r["pred"].strip().split()

    # condition pool: real barcodes seen as src or tgt for a given condition,
    # across BOTH transitions (RA appears as src of rat_RA_HO; HO appears as
    # tgt of rat_RA_HO AND src of rat_HO_AZI; AZI appears as tgt of rat_HO_AZI)
    cond_pool = {"RA": ("rat_RA_HO", "src_barcode"), "HO": None, "AZI": ("rat_HO_AZI", "tgt_barcode")}

    PS.apply()
    summary_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    for ax, (arm_name, pair_type, pred_cond) in zip(
        axes, [("RA -> HO (injury)", "rat_RA_HO", "HO"), ("HO -> AZI (rescue)", "rat_HO_AZI", "AZI")]
    ):
        sub = df[df["pair_type"] == pair_type]
        all_gt_gt, all_cross = [], []
        n_groups = 0
        for ct in C.CT_ORDER:
            ct_sub = sub[sub["ct"] == ct]
            if len(ct_sub) < C.MIN_N:
                continue
            # real pool for the TARGET condition of this transition (HO for RA->HO, AZI for HO->AZI)
            real_pool_bcs = pd.unique(ct_sub["tgt_barcode"])
            real_pool_X = np.stack([real_by_bc[bc] for bc in real_pool_bcs])
            real_pool_X = CR.truncate_topk(real_pool_X)

            pred_X = rem.reconstruct_batch([pred_by_idx[i] for i in ct_sub["idx"]], model, CR.K)

            dom = C.filtered_gene_domain(real_pool_X, pred_X)
            gt_f, pred_f = real_pool_X[:, dom], pred_X[:, dom]
            if len(gt_f) < 3:
                continue
            all_gt_gt.append(C.pairwise_dist(gt_f))
            all_cross.append(C.cdist(pred_f, gt_f).flatten())
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
        ax.set_title(f"{arm_name} (t1.0)", pad=12)
        ax.set_ylim(-0.02, 1.02); ax.set_xlim(left=0)
        ax.legend(loc="lower right", fontsize=8.5)

        print(f"[{arm_name}] groups={n_groups}  n_gt_gt={len(gt_gt)}  n_cross={len(cross)}  "
              f"Q90(real-real)={q90:.2f}  frac_within={frac_within:.3f}")
        summary_rows.append({"arm": arm_name, "n_groups": n_groups, "q90_real_real": q90,
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
