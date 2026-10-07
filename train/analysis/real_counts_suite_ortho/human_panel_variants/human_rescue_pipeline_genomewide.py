#!/usr/bin/env python3
"""
Genome-wide rescue table for the human cohorts, with the formulas of the rat
tables (Seurat log2FC, Mann-Whitney U with Bonferroni correction,
recovery_ratio, recovery_type).

Arms, per condition and cell type, on the model gene axis:
  control  measured Term0d + Term20d
  disease  measured Acute26 / BPD7mo / BPDPH7mo
  azi      model azithromycin prediction (reconstructed, at most 800 genes)

log2FC and p-values are computed with all arms truncated to their top 800
genes, so that recovery_ratio = |log2FC_azi| / |log2FC_disease| compares
quantities on the same gene budget; min.pct is computed on the untruncated
measured matrices. The disease arm is also reported untruncated
(*_disease_full columns).

GSE275938 has 1 Acute26, 2 BPD7mo, 2 BPDPH7mo, 1 Term0d and 1 Term20d donor;
the per-cell tests do not account for donor.

Output: reports/human_rescue_genomewide.csv,
reports/human_rescue_genomewide_summary.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ABS_DIR = Path(__file__).resolve().parents[2]  # train/analysis
PMA_DIR = ABS_DIR / "pathway_module_analysis"
sys.path.insert(0, str(ABS_DIR))
import common_human as CH
import rank_expr_model as rem

sys.path.insert(0, str(Path(__file__).parent.parent))
import common_real as CR

sys.path.insert(0, str(PMA_DIR))
_saved_common = sys.modules.pop("common", None)
import real_counts as RC
if _saved_common is not None:
    sys.modules["common"] = _saved_common

HERE = Path(__file__).parent
AZI_JSONL = Path(__file__).resolve().parents[3] / "results/inference_azi_results.jsonl"

MIN_PCT = 0.1
LOGFC_THRESHOLD = 0.25
RECOVERY_RATIO_THRESHOLD = 0.9

CONDITIONS = {"Acute26": "Acute", "BPD7mo": "BPD", "BPDPH7mo": "BPD-PH"}
MIN_AZI_CELLS = 3


def seurat_log2fc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """log2(mean(expm1(x1)) + 1) - log2(mean(expm1(x2)) + 1) per gene (axis 0)."""
    return np.log2(np.expm1(x1).mean(axis=0) + 1) - np.log2(np.expm1(x2).mean(axis=0) + 1)


def mwu_padj(x1: np.ndarray, x2: np.ndarray, n_genes_total: int) -> np.ndarray:
    """Two-sided Mann-Whitney U per gene, Bonferroni-adjusted by the total
    gene count. Genes constant in both groups (p = NaN) get p = 1.
    """
    p = stats.mannwhitneyu(x1, x2, alternative="two-sided", axis=0).pvalue
    p = np.where(np.isnan(p), 1.0, p)
    return np.minimum(p * n_genes_total, 1.0)


def main():
    print("[calibration] fitting rank->expr model on GSE275938 compiled counts (all human conditions)...")
    model = rem.fit_from_csv(CH.COUNTS_CSV)
    n_genes_total = len(model.gene_names)
    print(f"  r2={model.r2:.4f}  n_genes={n_genes_total}\n")

    print("[data] real Term0d+Term20d control (real counts, model gene axis)...")
    term_mats = RC.real_term_matrices(model)
    term_top800 = {ct: CR.truncate_topk(X) for ct, X in term_mats.items()}

    print("[data] model AZI-counterfactual (default variant, T=1.0)...")
    azi_groups = CH.load_azi_counterfactual(AZI_JSONL)

    rows = []
    for cond_csv, cond_model in CONDITIONS.items():
        print(f"\n[data] real {cond_csv} disease counts...")
        disease_mats = RC.real_disease_matrices(cond_csv, model)

        for ct in CH.CT_ORDER:
            if ct not in disease_mats or ct not in term_mats:
                continue
            arecs = azi_groups.get((ct, cond_model), [])
            if len(arecs) < MIN_AZI_CELLS:
                print(f"  {cond_csv}/{ct}: skipped, only {len(arecs)} AZI pred cells")
                continue

            ctrl_full = term_mats[ct]
            dis_full = disease_mats[ct]
            ctrl_t8 = term_top800[ct]
            dis_t8 = CR.truncate_topk(dis_full)
            azi_X = rem.reconstruct_batch([r["pred"] for r in arecs], model, CR.K)

            pct_ctrl = (ctrl_full > 0).mean(axis=0)
            pct_dis = (dis_full > 0).mean(axis=0)
            pct_azi = (azi_X > 0).mean(axis=0)
            pass_min_pct_dis = (pct_ctrl >= MIN_PCT) | (pct_dis >= MIN_PCT)
            pass_min_pct_azi = (pct_ctrl >= MIN_PCT) | (pct_azi >= MIN_PCT)

            lfc_dis_t8 = seurat_log2fc(dis_t8, ctrl_t8)
            padj_dis_t8 = mwu_padj(dis_t8, ctrl_t8, n_genes_total)
            lfc_azi_t8 = seurat_log2fc(azi_X, ctrl_t8)
            padj_azi_t8 = mwu_padj(azi_X, ctrl_t8, n_genes_total)

            lfc_dis_full = seurat_log2fc(dis_full, ctrl_full)
            padj_dis_full = mwu_padj(dis_full, ctrl_full, n_genes_total)

            with np.errstate(divide="ignore", invalid="ignore"):
                recovery_ratio = np.abs(lfc_azi_t8) / np.abs(lfc_dis_t8)
            recovery_ratio[~np.isfinite(recovery_ratio)] = np.nan

            dis_candidate = pass_min_pct_dis & (np.abs(lfc_dis_t8) > LOGFC_THRESHOLD)
            azi_candidate = pass_min_pct_azi & (np.abs(lfc_azi_t8) > LOGFC_THRESHOLD)
            rescued = dis_candidate & azi_candidate & (recovery_ratio < RECOVERY_RATIO_THRESHOLD)
            recovery_type = np.where(lfc_dis_t8 > 0, "Rescued_HO_up", "Rescued_HO_down")

            rows.append(pd.DataFrame({
                "gene": model.gene_names,
                "cell_type": ct,
                "condition": cond_csv,
                "n_control_cells": ctrl_full.shape[0],
                "n_disease_cells": dis_full.shape[0],
                "n_azi_pred_cells": azi_X.shape[0],
                "pct_control_full": pct_ctrl.round(4),
                "pct_disease_full": pct_dis.round(4),
                "pct_azi_pred": pct_azi.round(4),
                "passes_min_pct_disease": pass_min_pct_dis,
                "passes_min_pct_azi": pass_min_pct_azi,
                "avg_log2FC_disease_top800": lfc_dis_t8.round(4),
                "p_val_adj_disease_top800": padj_dis_t8,
                "avg_log2FC_azi_top800": lfc_azi_t8.round(4),
                "p_val_adj_azi_top800": padj_azi_t8,
                "avg_log2FC_disease_full": lfc_dis_full.round(4),
                "p_val_adj_disease_full": padj_dis_full,
                "disease_candidate": dis_candidate,
                "azi_candidate": azi_candidate,
                "recovery_ratio": recovery_ratio.round(4),
                "recovery_type": recovery_type,
                "rescued": rescued,
            }))
            print(f"  {cond_csv}/{ct}: n_ctrl={ctrl_full.shape[0]} n_dis={dis_full.shape[0]} "
                  f"n_azi={azi_X.shape[0]}  min.pct-eligible dis/azi="
                  f"{int(pass_min_pct_dis.sum())}/{int(pass_min_pct_azi.sum())}  "
                  f"rescued={int(rescued.sum())}")

    out = pd.concat(rows, ignore_index=True)
    out_path = HERE / "reports" / "human_rescue_genomewide.csv"
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}  ({len(out)} rows)")

    print("\n=== rescued genes per condition/cell type (rat's gate: both candidates + recovery_ratio<0.9) ===")
    summary = (out[out["rescued"]]
               .groupby(["condition", "cell_type", "recovery_type"])
               .size().rename("n_genes").reset_index())
    print(summary.to_string(index=False))
    summary_path = HERE / "reports" / "human_rescue_genomewide_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"-> {summary_path}")


if __name__ == "__main__":
    main()
