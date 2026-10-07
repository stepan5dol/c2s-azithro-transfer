#!/usr/bin/env python3
"""
Exports the arms of one human cohort as matrices for limma camera, cameraPR
and fry (limma_sets.py), per cell type:

    disease  control cells (Term0d + Term20d; He22 for Acute26), then disease
             cells; contrast disease vs control
    azi      disease cells, then the model's azithromycin predictions;
             contrast prediction vs disease

    matrix   genes x cells, log1p CPM10k, top-800 truncated (the values of
             human_rescue_pipeline_genomewide.py)
    genes    genes of human_rescue_genomewide.csv passing min.pct in both arms
    stat     avg_log2FC_disease_top800, in gene order

Columns are cells, so the p-values refer to groups of cells, not donors. In
the azi arm the measured and the reconstructed side differ in cell-to-cell
variance.

    python export_for_limma.py [--condition BPD7mo]
        -> reports/limma_input/{ct}.{bin,json}, {ct}_azi.{bin,json}, sets.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

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
SUITE = HERE.parent
OUT = HERE / "reports" / "limma_input"
CONDITION = "BPD7mo"
COND_MODEL = "BPD"

COND_MODEL_OF = {"BPD7mo": "BPD", "BPDPH7mo": "BPD-PH", "Acute26": "Acute"}
OUT_OF = {"BPD7mo": OUT}
CONTROL_OF = {"BPD7mo": "term", "BPDPH7mo": "term", "Acute26": "he22"}
AZI_JSONL = Path(__file__).resolve().parents[3] / "results/inference_azi_results.jsonl"
MIN_AZI_CELLS = 3
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022",
             "WikiPathways_2024_Human", "MSigDB_Hallmark_2020"]


def main():
    global CONDITION, COND_MODEL, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default=CONDITION, choices=sorted(COND_MODEL_OF),
                    help="cohort of human_rescue_genomewide.csv to export")
    CONDITION = ap.parse_args().condition
    COND_MODEL = COND_MODEL_OF[CONDITION]
    OUT = OUT_OF.get(CONDITION, HERE / "reports" / f"limma_input_{CONDITION.lower()}")
    print(f"[cohort] {CONDITION} (model calls it {COND_MODEL}) -> {OUT.name}")

    OUT.mkdir(parents=True, exist_ok=True)

    print("[calibration] fitting rank->expr model on GSE275938 compiled counts...")
    model = rem.fit_from_csv(CH.COUNTS_CSV)

    control = CONTROL_OF[CONDITION]
    if control == "term":
        print("[data] real Term0d+Term20d control...")
        term_mats = RC.real_term_matrices(model)
    else:
        print("[data] real He22 fetal GW22 control...")
        term_mats = RC.real_he22_matrices(model)
    print(f"[data] real {CONDITION}...")
    disease_mats = RC.real_disease_matrices(CONDITION, model)
    print("[data] model AZI-counterfactual predictions...")
    azi_groups = CH.load_azi_counterfactual(AZI_JSONL)

    gw = pd.read_csv(HERE / "reports" / "human_rescue_genomewide.csv")
    gw = gw[(gw.condition == CONDITION) & gw.passes_min_pct_disease & gw.passes_min_pct_azi]

    gene_index = {g: i for i, g in enumerate(model.gene_names)}

    for ct in CH.CT_ORDER:
        if ct not in disease_mats or ct not in term_mats:
            continue
        g = gw[gw.cell_type == ct]
        keep = np.array([gene_index[x] for x in g.gene])

        ctrl = CR.truncate_topk(term_mats[ct])[:, keep]
        dis = CR.truncate_topk(disease_mats[ct])[:, keep]

        arecs = azi_groups.get((ct, COND_MODEL), [])
        if len(arecs) < MIN_AZI_CELLS:
            print(f"  {ct}: no AZI arm, only {len(arecs)} predictions")
            azi = None
        else:
            azi = rem.reconstruct_batch([r["pred"] for r in arecs], model, CR.K)[:, keep]

        arms = {"disease": (ctrl, dis,
                            "term-born" if control == "term" else "fetal GW22", CONDITION)}
        if azi is not None:
            arms["azi"] = (dis, azi, CONDITION, "AZI-pred")

        for arm, (A, B, la, lb) in arms.items():
            X = np.vstack([A, B]).astype(np.float32)
            group = np.array([0] * A.shape[0] + [1] * B.shape[0])
            stem = ct if arm == "disease" else f"{ct}_azi"
            X.T.tofile(OUT / f"{stem}.bin")
            (OUT / f"{stem}.json").write_text(json.dumps({
                "cell_type": ct, "arm": arm,
                "n_genes": int(X.shape[1]), "n_cells": int(X.shape[0]),
                "n_ref": int(A.shape[0]), "n_test": int(B.shape[0]),
                "label_ref": la, "label_test": lb,
                "genes": list(g.gene), "group": group.tolist(),
                "stat": g.avg_log2FC_disease_top800.tolist(),
            }))
            print(f"  {ct} [{arm}]: {X.shape[1]} genes x {X.shape[0]} cells "
                  f"({A.shape[0]} {la} + {B.shape[0]} {lb})")

    cache = OUT / "all_terms.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        print(f"\n  reusing {cache.name}")
    else:
        blob = {}
        for lib in LIBRARIES:
            print(f"  [{lib}] downloading...", flush=True)
            blob[lib] = gp.get_library(name=lib)
        cache.write_text(json.dumps(blob))

    sets = {}
    for lib, terms in blob.items():
        for t, genes in terms.items():
            sets[f"{lib}__{t}"] = sorted(set(genes))
    (OUT / "sets.json").write_text(json.dumps(sets))
    print(f"\n  {len(sets)} library terms from {len(blob)} libraries -> sets.json")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
