#!/usr/bin/env python3
"""Reproduces Table 1 of Methods: unique cells that enter at least one forward
pair, per split and species. Replays the seeded pairing of
pipeline_short/stage1_sft_bidir.py through
real_counts_suite_ortho/human_panel_variants/recover_human_test_barcodes.py.
Pairing depends only on which barcodes are present, not on sentence text.

    python table1_cells_in_pairs.py
"""
import sys
import pandas as pd
sys.path.insert(0, "/Users/stepandolzhenko/Documents/AzithroGemma/model_comparison_grpo_sft/down_genes/perturbation_sensitivity/stat_tests/abs_analysis/real_counts_suite_ortho/human_panel_variants")
import recover_human_test_barcodes as R

S1 = R.S1
reg = pd.read_csv(S1.REGISTRY_PATH)
reg.index = reg["barcode"]
allbc = {bc: "X" for bc in reg.index}
rows = []
for split in ["train", "valid", "test"]:
    for sp, pairs in [("rat", R.rat_pairs_with_barcodes(split, reg, allbc)),
                      ("human", R.human_pairs_with_barcodes(split, reg, allbc, allbc))]:
        fwd = [p for p in pairs if not p["pair_type"].endswith("_rev")]
        cells = {p["src_barcode"] for p in fwd} | {p["tgt_barcode"] for p in fwd}
        n_reg = int(((reg["split"] == split) & ((reg["source"] == "rat") == (sp == "rat"))).sum())
        rows.append({"split": split, "species": sp, "forward_pairs": len(fwd),
                     "cells_in_registry": n_reg, "cells_in_pairs": len(cells)})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print(t[["cells_in_registry", "cells_in_pairs", "forward_pairs"]].sum().to_dict())
