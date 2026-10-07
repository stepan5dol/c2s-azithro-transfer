#!/usr/bin/env python3
"""
Writes the rescued genes of reports/human_rescue_genomewide.csv as one xlsx
per (condition, cell type), in the column layout of the rat workbooks:

    gene | avg_log2FC_HO | p_val_adj_HO | avg_log2FC_AZI | p_val_adj_AZI
         | recovery_ratio | recovery_type

Column mapping (top-800 values):

    avg_log2FC_disease_top800  ->  avg_log2FC_HO
    p_val_adj_disease_top800   ->  p_val_adj_HO
    avg_log2FC_azi_top800      ->  avg_log2FC_AZI
    p_val_adj_azi_top800       ->  p_val_adj_AZI

Rows: rescued (disease_candidate, azi_candidate and recovery_ratio < 0.9),
sorted by p_val_adj_HO.

Output: reports/rat_schema/human_{condition}_{ct}_rescue_predicted_azi.xlsx,
reports/rat_schema/index.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "reports" / "human_rescue_genomewide.csv"
OUT_DIR = HERE / "reports" / "rat_schema"

RAT_COLUMNS = ["gene", "avg_log2FC_HO", "p_val_adj_HO", "avg_log2FC_AZI",
               "p_val_adj_AZI", "recovery_ratio", "recovery_type"]

RENAME = {
    "avg_log2FC_disease_top800": "avg_log2FC_HO",
    "p_val_adj_disease_top800": "p_val_adj_HO",
    "avg_log2FC_azi_top800": "avg_log2FC_AZI",
    "p_val_adj_azi_top800": "p_val_adj_AZI",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[read] {SRC}")
    df = pd.read_csv(SRC)
    print(f"  {len(df)} rows in, {df['rescued'].sum()} rescued")

    summary = []
    for (cond, ct), sub in df[df["rescued"]].groupby(["condition", "cell_type"], sort=False):
        table = sub.rename(columns=RENAME)[RAT_COLUMNS].copy()
        table = table.sort_values("p_val_adj_HO", kind="mergesort").reset_index(drop=True)

        out_path = OUT_DIR / f"human_{cond}_{ct}_rescue_predicted_azi.xlsx"
        table.to_excel(out_path, index=False, sheet_name="Sheet1")

        n_down = int((table["recovery_type"] == "Rescued_HO_down").sum())
        n_up = int((table["recovery_type"] == "Rescued_HO_up").sum())
        summary.append({"condition": cond, "cell_type": ct, "n_genes": len(table),
                        "n_down": n_down, "n_up": n_up, "file": out_path.name})
        print(f"  {cond}/{ct}: {len(table)} genes ({n_down} down / {n_up} up) -> {out_path.name}")

    rep = pd.DataFrame(summary)
    rep_path = OUT_DIR / "index.csv"
    rep.to_csv(rep_path, index=False)
    print(f"\n-> {OUT_DIR}  ({len(rep)} workbooks)")
    print(f"-> {rep_path}")


if __name__ == "__main__":
    main()
