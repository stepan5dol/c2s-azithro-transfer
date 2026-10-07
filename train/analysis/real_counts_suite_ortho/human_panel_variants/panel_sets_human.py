#!/usr/bin/env python3
"""
Rescued-gene panels for the human figures, read from the workbooks written by
filter_to_rat_schema.py
(reports/rat_schema/human_{condition}_{ct}_rescue_predicted_azi.xlsx, rat
column layout). The AZI columns hold the model's prediction.

Gates, as for rat (rat_panel_variants/panel_sets.py):
  full_list  all rows, split by recovery_type
  gated      additionally p_val_adj_HO < 0.05 and p_val_adj_AZI >= 0.05
             (or missing)

Columns (human_rescue_pipeline_genomewide.py):
  p_val_adj_HO   Term control vs disease, both measured
  p_val_adj_AZI  Term control vs azithromycin prediction
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
XLSX_DIR = HERE / "reports" / "rat_schema"
CONDITIONS = ("Acute26", "BPD7mo", "BPDPH7mo")
CONDITION = "BPD7mo"
CT_ORDER = ["gCap", "aCap", "Pericyte", "VEC"]

GATES = ("full_list", "gated")
RECOVERY_TYPES = ("Rescued_HO_down", "Rescued_HO_up")

GATE_DESC = {
    "full_list": "all genes of the workbook, split by recovery_type, no p-value filter",
    "gated": "p_val_adj_HO<0.05 AND p_val_adj_AZI>=0.05",
}

EXPECTED = {
    ("Rescued_HO_down", "disease"): "falls", ("Rescued_HO_down", "azi"): "rises",
    ("Rescued_HO_up", "disease"): "rises", ("Rescued_HO_up", "azi"): "falls",
}


def load_panels(gate: str = "gated", condition: str = CONDITION,
                max_p_ho: float = 0.05,
                min_p_azi: float = 0.05) -> dict[str, dict[str, set[str]]]:
    """-> {cell_type: {"Rescued_HO_down": {...}, "Rescued_HO_up": {...}}}"""
    assert gate in GATES, f"unknown gate {gate!r}, expected one of {GATES}"
    assert condition in CONDITIONS, f"unknown condition {condition!r}"
    out: dict[str, dict[str, set[str]]] = {}
    for ct in CT_ORDER:
        path = XLSX_DIR / f"human_{condition}_{ct}_rescue_predicted_azi.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"{path} -- run filter_to_rat_schema.py first")
        df = pd.read_excel(path, sheet_name="Sheet1")
        if gate == "gated":
            df = df[(df["p_val_adj_HO"] < max_p_ho)
                    & (df["p_val_adj_AZI"].isna() | (df["p_val_adj_AZI"] >= min_p_azi))]
        out[ct] = {rt: set(df.loc[df["recovery_type"] == rt, "gene"]) for rt in RECOVERY_TYPES}
    return out


def main():
    tables = {g: load_panels(g) for g in GATES}
    print(f"{'cell type':10s} {'recovery_type':18s} {'full_list':>10s} {'gated':>7s}  removed by gate")
    for ct in CT_ORDER:
        for rt in RECOVERY_TYPES:
            a = len(tables["full_list"][ct][rt])
            b = len(tables["gated"][ct][rt])
            print(f"{ct:10s} {rt:18s} {a:10d} {b:7d}  {a - b}")


if __name__ == "__main__":
    main()
