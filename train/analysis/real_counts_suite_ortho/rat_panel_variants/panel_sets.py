#!/usr/bin/env python3
"""
Rescued-gene panels for the rat figures.

Reads the per-cell-type rescue workbooks (common.DGE_XLSX, Sheet1; columns
gene, avg_log2FC_HO, p_val_adj_HO, avg_log2FC_AZI, p_val_adj_AZI,
recovery_ratio, recovery_type) and returns Rescued_HO_down and Rescued_HO_up
as separate gene sets.

Gates:
  full_list  all rows, split by recovery_type
  gated      additionally p_val_adj_HO < 0.05 and p_val_adj_AZI >= 0.05
             (or missing), as in common.load_celltype_rescue_genes()
"""
from __future__ import annotations

import sys
from pathlib import Path

import openpyxl

ABS_DIR = Path(__file__).resolve().parents[2]  # train/analysis
sys.path.insert(0, str(ABS_DIR))
import common as C

GATES = ("full_list", "gated")
RECOVERY_TYPES = ("Rescued_HO_down", "Rescued_HO_up")

GATE_DESC = {
    "full_list": "all genes of the list, split by recovery_type, no p-value filter",
    "gated": "p_val_adj_HO<0.05 AND p_val_adj_AZI>=0.05",
}

EXPECTED = {
    ("Rescued_HO_down", "HO"): "falls", ("Rescued_HO_down", "AZI"): "rises",
    ("Rescued_HO_up", "HO"): "rises", ("Rescued_HO_up", "AZI"): "falls",
}


def load_panels(gate: str = "gated", max_p_ho: float = 0.05,
                min_p_azi: float = 0.05) -> dict[str, dict[str, set[str]]]:
    """-> {cell_type: {"Rescued_HO_down": {...}, "Rescued_HO_up": {...}}}"""
    assert gate in GATES, f"unknown gate {gate!r}, expected one of {GATES}"
    out: dict[str, dict[str, set[str]]] = {}
    for ct, path in C.DGE_XLSX.items():
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb["Sheet1"]
        data = list(ws.iter_rows(values_only=True))[1:]
        per_rt = {}
        for rt in RECOVERY_TYPES:
            keep = set()
            for r in data:
                if r[6] != rt:
                    continue
                if gate == "gated":
                    if r[2] is None or r[2] >= max_p_ho:
                        continue
                    if not (r[4] is None or r[4] >= min_p_azi):
                        continue
                keep.add(r[0])
            per_rt[rt] = keep
        out[ct] = per_rt
        wb.close()
    return out


def main():
    """Print panel sizes for every (cell type, recovery_type, variant)."""
    tables = {g: load_panels(g) for g in GATES}
    print(f"{'cell type':10s} {'recovery_type':18s} {'full_list':>10s} {'gated':>7s}  removed by gate")
    for ct in C.CT_ORDER:
        for rt in RECOVERY_TYPES:
            a = len(tables["full_list"][ct][rt])
            b = len(tables["gated"][ct][rt])
            print(f"{ct:10s} {rt:18s} {a:10d} {b:7d}  {a - b}")


if __name__ == "__main__":
    main()
