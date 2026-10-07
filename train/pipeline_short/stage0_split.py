#!/usr/bin/env python3
"""
Stage 0: data audit and cell split registry.

  0a  cell counts per source x condition x cell type
  0b  barcode-level train/valid/test split, seeded per stratum

Sanity checks S0-1..S0-6 run before the registry is written. If the registry
already exists, only the checks are run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline_short.common import (
    BASE, PIPELINE_DIR,
    RAT_PATH, ATLAS_CLEAN_PATH, BPD_META_PATH, CONFIG_PATH,
    DATASET_TO_CONDITION, HE_STAGE_TO_CONDITION,
    HARMONIZE, harmonize_cell_type,
    cell_type_broad,
    load_grpo_endo_types,
    stratum_seed,
    TARGET_RAT_CELL_TYPES, TARGET_HUMAN_CELL_TYPES,
    TARGET_RAT_CONDITIONS, TARGET_HUMAN_CONDITIONS,
)

REGISTRY_PATH = BASE / "cell_split_registry_short.csv"
AUDIT_PATH    = PIPELINE_DIR / "data_audit.csv"
SUMMARY_PATH  = PIPELINE_DIR / "split_summary.csv"

EXPECTED_BPD_COUNTS: dict[str, int] = {
    "Acute26":   3745,
    "BPD7mo":   14701,
    "BPDPH7mo":  8594,
    "Term0d":    8339,
    "Term20d":   8228,
}

ATLAS_INTENTIONALLY_UNMAPPED: set[str] = {
    "OMD+ endo",
    "Squamous", "MUC5AC+ ASCL1+",
    "GHRL+ neuroendocrine", "GHRL+ NE precursor",
    "Pulmonary neuroendocrine", "Pulmonary NE precursor", "Interm neuroendocrine",
    "Eosinophil",
    "HSC", "HSC/ELP", "CMP", "GMP", "MEP",
    "ILC2", "ILC3", "ILCP",
    "Cycling definitive erythroblast", "Definitive erythroblast", "Definitive erythrocyte",
    "Primitive erythroblast", "Primitive erythrocyte", "Definitive reticulocyte",
    "Megakaryocyte", "Platelet",
    "ASPN+ chondrocyte", "Interm chondrocyte", "Resting chondrocyte",
    "COL20A1+ Schwann", "Late Schwann", "Mid Schwann", "Proliferating Schwann", "Schwann precursor",
    "Early mesothelial", "Mid mesothelial", "Late mesothelial",
    "Late airway progenitor", "Mid airway progenitor", "Early airway progenitor",
    "Enteric neuron 1", "Enteric neuron 2", "KCNIP4+ neuron", "FGFBP2+ Neural progenitor",
    "Sympathoadrenal progenitor", "Chromaffin cell", "GHRL+ neuroendocrine"
}

HARMONIZE_UNCERTAIN: set[str] = set()

SCFID_RESERVATIONS: dict[tuple, int] = {}
SCFID_TRANSITIONS = [
    ("rat",   "RA",      "HO",       "gCAP"),
    ("rat",   "HO",      "AZI",      "gCAP"),
    ("human", "Acute26", "BPDPH7mo", "gCap"),
]
TEST_FRACTION   = 0.15
TEST_FLOOR      = 10
TEST_CAP        = 150
N_VALID_DEFAULT = 2
N_VALID_GCAP    = 5
GCAP_TYPES      = {"gCap", "gCAP"}
TRAIN_FLOOR     = 20


def _cond_to_source(cond: str) -> str:
    if cond in {"RA", "HO", "AZI"}:
        return "rat"
    if cond.startswith("He"):
        return "atlas"
    return "bpd"


class CellSplitRegistry:

    def __init__(self) -> None:
        self.grpo_endo = load_grpo_endo_types()
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


    def _load_rat(self) -> pd.DataFrame:
        print("Loading rat h5ad obs...")
        adata = anndata.read_h5ad(RAT_PATH)
        obs   = adata.obs[["cell_type_lab_fine", "sample"]].copy()
        del adata

        rows = []
        for barcode, row in obs.iterrows():
            cond = str(row["sample"])
            if cond not in TARGET_RAT_CONDITIONS:
                continue
            raw_ct = str(row["cell_type_lab_fine"])
            canon  = harmonize_cell_type(raw_ct)
            if canon not in TARGET_RAT_CELL_TYPES:
                continue
            rows.append({
                "barcode":       barcode,
                "source":        "rat",
                "condition":     cond,
                "cell_type_raw": raw_ct,
                "cell_type":     canon,
                "lineage":       cell_type_broad(canon),
            })
        df = pd.DataFrame(rows)
        print(f"  rat:   {len(df):6d} cells, "
              f"{df['cell_type'].nunique()} types, "
              f"{df['condition'].nunique()} conditions")
        return df

    def _load_atlas(self) -> pd.DataFrame:
        print("Loading atlas h5ad obs...")
        adata = anndata.read_h5ad(ATLAS_CLEAN_PATH)
        obs   = adata.obs[["new_celltype", "broad_celltype", "stage"]].copy()
        del adata

        rows = []
        for barcode, row in obs.iterrows():
            try:
                stage = float(row["stage"])
            except (ValueError, TypeError):
                continue
            cond = HE_STAGE_TO_CONDITION.get(stage)
            if cond is None or cond not in TARGET_HUMAN_CONDITIONS:
                continue
            raw_ct  = str(row["new_celltype"])
            if raw_ct in ATLAS_INTENTIONALLY_UNMAPPED:
                continue
            canon   = harmonize_cell_type(raw_ct)
            if canon not in TARGET_HUMAN_CELL_TYPES:
                continue
            lineage = str(row["broad_celltype"])
            rows.append({
                "barcode":       barcode,
                "source":        "atlas",
                "condition":     cond,
                "cell_type_raw": raw_ct,
                "cell_type":     canon,
                "lineage":       lineage,
            })
        df = pd.DataFrame(rows)
        print(f"  atlas: {len(df):6d} cells, "
              f"{df['cell_type'].nunique()} types, "
              f"{df['condition'].nunique()} conditions")
        return df

    def _load_bpd(self) -> pd.DataFrame:
        print("Loading BPD metadata CSV...")
        meta = pd.read_csv(BPD_META_PATH, index_col="id")

        rows = []
        for barcode, row in meta.iterrows():
            cond = DATASET_TO_CONDITION.get(str(row["dataset"]))
            if cond is None or cond not in TARGET_HUMAN_CONDITIONS:
                continue
            raw_ct  = str(row["celltype"])
            canon   = harmonize_cell_type(raw_ct)
            if canon not in TARGET_HUMAN_CELL_TYPES:
                continue
            lineage = str(row.get("celltype_lineage", "unknown"))
            rows.append({
                "barcode":       str(barcode),
                "source":        "bpd",
                "condition":     cond,
                "cell_type_raw": raw_ct,
                "cell_type":     canon,
                "lineage":       lineage,
            })
        df = pd.DataFrame(rows)
        print(f"  bpd:   {len(df):6d} cells, "
              f"{df['cell_type'].nunique()} types, "
              f"{df['condition'].nunique()} conditions")
        return df

    def load_all(self) -> pd.DataFrame:
        dfs = [self._load_rat(), self._load_atlas(), self._load_bpd()]
        df  = pd.concat(dfs, ignore_index=True)

        rat_endo   = self.grpo_endo["rat"]
        human_endo = self.grpo_endo["human"]
        df["grpo_endo"] = False
        rat_mask   = df["source"] == "rat"
        human_mask = df["source"].isin({"atlas", "bpd"})
        df.loc[rat_mask   & df["cell_type"].isin(rat_endo),   "grpo_endo"] = True
        df.loc[human_mask & df["cell_type"].isin(human_endo), "grpo_endo"] = True

        print(f"\nTotal: {len(df):,} cells  "
              f"(grpo_endo={df['grpo_endo'].sum():,})")
        return df


    def check_s01_audit(self, df: pd.DataFrame) -> None:
        print("\n" + "═" * 64)
        print("[S0-1] DATA AUDIT — counts per source × condition × cell_type")
        print("═" * 64)
        audit = (
            df.groupby(["source", "condition", "cell_type"])
            .size()
            .reset_index(name="n_cells")
            .sort_values(["source", "condition", "n_cells"], ascending=[True, True, False])
        )
        audit.to_csv(AUDIT_PATH, index=False)
        print(audit.to_string(index=False))
        print(f"\n  Saved → {AUDIT_PATH}")

        print("\n  BPD per-condition (expected vs actual):")
        bpd_counts = df[df["source"] == "bpd"].groupby("condition").size()
        for cond, exp in sorted(EXPECTED_BPD_COUNTS.items()):
            act  = bpd_counts.get(cond, 0)
            flag = "✓" if act == exp else "WARN"
            print(f"    {cond:12s}: expected={exp:5d}, actual={act:5d}  {flag}")

    def check_s02_config(self, df: pd.DataFrame) -> None:
        print("\n" + "═" * 64)
        print("[S0-2] CONFIG COVERAGE — config types found in obs?")
        print("═" * 64)
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)

        checks = [
            ("RAT",      "rat",   "cell_type_raw"),
            ("ATLAS_He", "atlas", "cell_type_raw"),
        ]
        for cfg_key, source, col in checks:
            obs_types = set(df[df["source"] == source][col].unique())
            for list_name in ("include", "ot_include"):
                wanted  = cfg.get(cfg_key, {}).get(list_name, [])
                missing = [t for t in wanted if t not in obs_types]
                tag = "OK  " if not missing else "WARN"
                print(f"  {tag} [{cfg_key}.{list_name}] "
                      f"{len(wanted) - len(missing)}/{len(wanted)} found"
                      + (f"  missing={missing}" if missing else ""))

    def check_s03_harmonize(self, df: pd.DataFrame) -> bool:
        """Bidirectional harmonize coverage. Returns True if STOP needed."""
        print("\n" + "═" * 64)
        print("[S0-3] HARMONIZE COVERAGE (bidirectional)")
        print("═" * 64)

        bpd_canon = set(df[df["source"] == "bpd"]["cell_type_raw"].unique())

        atlas_df  = df[df["source"] == "atlas"].copy()
        atlas_cts = (
            atlas_df.groupby(["cell_type_raw", "cell_type"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )

        unmapped = atlas_cts[~atlas_cts["cell_type"].isin(bpd_canon)]
        print(f"\n  (a) Atlas types NOT mapping to a BPD canon ({len(unmapped)} raw types):")
        stop_needed = False
        for _, row in unmapped.iterrows():
            raw, canon, n = row["cell_type_raw"], row["cell_type"], int(row["n"])
            intent = raw in ATLAS_INTENTIONALLY_UNMAPPED
            uncert = raw in HARMONIZE_UNCERTAIN
            if not intent and n > 500:
                tag = "✗ STOP  — unexpected unmapped, >500 cells"
                stop_needed = True
            elif not intent:
                tag = "WARN    — unexpected unmapped, check HARMONIZE"
            else:
                tag = "ok      — intentionally unmapped"
            mark = "?" if uncert else " "
            print(f"    {mark} {raw:50s} n={n:5d}  {tag}")

        mapped_bpd = set(atlas_cts["cell_type"].unique())
        unmatched  = sorted(bpd_canon - mapped_bpd)
        bpd_counts = df[df["source"] == "bpd"].groupby("cell_type_raw").size()
        print(f"\n  (b) BPD canonical types with NO atlas source ({len(unmatched)}):")
        for ct in unmatched:
            n = int(bpd_counts.get(ct, 0))
            print(f"      {ct:50s} n={n:5d}  (BPD-only, no atlas pairing)")

        uncertain_obs = HARMONIZE_UNCERTAIN & set(atlas_cts["cell_type_raw"].unique())
        if uncertain_obs:
            print(f"\n  (c) Uncertain ? mappings for human review ({len(uncertain_obs)}):")
            for raw in sorted(uncertain_obs):
                canon = harmonize_cell_type(raw)
                n = int(atlas_cts.loc[atlas_cts["cell_type_raw"] == raw, "n"].sum())
                print(f"      {raw:50s} → {canon}  n={n}")

        if stop_needed:
            print("\n  ✗ STOP: unexpected high-count unmapped atlas type(s). Fix HARMONIZE.")
        return stop_needed

    def check_s04_barcode_uniqueness(self, df: pd.DataFrame) -> None:
        print("\n" + "═" * 64)
        print("[S0-4] BARCODE UNIQUENESS between sources")
        print("═" * 64)
        sources = df["source"].unique().tolist()
        fail = False
        for i, s1 in enumerate(sources):
            for s2 in sources[i + 1:]:
                bc1     = set(df[df["source"] == s1]["barcode"])
                bc2     = set(df[df["source"] == s2]["barcode"])
                overlap = len(bc1 & bc2)
                if overlap:
                    print(f"  ✗ STOP: {s1} ∩ {s2} = {overlap} overlapping barcodes")
                    fail = True
                else:
                    print(f"  ✓ {s1} ∩ {s2} = 0")
        if fail:
            raise AssertionError("Barcode overlap between sources — check data loading.")


    def build_split(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n" + "═" * 64)
        print("BUILDING SPLIT (barcode-level, deterministic seed per stratum)")
        print("═" * 64)
        chunks = []
        for (source, condition, cell_type), grp in df.groupby(
            ["source", "condition", "cell_type"], sort=True
        ):
            n    = len(grp)
            seed = stratum_seed(source, condition, cell_type)
            rng  = np.random.default_rng(seed)
            idx  = rng.permutation(n)

            scfid_key = (source, condition, cell_type)
            if scfid_key in SCFID_RESERVATIONS:
                n_test  = min(SCFID_RESERVATIONS[scfid_key], n)
                n_valid = 0
            else:
                val_target = N_VALID_GCAP if cell_type in GCAP_TYPES else N_VALID_DEFAULT
                test_target = min(TEST_CAP, max(TEST_FLOOR, round(TEST_FRACTION * n)))
                avail   = max(0, n - TRAIN_FLOOR)
                n_test  = min(test_target,  avail)
                n_valid = min(val_target, max(0, avail - n_test))

            labels = np.empty(n, dtype=object)
            labels[idx[:n_test]]                      = "test"
            labels[idx[n_test:n_test + n_valid]]      = "valid"
            labels[idx[n_test + n_valid:]]            = "train"

            grp = grp.copy()
            grp["split"] = labels
            chunks.append(grp)

        registry = pd.concat(chunks, ignore_index=True)
        vc = registry["split"].value_counts()
        print(f"  train={vc.get('train',0):,}  valid={vc.get('valid',0):,}  test={vc.get('test',0):,}")
        return registry


    def check_s05_scfid(self, registry: pd.DataFrame) -> None:
        print("\n" + "═" * 64)
        print("[S0-5] scFID SETUP — held-out source cells + valid transitions")
        print("═" * 64)

        if not SCFID_RESERVATIONS:
            print("\n  SCFID_RESERVATIONS is empty — no scFID carve-out active, "
                  "all strata use the general train/valid/test rule.")
            return

        print("\n  Reserved test cells per scFID source stratum:")
        for (src, cond, ct), expected in SCFID_RESERVATIONS.items():
            mask  = ((registry["source"] == src) &
                     (registry["condition"] == cond) &
                     (registry["cell_type"] == ct))
            n_tot   = mask.sum()
            n_test  = (mask & (registry["split"] == "test")).sum()
            n_train = (mask & (registry["split"] == "train")).sum()
            flag    = "✓" if n_test == expected else f"WARN (expected {expected})"
            print(f"    {flag}  {src} [{cond}] {ct}: "
                  f"total={n_tot}  test={n_test}  train={n_train}")

        print("\n  Valid scFID transitions (gCap, ≥512 real cells in ref):")
        for sp, src_c, tgt_c, ct in SCFID_TRANSITIONS:
            tgt_src  = _cond_to_source(tgt_c)
            n_ref    = ((registry["source"] == tgt_src) &
                        (registry["condition"] == tgt_c) &
                        (registry["cell_type"] == ct)).sum()
            note = "256 test + ×2 stochastic = 512 at eval" if src_c == "HO" else "512 test"
            print(f"    ✓  {sp} {src_c}→{tgt_c} ({ct}): "
                  f"source={note},  ref={n_ref} real cells (all in train)")

        print("\n  Skipped scFID transitions (insufficient gCap):")
        print("    —  human He22→*  (He22 gCap ≤512, reserving would empty train)")
        print("    —  human Acute26→BPD7mo  (BPD7mo gCap=138 < 512 reference threshold)")

    def check_s06_isolation(self, registry: pd.DataFrame) -> None:
        print("\n" + "═" * 64)
        print("[S0-6] SPLIT ISOLATION")
        print("═" * 64)
        train_bc = set(registry[registry["split"] == "train"]["barcode"])
        valid_bc = set(registry[registry["split"] == "valid"]["barcode"])
        test_bc  = set(registry[registry["split"] == "test"]["barcode"])

        for name, a, b in [
            ("train ∩ valid", train_bc, valid_bc),
            ("valid ∩ test",  valid_bc, test_bc),
            ("train ∩ test",  train_bc, test_bc),
        ]:
            overlap = len(a & b)
            flag = "✓" if overlap == 0 else "✗ STOP"
            print(f"  {flag} {name} = {overlap}")
            assert overlap == 0, f"Leakage: {name} = {overlap} barcodes"


    def save(self, registry: pd.DataFrame) -> None:
        cols = [
            "barcode", "source", "condition",
            "cell_type_raw", "cell_type", "lineage",
            "grpo_endo", "split",
        ]
        registry[cols].to_csv(REGISTRY_PATH, index=False)
        print(f"\n  Saved: {REGISTRY_PATH}  ({len(registry):,} rows)")

        summary = (
            registry
            .groupby(["source", "condition", "cell_type", "split"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ("train", "valid", "test"):
            if col not in summary.columns:
                summary[col] = 0
        summary[["source", "condition", "cell_type", "train", "valid", "test"]].to_csv(
            SUMMARY_PATH, index=False
        )
        print(f"  Saved: {SUMMARY_PATH}")

    def run(self) -> pd.DataFrame:
        if REGISTRY_PATH.exists():
            print(f"Registry exists: {REGISTRY_PATH}")
            print("Skipping regeneration — running sanity checks only.\n")
            registry = pd.read_csv(REGISTRY_PATH)
            self.check_s06_isolation(registry)
            self.check_s05_scfid(registry)
            print("\n✓ Registry sanity checks passed.")
            return registry

        print("=" * 64)
        print("STAGE 0 — Data Audit + CellSplitRegistry")
        print("=" * 64 + "\n")

        df = self.load_all()
        self.check_s01_audit(df)
        self.check_s02_config(df)
        stop = self.check_s03_harmonize(df)
        self.check_s04_barcode_uniqueness(df)

        if stop:
            print("\n✗ Blocking check(s) failed. Fix HARMONIZE and re-run.")
            sys.exit(1)

        registry = self.build_split(df)
        self.check_s05_scfid(registry)
        self.check_s06_isolation(registry)
        self.save(registry)

        print("\n✓ Stage 0 complete.")
        return registry


if __name__ == "__main__":
    CellSplitRegistry().run()
