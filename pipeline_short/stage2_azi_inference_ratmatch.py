#!/usr/bin/env python3
"""
pipeline_short/stage2_azi_inference_ratmatch.py — AZI counterfactual inference builder (rat-matched prompt variant).

Same as stage2_azi_inference.py, but AZI_PERTURBATION is the literal, verbatim
rat perturbation string the model saw during rat SFT/GRPO training
(PERTURBATION_STR[("rat", "HO", "AZI")] in common.py):
  "Azithromycin treatment (30 mg/kg IP at P7, P10, P13) during after exposure
  to 85% O2 for 14 days."
No rewording for human dose/route/duration/condition — the point is to test
whether the model transfers the AZI signal when given the exact same
perturbation text it was trained on for rat, unchanged.

Human clinical translation: for every held-out (test + valid) BPD-source cell
in Acute26 / BPD7mo / BPDPH7mo, build a prompt asking the model to predict the
cell's expression under azithromycin treatment (AZI_PERTURBATION). There
is no ground truth for this counterfactual (azithromycin was never given to
these patients), so rows are prompt-only — no "completion" field.

Strictly split=="test" or split=="valid" barcodes from cell_split_registry_short.csv
(never train — checked below). Scope matches TARGET_HUMAN_CELL_TYPES (gCap,
aCap, Pulmonary venous EC, Pericyte), the scope shared with SFT/GRPO.

Writes a single combined file:
  pipeline_short/inference_azi_ratmatch.jsonl
Each line: {"prompt": ..., "condition": ..., "cell_type": ..., "split": ..., "barcode": ...}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline_short.common import (
    BASE, PIPELINE_DIR,
    COND_META, PERTURBATION_STR, SPECIES_STR,
    TARGET_HUMAN_CELL_TYPES,
    build_prompt, cell_type_broad,
)
from pipeline_short.stage1_sft_bidir import load_bpd_sentences

AZI_PERTURBATION = PERTURBATION_STR[("rat", "HO", "AZI")]

REGISTRY_PATH = BASE / "cell_split_registry_short.csv"
OUT_PATH      = PIPELINE_DIR / "inference_azi_ratmatch.jsonl"

INFERENCE_CONDITIONS = ["Acute26", "BPD7mo", "BPDPH7mo"]
HELD_OUT_SPLITS       = {"test", "valid"}


def build_rows(held_out: pd.DataFrame, bpd_sent: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for cond in INFERENCE_CONDITIONS:
        meta = COND_META[("human", cond)]
        sub  = held_out[held_out["condition"] == cond]

        n_before = len(rows)
        for ct, grp in sub.groupby("cell_type"):
            ct_broad = cell_type_broad(ct)
            for _, row in grp.iterrows():
                bc = row["barcode"]
                if bc not in bpd_sent:
                    continue
                prompt = build_prompt(
                    SPECIES_STR["human"], meta["age_str"], meta["pma_weeks"],
                    ct, ct_broad, AZI_PERTURBATION, bpd_sent[bc],
                )
                rows.append({
                    "prompt":    prompt,
                    "condition": cond,
                    "cell_type": ct,
                    "split":     row["split"],
                    "barcode":   bc,
                })

        cts = sorted(sub["cell_type"].unique())
        print(f"  [{cond}] {len(rows) - n_before:,} prompts, types: {cts}")
    return rows


def check_isolation(rows: list[dict], reg: pd.DataFrame) -> bool:
    """No inference barcode may appear in train."""
    train_bcs = set(reg[reg["split"] == "train"]["barcode"])
    inf_bcs   = {r["barcode"] for r in rows}
    overlap   = inf_bcs & train_bcs
    if overlap:
        print(f"  STOP: {len(overlap)} inference barcodes appear in train")
        return True
    print(f"  ✓ 0 inference barcodes in train ({len(inf_bcs):,} total)")
    return False


def check_no_completion(rows: list[dict]) -> bool:
    bad = sum(1 for r in rows if "completion" in r)
    if bad:
        print(f"  STOP: {bad} rows carry a completion field (should be prompt-only)")
        return True
    print("  ✓ all rows are prompt-only (no completion/ground truth)")
    return False


def main():
    print("=" * 64)
    print("STAGE 2 — AZI Counterfactual Inference Builder [rat-matched prompt] (test+valid, BPD source)")
    print("=" * 64)

    print("\nLoading registry...")
    reg = pd.read_csv(REGISTRY_PATH)

    held_out = reg[
        (reg["source"] == "bpd") &
        reg["condition"].isin(INFERENCE_CONDITIONS) &
        reg["cell_type"].isin(TARGET_HUMAN_CELL_TYPES) &
        reg["split"].isin(HELD_OUT_SPLITS)
    ]
    bpd_bcs = held_out["barcode"].tolist()
    print(f"  {len(bpd_bcs):,} held-out (test+valid) BPD barcodes to build sentences for")

    print("\n--- Building cell sentences ---")
    bpd_sent = load_bpd_sentences(bpd_bcs) if bpd_bcs else {}

    print("\n--- Building AZI inference prompts ---")
    rows = build_rows(held_out, bpd_sent)

    print("\n--- Sanity checks ---")
    blocking  = check_isolation(rows, reg)
    blocking |= check_no_completion(rows)
    if blocking:
        print("\n✗ Blocking check(s) failed.")
        sys.exit(1)

    if rows:
        print("\n--- Sample prompt ---")
        print(f"[{rows[0]['condition']}] cell_type={rows[0]['cell_type']} split={rows[0]['split']}")
        print(rows[0]["prompt"][:280])

    print("\n--- Saving ---")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {OUT_PATH}  ({len(rows):,} rows)")

    print(f"\n✓ Stage 2 complete. inference prompts = {len(rows):,}")


if __name__ == "__main__":
    main()
