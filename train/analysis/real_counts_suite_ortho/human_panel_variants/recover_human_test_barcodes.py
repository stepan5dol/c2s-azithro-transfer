#!/usr/bin/env python3
"""
Recovers the measured barcodes behind each forward human test example, as
rat/recover_rat_test_barcodes.py does for rat.

SFTBuilder.run() concatenates the rat pairs and then the human pairs before
_dedup_reverse, so both blocks are replayed in that order, deduplicated
together and then enumerated. Each row is verified by exact match of the
completion with the "gt" field of test_inference_results_t10.jsonl.

Output: recovered_human_test_pairs.csv (idx, pair_type, cell_type,
src_barcode, tgt_barcode, verified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_SHORT = Path(__file__).resolve().parents[3] / "pipeline_short"
sys.path.insert(0, str(PIPELINE_SHORT))
import stage1_sft_bidir as S1

HERE = Path(__file__).parent
TEST_INFERENCE_PATH = Path(__file__).resolve().parents[3] / "results/test_inference_results_t10.jsonl"


def _build_pairs_with_barcodes(
    src_bcs, tgt_bcs, src_sent, tgt_sent, species, age_str_src, pma_weeks_src,
    cell_type, pert_str, pair_type, split, n_per_ct, n_augment,
    age_str_tgt=None, pma_weeks_tgt=None, bidirectional=False, rev_pert_str=None,
):
    """stage1_sft_bidir._build_pairs with src_barcode and tgt_barcode added to
    each pair; RNG calls and their order are unchanged.
    """
    rev_pert_str = pert_str if rev_pert_str is None else rev_pert_str
    src = [bc for bc in src_bcs if bc in src_sent]
    tgt = [bc for bc in tgt_bcs if bc in tgt_sent]
    if not src or not tgt:
        return []

    rng = np.random.default_rng(S1._pair_seed(split, pair_type, cell_type))
    if n_per_ct is not None:
        if len(src) > n_per_ct:
            src = list(rng.choice(src, n_per_ct, replace=False))
        if len(tgt) > n_per_ct:
            tgt = list(rng.choice(tgt, n_per_ct, replace=False))

    ct_broad = S1.cell_type_broad(cell_type)
    pairs = []
    src_a, tgt_a = list(src), list(tgt)
    for _ in range(n_augment):
        rng.shuffle(src_a)
        rng.shuffle(tgt_a)
        n = min(len(src_a), len(tgt_a))
        for i in range(n):
            src_bc, tgt_bc = src_a[i], tgt_a[i]
            src_cs = src_sent[src_bc]
            tgt_cs = tgt_sent[tgt_bc]
            prompt = S1.build_prompt(species, age_str_src, pma_weeks_src,
                                      cell_type, ct_broad, pert_str, src_cs)
            pairs.append({"prompt": prompt, "completion": tgt_cs, "pair_type": pair_type,
                          "src_barcode": src_bc, "tgt_barcode": tgt_bc})
            if bidirectional:
                rprompt = S1.build_prompt_reverse(species, tgt_cs)
                rcompletion = S1.build_completion_reverse(
                    age_str_tgt, pma_weeks_tgt, cell_type, ct_broad, rev_pert_str)
                pairs.append({"prompt": rprompt, "completion": rcompletion,
                              "pair_type": pair_type + "_rev",
                              "src_barcode": None, "tgt_barcode": tgt_bc})
    return pairs


def rat_pairs_with_barcodes(split, reg, sent):
    """SFTBuilder._rat_pairs, replayed for the index offset of the human block."""
    reg_s = reg[(reg["source"] == "rat") & (reg["split"] == split)]
    pairs = []
    for sc, tc in [("RA", "HO"), ("HO", "AZI")]:
        pert = S1.PERTURBATION_STR[("rat", sc, tc)]
        meta_src, meta_tgt = S1.COND_META[("rat", sc)], S1.COND_META[("rat", tc)]
        src_r, tgt_r = reg_s[reg_s["condition"] == sc], reg_s[reg_s["condition"] == tc]
        for ct in sorted(set(src_r["cell_type"]) & set(tgt_r["cell_type"])):
            src_bcs = src_r[src_r["cell_type"] == ct].index.tolist()
            tgt_bcs = tgt_r[tgt_r["cell_type"] == ct].index.tolist()
            aug = S1.EVAL_AUG if split != "train" else (
                S1.RAT_AUG_SMALL if min(len(src_bcs), len(tgt_bcs)) < S1.RAT_AUG_THRESHOLD
                else S1.RAT_AUG_LARGE)
            pairs.extend(_build_pairs_with_barcodes(
                src_bcs, tgt_bcs, sent, sent,
                S1.SPECIES_STR["rat"], meta_src["age_str"], meta_src["pma_weeks"],
                ct, pert, f"rat_{sc}_{tc}", split, n_per_ct=None, n_augment=aug,
                age_str_tgt=meta_tgt["age_str"], pma_weeks_tgt=meta_tgt["pma_weeks"],
                bidirectional=True))
    return pairs


def human_pairs_with_barcodes(split, reg, atl_s, bpd_s):
    """SFTBuilder._human_pairs with the barcode-keeping builder."""
    atl = reg[(reg["source"] == "atlas") & (reg["split"] == split)]
    bpd = reg[(reg["source"] == "bpd") & (reg["split"] == split)]
    pairs = []
    for sc, tc in S1.HUMAN_SFT:
        key = ("human", sc, tc)
        if key not in S1.PERTURBATION_STR:
            continue
        pert = S1.PERTURBATION_STR[key]
        rev_pert = S1.REV_CONDITION_OVERRIDE.get(key, pert)
        meta_src, meta_tgt = S1.COND_META[("human", sc)], S1.COND_META[("human", tc)]
        src_reg = atl if S1._src_of_human(sc) == "atlas" else bpd
        tgt_reg = atl if S1._src_of_human(tc) == "atlas" else bpd
        ss = atl_s if S1._src_of_human(sc) == "atlas" else bpd_s
        ts = atl_s if S1._src_of_human(tc) == "atlas" else bpd_s
        src_r, tgt_r = src_reg[src_reg["condition"] == sc], tgt_reg[tgt_reg["condition"] == tc]
        for ct in sorted(set(src_r["cell_type"]) & set(tgt_r["cell_type"])):
            src_bcs = src_r[src_r["cell_type"] == ct].index.tolist()
            tgt_bcs = tgt_r[tgt_r["cell_type"] == ct].index.tolist()
            aug = S1.EVAL_AUG if split != "train" else (
                S1.HUMAN_AUG_SMALL if min(len(src_bcs), len(tgt_bcs)) < S1.HUMAN_AUG_THRESHOLD
                else S1.HUMAN_AUG_LARGE)
            pairs.extend(_build_pairs_with_barcodes(
                src_bcs, tgt_bcs, ss, ts,
                S1.SPECIES_STR["human"], meta_src["age_str"], meta_src["pma_weeks"],
                ct, pert, f"human_{sc}_{tc}", split, n_per_ct=None, n_augment=aug,
                age_str_tgt=meta_tgt["age_str"], pma_weeks_tgt=meta_tgt["pma_weeks"],
                bidirectional=True, rev_pert_str=rev_pert))
    return pairs


def dedup_reverse(pairs):
    """Same logic as SFTBuilder._dedup_reverse."""
    seen, out = set(), []
    for ex in pairs:
        if not ex["pair_type"].endswith("_rev"):
            out.append(ex)
            continue
        key = (ex["prompt"], ex["completion"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def main():
    reg = pd.read_csv(S1.REGISTRY_PATH)
    reg.index = reg["barcode"]

    rat_bcs = reg[reg["source"] == "rat"]["barcode"].tolist()
    atl_bcs = reg[reg["source"] == "atlas"]["barcode"].tolist()
    bpd_bcs = reg[reg["source"] == "bpd"]["barcode"].tolist()
    print(f"[load] registry: {len(rat_bcs)} rat / {len(atl_bcs)} atlas / {len(bpd_bcs)} bpd barcodes")

    print("[load] building cell sentences (rat + atlas + bpd)...")
    rat_s = S1.load_rat_sentences(rat_bcs)
    atl_s = S1.load_atlas_sentences(atl_bcs)
    bpd_s = S1.load_bpd_sentences(bpd_bcs)
    print(f"[load] sentences: {len(rat_s)} rat / {len(atl_s)} atlas / {len(bpd_s)} bpd")

    print("[replay] rat block then human block, run()'s order...")
    raw = rat_pairs_with_barcodes("test", reg, rat_s) + human_pairs_with_barcodes("test", reg, atl_s, bpd_s)
    deduped = dedup_reverse(raw)
    print(f"[replay] {len(raw)} raw -> {len(deduped)} after dedup_reverse")

    fwd_human = [(i, ex) for i, ex in enumerate(deduped)
                 if ex["pair_type"].startswith("human_") and not ex["pair_type"].endswith("_rev")]
    print(f"[filter] {len(fwd_human)} fwd human entries")

    print("[verify] matching against 'gt' of test_inference_results_t10.jsonl...")
    gt_by_idx = {}
    with open(TEST_INFERENCE_PATH) as f:
        for line in f:
            r = json.loads(line)
            gt_by_idx[r["idx"]] = r["gt"].strip().removesuffix("<|endoftext|>").strip()

    rows, n_ok, n_bad, n_missing = [], 0, 0, 0
    for idx, ex in fwd_human:
        real_gt = gt_by_idx.get(idx)
        if real_gt is None:
            n_missing += 1
            continue
        verified = real_gt == ex["completion"].strip()
        n_ok += verified
        n_bad += not verified
        rows.append({"idx": idx, "pair_type": ex["pair_type"],
                     "cell_type": ex["prompt"].split("Cell type:")[1].split("\n")[0].strip(),
                     "src_barcode": ex["src_barcode"], "tgt_barcode": ex["tgt_barcode"],
                     "verified": verified})

    print(f"[verify] match={n_ok}  mismatch={n_bad}  missing_idx={n_missing}  (of {len(fwd_human)})")
    df = pd.DataFrame(rows)
    if not df.empty:
        for pt, sub in df.groupby("pair_type"):
            print(f"  {pt}: {int(sub['verified'].sum())}/{len(sub)} verified")
    out_path = HERE / "recovered_human_test_pairs.csv"
    df.to_csv(out_path, index=False)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
