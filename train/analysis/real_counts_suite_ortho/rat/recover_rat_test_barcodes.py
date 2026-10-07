#!/usr/bin/env python3
"""
Recovers the measured rat barcodes behind each forward test example of both
transitions (RA -> HO, HO -> AZI).

stage1_sft_bidir.py samples source and target barcodes with a seeded RNG
(_pair_seed = MD5(split|pair_type|cell_type) + RANDOM_SEED) and writes only
prompt and completion. This script replays the same pairing keeping the
barcodes. The position in the replayed list is the idx of
test_inference_results_t10.jsonl, and each row is verified by exact match of
the completion with the "gt" field.

Output: recovered_rat_test_pairs.csv (idx, pair_type, cell_type,
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

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)

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
            pairs.append({
                "prompt": prompt, "completion": tgt_cs, "pair_type": pair_type,
                "src_barcode": src_bc, "tgt_barcode": tgt_bc,
            })
            if bidirectional:
                rprompt = S1.build_prompt_reverse(species, tgt_cs)
                rcompletion = S1.build_completion_reverse(
                    age_str_tgt, pma_weeks_tgt, cell_type, ct_broad, rev_pert_str)
                pairs.append({
                    "prompt": rprompt, "completion": rcompletion,
                    "pair_type": pair_type + "_rev",
                    "src_barcode": None, "tgt_barcode": tgt_bc,
                })
    return pairs


def rat_pairs_with_barcodes(split: str, sent: dict[str, str]) -> list[dict]:
    """SFTBuilder._rat_pairs with the barcode-keeping builder, in the same order."""
    reg = pd.read_csv(S1.REGISTRY_PATH)
    reg.index = reg["barcode"]
    reg_s = reg[(reg["source"] == "rat") & (reg["split"] == split)]
    pairs = []
    for sc, tc in [("RA", "HO"), ("HO", "AZI")]:
        pert = S1.PERTURBATION_STR[("rat", sc, tc)]
        meta_src = S1.COND_META[("rat", sc)]
        meta_tgt = S1.COND_META[("rat", tc)]
        src_r = reg_s[reg_s["condition"] == sc]
        tgt_r = reg_s[reg_s["condition"] == tc]
        shared = set(src_r["cell_type"]) & set(tgt_r["cell_type"])
        pt = f"rat_{sc}_{tc}"
        for ct in sorted(shared):
            src_bcs = src_r[src_r["cell_type"] == ct].index.tolist()
            tgt_bcs = tgt_r[tgt_r["cell_type"] == ct].index.tolist()
            if split != "train":
                aug = S1.EVAL_AUG
            else:
                aug = (S1.RAT_AUG_SMALL
                       if min(len(src_bcs), len(tgt_bcs)) < S1.RAT_AUG_THRESHOLD
                       else S1.RAT_AUG_LARGE)
            pairs.extend(_build_pairs_with_barcodes(
                src_bcs, tgt_bcs, sent, sent,
                S1.SPECIES_STR["rat"], meta_src["age_str"], meta_src["pma_weeks"],
                ct, pert, pt, split, n_per_ct=None, n_augment=aug,
                age_str_tgt=meta_tgt["age_str"], pma_weeks_tgt=meta_tgt["pma_weeks"],
                bidirectional=True,
            ))
    return pairs


def dedup_reverse(pairs: list[dict]) -> list[dict]:
    """Same logic as SFTBuilder._dedup_reverse."""
    seen = set()
    out = []
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
    rat_bcs = reg[reg["source"] == "rat"]["barcode"].tolist()
    print(f"[load] {len(rat_bcs)} rat barcodes in registry, loading real cell sentences...")
    rat_s = S1.load_rat_sentences(rat_bcs)
    print(f"[load] {len(rat_s)} rat sentences built")

    print("[replay] reproducing _rat_pairs('test', rat_s) with barcode capture...")
    raw_pairs = rat_pairs_with_barcodes("test", rat_s)
    deduped = dedup_reverse(raw_pairs)
    print(f"[replay] {len(raw_pairs)} raw -> {len(deduped)} after dedup_reverse "
          f"(index == idx in test_inference_results_t10.jsonl)")

    fwd_both = [(i, ex) for i, ex in enumerate(deduped)
                if ex["pair_type"] in ("rat_RA_HO", "rat_HO_AZI")]
    print(f"[filter] {len(fwd_both)} fwd rat_RA_HO + rat_HO_AZI entries recovered")

    print("[verify] loading gt by idx from test_inference_results_t10.jsonl...")
    gt_by_idx = {}
    with open(TEST_INFERENCE_PATH) as f:
        for line in f:
            r = json.loads(line)
            gt_by_idx[r["idx"]] = r["gt"].strip().removesuffix("<|endoftext|>").strip()

    n_ok, n_bad, n_missing = 0, 0, 0
    rows = []
    for idx, ex in fwd_both:
        real_gt_text = gt_by_idx.get(idx)
        if real_gt_text is None:
            n_missing += 1
            continue
        expected = ex["completion"].strip()
        verified = real_gt_text == expected
        if verified:
            n_ok += 1
        else:
            n_bad += 1
        rows.append({"idx": idx, "pair_type": ex["pair_type"],
                     "cell_type": ex["prompt"].split("Cell type:")[1].split("\n")[0].strip(),
                     "src_barcode": ex["src_barcode"], "tgt_barcode": ex["tgt_barcode"],
                     "verified": verified})

    print(f"[verify] match={n_ok}  mismatch={n_bad}  missing_idx={n_missing}  (of {len(fwd_both)})")
    for pt in ("rat_RA_HO", "rat_HO_AZI"):
        n = sum(1 for r in rows if r["pair_type"] == pt)
        n_v = sum(1 for r in rows if r["pair_type"] == pt and r["verified"])
        print(f"  {pt}: {n_v}/{n} verified")

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "recovered_rat_test_pairs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
