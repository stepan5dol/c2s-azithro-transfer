#!/usr/bin/env python3
"""
Converts sft_dataset_bidir/{train,valid,test}.jsonl ({"prompt", "completion"})
into the axolotl input_output format:

    {"segments": [{"label": false, "text": prompt},
                  {"label": true,  "text": " " + completion + "<|endoftext|>"}]}

Segments with label=false are masked from the loss. The leading space before
the completion matches the tokenization at inference.
"""
import json
from pathlib import Path

SFT_DIR = Path(__file__).resolve().parent / "sft_dataset_bidir"

for split in ["train", "valid", "test"]:
    src = SFT_DIR / f"{split}.jsonl"
    dst = SFT_DIR / f"{split}_segments.jsonl"
    n = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            d = json.loads(line)
            row = {
                "segments": [
                    {"label": False, "text": d["prompt"]},
                    {"label": True, "text": " " + d["completion"] + "<|endoftext|>"},
                ]
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{split}: {n:,} rows -> {dst}")
