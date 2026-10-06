#!/usr/bin/env python3
"""
Convert pipeline_short/sft_dataset_bidir/{train,valid,test}.jsonl
({"prompt":..., "completion":...}) into axolotl's `input_output` dataset
format: {"segments": [{"label": false, "text": ...}, {"label": true, "text": ...}]}.

label=false -> masked (no loss, -100), label=true -> loss computed.
Mirrors the manual masking already verified correct in sft_train-3.py:
prompt segment (unmasked->masked) + " " + completion segment (masked->loss),
leading space kept so the byte-level BPE tokenizer fuses it onto the first
completion token the same way at train time as it will at inference time.
No EOS appended here -- axolotl's input_output loader appends eos_token
itself per sequence when tokenizing (confirm with a preprocess dry run
before trusting this on a full run).
"""
import json
from pathlib import Path

SFT_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/pipeline_short/sft_dataset_bidir")

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
