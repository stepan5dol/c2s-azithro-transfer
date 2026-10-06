#!/usr/bin/env python3
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

MODEL_DIR = os.environ["MODEL_DIR"]
TEST_PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/inference_azi.jsonl"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "/workspace/inference_azi_results.jsonl"

MAX_NEW_TOKENS_FWD = 4096
MAX_NEW_TOKENS_REV = 256
TEMPERATURE = float(os.environ["TEMPERATURE"])
MAX_MODEL_LEN = 8192

def run_group(group, max_new, out_f, start, total_done_offset, total, llm, eos_id):
    if not group:
        return
    sp = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_new,
        stop_token_ids=[eos_id],
    )
    prompts = [e["prompt"] for e in group]
    outputs = llm.generate(prompts, sp)

    for e, out in zip(group, outputs):
        pred = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        rec = {
            "idx": e["idx"], "direction": e["direction"], "prompt": e["prompt"],
            "gt": e["gt"], "pred": pred, "finish_reason": finish_reason,
            "gen_tokens": len(out.outputs[0].token_ids),
        }
        out_f.write(json.dumps(rec) + "\n")
    out_f.flush()

    elapsed = time.time() - start
    done = total_done_offset + len(group)
    print(f"[{done}/{total}] elapsed={elapsed/60:.1f}min", flush=True)


def main():
    print("loading examples...", flush=True)
    examples = []
    with open(TEST_PATH) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            prompt = obj["prompt"]
            gt = obj.get("gt", "")
            direction = "fwd" if prompt.startswith("Determine the single cell's expression changes") else "rev"
            examples.append({"idx": i, "prompt": prompt, "gt": gt, "direction": direction})

    rev_examples = [e for e in examples if e["direction"] == "rev"]
    fwd_examples = [e for e in examples if e["direction"] == "fwd"]
    print(f"loaded {len(examples)} test examples ({len(fwd_examples)} fwd / {len(rev_examples)} rev)", flush=True)

    print("loading model...", flush=True)
    llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9, max_num_seqs=8)
    tokenizer = llm.get_tokenizer()
    eos_id = tokenizer.eos_token_id

    start = time.time()
    with open(OUT_PATH, "w") as out_f:
        print(f"--- rev pass: {len(rev_examples)} examples ---", flush=True)
        run_group(rev_examples, MAX_NEW_TOKENS_REV, out_f, start, 0, len(examples), llm, eos_id)

        print(f"--- fwd pass: {len(fwd_examples)} examples ---", flush=True)
        run_group(fwd_examples, MAX_NEW_TOKENS_FWD, out_f, start, len(rev_examples), len(examples), llm, eos_id)

    print("done, wrote", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
