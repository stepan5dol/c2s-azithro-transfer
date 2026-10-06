# run4 inference — checkpoint-264 (2026-10-05)

Model: `sft_pythia1b_short_full/checkpoint-264` from this repo (same checkpoint as run3).
Machine: 1x A100-SXM4-80GB, vLLM 0.30.0, torch 2.13.0+cu130 (full details in `env_snapshot/`).

## What was run (sequentially, one model load per run)
| # | run | input | runner | T | output |
|---|-----|-------|--------|---|--------|
| 1 | test | dataset/test_segments.jsonl | run_test.py | 1.0 | results/test_inference_results_t10.jsonl |
| 2 | test | dataset/test_segments.jsonl | run_test.py | 0.8 | results/test_inference_results_t08.jsonl |
| 3 | test | dataset/test_segments.jsonl | run_test.py | 1.2 | results/test_inference_results_t12.jsonl |
| 4 | azi | lastrunpls retrain_v2/inference_azi.jsonl | run_azi.py | 1.0 | results/inference_azi_results.jsonl |
| 5 | azi concise | lastrunpls retrain_v2/inference_azi_concise.jsonl | run_azi.py | 1.0 | results/inference_azi_concise_results.jsonl |
| 6 | azi ratmatch | lastrunpls retrain_v2/inference_azi_ratmatch.jsonl | run_azi.py | 1.0 | results/inference_azi_ratmatch_results.jsonl |

Temperature is varied only on test (the only set with references); azi is run once at T=1.0, as in run3.

## Runners
`run_test.py` / `run_azi.py` are run3's `run_test_inference_vllm.py` / `run_test_inference_vllm-2-2-2.py`
with only path/temperature plumbing changed (see `scripts/*.patch.diff`). Generation settings are unchanged:
max_model_len 8192, max_tokens fwd 4096 / rev 256, bf16, gpu_memory_utilization 0.9, stop on EOS;
azi runner keeps max_num_seqs=8.

## Control run (after the main queue)
run3 `checkpoint-264` + run3 `dataset/test_segments.jsonl`, T=0.8, on this machine's vLLM 0.30
(`scripts/run_control.sh`) → `results_control/run3model_test_t08_vllm030.jsonl`, log `logs/control_run3model_t08.log`.
fwd truncated by length: 93/783 (July, vLLM 0.25: 101/783; run4 model T=0.8: 51/783).

## Seed
The runners do not pass a seed explicitly, so vLLM's default engine seed is used: `seed=0`
(visible in every run log as `seed=0`; same in run3's July logs). SamplingParams.seed is unset (None),
so all requests draw from the engine RNG seeded with 0.
Bit-identical outputs require the same vLLM version, GPU type and batching parameters (see `env_snapshot/`).
For future runs set it explicitly: `LLM(..., seed=0)`.

## Layout
- `scripts/` — runners, `snapshot.sh`, orchestrators (`run_all.sh` launched run 1, `run_rest.sh` runs 2–6, `run_control.sh` control run), TG monitor, GPU logger, uploader
- `env_snapshot/` — hardware/OS/driver/CUDA, `nvidia-smi -q`, pip freeze (full + venv-only + with locations), dpkg list, redacted env vars, sha256 of inputs/model/scripts + HF revisions
- `logs/` — `timeline.log` (timestamped start/end/exit code of each run, status every 2 min, every Telegram message), per-run vLLM logs, `gpu.log` (every 60 s)
- `results/` — outputs of the 6 runs
- `results_control/` — control run output
- `analysis/` — `checks.py` (every sanity check done during the session: integrity, input/model hashes, EOS in data,
  run3 vs run4 train diff, predicted conditions vs train, truncation counts, training curves, seed) + `checks_output.txt`
