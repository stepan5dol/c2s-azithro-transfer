# Generation

Inference with vLLM 0.30.0 on one NVIDIA A100-SXM4-80GB (`env_hardware_software.txt`), model `sft_pythia1b_short_full/checkpoint-264`, 2026-10-05.

| # | input | runner | temperature | output |
|---|-------|--------|---|--------|
| 1 | `test_segments.jsonl` | `run_test.py` | 1.0 | `test_inference_results_t10.jsonl` |
| 2 | `test_segments.jsonl` | `run_test.py` | 0.8 | `test_inference_results_t08.jsonl` |
| 3 | `test_segments.jsonl` | `run_test.py` | 1.2 | `test_inference_results_t12.jsonl` |
| 4 | `train/pipeline_short/inference_azi.jsonl` | `run_azi.py` | 1.0 | `inference_azi_results.jsonl` |
| 5 | `train/pipeline_short/inference_azi_concise.jsonl` | `run_azi.py` | 1.0 | `inference_azi_concise_results.jsonl` |
| 6 | `train/pipeline_short/inference_azi_ratmatch.jsonl` | `run_azi.py` | 1.0 | `inference_azi_ratmatch_results.jsonl` |

The temperature is varied only on the test set, the only set with reference cells.

Generation settings: max_model_len 8192, max_tokens 4096 for forward and 256 for reverse prompts, bf16, gpu_memory_utilization 0.9, stop on EOS; `run_azi.py` uses max_num_seqs 8.

`run_test.py` reads `MODEL_DIR`, `TEST_PATH`, `OUT_PATH` and `TEMPERATURE` from the environment. `run_azi.py` reads `MODEL_DIR` and `TEMPERATURE` from the environment and the input and output paths from its arguments.

No seed is passed, so vLLM's default engine seed 0 is used. Identical outputs require the same vLLM version, GPU type and batching parameters.
