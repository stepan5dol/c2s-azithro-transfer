# Software and hardware environment

Lock files in this folder: `conda_base_environment_2026-10-01.yml`, `pip_freeze_base_2026-10-01.txt`, `R_sessionInfo_2026-10-01.txt`.

## Dataset assembly, analysis, figures (local)

| item | value |
|---|---|
| hardware | Apple M4 Max (arm64), 36 GB RAM |
| operating system | macOS 26.6 |
| Python | 3.13.12 (conda-forge, Miniforge base environment) |
| R | 4.5.2, limma 3.66.0 |
| numpy / pandas / scipy | 2.4.4 / 2.2.3 / 1.17.1 |
| anndata / scanpy / h5py | 0.12.10 / 1.12 / 3.15.1 |
| scikit-learn / statsmodels | 1.8.0 / 0.14.6 |
| matplotlib / matplotlib-venn / pillow | 3.10.8 / 1.1.2 / 12.2.0 |
| gseapy / rpy2 | 1.3.0 / 3.6.7 |

The versions are those installed on 2026-10-01, when the scripts of `pipeline_short/` and of the analysis folder ran unchanged in this environment.

## Training

| item | value |
|---|---|
| GPU | one NVIDIA A100 SXM4, 80 GB (vast.ai rental; 32 vCPU AMD EPYC 7513, 129 GB RAM) |
| framework | axolotl (Docker image `axolotlai/axolotl-cloud`, vast.ai template), `input_output` segments dataset, full fine-tuning without adapter |
| transformers / torch | 5.12.1 / 2.12.0 |
| attention | flash-attention 2 through `kernels-community/flash-attn2` |
| precision | bf16, tf32 |
| run | started 2026-07-14 04:56 UTC; stopped manually at step 367 of 828; best checkpoint 264 (validation loss 2.199) |

## Inference (vLLM)

| item | value |
|---|---|
| test set | vLLM 0.25.0 (V1 engine), bf16, max_model_len 8192, one sample per prompt, temperature 1.0 (log of the 0.8 run in `pod/logs/`), same pod and GPU as the training |
| Python / torch | 3.12.13 / 2.11.0+cu130 |
| azithromycin prompts | same script on a vast.ai pod from the `vastai/vllm` template; the vLLM version of this run is not logged separately |
