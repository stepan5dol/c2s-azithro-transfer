# c2s-azithro-transfer

Prediction of azithromycin treatment of BPD in neonatals using c2s-scale fine-tuning cross-species transfer

Code, configuration and logs for the fine-tuning of C2S-Scale-Pythia-1b-pt on paired single-cell profiles of lung endothelial cells and pericytes, and for the evaluation of the predictions.

Datasets: rat lung GSE300670 (room air, hyperoxia, hyperoxia + azithromycin), human fetal lung E-MTAB-11278 (He et al., 2022), human infant lung GSE275938 (Shirazi et al., 2025). Cell types: general capillary endothelial, aerocyte capillary endothelial, venous endothelial cells, pericytes. Model: vandijklab/C2S-Scale-Pythia-1b-pt (Rizvi et al., 2025), full fine-tuning with axolotl.

## Pipeline

| Step | Script or file | Output |
|---|---|---|
| Cell sentences, split, pairs | `train/pipeline_short/common.py`, `stage0_split.py`, `stage1_sft_bidir.py` | `cell_split_registry_short.csv`, `train/pipeline_short/data_audit.csv`, `split_summary.csv`, `sft_dataset_bidir/{train,valid,test}.jsonl` |
| Training segments | `train/pipeline_short/build_axolotl_segments.py` | `*_segments.jsonl` |
| Azithromycin prompts, three phrasings | `train/pipeline_short/stage2_azi_inference.py`, `stage2_azi_inference_concise.py`, `stage2_azi_inference_ratmatch.py` | `inference_azi*.jsonl` |
| Fine-tuning | axolotl with `training/axolotl_sft_pythia1b_A100_used.yaml` | checkpoint-264 (best validation loss 2.199) |
| Generation | `pod/run_test.py` and `pod/run_azi.py` (vLLM 0.30.0) | `test_inference_results_t10.jsonl` (test set, temperature 1.0; also t08 and t12), `inference_azi_results.jsonl` |

The header comment of the yaml predates the final dataset (13,886 / 60 / 892 examples); the run used 14,520 / 102 / 1,566.

## Analysis

`ANALYSIS` stands for `train/analysis` and `ortho` for `ANALYSIS/real_counts_suite_ortho`.

| Result | Scripts, in run order |
|---|---|
| Table 1 | `supplement_materials/05_split_rule/table1_cells_in_pairs.py` |
| Rank to expression reconstruction | `ANALYSIS/rank_expr_model.py`, loaders `ANALYSIS/common.py`, `common_human.py`, `pathway_module_analysis/{common,load_data,real_counts}.py` |
| Equivalence to real cells, rat and human | `ANALYSIS/a8_equivalence_beautiful_species.py`, `ANALYSIS/equivalence_human_beautiful.py`, `ortho/relabel_equivalence_titles.py` |
| Rescued-gene deltas, human | `ortho/human_panel_variants/`: `recover_human_test_barcodes.py`, `human_rescue_pipeline_genomewide.py`, `filter_to_rat_schema.py`, `panel_sets_human.py`, `build_disease_arms_trajectory.py` (with `human_updown_percell.py`) |
| Rescued-gene deltas, rat | `ortho/rat/recover_rat_test_barcodes.py`, `ortho/rat_panel_variants/build_all_panel_variants.py` (with `updown_core.py`, `panel_sets.py`) |
| cameraPR screen of the azithromycin prediction | `ortho/human_panel_variants/export_for_limma.py`, `limma_sets.py`, then `ortho/build_panelC_dotplot.py` |
| Main figure | `ortho/build_main_figure.py` |
| Figure typography | `ANALYSIS/plot_style.py` |

## Paths and inputs

The scripts build every path from their own location in the repository, so they run from a fresh clone and from any working directory without editing. The single-cell data, used both for training and for the analysis, are not part of this repository: download them from the archives below into the root of your local clone, under the names in the first column.

| Path in the repository | Dataset | Source |
|---|---|---|
| `rat.ho.azi.integrated.h5ad` | rat lung, room air, hyperoxia, hyperoxia + azithromycin | GEO [GSE300670](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE300670) |
| `he_lung_atlas.h5ad`, `2022FetalLungIntCounts.h5ad` | human fetal lung, E-MTAB-11278 (He et al., 2022) | [Human Cell Atlas](https://explore.data.humancellatlas.org/projects/2fe3c60b-ac1a-4c61-9b59-f6556c0fce63) |
| `BPD-PH/GSE275938_cell_metadata.csv`, `BPD-PH/GSE275938_compiled_counts.csv` | human infant lung, GSE275938 (Shirazi et al., 2025) | GEO [GSE275938](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275938), [article](https://doi.org/10.1038/s41467-025-60371-7) |

The archives use other file names: rename `Assembled10DomainsFiltered.h5ad` to `he_lung_atlas.h5ad`, and decompress `2022FetalLungIntCounts.h5ad.gz`, `GSE275938_cell_metadata.csv.gz` and `GSE275938_compiled_counts.csv.gz` (file list of the atlas: [fetal-lung.cellgeni.sanger.ac.uk](https://fetal-lung.cellgeni.sanger.ac.uk/scRNA.html)).

The rat rescued-gene tables read through `DGE_XLSX` in `ANALYSIS/common.py` go to the same place: `gcap rescue gene.xlsx`, `acap resuce gene.xlsx`, `pericyte_rescue gene.xlsx`, `Venous_rescue gene.xlsx`.

Model predictions are read from `train/results/`: `test_inference_results_t10.jsonl`, `test_inference_results_t08.jsonl`, `test_inference_results_t12.jsonl`, `inference_azi_results.jsonl`, the outputs of the generation step (`pod/README.md`).

Fine-tuning and generation ran on a GPU pod and keep its `/workspace` layout: the axolotl yaml lists the dataset and output paths under `/workspace/axolotl/mainwork/`, `pod/run_test.py` takes `MODEL_DIR`, `TEST_PATH` and `OUT_PATH` from the environment, and `pod/run_azi.py` takes the input and output paths as arguments. The logs keep the absolute paths of the machines where they were written.

Included: the split registry and its audit tables, `scripts/cell_types_config.json`, the gene lists read at import by `train/pipeline_short/common.py` (`train-after-grpo-analysis/rescued_*`), the training configuration and logs, and the environment lock files.

Not included:
- the single-cell data and the rat rescued-gene tables listed above;
- the SFT dataset (`train/pipeline_short/sft_dataset_bidir/`), rebuilt by `stage1_sft_bidir.py`;
- the model predictions in `train/results/`, produced by the generation step;
- Enrichr libraries, downloaded by `gseapy.get_library` at run time.

## Model weights

Hugging Face [`dolzhenkosv/c2s-pythia-1b-azithro-transfer`](https://huggingface.co/dolzhenkosv/c2s-pythia-1b-azithro-transfer), revision `27af97d2eae8e6be6de5272a269f67e378474a1c` (checkpoint-264, logs, predictions).

## Environment

See `environment/ENVIRONMENT.md` and the lock files in `environment/`.

## License

Code: MIT (`LICENSE`). Model weights: CC BY 4.0, the license of the base model vandijklab/C2S-Scale-Pythia-1b-pt; attribution to C2S-Scale and to EleutherAI Pythia (Apache-2.0) applies.
