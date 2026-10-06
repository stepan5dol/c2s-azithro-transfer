# azithropythia

Prediction of azithromycin treatment of BPD in neonatals using c2s-scale fine-tuning cross-species transfer

Code, configuration and logs for the fine-tuning of C2S-Scale-Pythia-1b-pt on paired single-cell profiles of lung endothelial cells and pericytes, and for the evaluation of the predictions.

Datasets: rat lung AA000 (room air, hyperoxia, hyperoxia + azithromycin), human fetal lung E-MTAB-11278 (He et al., 2022), human infant lung GSE275938 (Shirazi et al., 2025). Cell types: general capillary endothelial, aerocyte capillary endothelial, venous endothelial cells, pericytes. Model: vandijklab/C2S-Scale-Pythia-1b-pt (Rizvi et al., 2025), full fine-tuning with axolotl.

## Pipeline

| Step | Script or file | Output |
|---|---|---|
| Cell sentences, split, pairs | `retrain_v2/pipeline_short/common.py`, `stage0_split.py`, `stage1_sft_bidir.py` | `cell_split_registry_short.csv`, `retrain_v2/pipeline_short/data_audit.csv`, `split_summary.csv`, `sft_dataset_bidir/{train,valid,test}.jsonl` |
| Training segments | `retrain_v2/pipeline_short/build_axolotl_segments.py` | `*_segments.jsonl` |
| Azithromycin prompts, three phrasings | `retrain_v2/pipeline_short/stage2_azi_inference.py`, `stage2_azi_inference_concise.py`, `stage2_azi_inference_ratmatch.py` | `inference_azi*.jsonl` |
| Fine-tuning | axolotl with `training/axolotl_sft_pythia1b_A100_used.yaml` | checkpoint-264 (best validation loss 2.199) |
| Generation | `pod/run_test.py` and `pod/run_azi.py` (vLLM 0.30.0) | `test_inference_results_t10.jsonl` (test set, temperature 1.0; also t08 and t12), `inference_azi_results.jsonl` |

The header comment of the yaml predates the final dataset (13,886 / 60 / 892 examples); the run used 14,520 / 102 / 1,566.

## Analysis

`ANALYSIS` stands for `retrain_v2/analysis` and `ortho` for `ANALYSIS/real_counts_suite_ortho`.

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

The scripts contain absolute paths of the machine where they ran: the project root `/Users/stepandolzhenko/Documents/AzithroGemma` (the root of this repository) and `/Users/stepandolzhenko/Downloads` (generated predictions). To run them elsewhere, replace the prefixes, for example on macOS

    grep -rlI --exclude-dir=.git "/Users/stepandolzhenko/Documents/AzithroGemma" . | xargs sed -i '' "s#/Users/stepandolzhenko/Documents/AzithroGemma#$PWD#g"

(on Linux, `sed -i` without the empty string), and the same for `/Users/stepandolzhenko/Downloads`.

Included: the split registry and its audit tables, `scripts/cell_types_config.json`, the gene lists read at import by `retrain_v2/pipeline_short/common.py` (`train-after-grpo-analysis/rescued_*`), the training configuration and logs, and the environment lock files.

Not included:
- single-cell data: `rat.ho.azi.integrated.h5ad` (AA000), `he_lung_atlas.h5ad` and `2022FetalLungIntCounts.h5ad` (E-MTAB-11278), `BPD-PH/GSE275938_cell_metadata.csv` and `GSE275938_compiled_counts.csv` (GSE275938);
- the SFT dataset (`retrain_v2/pipeline_short/sft_dataset_bidir/`), rebuilt by `stage1_sft_bidir.py`;
- model predictions (`test_inference_results3.jsonl`, `test_inference_results_t08.jsonl`, `inference_azi_results.jsonl`), produced by the generation step;
- the rat rescued-gene tables read through `DGE_XLSX` in `ANALYSIS/common.py`;
- Enrichr libraries, downloaded by `gseapy.get_library` at run time.

## Model weights

Hugging Face `12mrch2023/sft-pythia1b-run4`, revision `27af97d2eae8e6be6de5272a269f67e378474a1c` (checkpoint-264, logs, predictions).

## Environment

See `environment/ENVIRONMENT.md` and the lock files in `environment/`.

## License

Code: MIT (`LICENSE`). Model weights: CC BY 4.0, the license of the base model vandijklab/C2S-Scale-Pythia-1b-pt; attribution to C2S-Scale and to EleutherAI Pythia (Apache-2.0) applies.
