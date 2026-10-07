"""
Human counterpart of common.py.

Cell records come from the same inference files as the rat analysis
(common.RUNS), filtered to Homo sapiens; calibration counts from
BPD-PH/GSE275938_compiled_counts.csv. Three transitions, each relative to the
baseline cell in the record's "Unexposed:" field:

  Acute   He22 fetal lung                 -> Acute26 acute preterm injury
  BPD     Acute26 acute preterm injury    -> BPD at 7 months
  BPD-PH  Acute26 acute preterm injury    -> BPD with pulmonary hypertension at 7 months
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(BASE))
from pipeline.common import BPD_COUNTS_PATH

from common import (
    K, MIN_N, SEED, RUNS,
    dedupe_first, reconstruct_group, filtered_gene_domain, pairwise_dist, cdist,
    fit_pca_pooled, fit_pca_shared, centroid_distance, centroid_perm_test,
    centroid_distance_bootstrap_ci, delta_and_mask, per_cell_rescue_auc,
    forest_plot, knn_mixing_lisi, unpaired_perm_test, bootstrap_ci_two_sample,
    bootstrap_ci, auc_mannwhitney, swap_perm_test, permdisp,
)

COUNTS_CSV = BPD_COUNTS_PATH

RESCUED_HUMAN_PATH = BASE / "train-after-grpo-analysis" / "rescued_human_genes.txt"


def load_rescue_genes() -> set[str]:
    return set(RESCUED_HUMAN_PATH.read_text().splitlines()) - {""}

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures_human"
FIG_DIR.mkdir(exist_ok=True)

CT_ORDER = ["gCap", "aCap", "Pericyte", "VEC"]
CT_MAP = {
    "general capillary endothelial cell":  "gCap",
    "aerocyte capillary endothelial cell": "aCap",
    "pericyte":                            "Pericyte",
    "pulmonary venous endothelial cell":   "VEC",
}
COND_ORDER = ["Acute", "BPD", "BPD-PH"]

_PROMPT_RE = {
    "species":      re.compile(r"Species:\s*([^\n]+)"),
    "cell_type":    re.compile(r"Cell type:\s*([^\(\n]+)"),
    "perturbation": re.compile(r"Perturbation:\s*([^\n]+)"),
    "unexposed":    re.compile(r"\nUnexposed:\s*([^\n]+)"),
}


def condition_of(perturbation: str) -> str | None:
    if perturbation.startswith("Acute preterm lung injury"):
        return "Acute"
    if "pulmonary hypertension" in perturbation:
        return "BPD-PH"
    if perturbation.startswith("Progression from acute preterm lung injury"):
        return "BPD"
    if perturbation.startswith("Disease progression to BPD with pulmonary hypertension"):
        return "BPD-PH"
    return None


def he22_reference_by_ct(groups: dict[tuple[str, str], list[dict]], model) -> dict[str, "np.ndarray"]:
    """Mean reconstructed He22 expression per cell type.

    Taken from the "Unexposed:" field of the Acute records, the only condition
    whose baseline is He22; used as a common healthy reference for all
    conditions.
    """
    import numpy as np
    import rank_expr_model as rem

    out = {}
    for ct in CT_ORDER:
        acute_group = groups.get((ct, "Acute"), [])
        if not acute_group:
            continue
        he22_X = rem.reconstruct_batch([c["unexposed"] for c in acute_group], model, K)
        out[ct] = he22_X.mean(axis=0)
    return out


AZI_RESULTS_PATH = BASE / "train/results/inference_azi_results.jsonl"
_AGE_RE = re.compile(r"Age:\s*([^\n]+)")


def condition_of_age(age: str) -> str | None:
    """Condition of a record in AZI_RESULTS_PATH from its Age: line; all of
    these records share the same Perturbation: line.
    """
    if "acute preterm injury" in age:
        return "Acute"
    if "BPD with pulmonary hypertension" in age:
        return "BPD-PH"
    if "(BPD)" in age:
        return "BPD"
    return None


def load_azi_counterfactual(jsonl_path: Path = AZI_RESULTS_PATH) -> dict[tuple[str, str], list[dict]]:
    """Human azithromycin predictions: the measured baseline cell
    ("Unexposed:") and the model's predicted cell sentence ("pred"). There is no
    ground truth: these patients did not receive azithromycin.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("direction") != "fwd":
                continue
            prompt = r["prompt"]
            species = _PROMPT_RE["species"].search(prompt).group(1).strip()
            if not species.startswith("Homo"):
                continue
            ct_raw = _PROMPT_RE["cell_type"].search(prompt).group(1).strip()
            ct = CT_MAP.get(ct_raw)
            if ct is None:
                continue
            age = _AGE_RE.search(prompt).group(1).strip()
            cond = condition_of_age(age)
            if cond is None:
                continue
            unexposed = _PROMPT_RE["unexposed"].search(prompt).group(1).strip().split()[:K]
            pred = dedupe_first(r["pred"].strip().split())[:K]
            groups[(ct, cond)].append({"unexposed": unexposed, "pred": pred})
    return groups


def load_cells(jsonl_path: Path) -> dict[tuple[str, str], list[dict]]:
    """Same shape/contract as common.load_cells, filtered to Homo sapiens."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("direction") != "fwd":
                continue
            prompt = r["prompt"]
            species = _PROMPT_RE["species"].search(prompt).group(1).strip()
            if not species.startswith("Homo"):
                continue
            ct_raw = _PROMPT_RE["cell_type"].search(prompt).group(1).strip()
            ct = CT_MAP.get(ct_raw)
            if ct is None:
                continue
            pert = _PROMPT_RE["perturbation"].search(prompt).group(1).strip()
            cond = condition_of(pert)
            if cond is None:
                continue
            unexposed = _PROMPT_RE["unexposed"].search(prompt).group(1).strip().split()[:K]
            gt = r["gt"].strip().split()[:K]
            pred = dedupe_first(r["pred"].strip().split())[:K]
            groups[(ct, cond)].append({"unexposed": unexposed, "gt": gt, "pred": pred})
    return groups
