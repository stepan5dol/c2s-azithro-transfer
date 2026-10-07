"""
Shared paths, constants and helpers for the rat and human data loaders.
Ribosomal, mitochondrial and haemoglobin genes are matched by
STRIP_PATTERN_RAT / STRIP_PATTERN_HUMAN.
"""
from pathlib import Path
import re

import numpy as np

BASE = Path(__file__).resolve().parents[3]  # repository root
RAT_PATH = BASE / "rat.ho.azi.integrated.h5ad"
ATLAS_CLEAN_PATH = BASE / "he_lung_atlas.h5ad"
ATLAS_RAW_PATH = BASE / "2022FetalLungIntCounts.h5ad"
BPD_META_PATH = BASE / "BPD-PH/GSE275938_cell_metadata.csv"
BPD_COUNTS_PATH = BASE / "BPD-PH/GSE275938_compiled_counts.csv"

HERE = Path(__file__).parent
OUT_DIR = HERE / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

TARGET_CELL_TYPES = {"gCap", "aCap", "Pulmonary venous EC", "Pericyte"}
DISEASE_CONDITIONS = ["Acute26", "BPD7mo", "BPDPH7mo"]

DATASET_TO_CONDITION = {
    "Acute preterm injury 1": "Acute26",
    "BPD 1": "BPD7mo", "BPD 2": "BPD7mo",
    "BPD+PH 1": "BPDPH7mo", "BPD+PH 2": "BPDPH7mo",
    "Term infant 1": "Term0d", "Term infant 2": "Term20d",
}

CT_MATCH_RAT_TO_HUMAN = {"Peri": "Pericyte", "VEC": "Pulmonary venous EC", "aCAP": "aCap", "gCAP": "gCap"}

STRIP_PATTERN_RAT = re.compile(r"^(Rpl|Rps|Mrpl|Mrps|Mt-|Hba|Hbb)", re.IGNORECASE)
STRIP_PATTERN_HUMAN = re.compile(r"^(RPL|RPS|MRPL|MRPS|MT-|HBA|HBB)", re.IGNORECASE)


def harmonize_atlas_celltype(raw: str) -> str:
    m = {
        "Aerocyte": "aCap", "Early cap": "gCap", "Mid cap": "gCap", "Late cap": "gCap",
        "Venous endo": "Pulmonary venous EC",
    }
    return m.get(raw, raw)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """x - y, pooled-SD standardized. Positive = higher in x."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled_std) if pooled_std > 0 else 0.0


def downsample_counts(X: np.ndarray, target_depth: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial downsample each row (cell) to target_depth total counts.
    Rows already at or below target_depth are left unchanged."""
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        row = X[i]
        total = row.sum()
        if total <= target_depth:
            out[i] = row
            continue
        p = (row / total).astype(np.float64)
        p = p / p.sum()
        out[i] = rng.multinomial(target_depth, p)
    return out
