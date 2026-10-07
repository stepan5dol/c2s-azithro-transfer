"""
Measured counts of the human disease, He22 and Term cells on the gene axis of
a rank_expr_model.RankExprModel.

Measured cells are used directly; only model predictions are reconstructed
with rank_expr_model. Values are log1p(CPM10k), the units of
rank_expr_model._fit_from_matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
_saved_common = sys.modules.pop("common", None)
import load_data as L
if _saved_common is not None:
    sys.modules["common"] = _saved_common

CT_REAL_NAME = {"VEC": "Pulmonary venous EC"}


def _normalize_log1p_cpm10k(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * 1e4)


def _reindex_to_model_genes(X_norm: np.ndarray, genes: list[str], model_gene_names: list[str]) -> np.ndarray:
    col_of = {g: i for i, g in enumerate(genes)}
    out = np.zeros((X_norm.shape[0], len(model_gene_names)), dtype=np.float32)
    for out_j, g in enumerate(model_gene_names):
        src_j = col_of.get(g)
        if src_j is not None:
            out[:, out_j] = X_norm[:, src_j]
    return out


CT_MODEL_ORDER = ["gCap", "aCap", "Pericyte", "VEC"]


def real_disease_matrices(condition: str, model) -> dict[str, np.ndarray]:
    """condition: Acute26 / BPD7mo / BPDPH7mo -> {cell_type: matrix}, from one
    chunked read of the counts CSV.
    """
    X, genes, ct, cond = L.load_human_bpd_raw([condition])
    Xn = _normalize_log1p_cpm10k(X)
    out = {}
    for ct_model in CT_MODEL_ORDER:
        ct_real = CT_REAL_NAME.get(ct_model, ct_model)
        mask = (ct == ct_real) & (cond == condition)
        if mask.sum() == 0:
            continue
        out[ct_model] = _reindex_to_model_genes(Xn[mask], genes, model.gene_names)
    return out


def real_he22_matrices(model) -> dict[str, np.ndarray]:
    X, genes, ct = L.load_human_he22_raw()
    Xn = _normalize_log1p_cpm10k(X)
    out = {}
    for ct_model in CT_MODEL_ORDER:
        ct_real = CT_REAL_NAME.get(ct_model, ct_model)
        mask = ct == ct_real
        if mask.sum() == 0:
            continue
        out[ct_model] = _reindex_to_model_genes(Xn[mask], genes, model.gene_names)
    return out


def real_term_matrices(model, pooled: bool = True) -> dict:
    """pooled=True: {cell_type: X}, Term0d and Term20d combined.
    pooled=False: {(term_label, cell_type): X}.
    """
    X, genes, ct, cond = L.load_human_bpd_raw(["Term0d", "Term20d"])
    Xn = _normalize_log1p_cpm10k(X)
    out = {}
    if pooled:
        for ct_model in CT_MODEL_ORDER:
            ct_real = CT_REAL_NAME.get(ct_model, ct_model)
            mask = ct == ct_real
            if mask.sum() == 0:
                continue
            out[ct_model] = _reindex_to_model_genes(Xn[mask], genes, model.gene_names)
    else:
        for term_label in ["Term0d", "Term20d"]:
            for ct_model in CT_MODEL_ORDER:
                ct_real = CT_REAL_NAME.get(ct_model, ct_model)
                mask = (ct == ct_real) & (cond == term_label)
                if mask.sum() == 0:
                    continue
                out[(term_label, ct_model)] = _reindex_to_model_genes(Xn[mask], genes, model.gene_names)
    return out
