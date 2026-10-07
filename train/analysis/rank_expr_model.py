"""
Rank -> expression reconstruction for cell sentences.

Cell sentences carry gene order only. As in C2S-Scale (Rizvi et al., 2025,
Supp. Fig. 8), expression is modelled as linear in log-rank:

    log1p_cpm10k(gene) ~= intercept + slope * log(rank + 1)

On rat.ho.azi.integrated.h5ad (counts layer, K=800, 2,000 cells, seed 0) the
fit gives R^2 = 0.836 for log-rank and 0.639 for raw rank.

The model is fitted once on measured counts and applied identically to
measured and predicted cell sentences. It is a single global curve, so r2 is
reported with every metric built on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import anndata


class RankExprModel(NamedTuple):
    slope: float
    intercept: float
    r2: float
    gene_to_idx: dict[str, int]
    gene_names: list[str]

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def _fit_from_matrix(X: np.ndarray, gene_names: list[str], k: int, n_cells: int, seed: int) -> RankExprModel:
    """Raw counts [n_cells, n_genes] -> RankExprModel; shared by fit() and fit_from_csv()."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_cells, len(X)), replace=False)
    X = X[idx]
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X_norm = np.log1p(X / lib * 1e4)

    log_ranks_parts, exprs_parts = [], []
    for row in X_norm:
        order = np.argsort(-row)[:k]
        e = row[order]
        mask = e > 0
        r = np.arange(k)[mask]
        log_ranks_parts.append(np.log(r + 1.0))
        exprs_parts.append(e[mask])
    log_ranks = np.concatenate(log_ranks_parts)
    exprs = np.concatenate(exprs_parts)

    A = np.column_stack([np.ones_like(log_ranks), log_ranks])
    (intercept, slope), *_ = np.linalg.lstsq(A, exprs, rcond=None)
    y_pred = intercept + slope * log_ranks
    ss_res = float(((exprs - y_pred) ** 2).sum())
    ss_tot = float(((exprs - exprs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    return RankExprModel(float(slope), float(intercept), float(r2), gene_to_idx, gene_names)


def fit(
    h5ad_path: str | Path,
    k: int = 800,
    n_cells: int = 2000,
    seed: int = 0,
    layer: str = "counts",
) -> RankExprModel:
    """Fit log-rank -> log1p(CPM10k) linear model on real counts (h5ad source)."""
    adata = anndata.read_h5ad(h5ad_path)
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    gene_names = list(adata.var_names)
    return _fit_from_matrix(X, gene_names, k, n_cells, seed)


def fit_from_csv(
    counts_csv_path: str | Path,
    k: int = 800,
    n_cells: int = 2000,
    seed: int = 0,
    id_col: str = "id",
) -> RankExprModel:
    """Fit log-rank -> log1p(CPM10k) on raw counts in a wide cells x genes CSV
    (e.g. BPD-PH/GSE275938_compiled_counts.csv). Reads a random subsample of
    n_cells rows without loading the whole file.
    """
    import pandas as pd

    with open(counts_csv_path) as f:
        total_rows = sum(1 for _ in f) - 1

    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(total_rows, size=min(n_cells, total_rows), replace=False).tolist())
    skip = lambda i: i != 0 and (i - 1) not in chosen

    df = pd.read_csv(counts_csv_path, skiprows=skip, index_col=id_col)
    X = df.to_numpy(dtype=np.float64)
    gene_names = df.columns.tolist()
    return _fit_from_matrix(X, gene_names, k, n_cells, seed)


def reconstruct(genes: list[str], model: RankExprModel, k: int = 800,
                 floor_rank: int | None = None) -> np.ndarray:
    """Rank-ordered gene list -> dense [n_genes] log1p(CPM10k) vector.

    Genes absent from the list are 0 or, with floor_rank, the model value at
    that rank.
    """
    vec = np.zeros(model.n_genes, dtype=np.float32)
    if floor_rank is not None:
        floor_val = model.intercept + model.slope * np.log(floor_rank + 1.0)
        if floor_val > 0:
            vec[:] = floor_val
    for rank, g in enumerate(genes[:k]):
        j = model.gene_to_idx.get(g)
        if j is None:
            continue
        val = model.intercept + model.slope * np.log(rank + 1.0)
        if val > 0:
            vec[j] = val
    return vec


def reconstruct_batch(gene_lists: list[list[str]], model: RankExprModel, k: int = 800,
                       floor_rank: int | None = None) -> np.ndarray:
    """[n_cells] gene-name lists -> [n_cells, n_genes] dense reconstructed matrix."""
    out = np.zeros((len(gene_lists), model.n_genes), dtype=np.float32)
    for i, genes in enumerate(gene_lists):
        out[i] = reconstruct(genes, model, k, floor_rank=floor_rank)
    return out


if __name__ == "__main__":
    H5AD = str(Path(__file__).resolve().parents[2] / "rat.ho.azi.integrated.h5ad")
    m = fit(H5AD)
    print(f"slope={m.slope:.5f} intercept={m.intercept:.4f} r2={m.r2:.4f} n_genes={m.n_genes}")
