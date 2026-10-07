"""
Shared primitives for the comparisons of measured counts with model
predictions. Measured cells are truncated to their top 800 genes, the length
of a cell sentence; predictions are reconstructed with rank_expr_model.

Model predictions (temperature 1.0):
  train/results/test_inference_results_t10.jsonl  rat and human test set
  train/results/inference_azi_results.jsonl       human azithromycin prompts
"""
from __future__ import annotations

import numpy as np

K = 800


def truncate_topk(X: np.ndarray, k: int = K) -> np.ndarray:
    """Keep each row's top-k values and set the rest to zero (K=800, the length of a cell sentence)."""
    Xt = np.zeros_like(X)
    if X.shape[1] <= k:
        return X.copy()
    part = np.argpartition(-X, k, axis=1)[:, :k]
    rows = np.arange(X.shape[0])[:, None]
    Xt[rows, part] = X[rows, part]
    return Xt


def expression_matched_null(anchor_vec: np.ndarray, gene_idx: np.ndarray, indicator: np.ndarray,
                             n_bins: int = 20, n_perm: int = 1000, seed: int = 0):
    """Expression-matched permutation null: genes are binned by anchor_vec
    (e.g. mean expression in the reference) into n_bins quantile bins; panel-sized
    gene sets are drawn from the same bins n_perm times, and the indicator rate
    of the panel is compared with these draws.

    Returns (observed_rate, null_median, fold, empirical_p).
    """
    rng = np.random.default_rng(seed)
    bin_edges = np.quantile(anchor_vec, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6
    gene_bins = np.digitize(anchor_vec, bin_edges[1:-1])
    bin_to_genes = {b: np.where(gene_bins == b)[0] for b in range(n_bins)}

    observed_bins = gene_bins[gene_idx]
    observed_rate = float(indicator[gene_idx].mean())
    null_rates = np.empty(n_perm)
    for p in range(n_perm):
        sampled = np.array([rng.choice(bin_to_genes[b]) for b in observed_bins])
        null_rates[p] = indicator[sampled].mean()
    null_median = float(np.median(null_rates))
    fold = observed_rate / null_median if null_median > 0 else float("nan")
    empirical_p = float((null_rates >= observed_rate).mean())
    return observed_rate, null_median, fold, empirical_p
