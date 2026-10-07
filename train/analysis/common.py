"""
Shared loaders and statistical primitives for the rat analyses on
reconstructed expression (rank_expr_model.py).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import rank_expr_model as rem

BASE = Path(__file__).resolve().parents[2]  # repository root
H5AD = str(BASE / "rat.ho.azi.integrated.h5ad")
RUNS = {
    "t1.0": BASE / "train/results/test_inference_results_t10.jsonl",
    "t0.8": BASE / "train/results/test_inference_results_t08.jsonl",
    "t1.2": BASE / "train/results/test_inference_results_t12.jsonl",
}
RESCUED_DOWN_JSON = str(BASE / "train-after-grpo-analysis/rescued_genes_down.json")

HERE = Path(__file__).parent
REP_DIR = HERE / "reports"
FIG_DIR = HERE / "figures"
REP_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

K = 800
MIN_N = 5
SEED = 0

CT_ORDER = ["gCap", "aCap", "Pericyte", "VEC"]
CT_MAP = {
    "general capillary endothelial cell":  "gCap",
    "aerocyte capillary endothelial cell": "aCap",
    "pericyte":                            "Pericyte",
    "pulmonary venous endothelial cell":   "VEC",
}
COND_ORDER = ["HO", "AZI"]

_PROMPT_RE = {
    "species":      re.compile(r"Species:\s*([^\n]+)"),
    "cell_type":    re.compile(r"Cell type:\s*([^\(\n]+)"),
    "perturbation": re.compile(r"Perturbation:\s*([^\n]+)"),
    "unexposed":    re.compile(r"\nUnexposed:\s*([^\n]+)"),
}


def condition_of(perturbation: str) -> str | None:
    if perturbation.startswith("Hyperoxia"):
        return "HO"
    if perturbation.startswith("Azithromycin"):
        return "AZI"
    return None


def dedupe_first(genes: list[str]) -> list[str]:
    seen, out = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def load_cells(jsonl_path: Path) -> dict[tuple[str, str], list[dict]]:
    """-> {(cell_type, condition): [{"unexposed": [...], "gt": [...], "pred": [...]}]}
    Rank-ordered top-K gene lists from the forward inference records: the
    baseline cell from the prompt's "Unexposed:" field, the measured and the
    predicted perturbed cell.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("direction") != "fwd":
                continue
            prompt = r["prompt"]
            species = _PROMPT_RE["species"].search(prompt).group(1).strip()
            if not species.startswith("Rattus"):
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


def rescued_genes(species: str = "rat") -> set[str]:
    return set(json.load(open(RESCUED_DOWN_JSON))[species])


DGE_XLSX = {
    "gCap":     BASE / "gcap rescue gene.xlsx",
    "aCap":     BASE / "acap resuce gene.xlsx",
    "Pericyte": BASE / "pericyte_rescue gene.xlsx",
    "VEC":      BASE / "Venous_rescue gene.xlsx",
}


def load_celltype_rescue_genes(max_p_ho: float = 0.05, min_p_azi: float = 0.05) -> dict[str, set[str]]:
    """-> {cell_type: {gene, ...}} from the per-cell-type rescue workbooks.

    Keeps genes with recovery_type == 'Rescued_HO_down', p_val_adj_HO < max_p_ho
    and p_val_adj_AZI >= min_p_azi (or missing).
    """
    import openpyxl
    out = {}
    for ct, path in DGE_XLSX.items():
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb["Sheet1"]
        rows = list(ws.iter_rows(values_only=True))
        header, data = rows[0], rows[1:]
        genes = {
            r[0] for r in data
            if r[6] == "Rescued_HO_down"
            and r[2] is not None and r[2] < max_p_ho
            and (r[4] is None or r[4] >= min_p_azi)
        }
        out[ct] = genes
        wb.close()
    return out


def reconstruct_group(group: list[dict], model: rem.RankExprModel):
    """cells -> (unexposed_X, gt_X, pred_X) each [n_cells, n_genes]."""
    unexp_X = rem.reconstruct_batch([c["unexposed"] for c in group], model, K)
    gt_X = rem.reconstruct_batch([c["gt"] for c in group], model, K)
    pred_X = rem.reconstruct_batch([c["pred"] for c in group], model, K)
    return unexp_X, gt_X, pred_X


def unpaired_perm_test(a: np.ndarray, b: np.ndarray, statfn=None, n_perm: int = 5000, seed: int = SEED):
    """Two-sided permutation test on statfn(a) - statfn(b) (default: mean).
    Returns (T_obs, p_value)."""
    if statfn is None:
        statfn = np.mean
    T_obs = float(statfn(a) - statfn(b))
    pool = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    T_null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pool)
        T_null[i] = statfn(perm[:n_a]) - statfn(perm[n_a:])
    p = float((np.abs(T_null) >= abs(T_obs)).mean())
    return T_obs, p


def bootstrap_ci_two_sample(a: np.ndarray, b: np.ndarray, statfn=None, n_boot: int = 5000,
                             seed: int = SEED, alpha: float = 0.05):
    """CI on statfn(a) - statfn(b), independently resampling each arm."""
    if statfn is None:
        statfn = np.mean
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = statfn(a[rng.integers(0, na, na)]) - statfn(b[rng.integers(0, nb, nb)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(values: np.ndarray, statfn=None, n_boot: int = 5000, seed: int = SEED, alpha: float = 0.05):
    if statfn is None:
        statfn = np.mean
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = statfn(values[idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def auc_mannwhitney(x: np.ndarray, y: np.ndarray) -> float:
    """AUC = P(X > Y) via Mann-Whitney U (ties count 0.5)."""
    from scipy.stats import mannwhitneyu
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(u / (len(x) * len(y)))


def swap_perm_test(within_dist: np.ndarray, cross_dist: np.ndarray, n_perm: int = 5000, seed: int = SEED):
    """Paired swap-permutation: T = mean(cross) - mean(within), per-pair sign flip.
    within_dist[i] = d(gt_i, pred_i); cross_dist[i] = d(gt_i, pred_j), j != i
    (cross distances need not be the same length as within).
    H0: pred_i indistinguishable from gt_i (T=0). Returns (T_obs, p)."""
    T_obs = float(cross_dist.mean() - within_dist.mean())
    rng = np.random.default_rng(seed)
    pool = np.concatenate([within_dist, cross_dist])
    n_w = len(within_dist)
    T_null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pool)
        T_null[i] = perm[n_w:].mean() - perm[:n_w].mean()
    p = float((np.abs(T_null) >= abs(T_obs)).mean())
    return T_obs, p


def permdisp(dist_a: np.ndarray, dist_b: np.ndarray, n_perm: int = 5000, seed: int = SEED):
    """PERMDISP-style test: compare within-group mean pairwise distance
    (dispersion) of group a vs group b via label-swap permutation.
    dist_a, dist_b: flattened upper-triangle pairwise distances within each group
    (not required to be equal length). Returns (T_obs=disp_a-disp_b, p)."""
    return unpaired_perm_test(dist_a, dist_b, statfn=np.mean, n_perm=n_perm, seed=seed)


def pairwise_dist(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    from scipy.spatial.distance import pdist
    return pdist(X, metric=metric)


def cdist(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    from scipy.spatial.distance import cdist as _cdist
    return _cdist(X, Y, metric=metric)


def fit_pca_shared(gt_X: np.ndarray, pred_X: np.ndarray, n_components: int = 30, seed: int = SEED):
    from sklearn.decomposition import PCA
    n_comp = max(2, min(n_components, len(gt_X) - 1, gt_X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed).fit(gt_X)
    return pca.transform(gt_X), pca.transform(pred_X), pca


def fit_pca_pooled(X_a: np.ndarray, X_b: np.ndarray, n_components: int = 30, seed: int = SEED):
    """PCA fitted on the pooled cells of two groups, for comparisons without a reference group."""
    from sklearn.decomposition import PCA
    pooled = np.vstack([X_a, X_b])
    n_comp = max(2, min(n_components, len(pooled) - 1, pooled.shape[1]))
    pca = PCA(n_components=n_comp, random_state=seed).fit(pooled)
    return pca.transform(X_a), pca.transform(X_b), pca


def forest_plot(ax, labels: list[str], values: list[float], los: list[float], his: list[float],
                 filled: list[bool] | None = None, ref: float = 0.0, xlabel: str = "", logx: bool = False):
    """Forest plot: one row per label, marker at values[i], error bar
    [los[i], his[i]], reference line at ref; filled[i]=False draws a hollow marker.
    """
    y = np.arange(len(labels))[::-1]
    filled = filled if filled is not None else [True] * len(labels)
    for i, (v, lo, hi, f) in enumerate(zip(values, los, his, filled)):
        color = "#4c72b0"
        xerr = [[max(v - lo, 0)], [max(hi - v, 0)]]
        ax.errorbar([v], [y[i]], xerr=xerr, fmt="o",
                    color=color, mfc=color if f else "white", mec=color,
                    markersize=7, capsize=3, lw=1.5)
    ax.axvline(ref, color="gray", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(-0.5, len(labels) - 0.5)


def knn_mixing_lisi(gt_emb: np.ndarray, pred_emb: np.ndarray, k: int = 30) -> np.ndarray:
    """Simplified integration LISI (Korsunsky et al. 2019) with hard k-NN
    instead of perplexity weighting: inverse Simpson index of the gt/pred label
    composition among each point's k nearest neighbours in the pooled embedding.
    1.0 = all neighbours share one label, 2.0 = even mix. Returned for the pred
    points only.
    """
    from sklearn.neighbors import NearestNeighbors
    pooled = np.vstack([gt_emb, pred_emb])
    labels = np.array([0] * len(gt_emb) + [1] * len(pred_emb))
    k_eff = min(k, len(pooled) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(pooled)
    _, idx = nn.kneighbors(pooled[len(gt_emb):])
    lisi = np.empty(len(pred_emb))
    for i, neighbours in enumerate(idx):
        neighbours = neighbours[neighbours != (len(gt_emb) + i)]
        lab = labels[neighbours]
        p = np.bincount(lab, minlength=2) / len(lab)
        lisi[i] = 1.0 / (p ** 2).sum() if (p ** 2).sum() > 0 else float("nan")
    return lisi


def delta_and_mask(unexp_X: np.ndarray, other_X: np.ndarray):
    """delta = other - unexposed (expression units); mask = genes nonzero on
    either side, which leaves out genes that are zero on both sides because of
    the K=800 cutoff.
    """
    delta = other_X - unexp_X
    mask = (unexp_X > 0) | (other_X > 0)
    return delta, mask


def filtered_gene_domain(X_a: np.ndarray, X_b: np.ndarray, min_frac: float = 0.2) -> np.ndarray:
    """Boolean gene mask: nonzero in >= min_frac of cells on at least one
    side. Removes genes that are zero in nearly all cells of both groups.
    """
    frac_a = (X_a != 0).mean(axis=0)
    frac_b = (X_b != 0).mean(axis=0)
    return (frac_a >= min_frac) | (frac_b >= min_frac)


def intersected_gene_domain(X_a: np.ndarray, X_b: np.ndarray, min_frac: float = 0.2) -> np.ndarray:
    """Boolean gene mask: nonzero in >= min_frac of cells on both sides (the
    intersection counterpart of filtered_gene_domain).
    """
    frac_a = (X_a != 0).mean(axis=0)
    frac_b = (X_b != 0).mean(axis=0)
    return (frac_a >= min_frac) & (frac_b >= min_frac)


def centroid_distance(X_a: np.ndarray, X_b: np.ndarray) -> float:
    return float(np.linalg.norm(X_a.mean(axis=0) - X_b.mean(axis=0)))


def centroid_perm_test(X_a: np.ndarray, X_b: np.ndarray, n_perm: int = 2000, seed: int = SEED):
    """One-sided permutation test on centroid separation ||mean(a)-mean(b)||.
    H0: a,b same centroid (any observed separation is as likely from a random
    relabelling of the pooled cells). Returns (T_obs, p)."""
    T_obs = centroid_distance(X_a, X_b)
    pool = np.vstack([X_a, X_b])
    n_a = len(X_a)
    rng = np.random.default_rng(seed)
    T_null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pool))
        T_null[i] = centroid_distance(pool[idx[:n_a]], pool[idx[n_a:]])
    p = float((T_null >= T_obs).mean())
    return T_obs, p


def centroid_distance_bootstrap_ci(X_a: np.ndarray, X_b: np.ndarray, n_boot: int = 2000,
                                    seed: int = SEED, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    na, nb = len(X_a), len(X_b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = centroid_distance(X_a[rng.integers(0, na, na)], X_b[rng.integers(0, nb, nb)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_cell_rescue_auc(delta_X: np.ndarray, mask_X: np.ndarray, gene_names: list[str],
                         rescued_set: set[str]) -> np.ndarray:
    """Per cell: AUC = P(delta_rescued > delta_background), restricted to genes
    'in play' (mask) for that cell. NaN where either arm has <2 genes."""
    rescued_bool = np.array([g in rescued_set for g in gene_names])
    n = len(delta_X)
    aucs = np.full(n, np.nan)
    for i in range(n):
        m = mask_X[i]
        r_idx = m & rescued_bool
        b_idx = m & ~rescued_bool
        if r_idx.sum() < 2 or b_idx.sum() < 2:
            continue
        aucs[i] = auc_mannwhitney(delta_X[i, r_idx], delta_X[i, b_idx])
    return aucs
