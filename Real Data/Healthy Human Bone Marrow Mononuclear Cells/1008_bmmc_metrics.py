"""Batch-mixing and biology-conservation metrics for the BMMC validation.

Pure functions on arrays (no file I/O), shared by the Stage 2 EDA (on PCA) and the Stage 5
evaluation (on Decipher v-space and z). Definitions follow scib / harmonypy so values are on
the usual benchmark scale:

- LISI (iLISI on batch, cLISI on cell type): Harmony's inverse Simpson index over a
  perplexity-calibrated Gaussian kernel on the ``3 * perplexity - 1`` nearest neighbours
  (self excluded), Euclidean distances. Per-cell values in [1, n_labels].
- kBET: per-cell chi-square test of the neighbourhood batch counts against the global batch
  frequencies; the score is the acceptance rate (1 - rejection rate).
- silhouette_label: plain ``sklearn.metrics.silhouette_score`` in [-1, 1], the same quantity the
  simulation pipeline reports as ``branch_asw`` (no scib (s + 1) / 2 rescale).
- silhouette_batch: scib batch ASW, mean over cell types of mean(1 - |s_batch|), in [0, 1].
- pc_regression: R^2 of each PC on a one-hot categorical covariate, and the variance-weighted
  total sum(R^2_i * var_i) / sum(var_i) (scib ``pcr``).

Load it by path (the name starts with a digit)::

    spec = importlib.util.spec_from_file_location("bmmc_metrics", path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
"""

from __future__ import annotations

from typing import Hashable, NamedTuple

import numpy as np
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

_LISI_TOL = 1e-5
_LISI_MAX_ITER = 50


def _check_emb_labels(emb: np.ndarray, labels: np.ndarray, name: str = "labels") -> None:
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D (cells x dims), got shape {emb.shape}")
    if emb.shape[0] < 2:
        raise ValueError(f"need at least 2 cells, got {emb.shape[0]}")
    if labels.shape != (emb.shape[0],):
        raise ValueError(f"{name} shape {labels.shape} does not match {emb.shape[0]} cells")


def _knn(emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(distances, indices) of the k nearest neighbours of each cell, self excluded."""
    k = min(k, emb.shape[0] - 1)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb)
    return dist[:, 1:], idx[:, 1:]


def _neighbour_counts(idx: np.ndarray, codes: np.ndarray, n_labels: int) -> np.ndarray:
    """(cells x labels) count of each label among each cell's neighbours."""
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    flat = rows * n_labels + codes[idx].ravel()
    return np.bincount(flat, minlength=idx.shape[0] * n_labels).reshape(idx.shape[0], n_labels)


def _perplexity_weights(dist: np.ndarray, perplexity: float) -> np.ndarray:
    """Row-normalized Gaussian weights whose entropy is log(perplexity) (harmonypy Hbeta)."""
    d = dist - dist[:, :1]  # shift by the nearest distance: P and H are invariant, avoids underflow
    n = d.shape[0]
    log_u = np.log(perplexity)
    beta = np.ones(n)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    for _ in range(_LISI_MAX_ITER):
        p = np.exp(-d * beta[:, None])
        sum_p = p.sum(axis=1)
        h = np.log(sum_p) + beta * (d * p).sum(axis=1) / sum_p
        diff = h - log_u
        active = np.abs(diff) >= _LISI_TOL
        if not active.any():
            break
        up = active & (diff > 0)
        down = active & (diff <= 0)
        lo[up] = beta[up]
        beta[up] = np.where(np.isfinite(hi[up]), (beta[up] + hi[up]) / 2, beta[up] * 2)
        hi[down] = beta[down]
        beta[down] = np.where(np.isfinite(lo[down]), (beta[down] + lo[down]) / 2, beta[down] / 2)
    p = np.exp(-d * beta[:, None])
    return p / p.sum(axis=1, keepdims=True)


def _lisi(emb: np.ndarray, labels: np.ndarray, perplexity: float) -> np.ndarray:
    codes = np.unique(labels, return_inverse=True)[1]
    dist, idx = _knn(emb, int(3 * perplexity) - 1)
    w = _perplexity_weights(dist, perplexity)
    n_labels = codes.max() + 1
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    mass = np.zeros((idx.shape[0], n_labels))
    np.add.at(mass, (rows, codes[idx].ravel()), w.ravel())
    return 1.0 / (mass**2).sum(axis=1)


def _per_group(within: np.ndarray, n_cells: int) -> dict[Hashable, np.ndarray]:
    within = np.asarray(within)
    if within.shape != (n_cells,):
        raise ValueError(f"within shape {within.shape} does not match {n_cells} cells")
    return {g: np.flatnonzero(within == g) for g in np.unique(within)}


def ilisi(
    emb: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30,
    within: np.ndarray | None = None,
) -> np.ndarray | dict[Hashable, np.ndarray]:
    """Per-cell integration LISI of ``labels`` (batch) in ``emb``; in [1, n_labels].

    With ``within`` (e.g. cell type), kNN and LISI are computed inside each group and a
    {group: per-cell array} dict is returned. Groups of one cell raise.
    """
    emb, labels = np.asarray(emb, dtype=float), np.asarray(labels)
    _check_emb_labels(emb, labels)
    if within is None:
        return _lisi(emb, labels, perplexity)
    out = {}
    for g, ix in _per_group(within, len(labels)).items():
        if len(ix) < 2:
            raise ValueError(f"within group {g!r} has {len(ix)} cell; LISI needs at least 2")
        out[g] = _lisi(emb[ix], labels[ix], perplexity)
    return out


def clisi(emb: np.ndarray, celltype: np.ndarray, perplexity: float = 30) -> np.ndarray:
    """Per-cell cell-type LISI; 1 means pure cell-type neighbourhoods (good)."""
    emb, celltype = np.asarray(emb, dtype=float), np.asarray(celltype)
    _check_emb_labels(emb, celltype, "celltype")
    return _lisi(emb, celltype, perplexity)


def normalize_lisi(lisi: np.ndarray, n_labels: int) -> np.ndarray:
    """(LISI - 1) / (n_labels - 1), in [0, 1]. 1 = all labels equally mixed, 0 = one label.

    For iLISI higher is better; for cLISI use ``1 - normalize_lisi(...)`` (scib's cLISI score).
    """
    if n_labels < 2:
        raise ValueError(f"normalized LISI needs at least 2 labels, got {n_labels}")
    return (np.asarray(lisi, dtype=float) - 1) / (n_labels - 1)


def lisi_median(lisi: np.ndarray, n_labels: int) -> dict[str, float]:
    """Median per-cell LISI and its normalized form (see ``normalize_lisi``)."""
    med = float(np.median(lisi))
    return {"median": med, "median_normalized": float(normalize_lisi(med, n_labels))}


def _kbet(emb: np.ndarray, labels: np.ndarray, k: int, alpha: float) -> float:
    uniq, codes = np.unique(labels, return_inverse=True)
    if len(uniq) < 2:
        return float("nan")
    _, idx = _knn(emb, k)
    observed = _neighbour_counts(idx, codes, len(uniq))
    expected = idx.shape[1] * np.bincount(codes) / len(codes)
    chi2 = ((observed - expected) ** 2 / expected).sum(axis=1)
    p = stats.chi2.sf(chi2, df=len(uniq) - 1)
    return float(np.mean(p >= alpha))


def kbet_acceptance(
    emb: np.ndarray,
    labels: np.ndarray,
    k: int,
    alpha: float = 0.05,
    within: np.ndarray | None = None,
) -> float | dict[Hashable, float]:
    """kBET acceptance rate: share of cells whose k-neighbourhood batch mix passes a chi-square
    test against the global batch frequencies at level ``alpha``. In [0, 1], higher = mixed.

    Every cell is tested (no subsampling); k is capped at n_cells - 1. With ``within``, the test
    runs inside each group against that group's batch frequencies and returns {group: rate};
    a group holding a single batch gets NaN. Without ``within``, a single batch raises.
    """
    emb, labels = np.asarray(emb, dtype=float), np.asarray(labels)
    _check_emb_labels(emb, labels)
    if within is None:
        if len(np.unique(labels)) < 2:
            raise ValueError("kBET needs at least 2 batches")
        return _kbet(emb, labels, k, alpha)
    return {
        g: _kbet(emb[ix], labels[ix], k, alpha) for g, ix in _per_group(within, len(labels)).items()
    }


def silhouette_label(
    emb: np.ndarray, labels: np.ndarray, sample_size: int | None = None, seed: int = 0
) -> float:
    """Mean silhouette of ``labels`` (Euclidean), in [-1, 1]; same as the simulation branch_asw.

    ``sample_size`` subsamples cells (sklearn), since the full score is O(n^2).
    """
    emb, labels = np.asarray(emb, dtype=float), np.asarray(labels)
    _check_emb_labels(emb, labels)
    n_labels = len(np.unique(labels))
    if not 2 <= n_labels < len(labels):
        raise ValueError(f"silhouette needs 2..n_cells-1 labels, got {n_labels}")
    return float(silhouette_score(emb, labels, sample_size=sample_size, random_state=seed))


def silhouette_batch(emb: np.ndarray, batch: np.ndarray, celltype: np.ndarray) -> float:
    """scib batch ASW: per cell type, mean of 1 - |silhouette(batch)|; then mean over cell types.

    In [0, 1], 1 = batches mixed. Cell types with one batch, or with as many batches as cells,
    are skipped (silhouette undefined); if all are skipped this raises.
    """
    emb, batch, celltype = np.asarray(emb, dtype=float), np.asarray(batch), np.asarray(celltype)
    _check_emb_labels(emb, batch, "batch")
    _check_emb_labels(emb, celltype, "celltype")
    scores = []
    for ct in np.unique(celltype):
        ix = celltype == ct
        n_b = len(np.unique(batch[ix]))
        if n_b < 2 or n_b == ix.sum():
            continue
        scores.append(np.mean(1 - np.abs(silhouette_samples(emb[ix], batch[ix]))))
    if not scores:
        raise ValueError("no cell type has 2..n_cells-1 batches; batch ASW is undefined")
    return float(np.mean(scores))


def pc_regression(
    pcs: np.ndarray, var_ratio: np.ndarray, covariate: np.ndarray
) -> tuple[np.ndarray, float]:
    """R^2 of each PC on a categorical covariate, and the variance-weighted total.

    R^2_i is the between-group share of PC i's variance (one-hot OLS with intercept).
    Total = sum(R^2_i * var_ratio_i) / sum(var_ratio_i), in [0, 1].
    """
    pcs, var_ratio, covariate = (
        np.asarray(pcs, float),
        np.asarray(var_ratio, float),
        np.asarray(covariate),
    )
    _check_emb_labels(pcs, covariate, "covariate")
    if var_ratio.shape != (pcs.shape[1],):
        raise ValueError(f"var_ratio shape {var_ratio.shape} does not match {pcs.shape[1]} PCs")
    codes = np.unique(covariate, return_inverse=True)[1]
    counts = np.bincount(codes)
    group_means = (
        np.stack([np.bincount(codes, weights=col) for col in pcs.T], axis=1) / counts[:, None]
    )
    centered = pcs - pcs.mean(axis=0)
    ss_tot = (centered**2).sum(axis=0)
    ss_between = (counts[:, None] * (group_means - pcs.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.divide(ss_between, ss_tot, out=np.zeros_like(ss_tot), where=ss_tot > 0)
    return r2, float((r2 * var_ratio).sum() / var_ratio.sum())


class TrajectoryCorr(NamedTuple):
    pearson: float
    r2: float
    spearman: float
    sign: int


def trajectory_corr(pred: np.ndarray, ref: np.ndarray) -> TrajectoryCorr:
    """Correlation of a predicted pseudotime/coordinate with a reference ordering.

    ``pred`` is multiplied by ``sign`` (+1 or -1), chosen to make Pearson non-negative; Pearson,
    R^2 (= Pearson^2) and Spearman are reported for ``sign * pred``.
    """
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    if pred.ndim != 1 or pred.shape != ref.shape or len(pred) < 3:
        raise ValueError(f"pred {pred.shape} and ref {ref.shape} must be equal-length 1-D, n >= 3")
    r = stats.pearsonr(pred, ref)[0]
    sign = 1 if r >= 0 else -1
    rho = stats.spearmanr(sign * pred, ref)[0]
    return TrajectoryCorr(pearson=float(abs(r)), r2=float(r**2), spearman=float(rho), sign=sign)
