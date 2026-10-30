"""scIB metrics (Luecken et al. 2022, Methods), decipher time on a trajectory, reconstruction R^2.

Definitions as implemented here; kNN graphs from scanpy `neighbors` with k = 15 on the space:
    batch_asw               mean over labels of mean_cells(1 - |silhouette on batch|), the
                            silhouette computed within the label
    graph_ilisi             (median LISI - 1) / (B - 1); LISI from the 90 nearest cells by
                            shortest-path distance on the kNN graph, perplexity 30
    graph_connectivity      mean over labels of (largest connected component / cells) in the
                            label's kNN subgraph
    trajectory_conservation (Spearman(pseudotime, reference) + 1) / 2 on cells where both are
                            finite; 0 when fewer than 3 such cells
DPT (`branch_dpt`) runs on the largest connected component of the kNN graph, rooted at the most
extreme root-type cell on diffusion components 1-3. `uncorrected_pca` is the 50-PC PCA of
library-size-normalised (1e4) log1p counts, the unintegrated baseline.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KNeighborsRegressor

N_PCS = 50
KNN_K = 15
LISI_NEIGHBORS, LISI_PERPLEXITY = 90, 30
TIME_KNN = 10  # the package's decipher_time KNN regression


def branch_time(
    v: np.ndarray, points: np.ndarray, times: np.ndarray, in_branch: np.ndarray
) -> np.ndarray:
    """Decipher time of every branch cell: KNN regression of v on the trajectory's (points, times),
    as the package's `decipher_time` does for path-cluster cells only. NaN outside the branch."""
    knn = KNeighborsRegressor(n_neighbors=TIME_KNN).fit(points, times)
    out = np.full(len(v), np.nan)
    out[in_branch] = knn.predict(v[in_branch])
    return out


def batch_asw(emb: np.ndarray, batch: np.ndarray, labels: np.ndarray) -> float:
    """scIB batch ASW: for each label, mean over its cells of 1 - |silhouette(batch)| computed
    within the label; then the mean over labels. 1 = batches mixed, 0 = separated."""
    scores = []
    for label in np.unique(labels):
        ix = labels == label
        n_batches = len(np.unique(batch[ix]))
        if not 2 <= n_batches < ix.sum():
            raise ValueError(f"label {label!r} has {n_batches} batches; batch ASW needs 2..n-1")
        scores.append(np.mean(1 - np.abs(silhouette_samples(emb[ix], batch[ix]))))
    return float(np.mean(scores))


def graph_connectivity(connectivities: sparse.spmatrix, labels: np.ndarray) -> float:
    """scIB graph connectivity: mean over labels of |largest connected component| / |label| in the
    kNN subgraph of that label's cells. In (0, 1], 1 = every label is one connected piece."""
    graph = sparse.csr_matrix(connectivities)
    shares = []
    for label in np.unique(labels):
        ix = np.flatnonzero(labels == label)
        _, comp = connected_components(graph[ix][:, ix], directed=False)
        shares.append(np.bincount(comp).max() / len(ix))
    return float(np.mean(shares))


def graph_neighbors(
    distances: sparse.spmatrix, n_neighbors: int, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """The `n_neighbors` nearest other cells of each cell by shortest-path length on the
    (undirected) kNN graph. Returns (dist, idx), both n x n_neighbors; unreachable slots are
    inf / -1."""
    n = distances.shape[0]
    m = min(n_neighbors, n - 1)
    out_d, out_i = np.full((n, m), np.inf), np.full((n, m), -1)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        d = dijkstra(distances, directed=False, indices=rows)
        d[np.arange(len(rows)), rows] = np.inf  # a cell is not its own neighbour
        idx = np.argpartition(d, m - 1, axis=1)[:, :m]
        dist = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(dist, axis=1)
        dist, idx = np.take_along_axis(dist, order, 1), np.take_along_axis(idx, order, 1)
        out_d[rows], out_i[rows] = dist, np.where(np.isfinite(dist), idx, -1)
    return out_d, out_i


def lisi(
    dist: np.ndarray,
    idx: np.ndarray,
    labels: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    n_iter: int = 50,
) -> np.ndarray:
    """Per-cell LISI = 1 / Simpson index of `labels` among each cell's neighbours, weighted by a
    Gaussian kernel whose bandwidth is binary-searched so the weights' entropy is log(perplexity)
    (Korsunsky et al. 2019). `dist`/`idx` are n x m neighbour distances and indices (-1 = none).
    In [1, number of labels]."""
    valid = idx >= 0
    if not valid.any(axis=1).all():
        raise ValueError("a cell has no neighbours; LISI is undefined")
    # shifting each row by its nearest distance leaves the weights unchanged and avoids underflow
    d = np.where(valid, dist - np.where(valid, dist, np.inf).min(axis=1, keepdims=True), 0.0)
    beta = np.ones(len(d))
    lo, hi = np.full(len(d), -np.inf), np.full(len(d), np.inf)
    target = np.log(perplexity)
    for _ in range(n_iter):
        p = np.where(valid, np.exp(-d * beta[:, None]), 0.0)
        sum_p = p.sum(axis=1)
        entropy = np.log(sum_p) + beta * (d * p).sum(axis=1) / sum_p
        diff = entropy - target
        if np.all(np.abs(diff) < tol):
            break
        up = diff > 0  # too flat: sharpen
        lo, hi = np.where(up, beta, lo), np.where(up, hi, beta)
        beta = np.where(
            up,
            np.where(np.isinf(hi), beta * 2, (beta + hi) / 2),
            np.where(np.isinf(lo), beta / 2, (beta + lo) / 2),
        )
    p = p / p.sum(axis=1, keepdims=True)
    codes = np.unique(labels, return_inverse=True)[1]
    neighbour_codes = np.where(valid, codes[np.where(valid, idx, 0)], -1)
    simpson = sum((p * (neighbour_codes == c)).sum(axis=1) ** 2 for c in range(codes.max() + 1))
    return 1 / simpson


def graph_ilisi(distances: sparse.spmatrix, batch: np.ndarray) -> float:
    """scIB graph iLISI: per-cell LISI of the batch over the 90 nearest cells by shortest-path
    length on the kNN graph (perplexity 30); median over cells rescaled as (median - 1) / (B - 1).
    1 = all batches equally mixed, 0 = one batch per neighbourhood."""
    n_batches = len(np.unique(batch))
    if n_batches < 2:
        raise ValueError("graph iLISI needs at least 2 batches")
    dist, idx = graph_neighbors(distances, LISI_NEIGHBORS)
    return float((np.median(lisi(dist, idx, batch, LISI_PERPLEXITY)) - 1) / (n_batches - 1))


def dpt_root(diffusion_components: np.ndarray, is_root: np.ndarray) -> int:
    """Index of the root cell: among `is_root` cells, the most extreme one on the diffusion
    component (of those given) where the root cell type sits farthest from the other cells, at the
    root type's end of it."""
    if not 0 < is_root.sum() < len(is_root):
        raise ValueError(f"need some but not all cells in the root type, got {is_root.sum()}")
    dcs = diffusion_components
    shift = (dcs[is_root].mean(axis=0) - dcs[~is_root].mean(axis=0)) / dcs.std(axis=0)
    dc = int(np.argmax(np.abs(shift)))
    root_ix = np.flatnonzero(is_root)
    values = np.sign(shift[dc]) * dcs[root_ix, dc]
    return int(root_ix[np.argmax(values)])


def trajectory_conservation(pseudotime: np.ndarray, reference: np.ndarray) -> float:
    """scIB trajectory conservation: (Spearman(pseudotime, reference) + 1) / 2 on cells where both
    are finite; 0 when fewer than 3 such cells (the trajectory could not be computed)."""
    ok = np.isfinite(pseudotime) & np.isfinite(reference)
    if ok.sum() < 3:
        return 0.0
    rho = pd.Series(pseudotime[ok]).corr(pd.Series(reference[ok]), method="spearman")
    return float((rho + 1) / 2)


def knn_graph(emb: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """scanpy `neighbors` (k = 15, Euclidean) on `emb`: (distances, connectivities)."""
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(np.zeros((len(emb), 1), dtype=np.float32), obsm={"rep": emb})
    sc.pp.neighbors(tmp, n_neighbors=KNN_K, use_rep="rep", random_state=0)
    return tmp.obsp["distances"], tmp.obsp["connectivities"]


def branch_dpt(emb: np.ndarray, is_root: np.ndarray) -> np.ndarray:
    """Diffusion pseudotime of `emb`'s cells (one branch) on the largest connected component of
    its kNN graph, rooted by `dpt_root` on diffusion components 1-3. NaN outside that component,
    and everywhere if the component holds no root-type cell."""
    import anndata as ad
    import scanpy as sc

    out = np.full(len(emb), np.nan)
    tmp = ad.AnnData(np.zeros((len(emb), 1), dtype=np.float32), obsm={"rep": emb})
    sc.pp.neighbors(tmp, n_neighbors=KNN_K, use_rep="rep", random_state=0)
    _, comp = connected_components(tmp.obsp["connectivities"], directed=False)
    lcc = comp == np.bincount(comp).argmax()
    if not is_root[lcc].any():
        return out
    tmp = tmp[lcc].copy()
    sc.tl.diffmap(tmp)
    tmp.uns["iroot"] = dpt_root(tmp.obsm["X_diffmap"][:, 1:4], is_root[lcc])
    sc.tl.dpt(tmp)
    out[lcc] = tmp.obs["dpt_pseudotime"].to_numpy()
    return out


def uncorrected_pca(counts: sparse.spmatrix) -> np.ndarray:
    """50-PC PCA of library-size-normalised (1e4) log1p counts: the unintegrated baseline."""
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(sparse.csr_matrix(counts, dtype=np.float32))
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.pca(tmp, n_comps=N_PCS, random_state=0)
    return tmp.obsm["X_pca"]


def reconstruction_r2(x: np.ndarray, x_hat: np.ndarray) -> dict:
    """R^2 of log1p(x_hat) against log1p(x): pooled and per gene (NaN for zero-variance genes)."""
    x_log = np.log1p(x.astype(float))
    x_hat_log = np.log1p(np.clip(x_hat, 0, None))
    resid = (x_log - x_hat_log) ** 2
    r2_overall = float(1 - resid.sum() / ((x_log - x_log.mean()) ** 2).sum())
    ss_tot_g = ((x_log - x_log.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_per_gene = np.where(ss_tot_g > 0, 1 - resid.sum(axis=0) / ss_tot_g, np.nan)
    return {
        "x_log": x_log,
        "x_hat_log": x_hat_log,
        "r2_overall": r2_overall,
        "r2_per_gene": r2_per_gene,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plot_reconstruction(loss: pd.DataFrame, r2: dict, title: str, seed: int = 0):
    """Three panels: loss curves, observed vs reconstructed log1p scatter, per-gene R^2 hist."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(loss["epoch"], loss["train_elbo"], label="train ELBO (neg., per cell)")
    ax.plot(loss["epoch"], loss["val_nll"], label="val NLL (per cell)")
    ax.set(xlabel="epoch", ylabel="loss", title="training curves")
    ax.legend()

    ax = axes[1]
    flat_x, flat_xhat = r2["x_log"].ravel(), r2["x_hat_log"].ravel()
    idx = np.random.default_rng(seed).choice(flat_x.size, min(20_000, flat_x.size), replace=False)
    ax.scatter(flat_x[idx], flat_xhat[idx], s=1, alpha=0.2)
    hi = float(max(flat_x[idx].max(), flat_xhat[idx].max()))
    ax.plot([0, hi], [0, hi], "r--", linewidth=1, label="y = x")
    ax.set(xlabel="observed log1p(x)", ylabel="reconstructed log1p(x_hat)")
    ax.set_title(f"reconstruction (overall R^2 = {r2['r2_overall']:.3f})")
    ax.legend()

    ax = axes[2]
    per_gene = r2["r2_per_gene"]
    median = float(np.nanmedian(per_gene))
    ax.hist(per_gene[~np.isnan(per_gene)], bins=30)
    ax.axvline(median, color="r", linestyle="--", label=f"median = {median:.3f}")
    ax.set(xlabel="per-gene R^2 (log1p scale)", ylabel="number of genes")
    ax.set_title("per-gene reconstruction quality")
    ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
