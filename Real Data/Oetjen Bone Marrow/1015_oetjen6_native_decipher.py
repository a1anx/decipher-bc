"""Native Decipher (published `scdecipher` from PyPI) on an Oetjen shared h5ad.

One process per seed: train -> rotate -> decipher time -> scIB metrics on v and z ->
reconstruction diagnostic -> v-space PNG.
Run from the repo root with the dedicated venv. oetjen6 (defaults):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-scdecipher/bin/python \
        "Real Data/Oetjen Bone Marrow/1015_oetjen6_native_decipher.py" --seed 0
oetjen5:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-scdecipher/bin/python \
        "Real Data/Oetjen Bone Marrow/1015_oetjen6_native_decipher.py" --seed 0 \
        --src "Real Data/Oetjen Bone Marrow/1016_oetjen5/oetjen5_shared.h5ad" --dataset oetjen5 \
        --out-dir "Real Data/Oetjen Bone Marrow/1016_oetjen5_native"

Input: `--src` (default `1015_oetjen6/oetjen6_shared.h5ad`; its sha256 is checked against the
`.sha256` file beside it), `hvg_2000` genes, raw counts as X, the file's own 90/10
`split` column (stratified by sample) instead of the package's random split. Native Decipher is
batch-blind. Hyper-parameters are the package defaults (early stopping patience 10 on val NLL)
except learning rate 1e-3 and beta 0.1. The package returns only validation losses, so `SVI` and
`predictive_log_likelihood` are wrapped (not edited) to record per epoch: train_elbo = mean
negative ELBO per training cell and val_nll = package validation NLL per cell.

Decipher time: trajectories HSPC (AVP) -> erythroid (GYPA) and HSPC (AVP) -> myeloid (CD14), each
restricted to clusters of its branch. The branch columns include the HSC lineage so the start
cluster survives the package's per-cluster subsetting. The package's `decipher_time` writes both
trajectories into one column (shared HSC clusters keep the last one) and leaves cells off the path
clusters NaN, so `decipher_time_<branch>` re-predicts each trajectory's time for every cell of its
branch with the package's own KNN regression on the trajectory points.

scIB metrics (Luecken et al. 2022, Methods), written here from the paper's definitions, on v, z and
the uncorrected PCA (log-normalised `hvg_2000`, 50 PCs) as the batch baseline; batch = `sample`,
label = `cell_type`, kNN graphs from scanpy `neighbors` with k = 15 on the space:
    batch_asw               mean over labels of mean_cells(1 - |silhouette on batch|)
    graph_ilisi             (median LISI - 1) / (B - 1); LISI from the 90 nearest cells by
                            shortest-path distance on the kNN graph, perplexity 30
    graph_connectivity      mean over labels of (largest connected component / cells) in the
                            label's kNN subgraph
    trajectory_conservation per branch, (Spearman + 1) / 2 of DPT on the space against the
                            reference DPT on the uncorrected PCA; DPT on the largest connected
                            component, rooted at the most extreme HSPC on the first three
                            diffusion components; 0 if the trajectory cannot be computed
`decipher_time_conservation` is the same score with `decipher_time_<branch>` in place of DPT on v.

Writes under --out-dir (default `1015_oetjen6_native/` next to this script):
    csv/seed<k>_loss.csv, csv/seed<k>_vspace.csv (obs, v, z, per-branch decipher time and DPTs),
    csv/seed<k>_metrics.csv (space, metric, branch, value), csv/seed<k>_traj_clusters.csv,
    reconstruction/seed<k>_recon.png, vspace/seed<k>_vspace.png, seed<k>_meta.json,
    _models/<run_id>/ (package save folder), logs/ (for nohup logs).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KNeighborsRegressor

HERE = Path(__file__).resolve().parent
SRC = HERE / "1015_oetjen6" / "oetjen6_shared.h5ad"
DATASET = "oetjen6"
LEARNING_RATE = 1e-3
BETA = 0.1
OBS_KEEP = ["split", "sample", "cell_type", "lineage", "ref_pseudotime"]
BRANCHES = {"erythroid_branch": ["HSC", "erythroid"], "myeloid_branch": ["HSC", "myeloid"]}
ROTATION = {
    "v1_col": "cell_type",
    "v1_order": ["HSPCs", "Erythroid progenitors", "Erythrocytes"],
    "v2_col": "lineage",
    "v2_order": ["erythroid", "HSC", "myeloid"],
}
# trajectory -> (start marker, end marker, branch column)
TRAJECTORIES = {
    "erythroid": ("AVP", "GYPA", "erythroid_branch"),
    "myeloid": ("AVP", "CD14", "myeloid_branch"),
}
# trajectory -> allowed majority cell types of its (start, end) cluster
SANITY_RULES = {
    "erythroid": ({"HSPCs"}, {"Erythroid progenitors", "Erythrocytes"}),
    "myeloid": ({"HSPCs"}, {"CD14+ Monocytes", "CD16+ Monocytes"}),
}
VSPACE_PANELS = ["sample", "cell_type", "lineage", "decipher_time"]
# scIB metric settings
BATCH_KEY, LABEL_KEY, ROOT_CELL_TYPE = "sample", "cell_type", "HSPCs"
N_PCS = 50
KNN_K = 15
LISI_NEIGHBORS, LISI_PERPLEXITY = 90, 30
TIME_KNN = 10  # the package's decipher_time KNN regression


def branch_columns(lineage: pd.Series) -> pd.DataFrame:
    """Boolean branch membership per cell; HSC cells belong to both branches."""
    return pd.DataFrame({col: lineage.isin(vals) for col, vals in BRANCHES.items()})


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


def trajectory_sanity(paths: dict[str, list[str]], counts: pd.DataFrame) -> dict:
    """Check each trajectory's start/end cluster majority cell type against SANITY_RULES.

    `paths` maps trajectory name -> cluster ids (start first); `counts` is a cluster x cell_type
    count table. Returns per-endpoint cluster, majority and pass flag, plus the overall `pass`.
    """
    result = {}
    for name, (start_ok, end_ok) in SANITY_RULES.items():
        for end, cluster, allowed in [
            ("start", paths[name][0], start_ok),
            ("end", paths[name][-1], end_ok),
        ]:
            majority = str(counts.loc[cluster].idxmax())
            result[f"{name}_{end}"] = {
                "cluster": str(cluster),
                "majority": majority,
                "pass": majority in allowed,
            }
    result["pass"] = all(v["pass"] for v in result.values())
    return result


def traj_cluster_table(paths: dict[str, list[str]], counts: pd.DataFrame) -> pd.DataFrame:
    """One row per (trajectory, cluster on its path): cell_type counts and majority type."""
    rows = []
    for name, clusters in paths.items():
        for step, cluster in enumerate(clusters):
            row = counts.loc[cluster]
            rows.append(
                {
                    "trajectory": name,
                    "step": step,
                    "cluster": str(cluster),
                    "n_cells": int(row.sum()),
                    "majority": str(row.idxmax()),
                    **row.astype(int).to_dict(),
                }
            )
    return pd.DataFrame(rows)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="decipher seed (init, batch order)")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=HERE / "1015_oetjen6_native")
    parser.add_argument(
        "--src", type=Path, default=SRC, help="shared h5ad; <src stem>.sha256 must sit beside it"
    )
    parser.add_argument("--dataset", default=DATASET, help="name for run name, plot title")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from importlib.metadata import version

    import anndata as ad
    import decipher as dc
    import matplotlib.pyplot as plt
    import torch
    from decipher.tools import decipher as dtool
    from decipher.utils import DECIPHER_GLOBALS

    torch.set_num_threads(1)
    out = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    for sub in ["csv", "reconstruction", "vspace", "_models", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    stages, t_stage = {}, time.time()

    def lap(name: str) -> None:
        nonlocal t_stage
        stages[name] = round(time.time() - t_stage, 1)
        t_stage = time.time()
        print(f"[{tag}] {name}: {stages[name]} s", flush=True)

    # 1. data
    h5ad_sha = sha256(args.src)
    expected_sha = args.src.with_suffix(".sha256").read_text().split()[0]
    adata = ad.read_h5ad(args.src)
    adata = adata[:, adata.var["hvg_2000"]].copy()
    adata.X = adata.layers["counts"].copy()
    del adata.layers["counts"]
    config = dc.tl.DecipherConfig(
        seed=args.seed, learning_rate=LEARNING_RATE, beta=BETA, n_epochs=args.max_epochs
    )

    lap("load")

    # 2. train (package wrapped: per-epoch losses, file split, save folder)
    rows, step = [], {"loss": 0.0, "n": 0}

    class RecordingSVI(dtool.SVI):
        def step(self, *a, **kw):
            loss = super().step(*a, **kw)
            step["loss"] += loss
            step["n"] += a[0].shape[0]
            return loss

    orig_pll = dtool.predictive_log_likelihood

    def recording_pll(*a, **kw):
        out_ll = orig_pll(*a, **kw)
        row = {
            "epoch": len(rows),
            "train_elbo": step["loss"] / step["n"],
            "val_nll": -out_ll / len(a[1].dataset),
        }
        rows.append(row)
        step.update(loss=0.0, n=0)
        return out_ll

    def file_split(adata, val_frac, seed):
        adata.obs["decipher_split"] = (
            adata.obs["split"].map({"train": "train", "val": "validation"}).astype("category")
        )

    dtool.SVI, dtool.predictive_log_likelihood = RecordingSVI, recording_pll
    dtool._make_train_val_split = file_split
    DECIPHER_GLOBALS["save_folder"] = str(out / "_models")

    decipher, val_losses = dtool.decipher_train(adata, config, plot_every_k_epochs=-2)
    plt.close("all")  # the package draws a final v-space figure
    loss = pd.DataFrame(rows)
    assert np.allclose(loss["val_nll"], val_losses), "recorded val_nll != package val_losses"
    loss.to_csv(out / "csv" / f"{tag}_loss.csv", index=False)
    lap("train")
    seconds_per_epoch = round(stages["train"] / len(loss), 2)

    # 3. rotate (reloads the saved model by run_id)
    dc.tl.decipher_rotate_space(adata, **ROTATION)
    lap("rotate")

    # 4. decipher time (package column, then one column per branch over all its cells)
    adata.obs[list(BRANCHES)] = branch_columns(adata.obs["lineage"])
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    dc.tl.trajectories(
        adata,
        *[dc.tl.TConfig(n, s, e, col, True) for n, (s, e, col) in TRAJECTORIES.items()],
    )
    dc.tl.decipher_time(adata)
    trajs = adata.uns["decipher"]["trajectories"]
    for name, (_, _, col) in TRAJECTORIES.items():
        adata.obs[f"decipher_time_{name}"] = branch_time(
            adata.obsm["decipher_v"],
            trajs[name]["points"],
            trajs[name]["times"],
            adata.obs[col].to_numpy(),
        )
    paths = {n: list(t["cluster_ids"]) for n, t in trajs.items()}
    counts = pd.crosstab(adata.obs["decipher_clusters"], adata.obs["cell_type"])
    traj_cluster_table(paths, counts).to_csv(out / "csv" / f"{tag}_traj_clusters.csv", index=False)
    sanity = trajectory_sanity(paths, counts)
    print(f"[{tag}] trajectory sanity: {json.dumps(sanity)}", flush=True)
    lap("decipher_time")

    # 5. scIB metrics on v, z and the uncorrected PCA baseline
    spaces = {
        "pca": uncorrected_pca(adata.X),
        "v": adata.obsm["decipher_v"],
        "z": adata.obsm["decipher_z"],
    }
    batch = adata.obs[BATCH_KEY].astype(str).to_numpy()
    labels = adata.obs[LABEL_KEY].astype(str).to_numpy()
    metric_rows = []
    for space, emb in spaces.items():
        distances, connectivities = knn_graph(emb)
        for metric, value in [
            ("batch_asw", batch_asw(emb, batch, labels)),
            ("graph_ilisi", graph_ilisi(distances, batch)),
            ("graph_connectivity", graph_connectivity(connectivities, labels)),
        ]:
            metric_rows.append({"space": space, "metric": metric, "branch": "all", "value": value})
    for name, (_, _, col) in TRAJECTORIES.items():
        in_branch = adata.obs[col].to_numpy()
        is_root = labels[in_branch] == ROOT_CELL_TYPE
        for space, emb in spaces.items():
            dpt = np.full(adata.n_obs, np.nan)
            dpt[in_branch] = branch_dpt(emb[in_branch], is_root)
            adata.obs[f"dpt_{space}_{name}"] = dpt
        reference = adata.obs[f"dpt_pca_{name}"].to_numpy()
        assert np.isfinite(reference).sum() >= 3, f"reference DPT for {name} could not be computed"
        for space, metric, pseudotime in [
            ("v", "trajectory_conservation", adata.obs[f"dpt_v_{name}"]),
            ("z", "trajectory_conservation", adata.obs[f"dpt_z_{name}"]),
            ("v", "decipher_time_conservation", adata.obs[f"decipher_time_{name}"]),
        ]:
            value = trajectory_conservation(pseudotime.to_numpy(), reference)
            metric_rows.append({"space": space, "metric": metric, "branch": name, "value": value})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "csv" / f"{tag}_metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    lap("metrics")

    # 6. reconstruction diagnostic on all cells
    decipher.eval()
    x = adata.X.toarray()
    with torch.no_grad():
        r2 = reconstruction_r2(x, decipher.impute_gene_expression_numpy(x))
    r2_median = float(np.nanmedian(r2["r2_per_gene"]))
    fig = plot_reconstruction(loss, r2, f"{args.dataset} native Decipher, seed {args.seed}")
    recon_png = out / "reconstruction" / f"{tag}_recon.png"
    fig.savefig(recon_png, dpi=120)
    plt.close(fig)
    lap("reconstruction")

    # 7. v-space PNG, cells in shuffled order
    order = np.random.default_rng(0).permutation(adata.n_obs)
    fig = dc.pl.decipher(adata[order], color=VSPACE_PANELS, ncols=2, wspace=0.6, sort_order=False)
    vspace_png = out / "vspace" / f"{tag}_vspace.png"
    fig.savefig(vspace_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    lap("vspace_png")

    # 8. tables, meta
    time_cols = [f"decipher_time_{n}" for n in TRAJECTORIES]
    dpt_cols = [f"dpt_{s}_{n}" for n in TRAJECTORIES for s in spaces]
    table = adata.obs[
        OBS_KEEP
        + ["decipher_split", *BRANCHES, "decipher_clusters", "decipher_time", *time_cols, *dpt_cols]
    ]
    table = table.copy()
    table[["v1", "v2"]] = adata.obsm["decipher_v"]
    table[["v1_unrotated", "v2_unrotated"]] = adata.obsm["decipher_v_not_rotated"]
    z = adata.obsm["decipher_z"]
    table[[f"z{i + 1}" for i in range(z.shape[1])]] = z
    table.index.name = "cell_id"
    table.to_csv(out / "csv" / f"{tag}_vspace.csv")

    metric_summary = {
        f"{r.space}/{r.metric}" + ("" if r.branch == "all" else f"/{r.branch}"): r.value
        for r in metrics.itertuples()
    }
    summary = {
        "r2_overall": r2["r2_overall"],
        "r2_median_per_gene": r2_median,
        "sanity_pass": sanity["pass"],
        "n_epochs_run": len(loss),
        "best_epoch": int(loss["val_nll"].idxmin()),
        "final_val_nll": float(loss["val_nll"].iloc[-1]),
        **metric_summary,
    }
    lap("save")
    meta = {
        "dataset": args.dataset,
        "src": str(args.src),
        "scdecipher": version("scdecipher"),
        "seed": args.seed,
        "config": config.to_dict(),
        "h5ad_sha256": h5ad_sha,
        "h5ad_sha256_matches": h5ad_sha == expected_sha,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "run_id": adata.uns["decipher"]["run_id"],
        **summary,
        "rotation": adata.uns["decipher"]["rotation"].tolist(),
        "trajectory_cluster_ids": paths,
        "sanity": sanity,
        "seconds": stages,
        "seconds_per_epoch": seconds_per_epoch,
    }
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
