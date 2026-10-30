"""Shared definitions of the aml4 runs: constants, the leukemic trajectory and the evaluation.

Imported by path by `1027_build_aml4.py`, `1027_aml4_d3_decipher.py` (decipher_models3) and
`1027_aml4_native_decipher.py` (published `scdecipher`), so it imports only numpy/pandas/scipy at
the top; the model package is passed in as `dc`. `scib_metrics.py` (same folder) is loaded by
path as `native` for the scIB metric functions, DPT, branch time and reconstruction plot.

Evaluation of one trained model (`evaluate`):
    rotate      v1 onto `stage_rank` (immature 0 < blast0 < blast1 < blast2 < blast3 4)
    trajectory  Leiden clusters on `decipher_z`; one "leukemic" trajectory from the cluster with the
                most `immature` cells to the cluster with the most `blast3` cells (by
                `cell_type_merged`, among clusters the package keeps, > `min_cell_per_cluster`);
                if both are the same cluster the end is the next cluster by blast3 count and
                `endpoint_conflict` is true (sanity then fails); package `decipher_time`, plus
                `decipher_time_leukemic` = the package's KNN time regression for every cell
    spearman    Spearman(decipher_time_leukemic, stage_rank) over ranked cells, overall and per
                patient (`immature_myelo` is unranked)
    metrics     on a fixed `METRIC_SUBSAMPLE`-cell subsample (seed 0, same cells for every run) of
                the uncorrected 50-PC PCA, v and z: batch ASW (labels present in >= 2 patients;
                the others are listed as skipped), graph iLISI, graph connectivity (batch
                `patient`, label `cell_type_merged`); trajectory conservation = (Spearman + 1) / 2
                of DPT (ranked cells, rooted at the most extreme immature cell) against
                `stage_rank`; on all cells `decipher_time_conservation` (same score for
                `decipher_time_leukemic`) and `decipher_time_stage_spearman` (branch `leukemic`
                overall, `leukemic_<patient>` per patient)
    recon       reconstruction R^2 (log1p) on the same subsample (all cells would hold several
                64,803 x 6,341 float64 arrays)
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_by_path(path: Path, name: str):
    """Import the module file at `path` under `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load_by_path(HERE / "scib_metrics.py", "aml4_scib_metrics")

SRC = HERE / "aml4" / "aml4_shared.h5ad"
DATASET = "aml4"
# scdecipher package defaults (= decipher_models3 DecipherConfig defaults); parser defaults of
# both drivers. The paper AML fit (lr 1e-2, beta 1e-3, batch_size 512) NaNs the PyPI native in
# its first epoch on seeds 0 and 1.

HYPERPARAMS = {"lr": 5e-3, "beta": 0.1, "batch_size": 64}
BATCH_SOURCE = "patient"
BATCH_COL = "batch"
LABEL_KEY = "cell_type_merged"
STAGE_ORDER = ["immature", "blast0", "blast1", "blast2", "blast3"]
TRAJECTORY = "leukemic"
METRIC_SUBSAMPLE = 10_000
SUBSAMPLE_SEED = 0
PANELS = ["patient", "cell_type_merged", "cell_type", "decipher_time", "library", "NPM1_mut"]
OBS_KEEP = [
    "patient",
    "cell_type",
    "cell_type_merged",
    "library",
    "prep_batch",
    "NPM1_mut",
    "batch",
    "split",
    "stage_rank",
]


def check_integer_counts(adata) -> None:
    """Drop-in for `decipher_models3.tools.decipher.check_adata_has_integer_counts` on a sparse X:
    the same `np.allclose(x, x.astype(int))` test on the stored values only (zeros pass), without
    densifying. The package version densifies X and builds int64/float64 copies (~16 GB peak RSS on
    64,803 x 6,341). Raises ValueError like it."""
    from scipy import sparse

    x = adata.X
    values = x.data if sparse.issparse(x) else np.asarray(x)
    if not np.allclose(values, values.astype(int)):
        raise ValueError(".X must contain integer counts but it contains non-integer values.")


def out_dir(dataset: str, arm: str, root: Path = HERE) -> Path:
    """`root / <arm>`."""
    return root / arm


def rel_path(path: Path, base: Path) -> str:
    """`path` (resolved) relative to `base` when it lies under it, else the resolved path."""
    path = Path(path).resolve()
    return str(path.relative_to(base)) if path.is_relative_to(base) else str(path)


def stage_rank(merged: pd.Series) -> pd.Series:
    """Position of each cell's `cell_type_merged` in `STAGE_ORDER` (float), NaN for other labels."""
    ranks = {label: float(i) for i, label in enumerate(STAGE_ORDER)}
    return merged.astype(str).map(ranks).astype(float).rename("stage_rank")


def trajectory_endpoints(
    cluster: pd.Series,
    label: pd.Series,
    start_label: str = STAGE_ORDER[0],
    end_label: str = STAGE_ORDER[-1],
) -> tuple[str, str, bool]:
    """(start, end, conflict): start = cluster with the most `start_label` cells, end = cluster with
    the most `end_label` cells (ties: the first cluster in order). When both are the same cluster,
    end is the next cluster by `end_label` count and conflict is True. Raises when a label has no
    cells or no other cluster holds `end_label` cells."""
    counts = pd.crosstab(cluster.astype(str), label.astype(str))
    for lab in (start_label, end_label):
        if lab not in counts.columns or counts[lab].max() == 0:
            raise ValueError(f"no {lab!r} cells in any cluster")
    start = counts[start_label].idxmax()
    end_counts = counts[end_label]
    conflict = bool(end_counts.idxmax() == start)
    if conflict:
        end_counts = end_counts.drop(start)
        if end_counts.empty or end_counts.max() == 0:
            raise ValueError(f"only cluster {start!r} holds {end_label!r} cells")
    return str(start), str(end_counts.idxmax()), conflict


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation over pairs where both are finite; NaN with < 3 pairs or a constant."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))


def time_stage_spearman(time: np.ndarray, rank: np.ndarray, patient: np.ndarray) -> dict:
    """Spearman(time, rank) on all cells (`"all"`) and within each patient (sorted)."""
    out = {"all": spearman(time, rank)}
    for p in sorted(set(patient)):
        ix = patient == p
        out[str(p)] = spearman(time[ix], rank[ix])
    return out


def metric_subsample(
    n_obs: int, k: int = METRIC_SUBSAMPLE, seed: int = SUBSAMPLE_SEED
) -> np.ndarray:
    """Sorted indices of min(k, n_obs) cells drawn without replacement (`default_rng(seed)`)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_obs, size=min(k, n_obs), replace=False))


def shared_label_mask(batch: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """(cells whose label occurs in >= 2 batches, the other labels sorted): batch ASW is undefined
    for a label seen in one batch."""
    n_batches = pd.Series(batch).groupby(labels).nunique()
    skipped = sorted(str(lab) for lab in n_batches.index[n_batches < 2])
    return ~np.isin(labels, skipped), skipped


def mixing_metrics(
    spaces: dict[str, np.ndarray], batch: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, list[str]]:
    """batch ASW (labels in >= 2 batches), graph iLISI and graph connectivity per space, through
    `scib_metrics.py`. (rows space/metric/branch="all"/value, skipped labels)."""
    keep, skipped = shared_label_mask(batch, labels)
    rows = []
    for space, emb in spaces.items():
        distances, connectivities = native.knn_graph(emb)
        for metric, value in [
            ("batch_asw", native.batch_asw(emb[keep], batch[keep], labels[keep])),
            ("graph_ilisi", native.graph_ilisi(distances, batch)),
            ("graph_connectivity", native.graph_connectivity(connectivities, labels)),
        ]:
            rows.append({"space": space, "metric": metric, "branch": "all", "value": value})
    return pd.DataFrame(rows), skipped


def conservation_metrics(
    spaces: dict[str, np.ndarray], rank: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per space, DPT on the ranked cells (rank finite) rooted at rank 0 and its trajectory
    conservation against `rank`. (rows space/metric/branch=TRAJECTORY/value, per-cell
    `dpt_<space>_leukemic` columns, NaN for unranked cells)."""
    ranked = np.isfinite(rank)
    rows, dpts = [], {}
    for space, emb in spaces.items():
        dpt = np.full(len(rank), np.nan)
        dpt[ranked] = native.branch_dpt(emb[ranked], rank[ranked] == 0)
        dpts[f"dpt_{space}_{TRAJECTORY}"] = dpt
        value = native.trajectory_conservation(dpt, rank)
        rows.append(
            {
                "space": space,
                "metric": "trajectory_conservation",
                "branch": TRAJECTORY,
                "value": value,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(dpts)


def time_metrics(time: np.ndarray, rank: np.ndarray, patient: np.ndarray) -> pd.DataFrame:
    """v rows on all cells: decipher_time_conservation (TRAJECTORY) and
    decipher_time_stage_spearman (TRAJECTORY overall, `leukemic_<patient>` per patient)."""
    rows = [
        {
            "space": "v",
            "metric": "decipher_time_conservation",
            "branch": TRAJECTORY,
            "value": native.trajectory_conservation(time, rank),
        }
    ]
    for key, rho in time_stage_spearman(time, rank, patient).items():
        branch = TRAJECTORY if key == "all" else f"{TRAJECTORY}_{key}"
        rows.append(
            {"space": "v", "metric": "decipher_time_stage_spearman", "branch": branch, "value": rho}
        )
    return pd.DataFrame(rows)


def traj_cluster_table(obs: pd.DataFrame, path: list[str]) -> pd.DataFrame:
    """One row per `decipher_clusters` level: size, 0-based position on `path` (-1 off it), mean
    stage rank, majority `cell_type_merged`, its count per label, top patient and its share."""
    cluster = obs["decipher_clusters"].astype(str)
    counts = pd.crosstab(cluster, obs[LABEL_KEY].astype(str))
    patients = pd.crosstab(cluster, obs[BATCH_SOURCE].astype(str))
    step = {c: i for i, c in enumerate(path)}
    table = pd.DataFrame(
        {
            "cluster": counts.index,
            "n_cells": counts.sum(axis=1).to_numpy(),
            "on_path_step": [step.get(c, -1) for c in counts.index],
            "mean_stage_rank": obs["stage_rank"].groupby(cluster).mean().reindex(counts.index),
            "majority": counts.idxmax(axis=1).to_numpy(),
            "top_patient": patients.idxmax(axis=1).reindex(counts.index).to_numpy(),
            "top_patient_share": (patients.max(axis=1) / patients.sum(axis=1))
            .reindex(counts.index)
            .to_numpy(),
        }
    ).reset_index(drop=True)
    return pd.concat([table, counts.reset_index(drop=True)], axis=1)


def metric_summary(metrics: pd.DataFrame) -> dict[str, float]:
    """`<space>/<metric>[/<branch>]` -> value (branch "all" left out)."""
    return {
        f"{r.space}/{r.metric}" + ("" if r.branch == "all" else f"/{r.branch}"): r.value
        for r in metrics.itertuples()
    }


def add_title_suffix(fig, suffix: str) -> None:
    """Append `suffix` to every non-colorbar axis title of `fig`."""
    for ax in fig.axes:
        if ax.get_label() != "<colorbar>":
            ax.set_title(f"{ax.get_title()}{suffix}")


def evaluate(
    adata,
    dc,
    out: Path,
    tag: str,
    lap: Callable[[str], None],
    loss: pd.DataFrame,
    recon: Callable,
    title: str,
    vspace_figure: Callable,
) -> dict:
    """Rotate, trajectory, metrics, reconstruction and v-space PNG of a trained model whose
    embeddings are in `adata` (see the module docstring). `dc` is the model package,
    `recon(adata_subset)` returns the reconstruction R^2 dict, `vspace_figure(adata)` the v-space
    figure. Writes `csv/<tag>_{traj_clusters,metrics,vspace}.csv`,
    `reconstruction/<tag>_recon.png`, `vspace/<tag>_vspace.png`; returns what the meta needs."""
    import matplotlib.pyplot as plt

    # rotate v1 onto the stage order (reloads the saved model by run_id)
    dc.tl.decipher_rotate_space(adata, v1_col="stage_rank")
    lap("rotate")

    # clusters, the leukemic trajectory, decipher time
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    min_cells = dc.tl.TConfig(TRAJECTORY).min_cell_per_cluster
    sizes = adata.obs["decipher_clusters"].value_counts()
    kept = adata.obs["decipher_clusters"].isin(sizes[sizes > min_cells].index).to_numpy()
    start, end, conflict = trajectory_endpoints(
        adata.obs.loc[kept, "decipher_clusters"], adata.obs.loc[kept, LABEL_KEY]
    )
    dc.tl.trajectories(adata, dc.tl.TConfig(TRAJECTORY, start, end))
    dc.tl.decipher_time(adata)
    traj = adata.uns["decipher"]["trajectories"][TRAJECTORY]
    path = [str(c) for c in traj["cluster_ids"]]
    time_col = f"decipher_time_{TRAJECTORY}"
    adata.obs[time_col] = native.branch_time(
        adata.obsm["decipher_v"], traj["points"], traj["times"], np.ones(adata.n_obs, bool)
    )
    traj_cluster_table(adata.obs, path).to_csv(
        out / "csv" / f"{tag}_traj_clusters.csv", index=False
    )
    rank = adata.obs["stage_rank"].to_numpy(dtype=float)
    patient = adata.obs[BATCH_SOURCE].astype(str).to_numpy()
    time_rho = time_stage_spearman(adata.obs[time_col].to_numpy(dtype=float), rank, patient)
    trajectory = {
        "start_cluster": start,
        "end_cluster": end,
        "endpoint_conflict": conflict,
        "path_cluster_ids": path,
        "n_path_cells": int(adata.obs["decipher_time"].notna().sum()),
        "time_vs_stage_spearman": time_rho["all"],
        "time_vs_stage_spearman_by_patient": {k: v for k, v in time_rho.items() if k != "all"},
        "sanity_pass": bool(not conflict and time_rho["all"] > 0),
    }
    print(f"[{tag}] trajectory: {trajectory}", flush=True)
    lap("decipher_time")

    # metrics: mixing and DPT conservation on the subsample, decipher time on all cells
    idx = metric_subsample(adata.n_obs)
    spaces = {
        "pca": native.uncorrected_pca(adata.X[idx]),
        "v": adata.obsm["decipher_v"][idx],
        "z": adata.obsm["decipher_z"][idx],
    }
    labels = adata.obs[LABEL_KEY].astype(str).to_numpy()
    mixing, skipped = mixing_metrics(spaces, patient[idx], labels[idx])
    conservation, dpt_sub = conservation_metrics(spaces, rank[idx])
    metrics = pd.concat(
        [mixing, conservation, time_metrics(adata.obs[time_col].to_numpy(float), rank, patient)],
        ignore_index=True,
    )
    metrics.to_csv(out / "csv" / f"{tag}_metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    lap("metrics")

    # reconstruction diagnostic on the subsample
    r2 = recon(adata[idx].copy())
    fig = native.plot_reconstruction(loss, r2, title)
    recon_png = out / "reconstruction" / f"{tag}_recon.png"
    fig.savefig(recon_png, dpi=120)
    plt.close(fig)
    lap("reconstruction")

    # v-space PNG, cells in a fixed shuffled order
    order = np.random.default_rng(0).permutation(adata.n_obs)
    fig = vspace_figure(adata[order])
    vspace_png = out / "vspace" / f"{tag}_vspace.png"
    fig.savefig(vspace_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    lap("vspace_png")

    # per-cell table
    table = adata.obs[
        OBS_KEEP + ["decipher_split", "decipher_clusters", "decipher_time", time_col]
    ].copy()
    table["in_metric_subsample"] = np.isin(np.arange(adata.n_obs), idx)
    for col in dpt_sub.columns:
        full = np.full(adata.n_obs, np.nan)
        full[idx] = dpt_sub[col].to_numpy()
        table[col] = full
    table[["v1", "v2"]] = adata.obsm["decipher_v"]
    table[["v1_unrotated", "v2_unrotated"]] = adata.obsm["decipher_v_not_rotated"]
    for key, prefix in [("decipher_z", "z"), ("decipher_z_raw", "z_raw")]:
        if key in adata.obsm:
            z = adata.obsm[key]
            table[[f"{prefix}{i + 1}" for i in range(z.shape[1])]] = z
    table.index.name = "cell_id"
    table.to_csv(out / "csv" / f"{tag}_vspace.csv")
    return {
        "metrics": metrics,
        "trajectory": trajectory,
        "r2_overall": r2["r2_overall"],
        "r2_median_per_gene": float(np.nanmedian(r2["r2_per_gene"])),
        "recon_png": recon_png,
        "vspace_png": vspace_png,
        "metric_subsample_n": len(idx),
        "batch_asw_skipped_labels": skipped,
        "rotation": np.asarray(adata.uns["decipher"]["rotation"]).tolist(),
    }


def run_summary(loss: pd.DataFrame, ev: dict, best_epoch: int) -> dict:
    """The meta summary fields shared by both drivers."""
    return {
        "r2_overall": ev["r2_overall"],
        "r2_median_per_gene": ev["r2_median_per_gene"],
        "sanity_pass": ev["trajectory"]["sanity_pass"],
        "time_vs_stage_spearman": ev["trajectory"]["time_vs_stage_spearman"],
        "n_epochs_run": len(loss),
        "best_epoch": best_epoch,
        "best_val_nll": float(loss["val_nll"].min()),
        "final_val_nll": float(loss["val_nll"].iloc[-1]),
        **metric_summary(ev["metrics"]),
    }
