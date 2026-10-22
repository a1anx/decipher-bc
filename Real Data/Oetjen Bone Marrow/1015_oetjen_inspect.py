"""Per-sample inspection of the full Oetjen bone marrow h5ad, to choose batches for a BC study.

Paper QC (>=500 genes, <8% mt) -> log-normalise -> 2000 HVGs -> PCA -> UMAP. Cell types come from
the scIB atlas labels on donors A/P/U, kNN-transferred to every other cell (leave-one-donor-out
recall is saved). Then, per sample: QC/depth, cell-type composition, trajectory cell counts
(HSPC -> erythroid / Mk / monocyte) and uncorrected kNN mixing between samples, computed within
each trajectory cell type so composition differences do not pose as batch effect.

Run from the repo root:
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1015_oetjen_inspect.py"
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent

TRAJECTORY = [
    "HSPCs",
    "Megakaryocyte progenitors",
    "Erythroid progenitors",
    "Erythrocytes",
    "Monocyte progenitors",
    "CD14+ Monocytes",
]


def transfer_labels(
    ref_emb: np.ndarray, ref_labels: np.ndarray, query_emb: np.ndarray, k: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """kNN majority vote of reference labels; returns (label, vote share of that label)."""
    ref_labels = np.asarray(ref_labels)
    if len(ref_labels) != len(ref_emb):
        raise ValueError(f"{len(ref_labels)} labels for {len(ref_emb)} reference cells")
    classes, codes = np.unique(ref_labels, return_inverse=True)
    idx = NearestNeighbors(n_neighbors=k).fit(ref_emb).kneighbors(query_emb, return_distance=False)
    votes = np.zeros((len(query_emb), len(classes)))
    np.add.at(votes, (np.repeat(np.arange(len(query_emb)), k), codes[idx].ravel()), 1)
    return classes[votes.argmax(1)], votes.max(1) / k


def knn_mixing(
    emb: np.ndarray, sample: np.ndarray, within: np.ndarray, k: int = 30
) -> pd.DataFrame:
    """Samples x samples neighbour enrichment (observed / expected), kNN found within each group.

    Expected counts use each group's own sample shares, so composition differences between
    samples do not count as batch effect. 1 = mixed; a diagonal >> 1 = the sample sits apart.
    Groups with no more than ``k`` cells are skipped. Self is excluded from its neighbours.
    """
    sample, within = np.asarray(sample), np.asarray(within)
    if not len(sample) == len(within) == len(emb):
        raise ValueError(f"{len(emb)} cells, {len(sample)} samples, {len(within)} groups")
    labels, codes = np.unique(sample, return_inverse=True)
    observed = np.zeros((len(labels), len(labels)))
    expected = np.zeros_like(observed)
    for group in np.unique(within):
        ix = np.flatnonzero(within == group)
        if len(ix) <= k:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(emb[ix])
        idx = nn.kneighbors(emb[ix], return_distance=False)[:, 1:]
        g_codes = codes[ix]
        np.add.at(observed, (np.repeat(g_codes, k), g_codes[idx].ravel()), 1)
        share = np.bincount(g_codes, minlength=len(labels)) / len(ix)
        expected += np.outer(np.bincount(g_codes, minlength=len(labels)) * k, share)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(observed / expected, index=labels, columns=labels)


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    adata = adata[(adata.obs.n_genes_by_counts >= 500) & (adata.obs.pct_counts_mt < 8)].copy()
    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000, flavor="seurat_v3", layer="counts", batch_key="sample"
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    return adata


def annotate(adata: ad.AnnData) -> pd.DataFrame:
    """Transfer scIB labels (donors A/P/U) to all cells; return leave-one-donor-out accuracy."""
    emb, obs = adata.obsm["X_pca"][:, :30], adata.obs
    ref = obs.scib_label.notna().values
    rows = []
    for donor in sorted(obs.donor[ref].unique()):
        held = ref & (obs.donor == donor).values
        pred, _ = transfer_labels(emb[ref & ~held], obs.scib_label[ref & ~held], emb[held])
        truth = obs.scib_label[held].values
        for label in np.unique(truth):
            hit = truth == label
            rows.append(
                {
                    "held_out": donor,
                    "label": label,
                    "n": hit.sum(),
                    "recall": (pred[hit] == label).mean().round(3),
                }
            )
    label, conf = transfer_labels(emb[ref], obs.scib_label[ref], emb)
    label[ref] = obs.scib_label[ref].values
    conf[ref] = 1.0
    adata.obs["celltype"] = pd.Categorical(label)
    adata.obs["celltype_conf"] = conf
    return pd.DataFrame(rows).pivot(index="label", columns="held_out", values="recall")


def sample_summary(adata: ad.AnnData, raw_cells: pd.Series) -> pd.DataFrame:
    obs = adata.obs
    g = obs.groupby("sample", observed=True)
    out = pd.DataFrame(
        {
            "donor": g.donor.first(),
            "age": g.age.first(),
            "cells_raw": raw_cells,
            "cells_qc": g.size(),
            "median_umi": g.total_counts.median().round(0),
            "median_genes": g.n_genes_by_counts.median().round(0),
            "median_pct_mt": g.pct_counts_mt.median().round(2),
        }
    )
    traj = obs.celltype.isin(TRAJECTORY)
    out["trajectory_cells"] = traj.groupby(obs["sample"], observed=True).sum()
    for label in TRAJECTORY:
        out[label] = (obs.celltype == label).groupby(obs["sample"], observed=True).sum()
    out["median_label_conf"] = g.celltype_conf.median().round(2)
    return out


def plot_all(adata: ad.AnnData, comp: pd.DataFrame, mixing: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc.pl.umap(adata, color="celltype", ax=axes[0], show=False, s=2)
    sc.pl.umap(adata, color="sample", ax=axes[1], show=False, s=2)
    fig.savefig(out / "umap_lineage_sample.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    ax = comp.plot.bar(stacked=True, figsize=(14, 6), colormap="tab20", width=0.85)
    ax.set_ylabel("fraction of QC cells")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.figure.savefig(out / "composition_by_sample.png", dpi=120, bbox_inches="tight")
    plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.log2(mixing.values.clip(1e-3)), cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_xticks(range(len(mixing)), mixing.columns, rotation=90)
    ax.set_yticks(range(len(mixing)), mixing.index)
    ax.set_title(
        "kNN neighbour enrichment, within trajectory cell types, uncorrected PCA (log2 obs/exp)"
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out / "knn_mixing_trajectory.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--h5ad", type=Path, default=HERE / "data" / "oetjen_bm_counts.h5ad")
    parser.add_argument("--out", type=Path, default=HERE / "1015_oetjen_inspect")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    raw_cells = adata.obs["sample"].value_counts()
    adata = preprocess(adata)
    recall = annotate(adata)
    recall.to_csv(args.out / "label_transfer_loo_recall.csv")

    summary = sample_summary(adata, raw_cells)
    comp = pd.crosstab(adata.obs["sample"], adata.obs.celltype, normalize="index")
    traj = adata.obs.celltype.isin(TRAJECTORY).values
    mixing = knn_mixing(
        adata.obsm["X_pca"][traj, :30],
        adata.obs["sample"].values[traj],
        adata.obs.celltype.values[traj],
    )
    summary["self_enrichment"] = pd.Series(np.diag(mixing), index=mixing.index).round(2)
    summary.to_csv(args.out / "sample_summary.csv")
    adata.obs[["sample", "celltype", "celltype_conf"]].to_csv(args.out / "cell_labels.csv")
    comp.round(4).to_csv(args.out / "composition_by_sample.csv")
    mixing.round(3).to_csv(args.out / "knn_mixing_trajectory.csv")
    plot_all(adata, comp, mixing, args.out)

    pd.set_option("display.width", 250)
    print(recall.to_string())
    print(summary.to_string())
    print((comp * 100).round(1).to_string())


if __name__ == "__main__":
    main()
