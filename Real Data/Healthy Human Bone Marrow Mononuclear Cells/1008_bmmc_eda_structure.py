"""Stage 2 EDA (structure) for the shared BMMC h5ad: QC, composition, HVG overlap, PC regression, UMAP.

Reads 1008_bmmc/bmmc_shared.h5ad read-only and writes to 1008_bmmc/eda/structure/.
Run: .venv/bin/python "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1008_bmmc_eda_structure.py"
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import anndata as ad  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "1008_bmmc"
SHARED = OUT / "bmmc_shared.h5ad"
STRUCT_DIR = OUT / "eda" / "structure"
COVARIATES = ["donor", "site", "sample", "cell_type"]
SEED = 0
N_PCS = 50
N_HVG = 2000


def load_metrics():
    spec = importlib.util.spec_from_file_location("bmmc_metrics", HERE / "1008_bmmc_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qc_table(counts: sp.csr_matrix, gene_names: pd.Index, obs: pd.DataFrame) -> pd.DataFrame:
    """Per-cell QC from raw counts on all genes: total counts, n genes detected, % mito."""
    mito = np.asarray(gene_names.str.upper().str.startswith("MT-"))
    if not mito.any():
        raise ValueError("no MT- genes found in var_names; cannot compute % mito")
    total = np.asarray(counts.sum(axis=1)).ravel()
    n_genes = np.diff(counts.indptr)
    mito_counts = np.asarray(counts[:, np.flatnonzero(mito)].sum(axis=1)).ravel()
    return pd.DataFrame(
        {
            "total_counts": total,
            "n_genes": n_genes,
            "pct_mito": 100 * mito_counts / np.maximum(total, 1),
            **{c: obs[c].astype(str).to_numpy() for c in COVARIATES},
        },
        index=obs.index,
    )


def qc_summary(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cov in ["donor", "site", "sample"]:
        g = qc.groupby(cov, observed=True)[["total_counts", "n_genes", "pct_mito"]]
        s = g.agg(["median", "mean"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        s.insert(0, "n_cells", g.size())
        s.insert(0, "level", s.index.astype(str))
        s.insert(0, "covariate", cov)
        rows.append(s.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def plot_qc_violins(qc: pd.DataFrame, out_dir: Path) -> None:
    for cov in ["donor", "site", "sample"]:
        levels = sorted(qc[cov].unique())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric in zip(axes, ["total_counts", "n_genes", "pct_mito"]):
            data = [qc.loc[qc[cov] == lv, metric].to_numpy() for lv in levels]
            ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(levels) + 1), levels, rotation=60)
            ax.set_title(f"{metric} by {cov}")
        fig.tight_layout()
        fig.savefig(out_dir / f"qc_violin_{cov}.png", dpi=120)
        plt.close(fig)


def composition_table(obs: pd.DataFrame) -> pd.DataFrame:
    """Cell-type fraction within each sample, with samples as rows."""
    return pd.crosstab(obs["sample"].astype(str), obs["cell_type"].astype(str), normalize="index")


def plot_composition(comp: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    comp.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.85)
    ax.set_ylabel("fraction of cells")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def per_donor_hvgs(counts: sp.csr_matrix, genes: pd.Index, donors: np.ndarray) -> dict[str, set]:
    """Top-N seurat_v3 HVGs computed on each donor's raw counts."""
    sets = {}
    for d in sorted(set(donors)):
        sub = ad.AnnData(counts[np.flatnonzero(donors == d)].astype(np.float32))
        sub.var_names = genes
        sc.pp.highly_variable_genes(sub, n_top_genes=N_HVG, flavor="seurat_v3")
        sets[d] = set(genes[sub.var["highly_variable"].to_numpy()])
    return sets


def jaccard_matrix(sets: dict[str, set]) -> pd.DataFrame:
    keys = list(sets)
    mat = np.array([[len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for b in keys] for a in keys])
    return pd.DataFrame(mat, index=keys, columns=keys)


def median_offdiag(mat: pd.DataFrame) -> float:
    return float(np.median(mat.to_numpy()[~np.eye(len(mat), dtype=bool)]))


def plot_heatmap(mat: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.to_numpy(), cmap="viridis")
    ax.set_xticks(range(len(mat)), mat.columns, rotation=60)
    ax.set_yticks(range(len(mat)), mat.index)
    for i in range(len(mat)):
        for j in range(len(mat)):
            ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, label="Jaccard")
    ax.set_title("Per-donor HVG overlap (top 2000, seurat_v3)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def normalized_pca(counts: sp.csr_matrix, hvg_mask: np.ndarray) -> ad.AnnData:
    """Library-size normalise to 1e4 on all genes, log1p, keep HVGs, PCA (50 comps, seed 0)."""
    total = np.asarray(counts.sum(axis=1)).ravel()
    norm = sp.diags(1e4 / np.maximum(total, 1)) @ counts
    adata = ad.AnnData(norm[:, np.flatnonzero(hvg_mask)].tocsr().astype(np.float32))
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=N_PCS, random_state=SEED)
    return adata


def pc_regression_table(pcs, var_ratio, obs, pc_regression) -> pd.DataFrame:
    cols = {}
    totals = {}
    for cov in COVARIATES:
        r2, totals[cov] = pc_regression(pcs, var_ratio, obs[cov].astype(str).to_numpy())
        cols[cov] = r2
    df = pd.DataFrame(cols, index=pd.Index(range(1, len(var_ratio) + 1), name="pc"))
    df.insert(0, "var_ratio", var_ratio)
    df.loc["weighted_total"] = [np.nan, *[totals[c] for c in COVARIATES]]
    return df


def plot_pc_regression(df: pd.DataFrame, path: Path) -> None:
    per_pc = df.drop(index="weighted_total")
    fig, ax = plt.subplots(figsize=(8, 4))
    for cov in COVARIATES:
        ax.plot(
            per_pc.index.astype(int),
            per_pc[cov],
            label=f"{cov} ({df.at['weighted_total', cov]:.3f})",
        )
    ax.set_xlabel("PC")
    ax.set_ylabel("R2")
    ax.legend(title="covariate (weighted total)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_umaps(adata: ad.AnnData, out_dir: Path) -> None:
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    for cov in COVARIATES:
        fig = sc.pl.umap(adata, color=cov, show=False, return_fig=True, size=2)
        fig.savefig(out_dir / f"umap_{cov}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(SHARED)
    obs = adata.obs.copy()
    counts = sp.csr_matrix(adata.layers["counts"]).astype(np.float32)
    genes = adata.var_names
    hvg_mask = adata.var["hvg_2000"].to_numpy()
    del adata

    qc = qc_table(counts, genes, obs)
    qc_summary(qc).to_csv(STRUCT_DIR / "qc_by_batch.csv", index=False)
    plot_qc_violins(qc, STRUCT_DIR)
    comp = composition_table(obs)
    comp.to_csv(STRUCT_DIR / "composition_by_batch.csv")
    plot_composition(comp, STRUCT_DIR / "composition_by_sample.png")

    jac = jaccard_matrix(per_donor_hvgs(counts, genes, obs["donor"].astype(str).to_numpy()))
    jac.to_csv(STRUCT_DIR / "hvg_jaccard_by_donor.csv")
    plot_heatmap(jac, STRUCT_DIR / "hvg_jaccard_by_donor.png")
    print(f"median off-diagonal HVG Jaccard: {median_offdiag(jac):.4f}")

    pca = normalized_pca(counts, hvg_mask)
    pca.obs = obs
    var_ratio = pca.uns["pca"]["variance_ratio"]
    pcr = pc_regression_table(pca.obsm["X_pca"], var_ratio, obs, metrics.pc_regression)
    pcr.to_csv(STRUCT_DIR / "pc_regression.csv")
    plot_pc_regression(pcr, STRUCT_DIR / "pc_regression.png")
    print("variance-weighted PC-regression R2:\n", pcr.loc["weighted_total", COVARIATES])

    plot_umaps(pca, STRUCT_DIR)


if __name__ == "__main__":
    main()
