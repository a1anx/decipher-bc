"""Stage 2 EDA (mixing) for the shared BMMC h5ad: is batch signal measurable beyond cell type?

Reads 1008_bmmc/bmmc_shared.h5ad read-only and writes to 1008_bmmc/eda/mixing/:

- mixing_pca.csv: iLISI, kBET and batch ASW of donor / site / sample on a 50-PC PCA, globally,
  within each lineage and within each cell type (plus site within donor1's cell types), each next
  to a within-cell-type label-permutation null (``*_perm``); cLISI and cell-type ASW as ceiling.
- baseline_pca.csv: the global and within-lineage rows of the above, the table the Stage 5
  model evaluation joins against on (scope, group, metric, variable).
- variance_partition.csv (+ .png): per-HVG fraction of log-expression variance from cell type,
  donor and site (type-II sums of squares, see ``variance_partition``).
- pseudobulk_groups.csv, pseudobulk_top_donor_genes.csv: donor x cell-type pseudobulk contrasts.
- ambient_sex_by_donor.csv: ambient (haemoglobin, platelet, mito) fractions and an X-escape-gene
  score per sample, with the recorded donor sex.

All metrics come from 1008_bmmc_metrics.py. Normalisation and PCA match
1008_bmmc_eda_structure.py (library size 1e4 on all genes, log1p, hvg_2000, 50 PCs, seed 0).
Run: .venv/bin/python "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1008_bmmc_eda_mixing.py"
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
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
MIX_DIR = OUT / "eda" / "mixing"
BATCH_VARS = ["donor", "site", "sample"]
SEED = 0
N_PCS = 50
KBET_K = 50
MIN_GROUP_CELLS = 100  # smallest group a within-group LISI / kBET is computed on
ASW_CELLS_PER_TYPE = 3000  # silhouette is O(n^2) per cell type, so subsample for it
LABEL_ASW_SAMPLE = 10000
PB_MIN_CELLS = 30
PB_MIN_DONORS = 3
PB_MIN_MEAN_CPM = 10.0
PB_TOP_N = 20
HB_GENES = r"^HB[AB]\d*$"  # HBA1, HBA2, HBB
PLATELET_GENES = ["PPBP", "PF4", "GP1BB", "TUBB1", "GNG11", "CAVIN2"]
SEX_GENES = ["XIST", "RPS4Y1", "DDX3Y"]
SEX_REF_CELLTYPE = "CD14+ Mono"  # X-escape score in one cell type so composition can't drive it
X_ESCAPE_GENES = ["KDM5C", "KDM6A", "DDX3X", "EIF1AX", "ZFX", "SMC1A", "USP9X", "PRKX", "JPX"]


def load_metrics() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bmmc_metrics", HERE / "1008_bmmc_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_shared(path: Path) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Raw counts (X), obs and var of the shared h5ad, without loading the duplicate layer."""
    with h5py.File(path, "r") as f:
        counts = ad.io.read_elem(f["X"])
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])
    return sp.csr_matrix(counts), obs, var


# ---------------------------------------------------------------- normalisation / PCA


def normalized_pca(counts: sp.csr_matrix, hvg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cells x 50 PCs, log-normalised HVG matrix) as in 1008_bmmc_eda_structure.py."""
    total = np.asarray(counts.sum(axis=1)).ravel()
    norm = sp.diags(1e4 / np.maximum(total, 1)) @ counts
    adata = ad.AnnData(norm[:, np.flatnonzero(hvg_mask)].tocsr().astype(np.float32))
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=N_PCS, random_state=SEED)
    return adata.obsm["X_pca"], adata.X


# ---------------------------------------------------------------- mixing metrics


def permute_within(labels: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle ``labels`` inside each stratum: keeps each cell type's batch composition and
    removes any batch structure beyond it (the null for 'beyond cell type')."""
    out = labels.copy()
    for s in np.unique(strata):
        ix = np.flatnonzero(strata == s)
        out[ix] = labels[rng.permutation(ix)]
    return out


def asw_subsample(celltype: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices keeping at most ``cap`` random cells of each cell type."""
    keep = []
    for ct in np.unique(celltype):
        ix = np.flatnonzero(celltype == ct)
        keep.append(ix if len(ix) <= cap else rng.choice(ix, cap, replace=False))
    return np.sort(np.concatenate(keep))


def group_mixing(
    metrics: ModuleType,
    emb: np.ndarray,
    batch: np.ndarray,
    celltype: np.ndarray,
    asw_ix: np.ndarray,
) -> dict[str, float]:
    """iLISI median (raw, normalised), kBET acceptance and batch ASW of ``batch`` in one group."""
    n_levels = len(np.unique(batch))
    lisi = metrics.lisi_median(metrics.ilisi(emb, batch), n_levels)
    return {
        "ilisi_median": lisi["median"],
        "ilisi_median_norm": lisi["median_normalized"],
        "kbet_acceptance": metrics.kbet_acceptance(emb, batch, KBET_K),
        "batch_asw": metrics.silhouette_batch(emb[asw_ix], batch[asw_ix], celltype[asw_ix]),
    }


def mixing_rows(
    metrics: ModuleType,
    emb: np.ndarray,
    obs: pd.DataFrame,
    scope: str,
    group_col: str | None,
    variables: list[str],
    seed: int,
) -> list[dict]:
    """Long-format rows of ``group_mixing`` and its permutation null for each group of
    ``group_col`` (None = all cells as one group 'all') and each batch variable."""
    groups = (
        np.full(len(obs), "all") if group_col is None else obs[group_col].astype(str).to_numpy()
    )
    celltype = obs["cell_type"].astype(str).to_numpy()
    rows = []
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        if len(ix) < MIN_GROUP_CELLS:
            continue
        rng = np.random.default_rng(seed)
        asw_ix = asw_subsample(celltype[ix], ASW_CELLS_PER_TYPE, rng)
        for var in variables:
            batch = obs[var].astype(str).to_numpy()[ix]
            n_levels = len(np.unique(batch))
            if n_levels < 2:
                continue
            perm = permute_within(batch, celltype[ix], rng)
            real = group_mixing(metrics, emb[ix], batch, celltype[ix], asw_ix)
            null = group_mixing(metrics, emb[ix], perm, celltype[ix], asw_ix)
            base = {
                "scope": scope,
                "group": g,
                "variable": var,
                "n_cells": len(ix),
                "n_levels": n_levels,
            }
            rows += [{**base, "metric": m, "value": v} for m, v in real.items()]
            rows += [{**base, "metric": f"{m}_perm", "value": v} for m, v in null.items()]
    return rows


def celltype_ceiling_rows(
    metrics: ModuleType, emb: np.ndarray, obs: pd.DataFrame, scope: str, group_col: str | None
) -> list[dict]:
    """cLISI (scib form, 1 - normalised median: 1 = pure) and cell-type ASW per group."""
    groups = (
        np.full(len(obs), "all") if group_col is None else obs[group_col].astype(str).to_numpy()
    )
    celltype = obs["cell_type"].astype(str).to_numpy()
    rows = []
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        n_types = len(np.unique(celltype[ix]))
        if n_types < 2:
            continue
        med = metrics.lisi_median(metrics.clisi(emb[ix], celltype[ix]), n_types)
        asw = metrics.silhouette_label(
            emb[ix], celltype[ix], sample_size=min(LABEL_ASW_SAMPLE, len(ix)), seed=SEED
        )
        base = {
            "scope": scope,
            "group": g,
            "variable": "cell_type",
            "n_cells": len(ix),
            "n_levels": n_types,
        }
        rows += [
            {**base, "metric": "clisi_median", "value": med["median"]},
            {**base, "metric": "clisi_score", "value": 1 - med["median_normalized"]},
            {**base, "metric": "celltype_asw", "value": asw},
        ]
    return rows


# ---------------------------------------------------------------- variance partitioning


def _dummies(labels: np.ndarray) -> np.ndarray:
    """One-hot columns of ``labels`` minus the first level (intercept handled by centring)."""
    return pd.get_dummies(pd.Series(labels)).to_numpy(dtype=float)[:, 1:]


def _explained_ss(design: np.ndarray, y_centered: np.ndarray) -> np.ndarray:
    """Per-column explained sum of squares of the OLS fit of centred ``y`` on ``design``.

    Uses an orthonormal basis of the centred design's column space, so collinear columns
    (donor + site are rank-deficient) are handled without choosing a reference level.
    """
    xc = design - design.mean(axis=0)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    q = u[:, s > s.max() * 1e-8].astype(np.float32)
    proj = q.T @ y_centered
    return (proj.astype(np.float64) ** 2).sum(axis=0)


def variance_partition(y: np.ndarray, factors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-gene variance fractions from cell type, donor and site with type-II sums of squares.

    Donor and site are confounded: every donor but donor1 is at one site, and donor + site
    together span exactly the 12 samples. So each factor's *unique* share is SS(full) minus
    SS(full without it) (type II): site's unique share is identified only by donor1's
    cross-site contrast, donor's by donors at the same site. ``donor_site_confounded`` is the
    batch variance (SS(full) - SS(cell type)) that neither claims uniquely, and
    ``celltype_batch_shared`` the part shared by cell type and batch through composition.
    Columns sum to 1 with ``residual``.
    """
    yc = y - y.mean(axis=0)
    total = (yc.astype(np.float64) ** 2).sum(axis=0)
    d = {k: _dummies(v) for k, v in factors.items()}
    ss = {
        "full": _explained_ss(np.hstack([d["cell_type"], d["donor"], d["site"]]), yc),
        "no_ct": _explained_ss(np.hstack([d["donor"], d["site"]]), yc),
        "no_donor": _explained_ss(np.hstack([d["cell_type"], d["site"]]), yc),
        "no_site": _explained_ss(np.hstack([d["cell_type"], d["donor"]]), yc),
        "ct": _explained_ss(d["cell_type"], yc),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        r = {k: v / total for k, v in ss.items()}
    ct_unique = r["full"] - r["no_ct"]
    donor = r["full"] - r["no_donor"]
    site = r["full"] - r["no_site"]
    return pd.DataFrame(
        {
            "cell_type": ct_unique,
            "celltype_batch_shared": r["ct"] - ct_unique,
            "donor": donor,
            "site": site,
            "donor_site_confounded": r["full"] - r["ct"] - donor - site,
            "residual": 1 - r["full"],
            "batch_total": r["full"] - r["ct"],
        }
    )


def plot_variance_partition(vp: pd.DataFrame, path: Path) -> None:
    cols = [
        "cell_type",
        "celltype_batch_shared",
        "donor",
        "site",
        "donor_site_confounded",
        "batch_total",
        "residual",
    ]
    batch_cols = ["donor", "site", "donor_site_confounded", "batch_total"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[7, 4])
    for ax, cs in zip(axes, [cols, batch_cols]):
        ax.boxplot([vp[c].dropna() for c in cs], showfliers=False)
        ax.set_xticks(range(1, len(cs) + 1), [c.replace("_", "\n") for c in cs], fontsize=8)
    axes[0].set_ylabel("fraction of variance (per HVG)")
    axes[0].set_title("Variance partition of log-normalised HVGs (type-II SS)", fontsize=10)
    axes[1].set_title("Batch components, zoomed", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- pseudobulk / ambient


def pseudobulk(counts: sp.csr_matrix, keys: pd.DataFrame) -> tuple[pd.DataFrame, sp.csr_matrix]:
    """Summed counts per unique row of ``keys``, groups in sorted key order:
    (group table with n_cells, groups x genes)."""
    codes, uniq = pd.factorize(pd.MultiIndex.from_frame(keys.astype(str)), sort=True)
    ind = sp.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes))
    )
    groups = uniq.to_frame(index=False, name=list(keys.columns))
    groups["n_cells"] = np.bincount(codes)
    return groups, sp.csr_matrix(ind @ counts)


def top_donor_genes(sums: np.ndarray, donors: np.ndarray, genes: np.ndarray) -> pd.DataFrame:
    """Genes ranked by across-donor SD of log1p(CPM) in one cell type, among expressed genes."""
    cpm = sums / sums.sum(axis=1, keepdims=True) * 1e6
    keep = cpm.mean(axis=0) >= PB_MIN_MEAN_CPM
    lcpm = np.log1p(cpm[:, keep])
    sd = lcpm.std(axis=0, ddof=1)
    order = np.argsort(-sd)[:PB_TOP_N]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "gene": genes[keep][order],
            "sd_log1p_cpm": sd[order],
            "mean_log1p_cpm": lcpm.mean(axis=0)[order],
            "max_donor": donors[lcpm[:, order].argmax(axis=0)],
            "min_donor": donors[lcpm[:, order].argmin(axis=0)],
            "n_genes_tested": int(keep.sum()),
            "median_sd_all_tested": float(np.median(sd)),
        }
    )


def pseudobulk_contrasts(
    counts: sp.csr_matrix, obs: pd.DataFrame, genes: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Donor x cell-type pseudobulk table and top donor-varying genes per eligible cell type
    (>= PB_MIN_DONORS donors with >= PB_MIN_CELLS cells; donor groups below that are dropped)."""
    groups, sums = pseudobulk(counts, obs[["cell_type", "donor"]])
    groups["total_counts"] = np.asarray(sums.sum(axis=1)).ravel()
    groups["used"] = groups["n_cells"] >= PB_MIN_CELLS
    tops = []
    for ct, g in groups[groups["used"]].groupby("cell_type"):
        if len(g) < PB_MIN_DONORS:
            groups.loc[g.index, "used"] = False
            continue
        top = top_donor_genes(sums[g.index.to_numpy()].toarray(), g["donor"].to_numpy(), genes)
        top.insert(0, "cell_type", ct)
        top.insert(1, "n_donors", len(g))
        tops.append(top)
    return groups, pd.concat(tops, ignore_index=True)


def ambient_sex_table(counts: sp.csr_matrix, obs: pd.DataFrame, genes: pd.Index) -> pd.DataFrame:
    """Per sample: haemoglobin fraction in non-erythroid cells, platelet-gene fraction in
    non-erythroid cells, mito fraction, CPM of each sex gene present, X-escape score
    (mean log1p CPM of the X-escape genes present, in SEX_REF_CELLTYPE cells only, since
    escape genes are expressed ~1.5x higher in females), and the recorded donor sex."""
    groups, sums = pseudobulk(counts, obs[["sample", "donor", "site"]])
    nonery = (obs["lineage"].astype(str) != "erythroid").to_numpy()
    groups_ne, sums_nonery = pseudobulk(
        counts[nonery], obs.loc[nonery, ["sample", "donor", "site"]]
    )
    if not groups_ne["sample"].equals(groups["sample"]):
        raise ValueError("some sample has no non-erythroid cells; cannot align rows")
    ref = (obs["cell_type"].astype(str) == SEX_REF_CELLTYPE).to_numpy()
    groups_ref, sums_ref = pseudobulk(counts[ref], obs.loc[ref, ["sample", "donor", "site"]])
    if not groups_ref["sample"].equals(groups["sample"]):
        raise ValueError(f"some sample has no {SEX_REF_CELLTYPE} cells; cannot align rows")
    total = np.asarray(sums.sum(axis=1)).ravel()
    total_ne = np.asarray(sums_nonery.sum(axis=1)).ravel()

    def frac(mat: sp.csr_matrix, tot: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.asarray(mat[:, np.flatnonzero(mask)].sum(axis=1)).ravel() / tot

    hb = genes.str.match(HB_GENES)
    plt_mask = genes.isin(PLATELET_GENES)
    out = groups.copy()
    out["hb_genes"] = ",".join(genes[hb])
    out["hb_frac_nonerythroid"] = frac(sums_nonery, total_ne, hb)
    out["hb_frac_all"] = frac(sums, total, hb)
    out["platelet_genes"] = ",".join(genes[plt_mask])
    out["platelet_frac_nonerythroid"] = frac(sums_nonery, total_ne, plt_mask)
    out["mito_frac"] = frac(sums, total, genes.str.startswith("MT-"))
    cpm = sums.multiply(1e6 / total[:, None]).tocsc()
    for g in SEX_GENES:
        out[f"cpm_{g}"] = (
            np.asarray(cpm[:, genes.get_loc(g)].todense()).ravel() if g in genes else np.nan
        )
    esc = [g for g in X_ESCAPE_GENES if g in genes]
    out["x_escape_genes"] = ",".join(esc)
    total_ref = np.asarray(sums_ref.sum(axis=1)).ravel()
    cpm_ref = sums_ref[:, genes.get_indexer(esc)].toarray() * (1e6 / total_ref[:, None])
    out["x_escape_score"] = np.log1p(cpm_ref).mean(axis=1)
    sex = obs.groupby("donor", observed=True)["DonorGender"].first().astype(str)
    out["recorded_sex"] = out["donor"].map(sex)
    return out.sort_values(["donor", "site"]).reset_index(drop=True)


# ---------------------------------------------------------------- driver


def main() -> None:
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    counts, obs, var = load_shared(SHARED)
    genes = var.index
    hvg = var["hvg_2000"].to_numpy()
    emb, lognorm_hvg = normalized_pca(counts, hvg)
    print("PCA done", emb.shape, flush=True)

    rows = mixing_rows(metrics, emb, obs, "global", None, BATCH_VARS, SEED)
    rows += mixing_rows(metrics, emb, obs, "lineage", "lineage", BATCH_VARS, SEED)
    rows += mixing_rows(metrics, emb, obs, "cell_type", "cell_type", BATCH_VARS, SEED)
    d1 = (obs["donor"].astype(str) == "donor1").to_numpy()
    rows += mixing_rows(metrics, emb[d1], obs[d1], "donor1_cell_type", "cell_type", ["site"], SEED)
    rows += celltype_ceiling_rows(metrics, emb, obs, "global", None)
    rows += celltype_ceiling_rows(metrics, emb, obs, "lineage", "lineage")
    cols = ["scope", "group", "metric", "variable", "value", "n_cells", "n_levels"]
    mix = pd.DataFrame(rows)[cols]
    mix.to_csv(MIX_DIR / "mixing_pca.csv", index=False)
    base = mix[mix["scope"].isin(["global", "lineage"])].copy()
    base.insert(0, "embedding", "pca")
    base.to_csv(MIX_DIR / "baseline_pca.csv", index=False)
    print("mixing done", flush=True)

    factors = {c: obs[c].astype(str).to_numpy() for c in ["cell_type", "donor", "site"]}
    vp = variance_partition(lognorm_hvg.toarray(), factors)
    vp.insert(0, "gene", genes[hvg])
    vp.to_csv(MIX_DIR / "variance_partition.csv", index=False)
    plot_variance_partition(vp, MIX_DIR / "variance_partition.png")
    del lognorm_hvg
    print("variance partition done", flush=True)

    groups, tops = pseudobulk_contrasts(counts, obs, genes.to_numpy())
    groups.to_csv(MIX_DIR / "pseudobulk_groups.csv", index=False)
    tops.to_csv(MIX_DIR / "pseudobulk_top_donor_genes.csv", index=False)
    ambient_sex_table(counts, obs, genes).to_csv(MIX_DIR / "ambient_sex_by_donor.csv", index=False)
    print("all outputs in", MIX_DIR, flush=True)


if __name__ == "__main__":
    main()
