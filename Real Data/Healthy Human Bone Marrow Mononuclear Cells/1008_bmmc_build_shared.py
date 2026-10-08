"""Build the shared, preprocessed BMMC h5ad that every run and analysis reads.

Steps: GEX features only, cells in a non-excluded lineage, raw counts in layers["counts"] and in
X (Decipher requires integer counts in X), nested HVG flags from one pooled seurat_v3 ranking,
a 90/10 train/val split stratified by sample, and renamed batch/lineage obs columns.

Run: .venv/bin/python "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1008_bmmc_build_shared.py"
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

HERE = Path(__file__).parent
FULL = HERE / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
OUT = HERE / "1008_bmmc"
LINEAGE_MAP = OUT / "stage0" / "lineage_map.csv"
SHARED = OUT / "bmmc_shared.h5ad"
CHECKSUM = OUT / "bmmc_shared.sha256"

# Source obs column -> shared obs column.
RENAME = {"DonorNumber": "donor", "Site": "site", "batch": "sample"}
N_HVG, N_HVG_SMALL, VAL_FRACTION, SEED = 5000, 2000, 0.1, 0


def filter_lineage_cells(
    adata: ad.AnnData, lineage_map: pd.DataFrame, exclude_lineages: tuple[str, ...] = ()
) -> ad.AnnData:
    """Keep cells whose `cell_type` maps to a lineage not `exclude` or in `exclude_lineages`.

    Adds `lineage` and `lineage_rank`.
    """
    mapping = lineage_map.set_index("cell_type")
    unknown = set(adata.obs["cell_type"].astype(str)) - set(mapping.index)
    if unknown:
        raise ValueError(f"cell types missing from the lineage map: {sorted(unknown)}")
    lineage = adata.obs["cell_type"].astype(str).map(mapping["lineage"])
    keep = (~lineage.isin(["exclude", *exclude_lineages])).to_numpy()
    out = adata[keep].copy()
    out.obs["lineage"] = lineage[keep].to_numpy()
    out.obs["lineage_rank"] = (
        out.obs["cell_type"].astype(str).map(mapping["rank"]).astype(int).to_numpy()
    )
    return out


def stratified_split(strata: pd.Series, val_fraction: float, seed: int) -> pd.Series:
    """'train'/'val' per cell, with round(val_fraction * n) val cells in every stratum."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=strata.index, dtype=object)
    for _, idx in strata.groupby(strata, observed=True).indices.items():
        n_val = int(round(val_fraction * len(idx)))
        split.iloc[rng.permutation(idx)[:n_val]] = "val"
    return split


def flag_nested_hvgs(adata: ad.AnnData, n_large: int, n_small: int) -> None:
    """Set var['hvg_<n_large>'] and var['hvg_<n_small>'] (top of the same seurat_v3 ranking)."""
    tmp = ad.AnnData(adata.layers["counts"].astype(np.float32))
    sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=n_large)
    rank = tmp.var["highly_variable_rank"].to_numpy()
    large = tmp.var["highly_variable"].to_numpy()
    small = large & (rank < n_small)
    assert small.sum() == n_small and large.sum() == n_large and not (small & ~large).any()
    adata.var[f"hvg_{n_large}"] = large
    adata.var[f"hvg_{n_small}"] = small


def build_shared(
    adata: ad.AnnData,
    lineage_map: pd.DataFrame,
    n_hvg: int = N_HVG,
    n_hvg_small: int = N_HVG_SMALL,
    exclude_lineages: tuple[str, ...] = (),
) -> ad.AnnData:
    """adata: GEX-only, with layers['counts'] and the source obs columns. Returns the shared file."""
    out = filter_lineage_cells(adata, lineage_map, exclude_lineages)
    out.layers["counts"] = sp.csr_matrix(out.layers["counts"]).astype(np.int32)
    out.X = out.layers["counts"].astype(np.float32)
    out.obs = out.obs.rename(columns=RENAME)
    out.obs["ref_pseudotime"] = out.obs["GEX_pseudotime_order"].astype(float)
    for col in ("donor", "site", "sample", "cell_type"):
        out.obs[col] = out.obs[col].astype(str).astype("category")
    out.obs["split"] = pd.Categorical(
        stratified_split(out.obs["sample"], VAL_FRACTION, SEED), categories=["train", "val"]
    )
    flag_nested_hvgs(out, n_hvg, n_hvg_small)
    return out


def load_gex_lineage_cells(
    lineage_map: pd.DataFrame, exclude_lineages: tuple[str, ...] = ()
) -> ad.AnnData:
    """Stream GEX counts of the kept cells from the legacy-format file (anndata's reader OOMs)."""
    spec = importlib.util.spec_from_file_location("load_bmmc_1002", HERE / "1002_load_bmmc_cite.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    with h5py.File(FULL, "r") as f:
        obs, var = helpers.read_frame(f, "obs"), helpers.read_frame(f, "var")
        gex = (var["feature_types"] == "GEX").to_numpy()
        kept = obs["cell_type"].astype(str).map(lineage_map.set_index("cell_type")["lineage"])
        rows = np.flatnonzero((~kept.isin(["exclude", *exclude_lineages])).to_numpy())
        counts = helpers.read_rows(f, "layers/counts", rows, gex, len(var))
    adata = ad.AnnData(X=counts, obs=obs.iloc[rows].copy(), var=var[gex].copy())
    adata.layers["counts"] = counts
    return adata


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exclude-lineages", nargs="*", default=[], help="lineages to drop")
    exclude = tuple(parser.parse_args().exclude_lineages)
    lineage_map = pd.read_csv(LINEAGE_MAP)
    unknown = set(exclude) - set(lineage_map["lineage"])
    if unknown:
        raise ValueError(f"--exclude-lineages not in the lineage map: {sorted(unknown)}")
    shared = build_shared(
        load_gex_lineage_cells(lineage_map, exclude), lineage_map, exclude_lineages=exclude
    )
    shared.write_h5ad(SHARED)
    CHECKSUM.write_text(f"{sha256_of(SHARED)}  {SHARED.name}\n")
    print(shared)
    print(CHECKSUM.read_text())


if __name__ == "__main__":
    main()
