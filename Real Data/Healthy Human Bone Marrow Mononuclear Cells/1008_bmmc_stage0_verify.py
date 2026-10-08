"""Stage 0 check of the NeurIPS 2021 BMMC CITE-seq h5ad (GSE194122) before any model run.

Settles the facts `dataset_handoff.md` left unchecked (obs columns, raw-count location, pseudotime
columns, feature types, totals), writes the batch-design tables and the proposed lineage map, and
evaluates the Stage 0 gate (raw counts exist and >= 3 donors have >= 100 cells in each lineage).

obs/var are decoded and counts are read with the h5py helpers of `1002_load_bmmc_cite.py` (the
file is a legacy-format h5ad). No model is trained.

Run: .venv/bin/python "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1008_bmmc_stage0_verify.py"
"""

import importlib.util
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
FULL = HERE / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
SUBSET_1002 = HERE / "1002_bmmc_cite_gex_hsc_ery_myeloid.h5ad"
OUT = HERE / "1008_bmmc" / "stage0"

DONOR, SITE, SAMPLE, CELL_TYPE = "DonorNumber", "Site", "batch", "cell_type"
MIN_DONORS, MIN_CELLS = 3, 100
SAMPLE_ROW_STARTS = (0, 20000, 45000, 70000, 89000)  # integrality check: 500 rows at each
SAMPLE_ROWS = 500

# Proposed lineage map: cell_type -> (lineage, rank within lineage; HSC = 0).
# Any cell type in the file but not listed here is "exclude" (reasons are in the P01 report).
LINEAGE_MAP = {
    "HSC": ("HSC", 0),
    "MK/E prog": ("erythroid", 1),
    "Proerythroblast": ("erythroid", 2),
    "Erythroblast": ("erythroid", 3),
    "Normoblast": ("erythroid", 4),
    "Reticulocyte": ("erythroid", 5),
    "G/M prog": ("myeloid", 1),
    "CD14+ Mono": ("myeloid", 2),
    "CD16+ Mono": ("myeloid", 3),
    "Lymph prog": ("lymphoid", 1),
    "Transitional B": ("lymphoid", 2),
    "Naive CD20+ B IGKC+": ("lymphoid", 3),
    "Naive CD20+ B IGKC-": ("lymphoid", 3),
}
LINEAGES = ["HSC", "erythroid", "myeloid", "lymphoid"]


def load_1002_helpers():
    """Import `1002_load_bmmc_cite.py` (its name starts with a digit, so no plain import)."""
    spec = importlib.util.spec_from_file_location("load_bmmc_1002", HERE / "1002_load_bmmc_cite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obs_summary(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per obs column: dtype, number of levels, NaN count and up to five example values."""
    rows = []
    for col in obs.columns:
        s = obs[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique": s.nunique(),
                "n_nan": int(s.isna().sum()),
                "examples": ", ".join(map(str, s.dropna().unique()[:5])),
            }
        )
    return pd.DataFrame(rows)


def read_row_slice(f: h5py.File, path: str, start: int, stop: int, n_vars: int) -> sp.csr_matrix:
    """Rows [start, stop) of a CSR group, read directly from disk."""
    g = f[path]
    indptr = g["indptr"][start : stop + 1]
    lo, hi = indptr[0], indptr[-1]
    return sp.csr_matrix(
        (g["data"][lo:hi], g["indices"][lo:hi], indptr - lo), shape=(stop - start, n_vars)
    )


def integrality(f: h5py.File, path: str, n_vars: int) -> dict:
    """dtype, integrality, min and max of the stored values on a spread sample of rows."""
    data = np.concatenate(
        [read_row_slice(f, path, s, s + SAMPLE_ROWS, n_vars).data for s in SAMPLE_ROW_STARTS]
    )
    return {
        "matrix": path,
        "dtype": str(data.dtype),
        "rows_checked": SAMPLE_ROWS * len(SAMPLE_ROW_STARTS),
        "all_integer": bool(np.all(data == np.round(data))),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def compare_1002_subset(f: h5py.File, obs: pd.DataFrame, var: pd.DataFrame, helpers) -> dict:
    """Whether the 1002 subset's X equals the full file's layers/counts for the same cells/genes."""
    sub = ad.read_h5ad(SUBSET_1002)
    rows = obs.index.get_indexer(sub.obs_names)
    if (rows < 0).any():
        raise ValueError(f"{(rows < 0).sum()} cells of the 1002 subset are not in the full file")
    gex = (var["feature_types"] == "GEX").values
    if not var.index[gex].equals(sub.var_names):
        raise ValueError("1002 subset genes differ from the full file's GEX features")
    order = np.argsort(rows)
    full = helpers.read_rows(f, "layers/counts", rows[order], gex, len(var))
    sub_x = sp.csr_matrix(sub.X)[order]
    return {
        "n_cells": sub.n_obs,
        "n_genes": sub.n_vars,
        "x_dtype": str(sub.X.dtype),
        "n_mismatched_entries": int((full != sub_x).nnz),
    }


def pseudotime_coverage(obs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per cell type: cells, and for each pseudotime column the non-NaN count and value range."""
    g = obs.groupby(CELL_TYPE, observed=True)
    table = g.size().rename("n_cells").to_frame()
    for c in cols:
        table[f"{c}_n_nonnan"] = g[c].count()
        table[f"{c}_min"] = g[c].min()
        table[f"{c}_max"] = g[c].max()
    return table


def lineage_map_table(cell_types: list[str]) -> pd.DataFrame:
    """cell_type, lineage, rank for every cell type in the file; unmapped types are excluded."""
    rows = [
        (ct, *LINEAGE_MAP[ct]) if ct in LINEAGE_MAP else (ct, "exclude", np.nan)
        for ct in cell_types
    ]
    table = pd.DataFrame(rows, columns=["cell_type", "lineage", "rank"])
    table["rank"] = table["rank"].astype("Int64")
    return table.sort_values(["lineage", "rank", "cell_type"]).reset_index(drop=True)


def gate(donor_by_lineage: pd.DataFrame, lineages: list[str]) -> pd.Series:
    """Per lineage: number of donors with >= MIN_CELLS cells."""
    return (donor_by_lineage[lineages] >= MIN_CELLS).sum()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    helpers = load_1002_helpers()
    with h5py.File(FULL, "r") as f:
        obs, var = helpers.read_frame(f, "obs"), helpers.read_frame(f, "var")
        n_vars = len(var)

        print(f"== 1. obs columns ({obs.shape[1]}) ==")
        print(obs_summary(obs).to_string(index=False))

        print("\n== 2. raw counts ==")
        print("layers:", list(f["layers"].keys()))
        checks = pd.DataFrame(
            [integrality(f, "X", n_vars), integrality(f, "layers/counts", n_vars)]
        )
        print(checks.to_string(index=False))
        print("1002 subset vs full layers/counts:", compare_1002_subset(f, obs, var, helpers))

    pt_cols = [c for c in obs.columns if "pseudotime" in c.lower()]
    print(f"\n== 3. pseudotime-like columns: {pt_cols} ==")
    coverage = pseudotime_coverage(obs, pt_cols)
    coverage.to_csv(OUT / "pseudotime_coverage.csv")
    print(coverage[coverage[[f"{c}_n_nonnan" for c in pt_cols]].sum(axis=1) > 0].to_string())
    for c in pt_cols:
        print(f"{c}: {obs[c].notna().sum()} non-NaN of {len(obs)}")

    print("\n== 4. features ==")
    print(f"var columns: {list(var.columns)}")
    print(var["feature_types"].value_counts().to_string())

    print("\n== 5. totals ==")
    for c in (DONOR, "DonorID", SITE, SAMPLE, "Samplename", CELL_TYPE):
        print(f"{c}: {obs[c].nunique()} levels")
    print(f"cells: {len(obs)}")

    design = (
        obs.groupby([DONOR, "DonorID", SITE, SAMPLE, "Samplename", "is_train"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    design.to_csv(OUT / "donor_site_sample_crosstab.csv", index=False)
    print("\n== donor x site x sample ==")
    print(design.to_string(index=False))
    print(pd.crosstab(obs[DONOR], obs[SITE]).to_string())
    multi_site = obs.groupby(DONOR, observed=True)[SITE].nunique()
    print("donors at > 1 site:", list(multi_site[multi_site > 1].index))

    pd.crosstab(obs[CELL_TYPE], obs[SAMPLE]).to_csv(OUT / "celltype_by_sample.csv")

    lineage_map = lineage_map_table(sorted(obs[CELL_TYPE].unique()))
    lineage_map.to_csv(OUT / "lineage_map.csv", index=False)
    print("\n== lineage map ==")
    print(lineage_map.to_string(index=False))

    lineage = obs[CELL_TYPE].astype(str).map(lineage_map.set_index("cell_type")["lineage"])
    kept = lineage != "exclude"
    donor_by_lineage = pd.crosstab(obs.loc[kept, DONOR], lineage[kept])[LINEAGES]
    donor_by_lineage["Lymph prog only"] = (
        (obs[CELL_TYPE] == "Lymph prog").groupby(obs[DONOR], observed=True).sum()
    )
    print("\n== cells per donor per lineage ==")
    print(donor_by_lineage.to_string())
    print(pd.crosstab(obs.loc[kept, SAMPLE], lineage[kept])[LINEAGES].to_string())

    n_ok = gate(donor_by_lineage, LINEAGES + ["Lymph prog only"])
    print(f"\n== gate: donors with >= {MIN_CELLS} cells ==\n{n_ok.to_string()}")
    counts_ok = bool(checks.set_index("matrix").loc["layers/counts", "all_integer"])
    lineages_ok = n_ok[LINEAGES] >= MIN_DONORS
    print(f"raw counts integer: {counts_ok}; lineages passing: {lineages_ok.to_dict()}")
    print(f"cells after lineage subset: {int(kept.sum())}")
    print(lineage[kept].value_counts().to_string())


if __name__ == "__main__":
    main()
