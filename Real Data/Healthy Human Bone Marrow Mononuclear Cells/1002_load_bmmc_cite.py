"""Unzip, load and inspect the NeurIPS 2021 BMMC CITE-seq h5ad (GSE194122), keep GEX only,
print obs columns plus batch/donor/site counts, and subset to HSC -> erythroid + myeloid.

The file is a legacy-format h5ad (obs columns stored as int codes + obs/__categories) whose X and
layers["counts"] each hold 141M nonzeros (~1.1 GB each in RAM). anndata's reader was OOM-killed on
the 3.6 GB VM, so obs/var are decoded with h5py and the raw-count matrix is streamed in row blocks,
keeping only the lineage cells and GEX columns.

Run: .venv/bin/python "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1002_load_bmmc_cite.py"
"""

import gzip
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
GZ = HERE / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
H5AD = GZ.with_suffix("")  # .h5ad
OUT = HERE / "1002_bmmc_cite_gex_hsc_ery_myeloid.h5ad"

# Cell-type labels (obs["cell_type"]) for the HSC -> erythroid + myeloid lineage.
# Labels not present in the file are reported and skipped.
LINEAGE = [
    "HSC",
    "MK/E prog",
    "Proerythroblast",
    "Erythroblast",
    "Normoblast",
    "G/M prog",
    "CD14+ Mono",
    "CD16+ Mono",
]
CELL_TYPE_COL = "cell_type"
BLOCK = 2000  # rows per streamed block (~25 MB of counts)


def unzip():
    if H5AD.exists():
        print(f"{H5AD.name} already unzipped")
        return
    print(f"unzipping {GZ.name} (streamed, not held in memory) ...")
    with gzip.open(GZ, "rb") as src, open(H5AD, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def read_frame(f, name):
    """Decode a legacy obs/var group (int codes + __categories) into a DataFrame."""
    g = f[name]
    cats = g["__categories"]
    cols = {}
    for k, d in g.items():
        if k in ("__categories", "_index"):
            continue
        v = d[:]
        if k in cats:
            labels = np.array([x.decode() if isinstance(x, bytes) else x for x in cats[k][:]])
            v = pd.Categorical.from_codes(v, labels)
        cols[k] = v
    index = [x.decode() if isinstance(x, bytes) else x for x in g["_index"][:]]
    return pd.DataFrame(cols, index=pd.Index(index, name=None))


def read_rows(f, path, rows, cols, n_vars):
    """Raw counts for `rows` (sorted) and `cols` (bool mask) from a CSR group, block by block.

    Output arrays are preallocated (upper bound = nnz of the selected rows) and filled in place,
    so peak memory is about one copy of the result plus one block.
    """
    g = f[path]
    indptr = g["indptr"][:]
    nnz = np.diff(indptr)[rows].sum()
    data = np.empty(nnz, dtype=np.float32)
    indices = np.empty(nnz, dtype=np.int32)
    new_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    pos = n_done = 0
    for start in range(0, len(indptr) - 1, BLOCK):
        end = min(start + BLOCK, len(indptr) - 1)
        sel = rows[(rows >= start) & (rows < end)] - start
        if len(sel) == 0:
            continue
        lo, hi = indptr[start], indptr[end]
        block = sp.csr_matrix(
            (g["data"][lo:hi], g["indices"][lo:hi], indptr[start : end + 1] - lo),
            shape=(end - start, n_vars),
        )[sel]
        block = block[:, np.flatnonzero(cols)]
        n = block.nnz
        data[pos : pos + n] = block.data
        indices[pos : pos + n] = block.indices
        new_indptr[n_done + 1 : n_done + 1 + len(sel)] = pos + block.indptr[1:]
        pos += n
        n_done += len(sel)
    return sp.csr_matrix(
        (data[:pos], indices[:pos], new_indptr), shape=(len(rows), int(cols.sum()))
    )


def counts(obs, col):
    print(f"\n--- cells per {col} ({obs[col].nunique()} levels) ---")
    print(obs[col].value_counts().sort_index().to_string())


def main():
    unzip()

    with h5py.File(H5AD, "r") as f:
        obs, var = read_frame(f, "obs"), read_frame(f, "var")
        print(f"\nfull file: {len(obs)} cells x {len(var)} features")
        print("layers:", list(f["layers"].keys()))
        print("var columns:", list(var.columns))

        # GEX only (drops ADT surface-protein features).
        print("\nfeature_types:\n" + var["feature_types"].value_counts().to_string())
        gex = (var["feature_types"] == "GEX").values

        print("\n--- obs columns ---")
        print(obs.dtypes.to_string())

        # Batch / donor / site, using whichever of these columns exist.
        cols = [c for c in ("batch", "DonorID", "DonorNumber", "Site") if c in obs.columns]
        for c in cols:
            counts(obs, c)
        if {"Site", "DonorID"} <= set(obs.columns):
            print("\n--- donor x site (cells) ---")
            print(pd.crosstab(obs["DonorID"], obs["Site"]).to_string())
        if {"batch", "Site", "DonorID"} <= set(obs.columns):
            print("\n--- batch -> n distinct (site, donor) ---")
            print(obs.groupby("batch", observed=True)[["Site", "DonorID"]].nunique().to_string())

        print("\npseudotime columns:", [c for c in obs.columns if "pseudotime" in c.lower()])
        counts(obs, CELL_TYPE_COL)

        # Lineage subset.
        present = [c for c in LINEAGE if c in set(obs[CELL_TYPE_COL])]
        print(f"\nlineage labels kept: {present}")
        print(f"lineage labels not in file: {sorted(set(LINEAGE) - set(present))}")
        rows = np.flatnonzero(obs[CELL_TYPE_COL].isin(present).values)

        X = read_rows(f, "layers/counts", rows, gex, len(var))

    sub = ad.AnnData(X=X, obs=obs.iloc[rows].copy(), var=var.loc[gex].copy())
    for c in sub.obs.select_dtypes("category"):
        sub.obs[c] = sub.obs[c].cat.remove_unused_categories()
    print(f"\nsubset: {sub.n_obs} cells x {sub.n_vars} genes (raw counts in .X)")
    for c in cols:
        counts(sub.obs, c)
    if {"Site", "DonorID"} <= set(sub.obs.columns):
        print("\n--- subset donor x site ---")
        print(pd.crosstab(sub.obs["DonorID"], sub.obs["Site"]).to_string())
    print("\n--- subset batch x cell type ---")
    print(pd.crosstab(sub.obs["batch"], sub.obs[CELL_TYPE_COL]).to_string())

    sub.write_h5ad(OUT)
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
