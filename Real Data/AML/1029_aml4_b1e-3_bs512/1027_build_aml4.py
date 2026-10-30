"""Build the shared aml4 h5ad (AML1-4, all cells, all genes) for the aml4 d3 and PyPI-native runs.

    python 1027_build_aml4.py

Run from this folder. Reads `../AML1-4_preprocessed.h5ad` (read only; 64,803 cells x 6,341 genes,
X = raw integer counts stored as float32) and writes `aml4/aml4_shared.h5ad` with
    X, layers["counts"]  the counts as int32 CSR (raises if any value is not an integer)
    obs                  the source's patient, cell_type, cell_type_merged, library, prep_batch,
                         NPM1_mut, plus `batch` (= patient, present levels), `split` (train/val,
                         round(10%) val per patient, seed `--split-seed`) and `stage_rank`
                         (immature 0, blast0 1, ..., blast3 4; NaN for immature_myelo)
    var                  all genes, no HVG mask: both drivers train on every gene
and `aml4_shared.sha256`, `patient_sizes.csv` (cells, val cells and cells per
`cell_type_merged` per patient) and `build_meta.json`.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
SRC_DEFAULT = HERE.parent / "AML1-4_preprocessed.h5ad"
OUT_DEFAULT = HERE / "aml4"
OUT_NAME = "aml4_shared.h5ad"
VAL_FRACTION = 0.1
OBS_SOURCE = ["patient", "cell_type", "cell_type_merged", "library", "prep_batch", "NPM1_mut"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load(HERE / "1027_aml4_common.py", "aml4_common")


def stratified_split(strata: pd.Series, val_fraction: float, seed: int) -> pd.Series:
    """'train'/'val' per cell, with round(val_fraction * n) val cells in every stratum."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=strata.index, dtype=object)
    for idx in strata.groupby(strata, observed=True).indices.values():
        n_val = round(val_fraction * len(idx))
        split.iloc[rng.permutation(idx)[:n_val]] = "val"
    return split


def integer_counts(x) -> sp.csr_matrix:
    """`x` (sparse or dense) as int32 CSR without explicit zeros. Raises if a value is not a
    non-negative integer."""
    csr = sp.csr_matrix(x)
    data = csr.data
    if np.any(data < 0) or np.any(data != np.round(data)):
        bad = data[(data < 0) | (data != np.round(data))]
        raise ValueError(f"{bad.size} counts are not non-negative integers, e.g. {bad[:3]}")
    out = sp.csr_matrix(
        (data.astype(np.int32), csr.indices.copy(), csr.indptr.copy()), shape=csr.shape
    )
    out.eliminate_zeros()
    return out


def build_obs(obs: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    """The source obs columns the runs need plus `batch`, `split` and `stage_rank`."""
    missing = [c for c in OBS_SOURCE if c not in obs.columns]
    if missing:
        raise ValueError(f"source obs lacks {missing}")
    out = obs[OBS_SOURCE].copy()
    out["batch"] = out["patient"].astype(str).astype("category")
    out["split"] = stratified_split(out["patient"], VAL_FRACTION, seed=split_seed).to_numpy()
    out["stage_rank"] = common.stage_rank(out["cell_type_merged"]).to_numpy()
    return out


def patient_sizes(obs: pd.DataFrame) -> pd.DataFrame:
    """Per patient: cells, val cells and cells per `cell_type_merged` level."""
    patient = obs["patient"].astype(str)
    sizes = pd.DataFrame(
        {
            "n_cells": patient.value_counts(),
            "n_val": patient[obs["split"] == "val"].value_counts(),
        }
    )
    per_label = pd.crosstab(patient, obs["cell_type_merged"].astype(str))
    out = sizes.join(per_label).fillna(0).astype(int).sort_index()
    out.index.name = "patient"
    return out.reset_index()


def sha256_file(path: Path, chunk: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import anndata as ad

    t0 = time.time()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--split-seed", type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = ad.read_h5ad(args.src)
    counts = integer_counts(src.X)
    obs = build_obs(src.obs, args.split_seed)
    adata = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=src.var_names.copy()),
        layers={"counts": counts.copy()},
    )
    out = args.out_dir / OUT_NAME
    adata.write_h5ad(out)
    digest = sha256_file(out)
    (args.out_dir / "aml4_shared.sha256").write_text(f"{digest}  {OUT_NAME}\n")
    patient_sizes(obs).to_csv(args.out_dir / "patient_sizes.csv", index=False)
    meta = {
        "source": os.path.relpath(args.src.resolve(), HERE),
        "source_sha256": sha256_file(args.src),
        "n_obs": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "nnz": int(counts.nnz),
        "split_seed": args.split_seed,
        "val_fraction": VAL_FRACTION,
        "n_val": int((obs["split"] == "val").sum()),
        "batch_levels": [str(c) for c in obs["batch"].cat.categories],
        "stage_order": common.STAGE_ORDER,
        "n_unranked": int(obs["stage_rank"].isna().sum()),
        "genes": "all (no HVG subset)",
        "wall_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
