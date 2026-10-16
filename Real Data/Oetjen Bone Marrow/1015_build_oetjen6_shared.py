"""Build a shared Oetjen h5ad for any sample subset in the BMMC shared-file layout.

Steps: QC'd cells of the chosen samples (``--samples``, default the six A, U, J, M, P, G; labels
from ``1015_oetjen_inspect/cell_labels.csv``: scIB labels on A/P/U, kNN-transferred on the rest),
cells in a non-excluded lineage of ``1015_oetjen6/lineage_map.csv``, raw integer counts in X and
layers["counts"], batch-aware seurat_v3 HVG flags (``batch_key="sample"``) and a 90/10 train/val
split stratified by sample (the BMMC builder's helper). ``ref_pseudotime`` is scIB's erythroid DPT
(A/P/U only). Writes ``<out-dir>/<name>_shared.h5ad``, ``<name>_shared.sha256``,
``celltype_by_sample.csv`` and, when the out-dir is not ``1015_oetjen6/``, a copy of the lineage map.

Run from the repo root (default drops lymphoid -> 2-branch HSC -> erythroid + myeloid).
oetjen6 (defaults, writes ``1015_oetjen6/oetjen6_shared.h5ad``):
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1015_build_oetjen6_shared.py"
oetjen5 (P, G, T, W, R):
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1015_build_oetjen6_shared.py" \
        --samples P G T W R --out-dir "Real Data/Oetjen Bone Marrow/1016_oetjen5" --name oetjen5
"""

import argparse
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "1015_oetjen6"
LINEAGE_MAP = OUT / "lineage_map.csv"
SAMPLES = ("A", "U", "J", "M", "P", "G")
N_HVG, N_HVG_SMALL, VAL_FRACTION, SEED = 5000, 2000, 0.1, 0

_spec = importlib.util.spec_from_file_location(
    "bmmc_build_shared",
    HERE.parent / "Healthy Human Bone Marrow Mononuclear Cells" / "1008_bmmc_build_shared.py",
)
bmmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bmmc)


def flag_hvgs(adata: ad.AnnData, batch_key: str, sizes: tuple[int, ...]) -> None:
    """Set var['hvg_<n>'] per size: scanpy seurat_v3 HVGs with ``batch_key``, on raw counts.

    seurat_v3 ranks genes by variance-stabilised variance per batch; with a batch key scanpy
    orders them by median rank across batches, ties broken by how many batches they are HVG in.
    """
    counts = ad.AnnData(adata.layers["counts"].astype(np.float32), obs=adata.obs[[batch_key]])
    counts.var_names = adata.var_names
    for n in sizes:
        res = sc.pp.highly_variable_genes(
            counts, flavor="seurat_v3", n_top_genes=n, batch_key=batch_key, inplace=False
        )
        adata.var[f"hvg_{n}"] = res["highly_variable"].to_numpy()
        adata.var[f"hvg_{n}_nbatches"] = res["highly_variable_nbatches"].to_numpy()


def build_shared(
    adata: ad.AnnData,
    labels: pd.DataFrame,
    lineage_map: pd.DataFrame,
    samples: tuple[str, ...] = SAMPLES,
    exclude_lineages: tuple[str, ...] = ("lymphoid",),
    n_hvg: int = N_HVG,
    n_hvg_small: int = N_HVG_SMALL,
) -> ad.AnnData:
    """adata: full Oetjen counts h5ad; labels: per-QC-cell ``celltype``/``celltype_conf``."""
    missing = set(samples) - set(adata.obs["sample"].astype(str))
    if missing:
        raise ValueError(f"samples not in the data: {sorted(missing)}")
    mapping = lineage_map.set_index("cell_type")
    unknown = set(labels["celltype"]) - set(mapping.index)
    if unknown:
        raise ValueError(f"cell types missing from the lineage map: {sorted(unknown)}")

    lineage = labels["celltype"].map(mapping["lineage"])
    keep_cells = labels.index[~lineage.isin(["exclude", *exclude_lineages])]
    in_samples = adata.obs_names[adata.obs["sample"].astype(str).isin(samples)]
    out = adata[in_samples.intersection(keep_cells, sort=False)].copy()

    out.obs["cell_type"] = pd.Categorical(labels.loc[out.obs_names, "celltype"])
    out.obs["celltype_conf"] = labels.loc[out.obs_names, "celltype_conf"].to_numpy()
    out.obs["lineage"] = out.obs["cell_type"].astype(str).map(mapping["lineage"]).to_numpy()
    out.obs["lineage_rank"] = (
        out.obs["cell_type"].astype(str).map(mapping["rank"]).astype(int).to_numpy()
    )
    out.obs["ref_pseudotime"] = out.obs["scib_dpt_pseudotime"].astype(float)
    for col in ("donor", "sample"):
        out.obs[col] = pd.Categorical(out.obs[col].astype(str), categories=list(samples))
    out.obs = out.obs[
        [
            "donor",
            "sample",
            "age",
            "cell_type",
            "celltype_conf",
            "lineage",
            "lineage_rank",
            "ref_pseudotime",
            "n_counts",
            "n_genes",
        ]
    ]

    out.layers["counts"] = sp.csr_matrix(out.layers["counts"]).astype(np.int32)
    out.X = out.layers["counts"].astype(np.float32)
    out.obs["split"] = pd.Categorical(
        bmmc.stratified_split(out.obs["sample"], VAL_FRACTION, SEED), categories=["train", "val"]
    )
    expressed = np.asarray((out.layers["counts"] > 0).sum(0)).ravel() > 0
    out = out[:, expressed].copy()
    flag_hvgs(out, "sample", (n_hvg, n_hvg_small))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5ad", type=Path, default=HERE / "data" / "oetjen_bm_counts.h5ad")
    parser.add_argument(
        "--labels", type=Path, default=HERE / "1015_oetjen_inspect" / "cell_labels.csv"
    )
    parser.add_argument(
        "--exclude-lineages", nargs="*", default=["lymphoid"], help="lineages to drop"
    )
    parser.add_argument(
        "--samples", nargs="+", default=list(SAMPLES), help="samples to keep, in category order"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT, help="output folder")
    parser.add_argument("--name", default="oetjen6", help="file prefix: <name>_shared.h5ad")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    samples = tuple(args.samples)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lineage_map = pd.read_csv(LINEAGE_MAP)
    unknown = set(args.exclude_lineages) - set(lineage_map["lineage"])
    if unknown:
        raise ValueError(f"--exclude-lineages not in the lineage map: {sorted(unknown)}")
    if out_dir != OUT:
        lineage_map.to_csv(out_dir / "lineage_map.csv", index=False)
    labels = pd.read_csv(args.labels, index_col=0)
    shared = build_shared(
        ad.read_h5ad(args.h5ad),
        labels,
        lineage_map,
        samples=samples,
        exclude_lineages=tuple(args.exclude_lineages),
    )

    path = out_dir / f"{args.name}_shared.h5ad"
    shared.write_h5ad(path)
    sha_path = out_dir / f"{args.name}_shared.sha256"
    sha_path.write_text(f"{bmmc.sha256_of(path)}  {path.name}\n")
    counts = pd.crosstab(shared.obs["cell_type"], shared.obs["sample"], margins=True)
    counts.to_csv(out_dir / "celltype_by_sample.csv")
    print(shared)
    print(counts.to_string())
    print(pd.crosstab(shared.obs["sample"], shared.obs["split"]).to_string())
    print("ref_pseudotime non-NaN:", shared.obs["ref_pseudotime"].notna().sum())
    for n in (N_HVG, N_HVG_SMALL):
        sel = shared.var[f"hvg_{n}"]
        tiers = shared.var.loc[sel, f"hvg_{n}_nbatches"].value_counts().sort_index(ascending=False)
        print(f"hvg_{n}: {sel.sum()} genes; HVG in k of {len(samples)} samples:", tiers.to_dict())
    small, large = shared.var[f"hvg_{N_HVG_SMALL}"], shared.var[f"hvg_{N_HVG}"]
    print(f"hvg_{N_HVG_SMALL} inside hvg_{N_HVG}: {(small & large).sum()} of {small.sum()}")
    print("Wrote", path, "and", sha_path)


if __name__ == "__main__":
    main()
