"""Uncorrected UMAP of an Oetjen shared h5ad, coloured by sample, cell type and more.

Reads ``<dir>/<name>_shared.h5ad`` and writes ``<dir>/umap_uncorrected.png``.
log-normalise -> ``hvg_2000`` -> PCA (50) -> kNN (15) -> UMAP. Cells are drawn in a shuffled
order so no sample is painted over the others.

Run from the repo root. oetjen6 (defaults):
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1015_oetjen6_umap.py"
oetjen5:
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1015_oetjen6_umap.py" \
        --dir "Real Data/Oetjen Bone Marrow/1016_oetjen5" --name oetjen5
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

OUT = Path(__file__).resolve().parent / "1015_oetjen6"
PANELS = ["sample", "cell_type", "lineage", "ref_pseudotime"]
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=OUT, help="folder holding <name>_shared.h5ad")
    parser.add_argument("--name", default="oetjen6", help="file prefix: <name>_shared.h5ad")
    args = parser.parse_args()

    out = args.dir
    adata = ad.read_h5ad(out / f"{args.name}_shared.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=50, mask_var="hvg_2000", random_state=SEED)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)

    order = np.random.default_rng(SEED).permutation(adata.n_obs)
    shuffled = adata[order]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, key in zip(axes.ravel(), PANELS):
        sc.pl.umap(shuffled, color=key, ax=ax, show=False, s=6, sort_order=False)
    n_samples = adata.obs["sample"].nunique()
    fig.suptitle(
        f"Oetjen {n_samples} samples, uncorrected ({adata.n_obs:,} cells, hvg_2000 seurat_v3)"
    )
    fig.savefig(out / "umap_uncorrected.png", dpi=130, bbox_inches="tight")
    print("Saved", out / "umap_uncorrected.png")


if __name__ == "__main__":
    main()
