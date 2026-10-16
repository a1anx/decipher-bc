"""Recreate the scIB human-immune integration-benchmark UMAP figure (cell type / batch / tissue).

Run from the repo root:
    .venv/bin/python "Real Data/scIB_human_immune/1015_recreate_integration_umaps.py"

Stage 1 computes one UMAP per method and caches it in out/umaps.npz; stage 2 only plots.
Re-running skips cached methods (delete out/umaps.npz or pass --recompute to redo them).

The original figure used Scanorama, fastMNN, scANVI, Harmony, DESC, SAUCIE (x2) and scGen.
fastMNN needs R and DESC/SAUCIE/scGen are not installed here, so the 8 panels are instead the
methods available in the repo .venv (Scanorama, Harmony, scANVI, scVI, BBKNN, ComBat), keeping
the scaled/unscaled variants the original labelled. Titles state what was actually run.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
H5AD = HERE / "data" / "Immune_ALL_human.h5ad"
OUT = HERE / "out"
CACHE = OUT / "umaps.npz"
SEED = 0
N_HVG = 2000
BATCH, CELLTYPE, TISSUE = "batch", "final_annotation", "tissue"

# (panel title, key) in figure order: 2 rows x 4 columns
PANELS = [
    ("Scanorama (embed)\nscaled/HVG", "scanorama_scaled"),
    ("Scanorama (embed)\nunscaled/HVG", "scanorama_unscaled"),
    ("scANVI (embed)\nunscaled/HVG", "scanvi"),
    ("Harmony (embed)\nunscaled/HVG", "harmony_unscaled"),
    ("Harmony (embed)\nscaled/HVG", "harmony_scaled"),
    ("scVI (embed)\nunscaled/HVG", "scvi"),
    ("BBKNN (graph)\nscaled/HVG", "bbknn"),
    ("ComBat (gene)\nscaled/HVG", "combat"),
]

# colours read off the original figure's legends
CELLTYPE_COLORS = {
    "CD10+ B cells": "#ffee33", "CD14+ Monocytes": "#22d3ee", "CD16+ Monocytes": "#ff33ee",
    "CD20+ B cells": "#ff4b3e", "CD4+ T cells": "#12853f", "CD8+ T cells": "#1b6aa5",
    "Erythrocytes": "#a3114d", "Erythroid progenitors": "#fbe3ea", "HSPCs": "#8c4a14",
    "Megakaryocyte progenitors": "#1c1ca0", "Monocyte progenitors": "#7ff0b8",
    "Monocyte-derived dendritic cells": "#b8a07a", "NK cells": "#0b5b3c",
    "NKT cells": "#8fa7e8", "Plasma cells": "#8a7a82", "Plasmacytoid dendritic cells": "#6b0f0f",
}
BATCH_COLORS = {
    "10X": "#ffee33", "Freytag": "#22d3ee", "Oetjen_A": "#ff33ee", "Oetjen_P": "#ff4b3e",
    "Oetjen_U": "#12853f", "Sun_sample1_CS": "#1b6aa5", "Sun_sample2_KC": "#a3114d",
    "Sun_sample3_TB": "#fbe3ea", "Sun_sample4_TC": "#8c4a14", "Villani": "#1c1ca0",
}
TISSUE_COLORS = {"Bone_Marrow": "#ffee33", "PBMCs": "#22d3ee"}


def load():
    adata = sc.read_h5ad(H5AD)  # X is log-normalised; layers['counts'] holds raw counts
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key=BATCH, subset=False)
    return adata[:, adata.var["highly_variable"]].copy()


def umap_of(adata, rep=None, graph=False):
    """UMAP coordinates from an obsm embedding (rep) or an existing neighbors graph."""
    if not graph:
        sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)
    return adata.obsm["X_umap"].copy()


def pca_of(adata, scale):
    ad = adata.copy()
    if scale:
        sc.pp.scale(ad, max_value=10)
    sc.pp.pca(ad, n_comps=50, random_state=SEED)
    return ad


def run_method(key, adata):
    import harmonypy
    import scanorama

    if key == "unintegrated":
        ad = pca_of(adata, scale=True)
        return umap_of(ad, "X_pca")

    if key.startswith("scanorama"):
        ad = adata.copy()
        if key.endswith("_scaled"):
            sc.pp.scale(ad, max_value=10)
        order = ad.obs[BATCH].astype(str).values
        names = sorted(set(order))
        idx = [np.where(order == b)[0] for b in names]
        corrected = [ad[i].copy() for i in idx]
        scanorama.integrate_scanpy(corrected, dimred=50)  # in place, returns None
        emb = np.zeros((ad.n_obs, corrected[0].obsm["X_scanorama"].shape[1]), dtype=np.float32)
        for i, c in zip(idx, corrected):
            emb[i] = c.obsm["X_scanorama"]
        ad.obsm["X_emb"] = emb
        return umap_of(ad, "X_emb")

    if key.startswith("harmony"):
        ad = pca_of(adata, scale=key.endswith("_scaled"))
        out = harmonypy.run_harmony(ad.obsm["X_pca"], ad.obs, BATCH, random_state=SEED)
        ad.obsm["X_emb"] = np.asarray(out.Z_corr)
        if ad.obsm["X_emb"].shape[0] != ad.n_obs:
            ad.obsm["X_emb"] = ad.obsm["X_emb"].T
        return umap_of(ad, "X_emb")

    if key == "bbknn":
        import bbknn

        ad = pca_of(adata, scale=True)
        bbknn.bbknn(ad, batch_key=BATCH)
        return umap_of(ad, graph=True)

    if key == "combat":
        ad = adata.copy()
        if hasattr(ad.X, "toarray"):  # scanpy's combat calls the removed scipy `.A`
            ad.X = ad.X.toarray()
        sc.pp.combat(ad, key=BATCH)
        ad = pca_of(ad, scale=True)
        return umap_of(ad, "X_pca")

    if key in ("scvi", "scanvi"):
        import scvi

        scvi.settings.seed = SEED
        ad = adata.copy()
        ad.X = ad.layers["counts"].copy()  # scVI models raw counts on the HVG subset
        scvi.model.SCVI.setup_anndata(ad, batch_key=BATCH)
        model = scvi.model.SCVI(ad, n_latent=30, n_layers=2)
        model.train(max_epochs=40, batch_size=512, accelerator="cpu")
        if key == "scanvi":
            model = scvi.model.SCANVI.from_scvi_model(model, labels_key=CELLTYPE, unlabeled_category="Unknown")
            model.train(max_epochs=20, batch_size=512, accelerator="cpu")
        ad.obsm["X_emb"] = model.get_latent_representation()
        return umap_of(ad, "X_emb")

    raise KeyError(key)


def scatter(ax, xy, codes, colors, order, size):
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[codes[order]], s=size, lw=0, rasterized=True)
    ax.set_axis_off()


def plot(umaps, obs, outdir):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(obs))  # shuffled draw order so no category hides another
    rows = [
        ("Cell Type", "a", CELLTYPE, CELLTYPE_COLORS),
        ("Batch", "b", BATCH, BATCH_COLORS),
        ("tissue", "c", TISSUE, TISSUE_COLORS),
    ]
    fig = plt.figure(figsize=(12.2, 13.4))
    outer = fig.add_gridspec(3, 2, width_ratios=[5.2, 1.4], hspace=0.07, wspace=0.0,
                             left=0.01, right=0.99, top=0.975, bottom=0.01)
    for r, (title, letter, col, cmap) in enumerate(rows):
        cats = list(cmap)
        codes = obs[col].astype(str).map({c: i for i, c in enumerate(cats)}).values
        colors = np.array([cmap[c] for c in cats])
        inner = outer[r, 0].subgridspec(2, 5, width_ratios=[1.25, 1, 1, 1, 1], hspace=0.28, wspace=0.03)
        ax0 = fig.add_subplot(inner[:, 0])
        scatter(ax0, umaps["unintegrated"], codes, colors, order, 1.0)
        ax0.set_title(title, fontsize=15, pad=2)
        ax0.text(-0.12, 1.0, letter, transform=ax0.transAxes, fontsize=8)
        for k, (ptitle, key) in enumerate(PANELS):
            ax = fig.add_subplot(inner[k // 4, 1 + k % 4])
            scatter(ax, umaps[key], codes, colors, order, 0.5)
            ax.set_title(ptitle, fontsize=9.5, pad=1, linespacing=1.15)
        lax = fig.add_subplot(outer[r, 1])
        lax.set_axis_off()
        handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=cmap[c], label=c) for c in cats]
        lax.legend(handles=handles, loc="center left", frameon=False, fontsize=7,
                   handletextpad=0.2, labelspacing=0.75, borderaxespad=0)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"1015_immune_integration_umaps.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    cached = {} if args.recompute or not CACHE.exists() else dict(np.load(CACHE))
    keys = ["unintegrated"] + [k for _, k in PANELS]
    if not args.plot_only and any(k not in cached for k in keys):
        adata = load()
        for k in keys:
            if k in cached:
                continue
            print(f"[{k}] running", flush=True)
            cached[k] = run_method(k, adata)
            np.savez(CACHE, **cached)  # save after each method so a crash loses nothing
    obs = sc.read_h5ad(H5AD, backed="r").obs.copy()
    plot(cached, obs, OUT)
    print("wrote", OUT / "1015_immune_integration_umaps.png")


if __name__ == "__main__":
    main()
