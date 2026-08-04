"""Plotting for post-training diagnostics."""

import numpy as np
from matplotlib import pyplot as plt

from decipher_vz.tools.diagnostics import reconstruction_r2_log1p as _tl_reconstruction_r2_log1p


def reconstruction_r2_log1p(
    decipher,
    adata,
    recompute=False,
    n_scatter_points=20_000,
    scatter_alpha=0.2,
    scatter_size=1,
    n_hist_bins=30,
    figsize=(15, 4),
    scatter_seed=0,
):
    """Three-panel reconstruction diagnostic: loss curves, x_hat vs x
    scatter, per-gene R^2 histogram.

    On first call, computes the diagnostic via
    dc.tl.reconstruction_r2_log1p and caches it on the decipher object as
    `decipher.reconstruction_r2_result_`. Subsequent calls reuse the
    cache unless `adata` has changed (checked via id(adata)) or
    `recompute=True` is passed.

    Loss curves are read from `decipher.train_losses_` and
    `decipher.val_losses_`, which are populated by decipher_train.

    Parameters
    ----------
    decipher : Decipher
        A trained decipher model.
    adata : sc.AnnData
        The data the model was trained on.
    recompute : bool, default False
        If True, recompute the diagnostic even if a cache is present.
    n_scatter_points : int
        Number of (cell, gene) entries to subsample for the scatter panel.
    scatter_alpha, scatter_size : float
        Matplotlib scatter styling.
    n_hist_bins : int
        Bins for the per-gene R^2 histogram.
    figsize : tuple
        Passed to plt.subplots.
    scatter_seed : int
        Seed for the scatter subsample. Fixed by default so reruns match.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (3,)
    """
    # Cache check
    cached_adata_id = getattr(decipher, "_reconstruction_r2_adata_id", None)
    if recompute or cached_adata_id != id(adata) or not hasattr(
        decipher, "reconstruction_r2_result_"
    ):
        decipher.reconstruction_r2_result_ = _tl_reconstruction_r2_log1p(decipher, adata)
        decipher._reconstruction_r2_adata_id = id(adata)
    result = decipher.reconstruction_r2_result_

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # (1) Loss curves
    ax = axes[0]
    train_losses = getattr(decipher, "train_losses_", [])
    val_losses = getattr(decipher, "val_losses_", [])
    if train_losses:
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="train ELBO (per obs)")
    if val_losses:
        epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(epochs, val_losses, label="val NLL (per obs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training curves")
    if train_losses or val_losses:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no loss curves available",
                ha="center", va="center", transform=ax.transAxes)

    # (2) Reconstruction scatter
    ax = axes[1]
    x_log = result["x_log"]
    x_hat_log = result["x_hat_log"]
    flat_x = x_log.ravel()
    flat_xhat = x_hat_log.ravel()
    rng = np.random.default_rng(scatter_seed)
    idx = rng.choice(
        flat_x.size, size=min(n_scatter_points, flat_x.size), replace=False
    )
    ax.scatter(flat_x[idx], flat_xhat[idx], s=scatter_size, alpha=scatter_alpha)
    lo = float(min(flat_x[idx].min(), flat_xhat[idx].min()))
    hi = float(max(flat_x[idx].max(), flat_xhat[idx].max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("observed log1p(x)")
    ax.set_ylabel("reconstructed log1p(x_hat)")
    ax.set_title(f"reconstruction (overall R^2 = {result['r2_overall']:.3f})")
    ax.legend()

    # (3) Per-gene R^2 histogram
    ax = axes[2]
    r2_per_gene = result["r2_per_gene"]
    valid = r2_per_gene[~np.isnan(r2_per_gene)]
    ax.hist(valid, bins=n_hist_bins)
    median_r2 = float(np.nanmedian(r2_per_gene))
    ax.axvline(
        median_r2, color="r", linestyle="--",
        label=f"median = {median_r2:.3f}",
    )
    ax.set_xlabel("per-gene R^2 (log1p scale)")
    ax.set_ylabel("number of genes")
    ax.set_title("per-gene reconstruction quality")
    ax.legend()

    fig.tight_layout()
    return fig, axes
