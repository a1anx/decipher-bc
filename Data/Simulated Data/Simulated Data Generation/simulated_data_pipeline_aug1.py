import logging
import pandas as pd
import scanpy as sc
import os
from sklearn.cluster import KMeans
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import scanpy as sc
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import spearmanr

import anndata as ad


from datetime import datetime


def simulate_multivariate(
    n_samples: int = 500,
    n_genes: int = 50,
    n_z_dims: int = 3,
    shift_cov: np.ndarray = None,   # (n_z_dims, n_z_dims) full dim now, not n_z_dims-1
    shift: np.ndarray = None,       # length n_z_dims, applied to all dims
    biological_sigma: float = 0.1,  # biological noise at z level
    seed: int = 0,                  # what is shared ACROSS batches. fixes w_bio (t→z projection) and W (z→x decoder)
    cell_seed: int = 0,             # per-cell / per-batch randomness - latent_t values, noise around z_mean, and the shift draw
):
    """
    Simulate one batch's worth of cells under mirroring Decipher-BC's generative hierarchy:
            v (latent_t)  →  z (n_z_dims)  →  x, 
    with a batch shift applied to z.

    Inputs
    ------
    n_samples : int
        Number of cells to simulate.
    n_genes : int
        Number of genes in the output count matrix.
    n_z_dims : int
        Dimensionality of latent z. Biological signal is projected into all
        n_z_dims; batch shift is applied to all n_z_dims.
    shift_cov : np.ndarray or None, shape (n_z_dims, n_z_dims)
        Covariance for the batch shift draw. Ignored if `shift` is provided.
        Defaults to the identity if both `shift` and `shift_cov` are None.
    shift : np.ndarray or None, shape (n_z_dims,)
        Precomputed raw shift vector for this batch. If provided, no draw
        happens and `shift_cov` is ignored. Normalization by sqrt(n_z_dims)
        is applied inside this function regardless of source.
    biological_sigma : float
        Standard deviation of the per-cell Gaussian noise around z_mean.
    seed : int
        Controls values shared across batches: the biological projection
        w_bio (latent_t to z) and the decoder W (z to x). Passing the same
        `seed` to different batches guarantees identical w_bio and W.
    cell_seed : int
        Controls per-batch, per-cell randomness: latent_t values, per-cell
        Gaussian noise around z_mean, and the shift draw (when `shift` is
        not passed in). Should differ across batches.

    Returns
    -------
    adata : AnnData
        X : (n_samples, n_genes) float array of pre-count expression.
        obs :
            latent_t   — ground-truth pseudotime, shape (n_samples,)
            shift      — string representation of the normalized shift vector
            batch      — string batch label built from the normalized shift
            latent_z0..latent_z{n_z_dims - 1} — ground-truth z per dimension
        uns :
            latent_z_names — list of the latent_z column names
            w_bio          — the biological projection matrix, shape (1, n_z_dims)

    """
    rng = np.random.default_rng(cell_seed)

    # --- v: batch-free pseudotime (1D) ---
    latent_t = rng.uniform(0, 1, size=(n_samples, 1))  # shape (n_samples, 1)

    # linear projection of latent_t into n_dim axes
    proj_rng = np.random.default_rng(seed) # uses seed so that all the batches sees the same physics of how pseudotime maps into the z-space and how z maps to x
    w_bio = proj_rng.standard_normal((1, n_z_dims))  # shape (1, n_z_dims)
    z_mean = latent_t @ w_bio  # shape (n_samples, n_z_dims)
    
    # batch shift: MVN over all n_z_dims 
    if shift is None:
        if shift_cov is None:
            shift_cov = np.eye(n_z_dims)               # identity default
        shift = rng.multivariate_normal(np.zeros(n_z_dims), shift_cov)

    # normalize by sqrt(n_z_dims) so σ has the same meaning across dim choices
    shift = shift / np.sqrt(n_z_dims)
    
    z_mean = z_mean + shift 
    
    # sample z ~ MVN(z_mean, biological_sigma^2 * I)
    latent_z = rng.normal(z_mean, biological_sigma)     # shape (n, n_z_dims)

    # --- x: z → x ---
    W = np.random.default_rng(seed + 1).standard_normal((n_z_dims, n_genes))
    pre_x = latent_z @ W

    adata = sc.AnnData(X=pre_x)
    adata.obs["latent_t"]   = latent_t[:, 0]
    adata.obs["shift"]      = str(np.round(shift, 2).tolist())
    adata.obs["batch"]      = "_".join(f"{s:.2f}" for s in shift)
    for i in range(latent_z.shape[1]):
        adata.obs[f"latent_z{i}"] = latent_z[:, i]
    adata.uns["latent_z_names"] = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata.uns["w_bio"] = w_bio                         # save for reproducibility

    return adata


def shift_magnitudes_multivariate_from_normal(
    n_batches: int = 5,          # number of batches (no separate baseline)
    shift_sigma: float = 0.0,    # std of the per-dim per-batch magnitude draw
    n_samples: int = 500,
    n_genes: int = 200,
    n_z_dims: int = 3,           # match simulate_multivariate default
    biological_sigma: float = 0.1,
    seed: int = 0,
):
    """
    Draw one shift vector per batch and concatenate simulated batches
    into a single AnnData. Wraps simulate_multivariate across n_batches
    with a shared biological projection and decoder.
    
    Inputs
    -------
    Those from simulate_multivariate, plus n_batches and shift_sigma
    n_batches : int
        Number of batches. All batches are drawn from the same distribution;
        no separate zero-shift baseline batch.
    shift_sigma : float
        Standard deviation of each dimension of the per-batch shift draw.
        The sweep knob. Scales the shift covariance as
        shift_sigma squared times the identity.
    
    Returns
    -------
    adata_concat : AnnData
        X : (n_batches * n_samples, n_genes) nonnegative integer count matrix.
        layers :
            counts — copy of X.
        obs : concatenated per-cell fields from all batches; see
            simulate_multivariate for the field list.
        uns :
            shift_sigma  — the value passed in
            shift_matrix — (n_batches, n_z_dims) raw shift matrix,
                        pre-normalization; row i is batch i's shift vector
            n_z_dims     — the value passed in

    The function no longer writes an on-disk copy. If a persisted
    copy is needed, save the return value in the caller.
    
    """
    # dedicated RNG stream so shift draws never collide with the projection
    # stream (`seed`), decoder stream (`seed+1`), or cell streams (`seed+i+1`)
    mag_rng = np.random.default_rng(seed + 1000)

    # (n_batches, n_z_dims): one shift vector per batch, all drawn together
    # shift distribution is MVN(0, shift_sigma^2 * I) in n_z_dims
    shift_matrix = mag_rng.multivariate_normal(
        mean=np.zeros(n_z_dims),
        cov=(shift_sigma ** 2) * np.eye(n_z_dims), # shift components across dimensions are independent. 
        size=n_batches,
    )

    adata_concat = None
    # i is the batch index, shift is the raw shift vector (not normalized by sqrt(n_z_dims))
    for i, shift in enumerate(shift_matrix):
        adata_sim = simulate_multivariate( # this function normalizes by sqrt(n_z_dims) internally
            n_samples=n_samples,
            n_genes=n_genes,
            n_z_dims=n_z_dims,
            shift=shift,    
            biological_sigma=biological_sigma,
            seed=seed,
            cell_seed=seed + i + 1,
        )
        if adata_concat is None:
            adata_concat = adata_sim
        else:
            adata_concat = ad.concat(
                [adata_concat, adata_sim],
                axis=0, join="outer", label=None, merge="same",
            )

    # After all batches concatenated: convert to non-negative integer counts
    X_all = adata_concat.X
    X_all = np.clip(np.round(X_all - X_all.min(axis=0)), 0, None).astype(int)
    adata_concat.X = X_all
    adata_concat.layers["counts"] = X_all.copy()

    # record swept parameters so downstream plots can read them back
    adata_concat.uns["shift_sigma"]  = shift_sigma
    adata_concat.uns["shift_matrix"] = shift_matrix    # raw, pre-normalization
    adata_concat.uns["n_z_dims"]     = n_z_dims

    # # Saving the adata so that we can review the training reconstruction
    # today = datetime.now().strftime("%m%d")
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # untrained_dir = os.path.join(script_dir,"..", "Simulated Adata", "reconstruction_eval", today, "untrained")
    # os.makedirs(untrained_dir, exist_ok=True)
    # untrained_h5ad_path = os.path.join(
    #     untrained_dir, f"sigma{shift_sigma}_seed{seed}_{n_samples}_{n_genes}.h5ad"
    # )
    # adata_concat.write(untrained_h5ad_path)
    # print(f"Saved untrained simulated adata to {untrained_h5ad_path}")

    return adata_concat


def train_and_compute_rho(model, 
                          decipher_seed,
                          n_batches: int = 5,          # number of batches (no separate baseline)
                          shift_sigma: float = 0.0,    # std of the per-dim per-batch magnitude draw
                          n_samples: int = 500,
                          n_genes: int = 50,
                          n_z_dims: int = 3,           # match simulate_multivariate default
                          biological_sigma: float = 0.1,
                          seed: int = 0,
                          dim_z: int = None,            # model's latent Z size; defaults to n_z_dims
                          ):

    """
    Inputs
    ------
    model : {"decipher", "decipher_vz", "decipher_vz2", "decipher_mf"}
        Which Decipher variant to import and train.
    decipher_seed : int
        Random seed for Decipher's SVI training. Independent of the
        simulation seed.
    n_batches, shift_sigma, n_samples, n_genes, n_z_dims, biological_sigma, seed :
        Passed through to shift_magnitudes_multivariate_from_normal.
        See that function's docstring.
    dim_z : int or None
        Size of the model's own latent Z (DecipherConfig.dim_z). Defaults
        to n_z_dims so model capacity matches the simulated ground-truth
        dimensionality; pass explicitly to decouple them for a model-
        capacity study (e.g. dim_z=10 on 3D-simulated data).

    Returns
    -------
    rho : float
        Spearman rank correlation (absolute value) between decipher_time
        and latent_t on cells where decipher_time is not NaN.
    trained_h5ad_path : str
        Absolute path to the written trained AnnData file.
    adata : AnnData
        The trained AnnData with Decipher outputs attached
        (obsm["decipher_v"], obsm["decipher_z"], obs["decipher_time"],
        obs["decipher_clusters"], uns["decipher"], uns["rho"], plus
        simulation ground truth). Same object that was written to disk.

    Side effects
    ------------
    Writes `adata` to:
        <script_dir>/../Simulated Adata/shift_sigma_sweep/<MMDD>/trained/
            sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad
    Creates the directory tree if it does not exist. Overwrites existing
    files with the same name. Consider adding n_z_dims to the filename if
    sweeping across simulation z-dims.
        Returns:
            _type_: _description_
    """
    
    adata = shift_magnitudes_multivariate_from_normal(
        n_batches=n_batches,
        shift_sigma=shift_sigma,
        n_samples=n_samples,
        n_genes=n_genes,
        n_z_dims=n_z_dims,
        biological_sigma=biological_sigma,
        seed=seed
    )

    if dim_z is None:
        dim_z = n_z_dims

    if model == 'decipher_vz':
        import decipher_vz as dc
        from decipher_vz.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz'

    elif model == 'decipher_vz2':
        import decipher_vz2 as dc
        from decipher_vz2.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz2'
    
    elif model == 'decipher_mf':
        import decipher_mf as dc
        from decipher_mf.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_mf'

    elif model == 'decipher':
        import decipher as dc
        from decipher.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher'

    config = DecipherConfig(learning_rate=1e-3, seed=decipher_seed, dim_z=dim_z)
    dc.tl.decipher_train(adata, config, plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"})
    
    #Compute ground truths
    dc.tl.cell_clusters(adata, leiden_resolution = 0.05, n_neighbors= 25, seed = 341)
    filtered = adata.obs["decipher_clusters"].value_counts()>10
    filtered_ids = set(filtered[filtered].index)
    ground_truths = adata.obs.groupby('decipher_clusters')['latent_t'].mean().sort_values().index.to_list()
    ground_truths = [c for c in ground_truths if c in filtered_ids]
    dc.tl.trajectories(adata, dc.tl.TConfig('trajectory', cluster_ids_list=ground_truths))
    dc.tl.decipher_rotate_space(adata) 

    #Compute decipher time
    dc.tl.decipher_time(adata)
    
    # Compute and save rho
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
    adata.uns["rho"] = abs(rho)
    
    # Save the TRAINED adata (has decipher_z, decipher_v, decipher_time) so we
    # never need to re-run decipher_train to get inferred z later.
    # trained_dir = os.path.join(os.path.dirname(gt_h5ad_path), "trained", model_tag)
    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trained_dir = os.path.join(script_dir,"..", "Simulated Adata", "shift_sigma_sweep", today, "trained")
    os.makedirs(trained_dir, exist_ok=True)
    trained_h5ad_path = os.path.join(
        trained_dir, f"sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad"
    )
    adata.write(trained_h5ad_path)

    return abs(rho), trained_h5ad_path, adata


def train_and_compute_rho_r2(model, 
                          decipher_seed,
                          n_batches: int = 5,
                          shift_sigma: float = 0.0,
                          n_samples: int = 500,
                          n_genes: int = 200,
                          n_z_dims: int = 3,
                          biological_sigma: float = 0.1,
                          seed: int = 0,
                          dim_z: int = None,
                          ):
    """
    Inputs
    ------
    model : {"decipher", "decipher_vz", "decipher_vz2", "decipher_mf"}
        Which Decipher variant to import and train.
    decipher_seed : int
        Random seed for Decipher's SVI training. Independent of the
        simulation seed.
    n_batches, shift_sigma, n_samples, n_genes, n_z_dims, biological_sigma, seed :
        Passed through to shift_magnitudes_multivariate_from_normal.
        See that function's docstring.
    dim_z : int or None
        Size of the model's own latent Z (DecipherConfig.dim_z). Defaults
        to n_z_dims so model capacity matches the simulated ground-truth
        dimensionality; pass explicitly to decouple them for a model-
        capacity study (e.g. dim_z=10 on 3D-simulated data).

    Returns
    -------
    rho : float
        Spearman rank correlation (absolute value) between decipher_time
        and latent_t on cells where decipher_time is not NaN.
    trained_h5ad_path : str
        Absolute path to the written trained AnnData file.
    r2_overall : float
        Reconstruction R^2 on log1p scale, pooled across all (cell, gene)
        entries.
    r2_per_gene_median : float
        Median across genes of the per-gene log1p reconstruction R^2.
        NaN genes (zero variance) are excluded.
    adata : AnnData
        The trained AnnData with Decipher outputs attached
        (obsm["decipher_v"], obsm["decipher_z"], obs["decipher_time"],
        obs["decipher_clusters"], uns["decipher"], uns["rho"],
        uns["r2_overall"], uns["r2_per_gene_median"], plus simulation
        ground truth). Same object that was written to disk.

    Side effects
    ------------
    Writes `adata` to:
        <script_dir>/../Simulated Adata/shift_sigma_sweep/<MMDD>/trained/
            sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad
    Creates the directory tree if it does not exist. Overwrites existing
    files with the same name.
    """

    adata = shift_magnitudes_multivariate_from_normal(
        n_batches=n_batches,
        shift_sigma=shift_sigma,
        n_samples=n_samples,
        n_genes=n_genes,
        n_z_dims=n_z_dims,
        biological_sigma=biological_sigma,
        seed=seed
    )

    if dim_z is None:
        dim_z = n_z_dims

    if model == 'decipher_vz':
        import decipher_vz as dc
        from decipher_vz.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz'

    elif model == 'decipher_vz2':
        import decipher_vz2 as dc
        from decipher_vz2.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz2'

    elif model == 'decipher_mf':
        import decipher_mf as dc
        from decipher_mf.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_mf'

    elif model == 'decipher':
        import decipher as dc
        from decipher.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher'

    config = DecipherConfig(learning_rate=1e-3, seed=decipher_seed, dim_z=dim_z)
    decipher, _ = dc.tl.decipher_train(
        adata, config,
        plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"},
    )

    # Reconstruction diagnostic — computed right after training so we still
    # have the in-memory decipher object with train_losses_ / val_losses_.
    r2_result = dc.tl.reconstruction_r2_log1p(decipher, adata)
    r2_overall = r2_result["r2_overall"]
    r2_per_gene_median = float(np.nanmedian(r2_result["r2_per_gene"]))
    adata.uns["r2_overall"] = r2_overall
    adata.uns["r2_per_gene_median"] = r2_per_gene_median

    # Save reconstruction diagnostic figure
    fig, axes = dc.pl.reconstruction_r2_log1p(decipher, adata, figsize=(21, 5))
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(labelsize=8)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(8)
    fig.suptitle(
        f"{model_tag} | sigma={shift_sigma} | decipher seed={decipher_seed}",
        y=1.02, fontsize=11,
    )
    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(
        script_dir, "..", "Simulated Adata", "shift_sigma_sweep", today, "figs",
    )
    os.makedirs(figs_dir, exist_ok=True)
    fig_path = os.path.join(
        figs_dir,
        f"reconstruction_sigma{shift_sigma}_seed{seed}_{model_tag}_decipherseed_{decipher_seed}.png",
    )
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Compute ground truths
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    filtered = adata.obs["decipher_clusters"].value_counts() > 10
    filtered_ids = set(filtered[filtered].index)
    ground_truths = adata.obs.groupby('decipher_clusters')['latent_t'].mean().sort_values().index.to_list()
    ground_truths = [c for c in ground_truths if c in filtered_ids]
    dc.tl.trajectories(adata, dc.tl.TConfig('trajectory', cluster_ids_list=ground_truths))
    dc.tl.decipher_rotate_space(adata)

    # Compute decipher time
    dc.tl.decipher_time(adata)

    # Compute and save rho
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
    adata.uns["rho"] = abs(rho)

    # Save trained adata
    trained_dir = os.path.join(
        script_dir, "..", "Simulated Adata", "shift_sigma_sweep", today, "trained",
    )
    os.makedirs(trained_dir, exist_ok=True)
    trained_h5ad_path = os.path.join(
        trained_dir, f"sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad"
    )
    adata.write(trained_h5ad_path)

    return abs(rho), trained_h5ad_path, r2_overall, r2_per_gene_median, adata


if __name__ == "__main__":

    # ---- sweep ----
    # shift distribution is MVN(0, shift_sigma^2 * I) in n_z_dims
    shift_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
    seeds  = [0, 1, 2]
    decipher_seeds = [1]  # one decipher_seed
    n_z_dims = 3
    n_samples = 500
    n_genes = 200
    biological_sigma = 0.1
    models = {  
        "Decipher-VZ": "decipher_vz",
        "Decipher-VZ2": "decipher_vz2",
        "Decipher-MF": "decipher_mf",
        "Base Decipher": "decipher",
    }
    colors = {
        "Decipher-VZ": "#2E7D32",
        "Decipher-VZ2": "#1565C0",
        "Decipher-MF": "#F9A825",
        "Base Decipher": "#C62828",
    }

    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "..", "Simulated Adata", "shift_sigma_sweep", today)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"sweep_log_{shift_sigmas}_{seeds}.csv")
    
    run_log = []
    for name, model in models.items():
        for shift_sigma in shift_sigmas:
            for sd in seeds:
                for decipher_seed in decipher_seeds:
                    record = {
                        "model": name, "n_z_dims": n_z_dims,
                        "shift_sigma": shift_sigma, "seed": sd,
                        "decipher_seed": decipher_seed,
                    }
                    try:
                        rho, trained_path, r2_overall, r2_per_gene_median, _ = train_and_compute_rho_r2(
                            model, decipher_seed,
                            shift_sigma=shift_sigma,
                            n_samples = n_samples,
                            n_genes = n_genes,
                            biological_sigma=biological_sigma,
                            seed=sd,
                            n_z_dims=n_z_dims,
                        )
                        record.update({
                            "rho": rho, "trained_h5ad": trained_path,
                            "r2_overall": r2_overall,
                            "r2_per_gene_median": r2_per_gene_median,
                            "error": None,
                        })
                    except Exception as e:
                        print(f"[{name}] n_z_dims={n_z_dims} shift_sigma={shift_sigma} "
                              f"seed={sd} decipher_seed={decipher_seed} "
                              f"failed: {type(e).__name__}: {e}")
                        record.update({
                            "rho": np.nan, "trained_h5ad": None,
                            "r2_overall": np.nan, "r2_per_gene_median": np.nan,
                            "error": f"{type(e).__name__}: {e}",
                        })
                    run_log.append(record)
                    pd.DataFrame(run_log).to_csv(csv_path, index=False)