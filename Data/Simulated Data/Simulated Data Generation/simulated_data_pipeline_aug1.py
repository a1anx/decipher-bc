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
    batch_label: str = None,        # what goes in adata.obs["batch"]; see the note at the assignment below
    gene_shift: np.ndarray = None,  # ADDED 2026-08-25 (A1/A2): length n_genes. When given, this
                                    # batch's offset is applied in GENE space after `latent_z @ W`
                                    # and NOTHING is added to z_mean. See the note at `pre_x`.
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
    
    # CHANGED 2026-08-25 -- A1/A2 (batch_shift_review_handoff.md Part 7).
    #
    # Two mutually exclusive places the batch effect can live. gene_shift=None keeps the
    # original z-mode path byte-for-byte, so every existing call is unchanged.
    #
    #   z-mode    (gene_shift is None) -- offset added to z_mean, BEFORE `@ W`.
    #             Reaches gene space only through W's n_z_dims rows, so it is confined to
    #             an n_z_dims-dimensional slice of the n_genes gene space.
    #   gene-mode (gene_shift given)   -- offset added to pre_x, AFTER `@ W`, and NOTHING
    #             is added to z_mean. A1/A2 decided replace, not alongside: running both
    #             confounds the two mechanisms and neither column of the 2x2 in Part 6.4
    #             would mean anything.
    #
    # Consequence worth knowing downstream: in gene-mode `latent_z` carries NO batch
    # offset, so obs["latent_z0..2"] is batch-free and the "true shift removed" ceiling in
    # Part 5.1 has nothing to remove.
    if gene_shift is None:
        # ---- z-mode (original path) ----
        # batch shift: MVN over all n_z_dims
        if shift is None:
            if shift_cov is None:
                shift_cov = np.eye(n_z_dims)               # identity default
            shift = rng.multivariate_normal(np.zeros(n_z_dims), shift_cov)

        # normalize by sqrt(n_z_dims) so σ has the same meaning across dim choices
        shift = shift / np.sqrt(n_z_dims)

        z_mean = z_mean + shift
    else:
        # ---- gene-mode ----
        # z is left batch-free; the offset is applied after the projection, below.
        gene_shift = np.asarray(gene_shift, dtype=float).ravel()
        if gene_shift.shape[0] != n_genes:
            raise ValueError(
                f"gene_shift has length {gene_shift.shape[0]}, expected n_genes={n_genes}"
            )
        shift = np.zeros(n_z_dims)

    # sample z ~ MVN(z_mean, biological_sigma^2 * I)
    latent_z = rng.normal(z_mean, biological_sigma)     # shape (n, n_z_dims)

    # --- x: z → x ---
    W = np.random.default_rng(seed + 1).standard_normal((n_z_dims, n_genes))
    pre_x = latent_z @ W
    pre_x_nobatch = None
    if gene_shift is not None:
        # ADDED 2026-08-25 -- the gene-mode analogue of Part 5.1's "true shift removed"
        # ceiling. In z-mode that ceiling is computed by removing the shift from
        # obs["latent_z0..2"], but in gene-mode latent_z carries NO batch offset (measured:
        # per-batch spread 0.006 against a within-batch std of 0.108), so there is nothing
        # there to remove and that row of the analysis has no analogue as written.
        #
        # Keeping the pre-offset matrix makes the ceiling EXACT rather than approximate.
        # Subtracting gene_shift from the final integer counts downstream would not
        # recover it, because round / min-subtract / clip happen after the offset is added
        # and are not invertible.
        pre_x_nobatch = pre_x.copy()

        # (n_samples, n_genes) + (n_genes,) broadcasts over cells: every cell in this
        # batch gets the same n_genes-long offset. This is the whole of gene-mode.
        pre_x = pre_x + gene_shift

    adata = sc.AnnData(X=pre_x)
    if pre_x_nobatch is not None:
        # carried through ad.concat, then converted to counts in the wrapper
        adata.layers["pre_x_nobatch"] = pre_x_nobatch
    adata.obs["latent_t"]   = latent_t[:, 0]
    # CHANGED 2026-08-25 (A1/A2): in gene-mode `shift` is all zeros by construction, and
    # writing "[0.0, 0.0, 0.0]" here would read as "this batch has no offset" when in fact
    # it has an n_genes-long one. Say so instead; the real offset is in
    # uns["gene_shift_matrix"], which is (n_batches, n_genes) and too large for an obs column.
    # WAS: adata.obs["shift"] = str(np.round(shift, 2).tolist())
    adata.obs["shift"]      = (
        "gene_mode -- see uns['gene_shift_matrix']" if gene_shift is not None
        else str(np.round(shift, 2).tolist())
    )
    # CHANGED 2026-08-25 -- batch label no longer encodes the shift vector.
    #
    # WAS: adata.obs["batch"] = "_".join(f"{s:.2f}" for s in shift)
    #
    # Why: pandas assigns category codes by sorting the label strings ALPHABETICALLY,
    # and batch_shift.weight row b belongs to code b. uns["shift_matrix"] is stored in
    # GENERATION order. With shift-vector labels the two orders disagree -- on
    # sigma0.5_seed0, zero of the five rows lined up, because '-0.09_...' sorts last.
    # Anything comparing batch_shift.weight to shift_matrix row-by-row was silently
    # comparing different batches, which reads as "the mechanism learned nothing".
    # "batch00" < "batch01" < ... sorts into generation order, so code order == gen
    # order and no permutation is ever needed.
    #
    # Nothing is lost: obs["shift"] on the line above already holds the same
    # normalized shift vector, in an easier form to parse than the label was.
    # Training is unaffected either way -- the model never reads shift_matrix, and
    # the codes were always internally consistent. This is a fix for ANALYSIS.
    #
    # batch_label=None keeps the old behaviour, so any direct call to
    # simulate_multivariate that does not pass it is unchanged.
    adata.obs["batch"]      = (
        batch_label if batch_label is not None
        else "_".join(f"{s:.2f}" for s in shift)
    )
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
    batch_mode: str = "z",       # ADDED 2026-08-25 (A1/A2): "z" or "genes". Default "z"
                                 # reproduces every pre-2026-08-25 call exactly.
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

    # CHANGED 2026-08-25 -- A1/A2. Exactly one of shift_matrix / gene_shift_matrix is
    # populated; the other stays None. A2 decided REPLACE, not alongside.
    if batch_mode not in ("z", "genes"):
        raise ValueError(f"batch_mode must be 'z' or 'genes', got {batch_mode!r}")

    if batch_mode == "z":
        # (n_batches, n_z_dims): one shift vector per batch, all drawn together
        # shift distribution is MVN(0, shift_sigma^2 * I) in n_z_dims
        shift_matrix = mag_rng.multivariate_normal(
            mean=np.zeros(n_z_dims),
            cov=(shift_sigma ** 2) * np.eye(n_z_dims), # shift components across dimensions are independent.
            size=n_batches,
        )
        gene_shift_matrix = None
    else:
        # (n_batches, n_genes): a free offset per gene per batch, added after `@ W`.
        # No /sqrt() normalization here -- that existed in z-mode so shift_sigma meant the
        # same thing across n_z_dims choices, and there is no such choice to normalize
        # against in gene space. Scales are already comparable: at shift_sigma=0.5 z-mode
        # produces a per-gene offset with std 0.464 and this draw gives 0.500 (Part 6.4).
        shift_matrix = None
        gene_shift_matrix = mag_rng.normal(0.0, shift_sigma, size=(n_batches, n_genes))

    adata_concat = None
    # i is the batch index. In z-mode `shift` is that batch's raw (un-normalized) z-space
    # vector and gene_shift is None; in gene-mode the reverse.
    for i in range(n_batches):
        shift = None if shift_matrix is None else shift_matrix[i]
        gene_shift = None if gene_shift_matrix is None else gene_shift_matrix[i]
        adata_sim = simulate_multivariate( # this function normalizes by sqrt(n_z_dims) internally
            n_samples=n_samples,
            n_genes=n_genes,
            n_z_dims=n_z_dims,
            shift=shift,
            gene_shift=gene_shift,
            biological_sigma=biological_sigma,
            seed=seed,
            cell_seed=seed + i + 1,
            # ADDED 2026-08-25 -- makes pandas category codes match generation order,
            # so batch_shift.weight[b] and shift_matrix[b] are the same batch.
            # See the note at the obs["batch"] assignment in simulate_multivariate.
            batch_label=f"batch{i:02d}",
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

    # ADDED 2026-08-25: the exact gene-mode ceiling. Identical count conversion applied to
    # the batch-free pre_x, so layers["counts_nobatch"] is what this dataset would have
    # been with gene_shift = 0 and everything else held fixed. Batch silhouette measured
    # on it is the "perfect correction" reference that Part 5.1 gets from latent_z in
    # z-mode. The per-gene min-subtract differs between the two matrices, but that is a
    # per-gene constant and cannot move batch structure.
    if batch_mode == "genes":
        C = np.asarray(adata_concat.layers["pre_x_nobatch"])
        adata_concat.layers["counts_nobatch"] = np.clip(
            np.round(C - C.min(axis=0)), 0, None
        ).astype(int)
        del adata_concat.layers["pre_x_nobatch"]

    # record swept parameters so downstream plots can read them back
    adata_concat.uns["shift_sigma"]  = shift_sigma
    adata_concat.uns["n_z_dims"]     = n_z_dims
    # CHANGED 2026-08-25 (A1/A2): uns cannot hold None, so store only the key that applies,
    # plus batch_mode so downstream code can tell which one to look for. Reading
    # uns["shift_matrix"] unconditionally will now KeyError on a gene-mode file -- that is
    # deliberate and loud, rather than silently handing back a stale or wrong-shaped array.
    adata_concat.uns["batch_mode"]   = batch_mode
    if batch_mode == "z":
        # (n_batches, n_z_dims), raw / pre-normalization. simulate_multivariate divides by
        # sqrt(n_z_dims) internally, so this needs /sqrt(n_z_dims) before comparing it to
        # anything measured off latent_z. See Part 4 verification step 5.
        adata_concat.uns["shift_matrix"] = shift_matrix
    else:
        # (n_batches, n_genes), applied as-is -- no normalization to undo.
        adata_concat.uns["gene_shift_matrix"] = gene_shift_matrix

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

    elif model == 'decipher_vz_add':
        import decipher_vz_add as dc
        from decipher_vz_add.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz_add'

    elif model == 'decipher_vz2_add':
        import decipher_vz2_add as dc
        from decipher_vz2_add.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz2_add'

    elif model == 'decipher_mf_add':
        import decipher_mf_add as dc
        from decipher_mf_add.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_mf_add'

    config = DecipherConfig(learning_rate=1e-3, seed=decipher_seed, dim_z=dim_z)
    dc.tl.decipher_train(adata, config, plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"})

    #Compute ground truths
    dc.tl.cell_clusters(adata, leiden_resolution = 1.0, n_neighbors= 25, seed = 341)
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
                          beta: float = 0.1,
                          notebook_tag: str = None,
                          batch_mode: str = "z",   # ADDED 2026-08-25 (C13b): "z" or
                                                   # "genes", passed straight to the
                                                   # generator. Default leaves 9.7 /
                                                   # 9.8.1 unchanged.
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
            {notebook_tag}_sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad
    with the `{notebook_tag}_` prefix omitted when `notebook_tag` is None.
    Creates the directory tree if it does not exist. Overwrites existing
    files with the same name -- so two sweeps on the same calendar day that
    both leave `notebook_tag` unset will overwrite each other's h5ads.
    Pass a distinct `notebook_tag` per notebook to keep runs separate.
    """

    adata = shift_magnitudes_multivariate_from_normal(
        n_batches=n_batches,
        shift_sigma=shift_sigma,
        n_samples=n_samples,
        n_genes=n_genes,
        n_z_dims=n_z_dims,
        biological_sigma=biological_sigma,
        seed=seed,
        batch_mode=batch_mode,   # ADDED 2026-08-25 (C13b)
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

    elif model == 'decipher_vz_add':
        import decipher_vz_add as dc
        from decipher_vz_add.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz_add'

    elif model == 'decipher_vz2_add':
        import decipher_vz2_add as dc
        from decipher_vz2_add.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz2_add'

    elif model == 'decipher_mf_add':
        import decipher_mf_add as dc
        from decipher_mf_add.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_mf_add'

    # ADDED 2026-08-25 -- B4. The p(x|z,b) arms. No `_add` suffix, because B6 deletes the
    # additive batch_shift table that suffix referred to.
    elif model == 'decipher_zx':
        import decipher_zx as dc
        from decipher_zx.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_zx'

    elif model == 'decipher_zx2':
        import decipher_zx2 as dc
        from decipher_zx2.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_zx2'

    elif model == 'decipher_mf2':
        import decipher_mf2 as dc
        from decipher_mf2.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_mf2'

    else:
        # ADDED 2026-08-25: an unrecognised name used to fall through the chain and die on
        # `NameError: DecipherConfig` below, which reads like an import problem rather than
        # a typo in the model name.
        raise ValueError(
            f"unknown model {model!r}. Expected one of: decipher, decipher_vz, "
            f"decipher_vz2, decipher_mf, decipher_vz_add, decipher_vz2_add, "
            f"decipher_mf_add, decipher_zx, decipher_zx2, decipher_mf2"
        )

    config = DecipherConfig(learning_rate=1e-3, seed=decipher_seed, dim_z=dim_z, beta=beta)
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
    # CHANGED 2026-08-24 -- the h5ad filename carried no notebook tag, so a second sweep on
    # the same calendar day silently overwrote the first one's trained files in place.
    # NOTEBOOK_TAG in the notebooks guards only the CSVs and PNGs, not this path. 9.8.1
    # (beta=0.3) destroyed 6 of 9.7's (beta=0.1) h5ads that way before the remaining 54 were
    # copied to shift_sigma_sweep/0824_9.7_beta0.1/. Pass notebook_tag="9.8.1" to prefix the
    # filename. Default None keeps the old name, so 9.7's existing sweep log -- which stores
    # these paths in its trained_h5ad column -- still resolves.
    # WAS: trained_h5ad_path = os.path.join(
    # WAS:     trained_dir, f"sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad"
    # WAS: )
    prefix = f"{notebook_tag}_" if notebook_tag else ""
    trained_h5ad_path = os.path.join(
        trained_dir, f"{prefix}sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad"
    )
    adata.write(trained_h5ad_path)

    return abs(rho), trained_h5ad_path, r2_overall, r2_per_gene_median, adata


if __name__ == "__main__":

    # ---- sweep ----
    # shift distribution is MVN(0, shift_sigma^2 * I) in n_z_dims
    shift_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
    seeds  = [3, 4]
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
        "Decipher-VZ-Add": "decipher_vz_add",
        "Decipher-VZ2-Add": "decipher_vz2_add",
        "Decipher-MF-Add": "decipher_mf_add",
    }
    colors = {
        "Decipher-VZ": "#2E7D32",
        "Decipher-VZ2": "#1565C0",
        "Decipher-MF": "#F9A825",
        "Base Decipher": "#C62828",
        "Decipher-VZ-Add": "#00897B",
        "Decipher-VZ2-Add": "#7B1FA2",
        "Decipher-MF-Add": "#AD1457",
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