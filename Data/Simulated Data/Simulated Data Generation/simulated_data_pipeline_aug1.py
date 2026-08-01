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
from simulation_functions import simulation_correlated_shift, simulation_correlated_shift2, simulation_correlated_shift_v_to_z
from simulation_functions import simulate_simple, simulate_simple2
import params
from params import SIMUL_PARAMS


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
    Minimal linear simulation mirroring Decipher-BC's generative hierarchy:
        v (latent_t)  →  z (n_z_dims)  →  x

    Biological signal (latent_t) is projected linearly into ALL n_z_dims
    z-dimensions and z is sampled from MVN(w_bio · t, biological_sigma^2 · I).
    The per-batch shift is drawn from an n_z_dims-dim MVN with covariance
    shift_cov (default: identity), normalized by sqrt(n_z_dims), and added
    to all dims of z_mean before z is sampled.

    Both w_bio and W are fixed by `seed`, so all batches share the same
    projection into z-space and the same z→x decoder — only the injected
    shift differs.
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

