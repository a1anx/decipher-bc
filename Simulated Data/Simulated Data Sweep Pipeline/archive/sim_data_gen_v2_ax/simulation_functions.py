import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# REMOVED: torch, torch.nn, torch.nn.functional
#   These were only needed for RandomNet. RandomNet has been replaced by a fixed
#   random linear transformation W, implemented with plain numpy.

# REMOVED: scvi
#   Was imported for the commented-out scVI runs in run_methods(). Not needed in v2.

# REMOVED: sklearn.cluster.KMeans
#   Was used for cluster_true labels on the branching latent_z.
#   Removed with branching (z1/z3/z5 → z_mu = latent_t).

# REMOVED: seaborn
#   Was imported but only used implicitly; not needed in v2 simulation functions.

_LOGGER = logging.getLogger(__name__)

sc.set_figure_params(figsize=[3, 3])


# REMOVED: class RandomNet
#   RandomNet was a 3-layer nonlinear MLP (Linear→ReLU→Linear→ReLU→Linear→Softmax)
#   with fixed random weights, used to decode latent_z_shifted → gene expression counts.
#   It was seeded for reproducibility but its nonlinearity made the mapping from
#   latent space to gene space opaque.
#
#   REPLACED WITH: a fixed random linear transformation W (see simulation_linear_trajectory).
#   W @ latent_z_sampled gives a transparent, interpretable linear map from pseudotime
#   to gene expression, sufficient for testing decoder methods.


def simulation_linear_trajectory(
    n_samples: int = 500,
    n_genes: int = 50,
    seed: int = 0,
    sigma: float = 0.03,
    shift_z: float = 0.0,
) -> sc.AnnData:
    """
    Simplified linear trajectory simulation (v2).

    Generative process:
        latent_t        ~ Uniform[0, 1]                      # pseudotime
        z_mu             = latent_t                           # 1D latent mean (no branching)
        shifted_z_mu     = z_mu + shift_z                    # inject batch shift as bias
        latent_z_sampled ~ N(shifted_z_mu, sigma)            # add biological noise
        adata.X          = clip(latent_z_sampled @ W, 0)     # linear decode to gene space

    Args:
        n_samples : number of cells to simulate
        n_genes   : number of genes (output dimensionality)
        seed      : random seed for reproducibility
        sigma     : std of the latent noise distribution
        shift_z   : batch shift magnitude added to z_mu before sampling

    Returns:
        AnnData with:
            .X                        : (n_samples, n_genes) integer count matrix
            .obs["latent_t"]          : ground-truth pseudotime in [0, 1]
            .obs["shift_z"]           : batch shift applied
            .obsm["latent_z_mu"]      : unshifted latent mean (= latent_t)
            .obsm["shifted_latent_z_mu"] : latent mean after batch shift
            .obsm["latent_z_sampled"] : sampled latent values (noisy, shifted)
            .uns["W"]                 : the linear transformation matrix (1, n_genes)
    """
    np.random.seed(seed)

    # ── Step 1: Sample pseudotime ────────────────────────────────────────────
    # NEW v2: sample latent_t directly from Uniform[0, 1].
    # REMOVED v1: chunk/hole sampling (chunk_size, chunk_offset, chunk_prob, total_size).
    #   Chunk sampling created sparse "holes" in the pseudotime distribution to simulate
    #   uneven cell density along a trajectory. Removed for simplicity.
    latent_t = np.random.uniform(0, 1, size=(n_samples, 1))  # shape: (n_samples, 1)

    # ── Step 2: Define latent mean z_mu = latent_t (1D, no branching) ───────
    # NEW v2: z_mu is simply latent_t, giving a clean 1D linear trajectory.
    # REMOVED v1: branching logic — z1, z3, z5 construction and concatenation:
    #   branching_t1, branching_t2 thresholds
    #   branch_id = Binomial(1, branch_prob) * 2 - 1  (±1 branch assignment)
    #   z1 = latent_t * total_size          (time component)
    #   z3 = (branch_id==1) * clip(...)     (branch A activation)
    #   z5 = branch_id * (latent_t - ...)   (branch direction)
    #   latent_z = concat([z1, z3, z5])     (3D branching latent)
    # These were removed because branching introduces unneeded complexity for
    # testing decoder methods on a simple trajectory.
    z_mu = latent_t.copy()  # shape: (n_samples, 1); z_mu[i] = pseudotime of cell i

    # ── Step 3: Inject batch shift as a bias into z_mu BEFORE sampling ──────
    # NEW v2: batch shift is added to the latent mean before sampling.
    #   This shifts the entire latent distribution of the batch — a cleaner model
    #   of a technical batch effect than post-hoc injection.
    # CHANGED FROM v1 (simulation_correlated_shift_v_to_z):
    #   v1 injected shift_z AFTER sampling: latent_z_sampled[:, 0] += shift_z
    #   v2 injects shift_z INTO the mean:   shifted_z_mu = z_mu + shift_z
    # REMOVED: shift_x (expression-space batch shift, latent_z → X).
    #   v1 also had a second shift at the gene expression level (data = data_clean + shift_x).
    #   Removed in v2 — a single latent-space shift is sufficient and easier to interpret.
    shifted_z_mu = z_mu + shift_z  # shape: (n_samples, 1)

    # ── Step 4: Sample latent_z ~ N(shifted_z_mu, sigma) ────────────────────
    # The noise sigma captures biological cell-to-cell variability around the
    # trajectory mean. Cells at the same pseudotime will have slightly different
    # latent representations.
    latent_z_sampled = np.random.normal(shifted_z_mu, sigma)  # shape: (n_samples, 1)

    # NOTE: REMOVED "normalize dim 0" step from v1 (latent_z_sampled[:, 0] /= total_size).
    #   In v1 this was needed because z1 = latent_t * total_size (range [0, total_size]).
    #   In v2 z_mu = latent_t is already in [0, 1], so normalization is unnecessary.

    # ── Step 5: Linear transformation from latent space to gene expression ───
    # NEW v2: fixed random linear transformation W of shape (1, n_genes).
    #   data_raw = latent_z_sampled @ W   (shape: n_samples × n_genes)
    # REMOVED v1: RandomNet — a 3-layer nonlinear MLP (fc1→ReLU→fc2→ReLU→fc3→Softmax)
    #   followed by Poisson-like scaling (softmax * exp(N(log(10000), 0.1))).
    #   Replaced because the linear map is transparent: each gene's expression is a
    #   known linear function of pseudotime, making ground truth easy to verify.
    #
    # W is seeded separately (seed + 1) so the linear map is independent of the
    # cell sampling above and is fixed across calls with the same seed.
    np.random.seed(seed + 1)
    W = np.random.randn(1, n_genes) * 100  # shape: (1, n_genes); scale for count range ~[0, 300]

    data_raw = latent_z_sampled @ W       # shape: (n_samples, n_genes); linear decode
    data = np.clip(np.round(data_raw), 0, None).astype(int)  # non-negative integer counts

    # ── Build AnnData ────────────────────────────────────────────────────────
    adata_sim = sc.AnnData(data)

    adata_sim.obs["latent_t"] = latent_t[:, 0]          # ground-truth pseudotime
    adata_sim.obs["shift_z"] = shift_z                   # batch shift applied

    adata_sim.obsm["latent_z_mu"] = z_mu                 # unshifted latent mean
    adata_sim.obsm["shifted_latent_z_mu"] = shifted_z_mu # mean after batch shift
    adata_sim.obsm["latent_z_sampled"] = latent_z_sampled # noisy sampled latent

    adata_sim.uns["W"] = W  # store W so the linear map is inspectable

    # REMOVED v1 obs fields: "branch_id", "cluster_true", "latent_z0/1/2"
    #   These were outputs of the branching logic and KMeans clustering, both removed.
    # REMOVED v1 obsm fields: "latent_z" (the 3D branching latent), "latent" (= latent_z_sampled)
    # REMOVED v1 uns fields:  "latent_z_names", "cluster_rank"

    return adata_sim


# REMOVED: simulation_correlated_shift (v1 function)
#   Used chunk/hole trajectory sampling, z1/z3/z5 branching, inject shift AFTER sampling
#   into latent space (latent_z_sampled[:, 0] += shift), then RandomNet decode.
#   Replaced by simulation_linear_trajectory.

# REMOVED: simulation_correlated_shift2 (v1 function)
#   Same as simulation_correlated_shift but injected the shift at gene expression space
#   instead of latent space (data = data_clean + shift_input).
#   Replaced by simulation_linear_trajectory.

# REMOVED: simulation_correlated_shift_v_to_z (v1 function)
#   Extended v1 with two injection points: shift_z (latent space) and shift_x (gene space).
#   Replaced by simulation_linear_trajectory, which uses only shift_z (latent space),
#   injected into the mean before sampling rather than after.

# REMOVED: run_methods
#   Ran UMAP and Decipher on a simulated adata. Removed because it depended on
#   the decipher package and was not part of the data generation pipeline.

# REMOVED: combined_embeddings
#   Plotted latent/UMAP/cluster embeddings. Removed as it depended on run_methods
#   and the now-removed cluster_true obs field.

# REMOVED: run_delta_shift
#   Orchestrated simulation + run_methods for multiple delta values. Removed because
#   it called simulation_correlated_shift (removed above) and run_methods (removed above).
