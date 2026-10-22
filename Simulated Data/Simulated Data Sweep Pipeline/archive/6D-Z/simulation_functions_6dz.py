import logging

import numpy as np
import scanpy as sc

_LOGGER = logging.getLogger(__name__)


def simulate_multivariate(
    n_samples: int = 500,
    n_genes: int = 50,
    n_z_dims: int = 6,
    shift_type: str = "none",   # "vz" | "none"
    shift_cov: np.ndarray = None,   # (n_z_dims-1, n_z_dims-1) covariance for the batch shift draw
    shift: np.ndarray = None,       # precomputed shift vector, length n_z_dims-1 (overrides drawing one)
    biological_sigma: float = 0.1,  # biological noise at z level
    seed: int = 0,
    cell_seed: int = 0,     # controls latent_t, latent_z — different per batch
):
    """
    Minimal linear simulation mirroring Decipher-BC's generative hierarchy:
        v (latent_t)  →  z (n_z_dims)  →  x

    Generalizes simulate_simple2 from a 2D z (1 biological + 1 batch dim) to
    an n_z_dims z, where the batch shift is a vector drawn from a
    multivariate normal and applied to the n_z_dims-1 batch-direction z
    dimensions before sampling. Dim 0 (latent_t/pseudotime) stays batch-free,
    same as simulate_simple2 only ever shifting dim 1.

    Batch can be injected at:
        shift_type="vz"   : v→z step, shift applied to z_mean before sampling
        shift_type="none" : unshifted baseline

    The linear projection W is fixed by seed, so all batches share
    the same W — only the injected shift differs.
    """
    rng = np.random.default_rng(cell_seed)

    # --- v: batch-free pseudotime (1D) ---
    latent_t = rng.uniform(0, 1, size=(n_samples, 1))  # shape (n_cells, 1)

    # --- z: v → z, now n_z_dims ---
    # dim 0 = biological (driven by latent_t)
    # dims 1..n_z_dims-1 = batch-direction (zero in baseline, shifted per batch)
    z_mean = np.concatenate(
        [latent_t, np.zeros((n_samples, n_z_dims - 1))], axis=1
    )  # (n, n_z_dims)

    valid_types = {"vz", "none"}

    if shift_type == "vz":
        if shift is None:
            if shift_cov is None:
                raise ValueError("shift_type='vz' requires shift_cov or shift")
            shift = rng.multivariate_normal(np.zeros(n_z_dims - 1), shift_cov)
        z_mean[:, 1:] += shift                              # shift applied only to batch-direction dims, before sampling
        latent_z = rng.normal(z_mean, biological_sigma)     # shape (n, n_z_dims)
        z_for_decoder = latent_z.copy()
    elif shift_type == "none":
        latent_z = rng.normal(z_mean, biological_sigma)     # shape (n, n_z_dims)
        z_for_decoder = latent_z.copy()
        shift = np.zeros(n_z_dims - 1)
    else:
        raise ValueError(f"shift_type must be one of {valid_types}, got {shift_type!r}")

    # --- x: z → x, W now (n_z_dims, n_genes) ---
    W = np.random.default_rng(seed + 1).standard_normal((n_z_dims, n_genes))
    pre_x = z_for_decoder @ W

    adata = sc.AnnData(X=pre_x)
    adata.obs["latent_t"]   = latent_t[:, 0]
    adata.obs["shift"]      = str(np.round(shift, 2).tolist())
    adata.obs["shift_type"] = shift_type
    adata.obs["batch"]      = "_".join(f"{s:.2f}" for s in shift)
    for i in range(latent_z.shape[1]):
        adata.obs[f"latent_z{i}"] = latent_z[:, i]
    latent_names = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata.uns["latent_z_names"] = latent_names

    return adata
