import logging
import os
from datetime import datetime

import numpy as np
import anndata as ad

from simulation_functions_6dz import simulate_multivariate

_LOGGER = logging.getLogger(__name__)


def shift_magnitudes_multivariate_from_normal(
    shift_type: str,          # "vz" | "none"
    n_batches: int = 5,         # number of shifted batches (including baseline)
    shift_sigma: float = 0.0,   # std of the per-batch, per-dim magnitude draw  <-- the swept knob
    n_z_dims: int = 6,
    n_samples: int = 500,
    n_genes: int = 50,
    biological_sigma: float = 0.1,
    seed: int = 0,
):
    """
    Each shifted batch draws an (n_z_dims-1)-length shift vector ~ N(0, shift_sigma^2 * I)
    (diagonal covariance — independent batches), applied only to the
    batch-direction z dimensions (dims 1..n_z_dims-1) before sampling. Dim 0
    (latent_t/pseudotime) stays batch-free. Sweeping shift_sigma titrates how
    large a batch effect the model can absorb before it distorts the shared
    trajectory.
    """
    # dedicated RNG stream so batch-magnitude draws never collide with the other seeds (W uses seed+1 and cell_seed uses seed+i+1)
    mag_rng = np.random.default_rng(seed + 1000)
    shift_cov = shift_sigma ** 2 * np.eye(n_z_dims - 1)
    # one (n_z_dims-1)-length shift vector per shifted batch (excluding baseline, so n_batches-1)
    shift_vecs = mag_rng.multivariate_normal(np.zeros(n_z_dims - 1), shift_cov, size=n_batches - 1)
    shift_vecs = np.round(shift_vecs, 2)

    # unshifted baseline
    adata_base = simulate_multivariate(
        n_samples=n_samples, n_genes=n_genes, n_z_dims=n_z_dims,
        shift_type="none",
        biological_sigma=biological_sigma, seed=seed, cell_seed=seed,
    )
    adata_concat = adata_base.copy()

    # Each shifted batch gets its own cell_seed
    for i, shift in enumerate(shift_vecs):
        adata_sim = simulate_multivariate(
            n_samples=n_samples, n_genes=n_genes, n_z_dims=n_z_dims,
            shift_type=shift_type, shift=shift,
            biological_sigma=biological_sigma, seed=seed, cell_seed=seed + i + 1,
        )
        adata_concat = ad.concat(
            [adata_concat, adata_sim],
            axis=0, join="outer", label=None, merge="same",
        )

    # After all batches concatenated
    X_all = adata_concat.X
    X_all = np.clip(np.round(X_all - X_all.min(axis=0)), 0, None).astype(int)
    adata_concat.X = X_all
    adata_concat.layers["counts"] = X_all.copy()

    # record the swept parameter so downstream plots can read it back
    adata_concat.uns["shift_sigma"] = shift_sigma
    adata_concat.uns["shift_vecs"]  = shift_vecs

    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    adata_folder = os.path.join(script_dir, "..", "Simulated Adata", "6dz_shift_sigma_sweep")

    os.makedirs(adata_folder, exist_ok=True)
    out_path = os.path.join(
        adata_folder, f"{today}_{shift_type}_sigma{shift_sigma:.2f}.h5ad"
    )
    adata_concat.write(out_path)
    _LOGGER.info(f"Combined adata saved: {out_path}")

    return adata_concat, out_path


def sweep_shift_sigma_6dz(
    shift_sigmas: np.ndarray,   # e.g. np.linspace(0.0, 3.0, 7)
    shift_type: str = "vz",
    n_batches: int = 5,
    n_z_dims: int = 6,
    n_samples: int = 500,
    n_genes: int = 50,
    sigma: float = 0.1,
    seed: int = 0,
):
    """Run one dataset per batch-effect level. Returns {shift_sigma: (adata, out_path)}."""

    out = {}
    for ss in shift_sigmas:
        out[float(ss)] = shift_magnitudes_multivariate_from_normal(
            shift_type=shift_type, n_batches=n_batches, n_z_dims=n_z_dims,
            shift_sigma=float(ss),
            n_samples=n_samples, n_genes=n_genes,
            biological_sigma=sigma, seed=seed,
        )

    return out


if __name__ == "__main__":
    sweep_shift_sigma_6dz(shift_sigmas=np.linspace(0.0, 3.0, 7))
