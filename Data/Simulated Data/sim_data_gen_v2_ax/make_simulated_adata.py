import logging
import os
from pathlib import Path

import numpy as np
import anndata as ad
import scanpy as sc

# REMOVED: pandas, matplotlib, seaborn, sklearn — were used by run_methods/combined_embeddings
#   which have been removed in v2.
# REMOVED: sys path manipulation for decipher-batch-correction parent dir — not needed in v2.

from simulation_functions import simulation_linear_trajectory
# REMOVED: simulation_correlated_shift, simulation_correlated_shift2,
#          simulation_correlated_shift_v_to_z
#   All three v1 simulation functions have been replaced by simulation_linear_trajectory.

import params
from params import SIMUL_PARAMS

_LOGGER = logging.getLogger(__name__)


def shift_magnitudes_linear_trajectory(
    adata_folder: str,
    shifts: np.ndarray,
    mag: float,
) -> ad.AnnData:
    """
    Generate and save a concatenated AnnData across multiple batch shift magnitudes.

    For each shift value in (mag * shifts), produces one simulated dataset via
    simulation_linear_trajectory and concatenates them with the unshifted base (shift_z=0).

    Args:
        adata_folder : directory to write the output .h5ad file
        shifts       : array of relative shift levels (e.g. [0.2, 0.4, 0.6])
        mag          : scalar multiplier applied to shifts

    Returns:
        adata_concat : concatenated AnnData (base + all shifted datasets)
    """
    os.makedirs(adata_folder, exist_ok=True)

    # Round to 2 decimal places to keep filenames clean
    shift_vec = np.round(mag * shifts, 2)

    # ── Unshifted base (shift_z = 0) ────────────────────────────────────────
    adata_base = simulation_linear_trajectory(
        n_samples=SIMUL_PARAMS['n_samples'],
        n_genes=SIMUL_PARAMS['n_genes'],
        seed=SIMUL_PARAMS['seed'],
        sigma=SIMUL_PARAMS['sigma'],
        shift_z=0.0,
    )
    adata_concat = adata_base.copy()

    # ── Shifted datasets ─────────────────────────────────────────────────────
    for shift in shift_vec:
        adata_sim = simulation_linear_trajectory(
            n_samples=SIMUL_PARAMS['n_samples'],
            n_genes=SIMUL_PARAMS['n_genes'],
            seed=SIMUL_PARAMS['seed'],
            sigma=SIMUL_PARAMS['sigma'],
            shift_z=float(shift),  # batch shift injected into z_mu before sampling
        )

        adata_concat = ad.concat(
            [adata_concat, adata_sim],
            axis=0,
            join="outer",
            label=None,
            merge="same",
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    shift_vec_str = "_".join([f"{s:.2f}" for s in shift_vec])
    out_path = os.path.join(adata_folder, f"linear_traj_{shift_vec_str}.h5ad")
    adata_concat.write(out_path)
    _LOGGER.info(f"Saved: {out_path}")

    return adata_concat


# REMOVED: shift_magnitudes (v1 function)
#   Called simulation_correlated_shift2 with shift_type="delta", injecting the shift
#   at gene expression space (data = data_clean + shift). Also ran KMeans clustering
#   and preserved cluster labels across shifted datasets.
#   Replaced by shift_magnitudes_linear_trajectory.

# REMOVED: shift_magnitudes_v_to_z (v1 function)
#   Called simulation_correlated_shift_v_to_z with both shift_z (latent) and shift_x
#   (expression space). Also preserved KMeans cluster labels across datasets.
#   Replaced by shift_magnitudes_linear_trajectory, which uses only shift_z and
#   removes the cluster tracking (no branching → no meaningful clusters).


if __name__ == "__main__":
    shifts = np.array([0.2, 0.4, 0.6])
    adata_folder = "../Simulated Adata/linear-trajectory"

    adata_concat = shift_magnitudes_linear_trajectory(
        adata_folder=adata_folder,
        shifts=shifts,
        mag=0.6,
    )
    print(f"adata_concat shape: {adata_concat.shape}")
    print(f"shift_z values: {adata_concat.obs['shift_z'].unique().tolist()}")
