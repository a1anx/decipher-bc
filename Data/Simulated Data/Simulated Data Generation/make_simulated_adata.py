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



parent_dir = Path(__file__).parent.parent / "decipher-batch-correction" / "decipher-bc"
sys.path.insert(0, str(parent_dir))
#from decipher_batch_corrected import DecipherBatchCorrectedConfig
#from train_batch_corrected import train_batch_corrected_decipher, evaluate_batch_correction


_LOGGER = logging.getLogger(__name__)


def shift_magnitudes(
    adata_folder,
    shift_type,
    shifts,
    mag
):
    """
    Compute comprehensive metrics to assess batch mixing quality.
    
    Calls:
    - simulation_correlated_shift from simulations
    - run_methods from simulations

    Metrics:
    - batch_silhouette: Lower = better batch mixing
    - pseudotime_correlation: Higher = better biological preservation
    - cluster_ari: Higher = better biological structure preservation
    - mean_attention_strength: Model's batch correction effort

    Returns:
        adata_concat: concatenated adata of unshifted base data and shifted adatas
    """
    os.makedirs(adata_folder, exist_ok=True)
    shift_vec = mag * shifts
    shift_vec = np.round(mag * shifts, 2)
    
    
    # Unshifted Base (shift=0)
    adata_base = simulation_correlated_shift2(
        n_samples=500,
        n_genes=50,
        seed=0,
        sigma=0.03,
        branch_prob=0.7,
        k_clusters=20,
        hole_size=2,
        n_holes=2,
        hole_density=0.5,
        shift_type="none",
        shift=0,
    )
    ground_truth_clusters = adata_base.obs["cluster_true"]
    ground_truth_cluster_rank = adata_base.uns["cluster_rank"] 
    
    # Run all methods on the dataset
    # latent_spaces = run_methods(adata_base, seed=params.SEED)
    # _LOGGER.info(f"Latent spaces: {latent_spaces}")
    # adata_base.write(f"{adata_folder}/adata_base.h5ad")
    # _LOGGER.info(f"Adata saved: {adata_folder}/adata_base.h5ad")
    
    
    adata_concat = adata_base.copy()

    # Shifted adatas
    for shift in shift_vec:
        adata_sim = simulation_correlated_shift2(
            n_samples=500,
            n_genes=50,
            seed=0,
            sigma=0.03,
            branch_prob=0.7,
            k_clusters=20,
            hole_size=2,
            n_holes=2,
            hole_density=0.5,
            shift_type=shift_type,
            shift=shift,
        )
        # override the clusters
        adata_sim.obs["cluster_true"] = ground_truth_clusters
        adata_sim.uns["cluster_rank"] = ground_truth_cluster_rank
        
        # 4Run all methods on the dataset
        #latent_spaces = run_methods(adata_sim, seed=params.SEED)
        
        adata_concat = ad.concat([adata_concat, adata_sim], axis=0,
                                    join="outer", label=None, merge="same")
        
    shift_vec_str = "_".join([f"{s:.2f}" for s in shift_vec])

    adata_concat.write(os.path.join(adata_folder, f"adata_combined_{shift_type}_{shift_vec_str}.h5ad"))
    _LOGGER.info(f"Combined adata saved: {adata_folder}/adata_combined_{shift_type}_{shift_vec_str}.h5ad")

    return adata_concat

def shift_magnitudes_v_to_z(
    adata_folder,
    shifts,
    mag
):
   
    os.makedirs(adata_folder, exist_ok=True)
    shift_vec = mag * shifts
    shift_vec = np.round(mag * shifts, 2)
    
    # Unshifted Base (shift=0)
    adata_base = simulation_correlated_shift_v_to_z(
        n_samples=500,
        n_genes=50,
        seed=0,
        sigma=0.03,
        branch_prob=0.7,
        k_clusters=20,
        hole_size=2,
        n_holes=2,
        hole_density=0.5,
        shift_z = 0.0,   # delta shift at v→z (latent space)
        shift_x = 0.0,   # delta shift at z→x (expression space)
    )
    ground_truth_clusters = adata_base.obs["cluster_true"]
    ground_truth_cluster_rank = adata_base.uns["cluster_rank"] 
    
    
    adata_concat = adata_base.copy()

    # Shifted adatas
    for shift in shift_vec:
        adata_sim = simulation_correlated_shift_v_to_z(
            n_samples=500,
            n_genes=50,
            seed=0,
            sigma=0.03,
            branch_prob=0.7,
            k_clusters=20,
            hole_size=2,
            n_holes=2,
            hole_density=0.5,
            shift_z = shift,   # delta shift at v→z (latent space)
            shift_x = shift,   # delta shift at z→x (expression space)
        )
        # override the clusters
        adata_sim.obs["cluster_true"] = ground_truth_clusters
        adata_sim.uns["cluster_rank"] = ground_truth_cluster_rank
        
        # 4Run all methods on the dataset
        #latent_spaces = run_methods(adata_sim, seed=params.SEED)
        
        adata_concat = ad.concat([adata_concat, adata_sim], axis=0,
                                    join="outer", label=None, merge="same")
        
    shift_vec_str = "_".join([f"{s:.2f}" for s in shift_vec])

    adata_concat.write(os.path.join(adata_folder, f"v_to_z_{shift_vec_str}.h5ad"))
    _LOGGER.info(f"Combined adata saved: {adata_folder}/v_to_z_{shift_vec_str}.h5ad")

    return adata_concat

def shift_magnitudes_simple(
    #adata_folder: str,
    shift_type: str,          # "vz" | "zx" | "none"
    shifts: np.ndarray,       # unit direction vector, e.g. np.array([1.0])
    mag: float,               # scalar multiplier titrated across experiments
    n_samples: int = 500,
    n_genes: int = 50,
    sigma: float = 0.1,
    seed: int = 0,
):
    """
    Builds a multi-batch AnnData by concatenating:
        - one unshifted baseline  (shift=0, batch="batch_0.00")
        - one per entry in mag * shifts

    Each resulting obs has columns: latent_t, shift, shift_type, batch.
    The 'batch' column is what you pass as batch_key to decipher_train.

    Mirrors the interface of the original shift_magnitudes() function.
    """
    #os.makedirs(adata_folder, exist_ok=True)
    shift_vec = np.round(mag * shifts, 2)

    # Unshifted baseline — cell_seed=seed
    adata_base = simulate_simple2(
        n_samples=n_samples, n_genes=n_genes,
        shift_type="none", shift=0.0,
        sigma=sigma, seed=seed, cell_seed=seed,
    )
    adata_concat = adata_base.copy()

    # Each shifted batch gets its own cell_seed
    for i, shift in enumerate(shift_vec):
        adata_sim = simulate_simple2(
            n_samples=n_samples, n_genes=n_genes,
            shift_type=shift_type, shift=float(shift),
            sigma=sigma, seed=seed, cell_seed=seed + i + 1,
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

    shift_vec_str = "_".join([f"{s:.2f}" for s in shift_vec])
    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if shift_type == "vz":
        adata_folder = os.path.join(script_dir, "..", "Simulated Adata", "v-to-z")
    elif shift_type == "zx":
        adata_folder = os.path.join(script_dir, "..", "Simulated Adata", "z-to-x")
    else:
        adata_folder = os.path.join(script_dir,"..", "Simulated Adata", "none")
    os.makedirs(adata_folder, exist_ok=True)
    out_path = os.path.join(
        adata_folder, f"{today}_{shift_type}_{shift_vec_str}.h5ad"
    )
    adata_concat.write(out_path)
    _LOGGER.info(f"Combined adata saved: {out_path}")

    return adata_concat



if __name__ == "__main__":
    shifts = np.array([0.20, 0.40, 0.60, 0.80])
    shift_magnitudes_simple(shift_type = "vz",          # "vz" | "zx" | "none"
                            shifts = shifts,       # unit direction vector, e.g. np.array([1.0])
                            mag = 1.0)
