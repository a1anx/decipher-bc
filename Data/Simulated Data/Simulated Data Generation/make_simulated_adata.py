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
import params
from params import SIMUL_PARAMS



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



if __name__ == "__main__":
    shifts = np.array([0.2, 0.4, 0.6])
    adata_folder = "Simulated Adata/v-to-z"
    shift_magnitudes_v_to_z(adata_folder,
                     shifts = shifts,
                     mag = 0.6)
