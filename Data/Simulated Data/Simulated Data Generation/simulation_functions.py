import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#import phate
import scanpy as sc
import scvi
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

import decipher as dc

sc.set_figure_params(figsize=[3, 3])

class RandomNet(nn.Module):
    def __init__(self, n_in, n_out, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_out)
        self.n_out = n_out

        # initialize weights
        nn.init.normal_(self.fc1.weight, 0, 10)
        nn.init.normal_(self.fc1.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc3.bias, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        l = torch.normal(np.log(10_000), 0.1, size=(self.n_out,)).exp()
        x = l * x
        return x


def run_methods(adata, seed=0):
    latent_spaces = []
    
    # ---------- 1) Make a normalized copy for UMAP/PCA/PHATE ----------
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    # optionally: sc.pp.highly_variable_genes, sc.pp.scale, etc.

    # ---------- 2) UMAP on normalized data ----------
    _LOGGER.info("Computing UMAP on normalized data")
    sc.pp.neighbors(adata_norm, random_state=seed)
    sc.tl.umap(adata_norm, random_state=seed)
    adata.obsm["X_umap"] = adata_norm.obsm["X_umap"]
    latent_spaces.append("X_umap")
    _LOGGER.info("Norm UMAP computed")

    # Compute scVIe
    # _LOGGER.info("Computing scVI")
    # scvi.settings.seed = seed
    # scvi.model.SCVI.setup_anndata(adata)
    # sim_model = scvi.model.SCVI(adata, gene_likelihood="nb", n_latent=10)
    # sim_model.train()
    # adata.obsm["X_scVI"] = sim_model.get_latent_representation()
    # latent_spaces.append("X_scVI")
    # _LOGGER.info("scVI computed")

    # # Compute scVI-umap
    # _LOGGER.info("Computing scVI-UMAP")
    # sc.pp.neighbors(adata, use_rep="X_scVI", key_added="scVI_neighbors", random_state=seed)
    # sc.tl.umap(adata, neighbors_key="scVI_neighbors", random_state=seed)
    # adata.obsm["X_scVI_umap"] = adata.obsm["X_umap"]
    # latent_spaces.append("X_scVI_umap")
    # _LOGGER.info("scVI-UMAP computed")

    # # Compute scVI-linear
    # _LOGGER.info("Computing scVI-linear")
    # scvi.settings.seed = seed
    # scvi.model.LinearSCVI.setup_anndata(adata)
    # sim_model = scvi.model.LinearSCVI(adata, gene_likelihood="nb", n_latent=10)
    # sim_model.train()
    # adata.obsm["X_scVI_linear"] = sim_model.get_latent_representation()
    # latent_spaces.append("X_scVI_linear")
    # _LOGGER.info("scVI-linear computed")

    # # Compute scVI-linear-umap
    # _LOGGER.info("Computing scVI-linear-UMAP")
    # sc.pp.neighbors(
    #     adata, use_rep="X_scVI_linear", key_added="scVI_linear_neighbors", random_state=seed
    # )
    # sc.tl.umap(adata, neighbors_key="scVI_linear_neighbors", random_state=seed)
    # adata.obsm["X_scVI_linear_umap"] = adata.obsm["X_umap"]
    # latent_spaces.append("X_scVI_linear_umap")
    # _LOGGER.info("scVI-linear-UMAP computed")

    # # Compute PCA
    # _LOGGER.info("Computing PCA")
    # sc.tl.pca(adata, n_comps=10, random_state=seed)
    # latent_spaces.append("X_pca")
    # _LOGGER.info("PCA computed")

    # # Compute Phate
    # _LOGGER.info("Computing Phate")
    # phate_op = phate.PHATE(n_components=10, random_state=seed)
    # adata.obsm["X_phate"] = phate_op.fit_transform(adata.X)
    # latent_spaces.append("X_phate")
    # _LOGGER.info("Phate computed")
    # #
    # _LOGGER.info("Computing Phate-UMAP")
    # sc.pp.neighbors(adata, use_rep="X_phate", key_added="phate_neighbors", random_state=seed)
    # sc.tl.umap(adata, neighbors_key="phate_neighbors", random_state=seed)
    # adata.obsm["X_phate_umap"] = adata.obsm["X_umap"]
    # latent_spaces.append("X_phate_umap")
    # _LOGGER.info("Phate-UMAP computed")

    configuration = {"dim_v": 2, "dim_z": 10}
    config = dc.tl.DecipherConfig(
        **configuration, n_epochs=1000, early_stopping_patience=10, seed=seed
    )
    a = adata.copy()
    res = dc.tl.decipher_train(a, config, plot_every_k_epochs=-2)
    k = "decipher"
    if config.dim_v > 0:
        adata.obsm[k + "_decipher_v"] = a.obsm["decipher_v"]
        latent_spaces.append(k + "_decipher_v")
    adata.obsm[k + "_decipher_z"] = a.obsm["decipher_z"]
    latent_spaces.append(k + "_decipher_z")

    return latent_spaces


def simulation_correlated_shift(
    n_samples=500,
    n_genes=50,
    seed=0,
    sigma=0.03,
    branch_prob=0.7,
    k_clusters=20,
    hole_size=0,
    n_holes=0,
    hole_density=0.0,
    shift_type: str = "none",
    shift: float = 0.0, # shift
) -> sc.AnnData:
    np.random.seed(seed=seed)

    chunk_size = np.array([1, hole_size] * n_holes + [1])
    chunk_offset = np.cumsum(chunk_size) - chunk_size[0]
    chunk_prob = np.array([1, hole_size * hole_density] * n_holes + [1])
    chunk_prob = chunk_prob / chunk_prob.sum()
    total_size = sum(chunk_size)
    # sample which chunk the latent time is in
    chunk = np.random.choice(np.arange(len(chunk_prob)), size=(n_samples,), p=chunk_prob)
    # sample the latent time within the chunk
    latent_t_chunk = np.random.uniform(0, 1, size=(n_samples,))
    latent_t = latent_t_chunk * chunk_size[chunk] + chunk_offset[chunk]
    latent_t = latent_t / total_size
    latent_t = latent_t[:, None]

    # branching
    branching_t1 = 0.3
    branching_t2 = 0.5
    branch_id = np.random.binomial(1, branch_prob, size=(n_samples, 1)) * 2 - 1
    branch_id = branch_id * (latent_t > branching_t1)

    z1 = latent_t * total_size
    z5 = branch_id * (latent_t - branching_t1)
    z3 = (branch_id == 1) * np.clip(latent_t - 0.3, 0, 0.5)
    z4 = (branch_id == 1) * np.clip(latent_t - branching_t2, 0, branching_t1)
    z2 = (branch_id >= 0) * (np.clip(latent_t - 0.2, 0, 0.5) - np.clip(latent_t - 0.5, 0, 0.5))

    # Create latent_z
    latent_z = np.concatenate([z1, z3, z5], axis=1)
    latent_z_sampled = np.random.normal(latent_z, sigma)
    latent_z_sampled[:, 0] /= total_size
    
    # apply the shift
    if shift_type == "alpha":
        shift_input = shift * (latent_t[:,0])
    elif shift_type == "delta":
        shift_input = shift
    elif shift_type == "none":
        shift_input = 0
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")
        
    latent_z_sampled[:, 0] += shift_input

    net = RandomNet(
        latent_z_sampled.shape[1],
        n_genes,
    )
    data = net(torch.tensor(latent_z_sampled).float()).detach().numpy().astype(int)

    adata_sim = sc.AnnData(data)
    adata_sim.obs["latent_t"] = latent_t
    adata_sim.obs["branch_id"] = branch_id
    adata_sim.obsm["latent_z"] = latent_z
    
    k_means = KMeans(k_clusters)
    k_means.fit(latent_z)
    adata_sim.obs["cluster_true"] = k_means.labels_

    for i in range(latent_z.shape[1]):
        adata_sim.obs[f"latent_z{i}"] = latent_z[:, i]
    adata_sim.obsm["latent"] = latent_z_sampled
    latent_names = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata_sim.uns["latent_z_names"] = latent_names

    # for each cluster, order the other clusters by distance of their kmeans center
    cluster_centers = k_means.cluster_centers_
    cluster_rank = np.argsort(
        np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    )
    adata_sim.uns["cluster_rank"] = cluster_rank[:, 1:]
    
    adata_sim.obs["shift"] = f"{shift}"

    return adata_sim

def simulation_correlated_shift2(
    n_samples=500,
    n_genes=50,
    seed=0,
    sigma=0.03,
    branch_prob=0.7,
    k_clusters=20,
    hole_size=0,
    n_holes=0,
    hole_density=0.0,
    shift_type: str = "none",
    shift: float = 0.0, # shift
) -> sc.AnnData:
    np.random.seed(seed=seed)

    chunk_size = np.array([1, hole_size] * n_holes + [1])
    chunk_offset = np.cumsum(chunk_size) - chunk_size[0]
    chunk_prob = np.array([1, hole_size * hole_density] * n_holes + [1])
    chunk_prob = chunk_prob / chunk_prob.sum()
    total_size = sum(chunk_size)
    # sample which chunk the latent time is in
    chunk = np.random.choice(np.arange(len(chunk_prob)), size=(n_samples,), p=chunk_prob)
    # sample the latent time within the chunk
    latent_t_chunk = np.random.uniform(0, 1, size=(n_samples,))
    latent_t = latent_t_chunk * chunk_size[chunk] + chunk_offset[chunk]
    latent_t = latent_t / total_size
    latent_t = latent_t[:, None]

    # branching
    branching_t1 = 0.3
    branching_t2 = 0.5
    branch_id = np.random.binomial(1, branch_prob, size=(n_samples, 1)) * 2 - 1
    branch_id = branch_id * (latent_t > branching_t1)

    z1 = latent_t * total_size
    z5 = branch_id * (latent_t - branching_t1)
    z3 = (branch_id == 1) * np.clip(latent_t - 0.3, 0, 0.5)
    z4 = (branch_id == 1) * np.clip(latent_t - branching_t2, 0, branching_t1)
    z2 = (branch_id >= 0) * (np.clip(latent_t - 0.2, 0, 0.5) - np.clip(latent_t - 0.5, 0, 0.5))

    # Create latent_z
    latent_z = np.concatenate([z1, z3, z5], axis=1)
    latent_z_sampled = np.random.normal(latent_z, sigma)
    latent_z_sampled[:, 0] /= total_size
    
    
    net = RandomNet(
        latent_z_sampled.shape[1],
        n_genes,
    )
    
    # Decode clean z → gene expression
    data_clean = net(torch.tensor(latent_z_sampled).float()).detach().numpy()

    if shift_type == "delta":
        shift_input = shift  # scalar, broadcasts uniformly over all cells and genes
    elif shift_type == "none":
        shift_input = 0
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")

    data = np.clip(data_clean + shift_input, 0, None).astype(int)

    adata_sim = sc.AnnData(data)
    adata_sim.obs["latent_t"] = latent_t
    adata_sim.obs["branch_id"] = branch_id
    adata_sim.obsm["latent_z"] = latent_z
    
    k_means = KMeans(k_clusters)
    k_means.fit(latent_z)
    adata_sim.obs["cluster_true"] = k_means.labels_

    for i in range(latent_z.shape[1]):
        adata_sim.obs[f"latent_z{i}"] = latent_z[:, i]
    adata_sim.obsm["latent"] = latent_z_sampled
    latent_names = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata_sim.uns["latent_z_names"] = latent_names

    # for each cluster, order the other clusters by distance of their kmeans center
    cluster_centers = k_means.cluster_centers_
    cluster_rank = np.argsort(
        np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    )
    adata_sim.uns["cluster_rank"] = cluster_rank[:, 1:]
    
    adata_sim.obs["shift"] = f"{shift}"

    return adata_sim


def simulation_correlated_shift_v_to_z(
    n_samples=500,
    n_genes=50,
    seed=0,
    sigma=0.03,
    branch_prob=0.7,
    k_clusters=20,
    hole_size=0,
    n_holes=0,
    hole_density=0.0,
    shift_type: str = "none",
    shift_z: float = 0.0,   # delta shift at v→z (latent space)
    shift_x: float = 0.0,   # delta shift at z→x (expression space)
) -> sc.AnnData:
    np.random.seed(seed=seed)


    # ... trajectory construction unchanged ...
    
    chunk_size = np.array([1, hole_size] * n_holes + [1])
    chunk_offset = np.cumsum(chunk_size) - chunk_size[0]
    chunk_prob = np.array([1, hole_size * hole_density] * n_holes + [1])
    chunk_prob = chunk_prob / chunk_prob.sum()
    total_size = sum(chunk_size)
    # sample which chunk the latent time is in
    chunk = np.random.choice(np.arange(len(chunk_prob)), size=(n_samples,), p=chunk_prob)
    # sample the latent time within the chunk
    latent_t_chunk = np.random.uniform(0, 1, size=(n_samples,))
    latent_t = latent_t_chunk * chunk_size[chunk] + chunk_offset[chunk]
    latent_t = latent_t / total_size
    latent_t = latent_t[:, None]

    # branching
    branching_t1 = 0.3
    branching_t2 = 0.5
    branch_id = np.random.binomial(1, branch_prob, size=(n_samples, 1)) * 2 - 1
    branch_id = branch_id * (latent_t > branching_t1)

    z1 = latent_t * total_size
    z5 = branch_id * (latent_t - branching_t1)
    z3 = (branch_id == 1) * np.clip(latent_t - 0.3, 0, 0.5)
    z4 = (branch_id == 1) * np.clip(latent_t - branching_t2, 0, branching_t1)
    z2 = (branch_id >= 0) * (np.clip(latent_t - 0.2, 0, 0.5) - np.clip(latent_t - 0.5, 0, 0.5))

    # Create latent_z
    latent_z = np.concatenate([z1, z3, z5], axis=1)
    latent_z_sampled = np.random.normal(latent_z, sigma)
    latent_z_sampled[:, 0] /= total_size

    # ── Injection 1: v→z — uniform offset, batch-independent of pseudotime ──
    latent_z_sampled[:, 0] += shift_z

    net = RandomNet(latent_z_sampled.shape[1], n_genes)
    data_clean = net(torch.tensor(latent_z_sampled).float()).detach().numpy()

    # ── Injection 2: z→x — uniform offset across all cells and genes ──
    data = np.clip(data_clean + shift_x, 0, None).astype(int)


    adata_sim = sc.AnnData(data)
    adata_sim.obs["latent_t"] = latent_t
    adata_sim.obs["branch_id"] = branch_id
    adata_sim.obsm["latent_z"] = latent_z
    
    k_means = KMeans(k_clusters)
    k_means.fit(latent_z)
    adata_sim.obs["cluster_true"] = k_means.labels_

    for i in range(latent_z.shape[1]):
        adata_sim.obs[f"latent_z{i}"] = latent_z[:, i]
    adata_sim.obsm["latent"] = latent_z_sampled
    latent_names = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata_sim.uns["latent_z_names"] = latent_names

    # for each cluster, order the other clusters by distance of their kmeans center
    cluster_centers = k_means.cluster_centers_
    cluster_rank = np.argsort(
        np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    )
    adata_sim.uns["cluster_rank"] = cluster_rank[:, 1:]
    
    adata_sim.obs["shift"] = f"z={shift_z},x={shift_x}"

    return adata_sim



def combined_embeddings(
    adata_sim,
    out_folder,
    delta):
    fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(4, 3*3)
        )

    # --- Row 1 ---
    sc.pl.embedding(
        adata_sim,
        basis="latent",
        color="latent_t",
        show=False,
        ax=axes[0],
        size=30
    )

    #--- Row 2 ---
    sc.pl.embedding(
        adata_sim,
        basis="X_umap",
        color="latent_t",
        show=False,
        ax=axes[1],
        size=30
    )
    #--- Row 3 ---
    sc.pl.embedding(
        adata_sim,
        basis="X_umap",
        color="cluster_true",
        show=False,
        ax=axes[2],
        size=30
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_folder, f"combined_embeddings_{delta}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    _LOGGER.info(f"Plot saved: {out_folder}/combined_embeddings_{delta}.png")


def run_delta_shift(
    n_samples: int = 1000,
    n_genes: int = 50,
    sigma: float = 0.05,
    branch_prob: float = 0.7,
    k_clusters: int = 20,
    hole_size: int = 1,
    n_holes: int = 3,
    hole_density: float = 0.05,
    seed: int = 0,
    out_folder: str = "",
    adata_folder: str = "",
    delta_vec: np.ndarray = np.array([0.0]), # slope of the linear shift,
):
    """
    Implements the TA's suggestion:

    - Simulate synthetic data (delta = 0) using simulation_correlated_1
    - Create a delta-shifted version by shifting only the stored latent
    - Run run_methods on each
    - Compute ARI for each method vs ground-truth cluster latent (cluster_true)
    - Plot:
        * true vs shifted latent space
        * UMAP and Decipher visible embeddings
    - Compute an MST on Decipher's visible embedding as a simple trajectory
    """
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(adata_folder, exist_ok=True)
    
    # Unshifted Base (delta=0)
    adata_base = simulation_correlated_shift(
        n_samples=n_samples,
        n_genes=n_genes,
        seed=seed,
        sigma=sigma,
        branch_prob=branch_prob,
        k_clusters=k_clusters,
        hole_size=hole_size,
        n_holes=n_holes,
        hole_density=hole_density,
        delta=0,
    )
    ground_truth_clusters = adata_base.obs["cluster_true"]
    ground_truth_cluster_rank = adata_base.uns["cluster_rank"] 
    
    # Run all methods on the dataset
    latent_spaces = run_methods(adata_base, seed=seed)
    _LOGGER.info(f"Latent spaces: {latent_spaces}")
    adata_base.write(f"{adata_folder}/adata_base.h5ad")
    _LOGGER.info(f"Adata saved: {adata_folder}/adata_base.h5ad")
    
    combined_embeddings(adata_base, out_folder, delta=0)

    # Shifted adatas
    for delta in delta_vec:
        _LOGGER.info(f"Delta={delta}")
        adata_sim = simulation_correlated_shift(
            n_samples=n_samples,
            n_genes=n_genes,
            seed=seed,
            sigma=sigma,
            branch_prob=branch_prob,
            k_clusters=k_clusters,
            hole_size=hole_size,
            n_holes=n_holes,
            hole_density=hole_density,
            delta=delta,
        )
        # override the clusters
        adata_sim.obs["cluster_true"] = ground_truth_clusters
        adata_sim.uns["cluster_rank"] = ground_truth_cluster_rank
        
        # 4Run all methods on the dataset
        latent_spaces = run_methods(adata_sim, seed=seed)
        _LOGGER.info(f"Latent spaces: {latent_spaces}")
        adata_sim.write(f"{adata_folder}/adata_shift_delta_{delta}.h5ad")
        _LOGGER.info(f"Adata saved: {adata_folder}/adata_shift_delta_{delta}.h5ad")

        # (Optional) keep only 2D spaces if you want pure visual spaces
        latent_spaces_2d = [k for k in latent_spaces if adata_sim.obsm[k].shape[1] == 2]
        
        combined_embeddings(adata_sim=adata_sim,
                            out_folder=out_folder, 
                            delta=delta)
        return adata_sim
    
_LOGGER = logging.getLogger(__name__)

#if __name__ == "__main__":
    # delta_vec = [0.01, 0.05, 0.1]
    # run_delta_shift(
    #     n_samples=1000,
    #     n_genes=50,
    #     sigma=0.05,
    #     branch_prob=0.7,
    #     k_clusters=20,
    #     hole_size=1,
    #     n_holes=3,
    #     hole_density=0.05,
    #     seed=0,
    #     out_folder="new_figures_delta",
    #     adata_folder= "newadata_delta",
    #     delta_vec = delta_vec,
    #     )
        
    
