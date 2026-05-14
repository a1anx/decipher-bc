import logging
import pandas as pd
import scanpy as sc
import os
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import silhouette_samples

import anndata as ad
#Import tools necessary to call decipher_train()
#decipher train is training function and DecipherConfig holds hyperparameters
from decipher.tools import decipher_train, DecipherConfig 

#alpha_vec = [0.01, 0.05, 0.1]
adata_base = sc.read("newadata_delta/adata_base.h5ad")
adata_0_01 = sc.read("newadata_delta/adata_shift_delta_0.01.h5ad")
adata_0_05 = sc.read("newadata_delta/adata_shift_delta_0.05.h5ad")
adata_0_1 = sc.read("newadata_delta/adata_shift_delta_0.1.h5ad")


def concat_adata(
    adata_array, 
    output_name,
    #k_clusters = 20,
    batch_keys=None
    ):
    adata_combined = ad.concat(
        adata_array,
        axis=0,                                 # concatenate cells
        join="outer",                           # union of var names (or use "inner")
        label="batch" if batch_keys else None,  # Add option to set label as batch
        keys=batch_keys,
        merge="same"                            # requires identical obsm keys
    )
    # k_means = KMeans(k_clusters)
    # latent_z = adata_combined.obsm["latent_z"]
    # k_means.fit(latent_z)
    # adata_combined.obs["cluster_latent"] = k_means.labels_
    # # for each cluster, order the other clusters by distance of their kmeans center
    # cluster_centers = k_means.cluster_centers_
    # cluster_rank = np.argsort(
    #     np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    # )
    # adata_combined.uns["cluster_rank"] = cluster_rank[:, 1:]
    adata_combined.write(f"newadata_delta/adata_combined_{output_name}.h5ad")
    _LOGGER.info(f"Combined adata saved: newadata_delta/adata_combined_{output_name}.h5ad")
    return adata_combined

_LOGGER = logging.getLogger(__name__)

def run_and_save(adata_combined, suffix, out_folder, adata_folder, k_clusters=20):
    decipher_z = adata_combined.obsm["decipher_z"]
    k_means = KMeans(k_clusters)
    k_means.fit(decipher_z)
    adata_combined.obs["cluster_latent"] = k_means.labels_

    aris = {}
    for a in adata_combined.obs["delta"].unique():
        mask = adata_combined.obs["delta"] == a
        aris[a] = adjusted_rand_score(
            adata_combined.obs["cluster_true"][mask],
            adata_combined.obs["cluster_latent"][mask],
        )
    print(f"[{suffix}] aris values: {aris}")

    delta_labels = adata_combined.obs["delta"]
    sil_values = silhouette_samples(decipher_z, delta_labels)
    df = pd.DataFrame({"silhouette_value": sil_values, "alpha": delta_labels.values})
    print(df.groupby("alpha", observed=True)["silhouette_value"].mean())

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 3*4))
    sc.pl.embedding(adata_combined, basis="decipher_v", color="batch",
                    show=False, ax=axes[0], size=30)
    sc.pl.embedding(adata_combined, basis="decipher_v", color="delta",
                    show=False, ax=axes[1], size=30)
    sc.pl.embedding(adata_combined, basis="decipher_v", color="cluster_true",
                    show=False, ax=axes[2], size=30)
    sc.pl.embedding(adata_combined, basis="decipher_v", color="latent_t",
                    show=False, ax=axes[3], size=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"combined_embeddings_{suffix}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    adata_combined.write(f"{adata_folder}/adata_combined_{suffix}.h5ad")

if __name__ == "__main__":
    seed = 0
    out_folder = "concat_figures"
    adata_folder = "concat_adata"
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(adata_folder, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    adata_array = [adata_base, adata_0_01, adata_0_05, adata_0_1]
    adata_combined = concat_adata(
        adata_array=adata_array,
        output_name="2_delta",
        batch_keys=["base", "delta_0.01", "delta_0.05", "delta_0.1"]
    )

    # Native decipher (no batch correction)
    decipher_train(adata_combined, decipher_config=DecipherConfig())
    run_and_save(adata_combined, "native", out_folder, adata_folder)

    # Decipher-bc (simple concatenation with batch correction)
    decipher_train(adata_combined, decipher_config=DecipherConfig(), batch_key="batch")
    run_and_save(adata_combined, "bc", out_folder, adata_folder)


# if __name__ == "__main__":
#     #print(adata_0_0.obs.columns)
#     seed=0
#     adata_array = [adata_base, adata_0_01, adata_0_05, adata_0_1]
#     #Log message
#     logging.basicConfig(level=logging.INFO)
#     adata_combined = concat_adata(
#         adata_array = adata_array,
#         output_name="2_delta",
#         batch_keys=["base", "delta_0.01", "delta_0.05", "delta_0.1"] #Pass batch keys into all simulated data
#         )                                                            #Creates new column where cells get key that corresponds to AnnData
    
#     #Run decipher-bc training step
#     decipher_train(adata_combined, decipher_config=DecipherConfig(), batch_key="batch")

#     k_clusters = 20
#     #Switch evaluation target to trained decipher_z space
#     decipher_z = adata_combined.obsm["decipher_z"]
#     k_means = KMeans(k_clusters)
#     #Latent z is only pre-existing embeddings 
#     #latent_z = adata_combined.obsm["latent_z"]
#     k_means.fit(decipher_z)
#     adata_combined.obs["cluster_latent"] = k_means.labels_
    
#     # cluster preservation vs ground-truth clusters
#     aris = {}
#     for a in adata_combined.obs["delta"].unique():
#         mask = adata_combined.obs["delta"] == a
#         aris[a] = adjusted_rand_score(
#             adata_combined.obs["cluster_true"][mask],
#             adata_combined.obs["cluster_latent"][mask],
#         )
#     print(f"aris values: {aris}")
    
#     delta_labels = adata_combined.obs["delta"]
#     sil_values = silhouette_samples(decipher_z, delta_labels)

#     # Build a DataFrame
#     df = pd.DataFrame({
#         "silhouette_value": sil_values,
#         "alpha": delta_labels.values   # or parse numeric α from the string
#     })
#     silhouette_by_alpha = df.groupby("alpha")["silhouette_value"].mean()
#     print(silhouette_by_alpha)
    
#     #Umap
#     """"
#     Decipher_v is already in 2D space, so coment out normalize/neighbor/umap process
#     # ---------- 1) Make a normalized copy for UMAP/PCA/PHATE ----------
#     adata_norm = adata_combined.copy()
#     sc.pp.normalize_total(adata_norm, target_sum=1e4)
#     sc.pp.log1p(adata_norm)
#     # optionally: sc.pp.highly_variable_genes, sc.pp.scale, etc.

#     # ---------- 2) UMAP on normalized data ----------
#     _LOGGER.info("Computing UMAP on normalized data")
#     sc.pp.neighbors(adata_norm, random_state=seed)
#     sc.tl.umap(adata_norm, random_state=seed)
#     adata_norm.obsm["X_norm_umap"] = adata_norm.obsm["X_umap"]
#     #latent_spaces.append("X_norm_umap")
#     _LOGGER.info("Norm UMAP computed")
    
#     fig, axes = plt.subplots(
#             nrows=4,
#             ncols=1,
#             figsize=(4, 3*4)
#         )
    
#     sc.pl.embedding(
#         adata_combined,
#         basis="X_norm_umap",
#         color="delta",
#         show=False,
#         ax=axes[0],
#         size=30
#     )
#     # Umap normalized in entire dataset
#     sc.pl.embedding(
#         adata_norm,
#         basis="X_norm_umap",
#         color="delta",
#         show=False,
#         ax=axes[1],
#         size=30
#     )
#     #--- Row 3 ---
#     sc.pl.embedding(
#         adata_norm,
#         basis="X_norm_umap",
#         color="cluster_true",
#         show=False,
#         ax=axes[2],
#         size=30
#     )
#     #--- Row 4 ---
#     sc.pl.embedding(
#         adata_norm,
#         basis="X_norm_umap",
#         color="latent_t",
#         show=False,
#         ax=axes[3],
#         size=30
#     )
#     """

#     fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 3*4))

#     sc.pl.embedding(adata_combined, basis="decipher_v", color="batch",
#                 show=False, ax=axes[0], size=30)
#     sc.pl.embedding(adata_combined, basis="decipher_v", color="delta",
#                 show=False, ax=axes[1], size=30)
#     sc.pl.embedding(adata_combined, basis="decipher_v", color="cluster_true",
#                 show=False, ax=axes[2], size=30)
#     sc.pl.embedding(adata_combined, basis="decipher_v", color="latent_t",
#                 show=False, ax=axes[3], size=30)

#     plt.tight_layout()
#     out_folder="concat_figures"
#     adata_folder="concat_adata"
#     suffix = "2"
#     os.makedirs(out_folder, exist_ok=True)
#     os.makedirs(adata_folder, exist_ok=True)
#     plt.savefig(
#         os.path.join(out_folder, f"combined_embeddings_{suffix}.png"),
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.close()
#     _LOGGER.info(f"Plot saved: {out_folder}/combined_embeddings_{suffix}.png")
#     adata_combined.write(f"{adata_folder}/adata_combined_{suffix}.h5ad")
#     _LOGGER.info(f"Adata saved: {adata_folder}/adata_combined_{suffix}.h5ad")
    
#     # adata_combined = ad.concat(
#     #     adata_array,
#     #     axis=0,                       # concatenate cells
#     #     join="outer",                 # union of var names (or use "inner")
#     #     label=None,                   # since we already added .obs["patient"]
#     #     merge="same"                  # requires identical obsm keys
#     # )
#     # print("combined obsm:", adata_combined.obsm.keys())
#     # latent = adata_combined.obsm['latent_z']
#     # print("Any NaNs?", np.isnan(latent).any())
#     # print("Number of NaNs:", np.isnan(latent).sum())
    
#     # concat_adata(
#     #     adata_array = [adata_nonshifted, adata_shifted_001, adata_shifted_005, adata_shifted_01],
#     #     output_name="c2"
#     # )
#     # concat_adata(
#     #     adata_array = [adata_nonshifted, adata_shifted_01, adata_shifted_05, adata_shifted_1],
#     #     output_name="c3"
#     # )
    