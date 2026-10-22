# v2 params: simplified for linear trajectory (no branching, no holes)
#
# REMOVED from v1:
#   - branch_prob  : controlled branching probability for z3/z5 components; removed because
#                    we replaced the z1/z3/z5 branching trajectory with z_mu = latent_t (1D)
#   - k_clusters   : used for KMeans on the branching latent_z; removed with branching
#   - hole_size    : controlled gap size in the chunk-based trajectory sampling; removed
#   - n_holes      : number of gaps in the trajectory; removed
#   - hole_density : density of samples inside holes; removed
#                    (hole/chunk sampling was a way to create sparse regions in pseudotime;
#                     not needed for a clean linear trajectory)

SIMUL_PARAMS = {
    'n_samples': 500,
    'n_genes': 50,
    'seed': 0,
    'sigma': 0.03,
}

SEED = 0
