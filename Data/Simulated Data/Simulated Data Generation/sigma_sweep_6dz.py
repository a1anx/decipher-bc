import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from make_simulated_adata_6dz import shift_magnitudes_multivariate_from_normal

N_Z_DIMS = 6


def build_sim_normal_shifts(sigma, n_batches, seed, sigma_biological=0.10,
                            n_samples=500, n_genes=50, n_z_dims=N_Z_DIMS):
    """One dataset: n_batches-1 shift vectors drawn ~ N(0, sigma^2 * I) over n_z_dims dims."""
    adata, h5ad_path = shift_magnitudes_multivariate_from_normal(
        shift_type="vz", n_batches=n_batches, shift_sigma=sigma, n_z_dims=n_z_dims,
        n_samples=n_samples, n_genes=n_genes, biological_sigma=sigma_biological, seed=seed,
    )
    return adata, h5ad_path

def rho_for_run(sigma, seed, model, decipher_seed, n_batches=5):
    adata, gt_h5ad_path = build_sim_normal_shifts(sigma, n_batches, seed)

    if model == 'decipher_vz':
        import decipher_vz as dc
        from decipher_vz.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz'

    elif model == 'decipher_vz2':
        import decipher_vz2 as dc
        from decipher_vz2.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_vz2'

    elif model == 'decipher_mf':
        import decipher_mf as dc
        from decipher_mf.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher_mf'

    elif model == 'decipher':
        import decipher as dc
        from decipher.tools._decipher import DecipherConfig as DecipherConfig
        model_tag = 'decipher'

    config = DecipherConfig(learning_rate=1e-3, seed=decipher_seed, dim_z=N_Z_DIMS)
    dc.tl.decipher_train(adata, config, plot_kwargs={"color": "batch", "title": f"sigma={sigma}"})

    #Compute ground truths and manually plot trajectories
    dc.tl.cell_clusters(adata, leiden_resolution = 0.05, n_neighbors= 25, seed = 341)
    filtered = adata.obs["decipher_clusters"].value_counts()>10
    filtered_ids = set(filtered[filtered].index)
    ground_truths = adata.obs.groupby('decipher_clusters')['latent_t'].mean().sort_values().index.to_list()
    ground_truths = [c for c in ground_truths if c in filtered_ids]
    dc.tl.trajectories(adata, dc.tl.TConfig('trajectory', cluster_ids_list=ground_truths))
    dc.tl.decipher_rotate_space(adata)

    #Compute decipher time
    # decipher_v can end up compressed into a very short path (e.g. for decipher_mf),
    # so the trajectory can have fewer points than the default n_neighbors=10 requires.
    n_trajectory_points = len(adata.uns["decipher"]["trajectories"]["trajectory"]["times"])
    dc.tl.decipher_time(adata, n_neighbors=min(10, n_trajectory_points))
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])

    # Save the TRAINED adata (has decipher_z, decipher_v, decipher_time) so we
    # never need to re-run decipher_train to get inferred z later.
    trained_dir = os.path.join(os.path.dirname(gt_h5ad_path), "trained", model_tag)
    os.makedirs(trained_dir, exist_ok=True)
    trained_h5ad_path = os.path.join(
        trained_dir, f"sigma{sigma}_seed{seed}_{model_tag}.h5ad"
    )
    adata.write(trained_h5ad_path)

    return abs(rho), gt_h5ad_path, trained_h5ad_path, adata


if __name__ == "__main__":
    # ---- sweep ----
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
    seeds  = [0, 1, 2, 3, 4]                                  # multiple seeds: Decipher is non-identifiable
    decipher_seeds = [1]                                      # one decipher_seed
    models = {
        "Decipher-VZ": "decipher_vz",
        "Decipher-VZ2": "decipher_vz2",
        "Decipher-MF": "decipher_mf",
        "Base Decipher": "decipher",
    }
    colors = {
        "Decipher-VZ": "#2E7D32",
        "Decipher-VZ2": "#1565C0",
        "Decipher-MF": "#F9A825",
        "Base Decipher": "#C62828",
    }

    results = {name: np.full((len(sigmas), len(seeds) * len(decipher_seeds)), np.nan) for name in models}
    run_log = []
    for name, model in models.items():
        for i, sig in enumerate(sigmas):
            run_idx = 0
            for sd in seeds:
                for dsd in decipher_seeds:
                    try:
                        rho, gt_path, trained_path, _ = rho_for_run(sig, sd, model, dsd)
                        results[name][i, run_idx] = rho
                        run_log.append({
                            "model": name, "sigma": sig, "seed": sd, "decipher_seed": dsd, "rho": rho,
                            "ground_truth_h5ad": gt_path, "trained_h5ad": trained_path,
                        })
                    except Exception as e:
                        print(f"[{name}] sigma={sig} seed={sd} decipher_seed={dsd} failed: {type(e).__name__}: {e}")
                    run_idx += 1

    # ---- plot: correlation vs sigma, mean +/- band, per model ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in models:
        R = results[name]
        mean = np.nanmean(R, axis=1)
        sd_  = np.nanstd(R, axis=1)
        ax.plot(sigmas, mean, marker="o", color=colors[name], label=name, linewidth=2)
        ax.fill_between(sigmas, mean - sd_, mean + sd_, color=colors[name], alpha=0.18)

    ax.set_xscale("log")                                     # sigma spans 0.1–10; log reads better
    ax.set_xlabel(r"batch-shift noise  $\sigma$   (shifts $\sim \mathcal{N}(0,\sigma^2)$, 6D)")
    ax.set_ylabel(r"Spearman $|\rho|$  (decipher_time vs latent_t)")
    ax.set_title("Trajectory recovery vs batch noise (6D z)")
    ax.set_ylim(0, 1); ax.axhline(0, color="0.8", lw=0.8)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("sigma_vs_correlation_6dz.png", dpi=200)
    pd.DataFrame(run_log).to_csv("sigma_sweep_results_6dz.csv", index=False)
    plt.show()
