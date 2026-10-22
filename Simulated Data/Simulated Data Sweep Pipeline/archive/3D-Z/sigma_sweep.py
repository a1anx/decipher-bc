import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from simulation_functions import simulate_simple2
from make_simulated_adata import shift_magnitudes_simple_from_linear


def build_sim_normal_shifts(shift_sigma, n_batches, seed, sigma_biological=0.10,
                            n_samples=500, n_genes=50):
    """One dataset: n_batches shifts drawn ~ N(0, sigma^2) (Josh's prescription)."""
    # dedicated RNG stream so batch-magnitude draws never collide with the other seeds (W uses seed+1 and cell_seed uses seed+i+1)
    mag_rng = np.random.default_rng(seed + 1000)
    # samples different shifts for each batch (excluding baseline, so n_batches-1) from a normal distribution with mean 0 and standard deviation shift_sigma
    shift_vec = np.round(mag_rng.normal(loc=0, scale=shift_sigma, size=n_batches-1), 2)
    # baseline (shift 0) + normally-drawn shifted batches = total n_batches, concatenated & count-transformed
    adata = shift_magnitudes_simple_from_linear(
        shift_type="vz", shifts=shift_vec, mag=1.0,
        n_samples=n_samples, n_genes=n_genes, sigma=sigma_biological, seed=seed,
    )
    # record the swept parameter so downstream plots can read it back
    adata.uns["shift_sigma"] = shift_sigma
    adata.uns["shift_vec"]   = shift_vec
    return adata

def rho_for_run(shift_sigma, seed, model, decipher_seed, n_batches=5):
    adata = build_sim_normal_shifts(shift_sigma, n_batches, seed)

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

    config = DecipherConfig(learning_rate=1e-3, seed=decipher_seed)
    dc.tl.decipher_train(adata, config, plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"})
    
    #Compute ground truths and manually plot trajectories
    dc.tl.cell_clusters(adata, leiden_resolution = 0.05, n_neighbors= 25, seed = 341)
    filtered = adata.obs["decipher_clusters"].value_counts()>10
    filtered_ids = set(filtered[filtered].index)
    ground_truths = adata.obs.groupby('decipher_clusters')['latent_t'].mean().sort_values().index.to_list()
    ground_truths = [c for c in ground_truths if c in filtered_ids]
    dc.tl.trajectories(adata, dc.tl.TConfig('trajectory', cluster_ids_list=ground_truths))
    dc.tl.decipher_rotate_space(adata) 

    #Compute decipher time
    dc.tl.decipher_time(adata)
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
    adata.uns["rho"] = abs(rho)

    # Save the TRAINED adata (has decipher_z, decipher_v, decipher_time) so we
    # never need to re-run decipher_train to get inferred z later.
    # trained_dir = os.path.join(os.path.dirname(gt_h5ad_path), "trained", model_tag)
    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trained_dir = os.path.join(script_dir,"..", "Simulated Adata", "shift_sigma_sweep", today, "trained")
    os.makedirs(trained_dir, exist_ok=True)
    trained_h5ad_path = os.path.join(
        trained_dir, f"sigma{shift_sigma}_seed{seed}_{model_tag}.h5ad"
    )
    adata.write(trained_h5ad_path)

    return abs(rho), trained_h5ad_path, adata 


if __name__ == "__main__":
    # ---- sweep ----
    #sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
    sigmas = [0.5,2.0, 5.0]
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
                            "trained_h5ad": trained_path,
                        })
                    except Exception as e:
                        print(f"[{name}] sigma={sig} seed={sd} decipher_seed={dsd} failed: {type(e).__name__}: {e}")
                    run_idx += 1

    # # ---- plot: correlation vs sigma, mean +/- band, per model ----
    # fig, ax = plt.subplots(figsize=(7, 5))
    for name in models:
        R = results[name]
        mean = np.nanmean(R, axis=1)
        sd_  = np.nanstd(R, axis=1)
    #     ax.plot(sigmas, mean, marker="o", color=colors[name], label=name, linewidth=2)
    #     ax.fill_between(sigmas, mean - sd_, mean + sd_, color=colors[name], alpha=0.18)

    # ax.set_xscale("log")                                     # sigma spans 0.1–10; log reads better
    # ax.set_xlabel(r"batch-shift noise  $\sigma$   (shifts $\sim \mathcal{N}(0,\sigma^2)$)")
    # ax.set_ylabel(r"Spearman $|\rho|$  (decipher_time vs latent_t)")
    # ax.set_title("Trajectory recovery vs batch noise")
    # ax.set_ylim(0, 1); ax.axhline(0, color="0.8", lw=0.8)
    # ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    # fig.tight_layout()
    # fig.savefig("sigma_vs_correlation.png", dpi=200)
    # pd.DataFrame(run_log).to_csv("sigma_sweep_results.csv", index=False)
    # plt.show()