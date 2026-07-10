import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from simulation_functions import simulate_simple2
from make_simulated_adata import shift_magnitudes_simple


def build_sim_normal_shifts(sigma, n_batches, seed, sigma_biological=0.10,
                            n_samples=500, n_genes=50):
    """One dataset: n_batches shifts drawn ~ N(0, sigma^2) (Josh's prescription)."""
    rng = np.random.default_rng(seed)
    shifts = rng.normal(loc=0.0, scale=sigma, size=n_batches)   # scale = sigma (std)
    # baseline (shift 0) + normally-drawn shifted batches, concatenated & count-transformed
    return shift_magnitudes_simple(
        shift_type="vz", shifts=shifts, mag=1.0,
        n_samples=n_samples, n_genes=n_genes, sigma=sigma_biological, seed=seed,
    )

def rho_for_run(sigma, seed, batch_aware, n_batches=5):
    adata = build_sim_normal_shifts(sigma, n_batches, seed)
    
    # alan's code for calculating decipher_time
    
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
    return abs(rho)                                       # |rho|: direction is arbitrary

if __name__ == "__main__":
    # ---- sweep ----
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    seeds  = [0, 1, 2, 3, 4]                                  # multiple seeds: Decipher is non-identifiable
    models = {"Decipher-BC": True, "Base Decipher": False}
    colors = {"Decipher-BC": "#2E7D32", "Base Decipher": "#C62828"}

    results = {name: np.full((len(sigmas), len(seeds)), np.nan) for name in models}
    for name, batch_aware in models.items():
        for i, sig in enumerate(sigmas):
            for j, sd in enumerate(seeds):
                try:
                    results[name][i, j] = rho_for_run(sig, sd, batch_aware)
                except Exception as e:
                    print(f"[{name}] sigma={sig} seed={sd} failed: {type(e).__name__}: {e}")

    # ---- plot: correlation vs sigma, mean +/- band, BC vs base ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in models:
        R = results[name]
        mean = np.nanmean(R, axis=1)
        sd_  = np.nanstd(R, axis=1)
        ax.plot(sigmas, mean, marker="o", color=colors[name], label=name, linewidth=2)
        ax.fill_between(sigmas, mean - sd_, mean + sd_, color=colors[name], alpha=0.18)

    ax.set_xscale("log")                                     # sigma spans 0.1–10; log reads better
    ax.set_xlabel(r"batch-shift noise  $\sigma$   (shifts $\sim \mathcal{N}(0,\sigma^2)$)")
    ax.set_ylabel(r"Spearman $|\rho|$  (decipher_time vs latent_t)")
    ax.set_title("Trajectory recovery vs batch noise")
    ax.set_ylim(0, 1); ax.axhline(0, color="0.8", lw=0.8)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("sigma_vs_correlation.png", dpi=200)
    plt.show()