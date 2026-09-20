import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

import wandb
import decipher_models as dc

from simulated_data_pipeline_aug1 import shift_magnitudes_multivariate_from_normal

WANDB_ENTITY = "jpark-columbia"
WANDB_PROJECT = "decipher-bc-sigma-sweep"

# Verified batch-conditioning preset mapping from decipher-model-consolidation-plan.md,
# checked against decipher_models/tools/_decipher/decipher.py's Decipher.__init__/model()/guide().
PRESETS = {
    "decipher": dict(batch_conditioning="none"),
    # Set 1 -- concat -> z
    "decipher_vz": dict(batch_conditioning="decoder_only", batch_embedding_mode="concat_z"),
    "decipher_vz2": dict(batch_conditioning="decoder_encoder", batch_embedding_mode="concat_z"),
    "decipher_mf": dict(
        batch_conditioning="decoder_encoder", batch_embedding_mode="concat_z", mean_field_v=True
    ),
    # Set 2 -- add -> z
    "decipher_vz_add": dict(batch_conditioning="decoder_only", batch_embedding_mode="additive"),
    "decipher_vz2_add": dict(batch_conditioning="decoder_encoder", batch_embedding_mode="additive"),
    "decipher_mf_add": dict(
        batch_conditioning="decoder_encoder", batch_embedding_mode="additive", mean_field_v=True
    ),
    # Set 3 -- concat -> x. The b -> z edge is deleted; batch reaches only the reconstruction.
    "decipher_zx": dict(batch_conditioning="decoder_only", batch_embedding_mode="concat_x"),
    "decipher_zx2": dict(batch_conditioning="decoder_encoder", batch_embedding_mode="concat_x"),
    "decipher_mf2": dict(
        batch_conditioning="decoder_encoder", batch_embedding_mode="concat_x", mean_field_v=True
    ),
}

DISPLAY_NAMES = {
    "decipher": "Base Decipher",
    "decipher_vz": "Decipher-VZ",
    "decipher_vz2": "Decipher-VZ2",
    "decipher_mf": "Decipher-MF",
    "decipher_vz_add": "Decipher-VZ-Add",
    "decipher_vz2_add": "Decipher-VZ2-Add",
    "decipher_mf_add": "Decipher-MF-Add",
    "decipher_zx": "Decipher-ZX",
    "decipher_zx2": "Decipher-ZX2",
    "decipher_mf2": "Decipher-MF2",
}

# same hex values the earlier sweep plots use, so figures stay comparable across scripts
MODEL_COLORS = {
    "decipher": "#C62828",
    "decipher_vz": "#2E7D32",
    "decipher_vz2": "#1565C0",
    "decipher_mf": "#F9A825",
    "decipher_vz_add": "#00897B",
    "decipher_vz2_add": "#7B1FA2",
    "decipher_mf_add": "#AD1457",
    "decipher_zx": "#EF6C00",
    "decipher_zx2": "#4527A0",
    "decipher_mf2": "#558B2F",
}

# Simulation mechanisms, keyed by where the batch effect enters the generative process.
# Every function takes the SIM_SETTINGS keys plus shift_sigma and seed, and returns an AnnData
# with integer counts in X and obs["batch"] / obs["latent_t"].
SIMULATIONS = {
    # batch adds a shift in z-space before the z->x map (v->z entry point)
    "z_shift": shift_magnitudes_multivariate_from_normal,
    # batch adds a free per-gene offset AFTER `W @ z`, so it lives in gene space and never
    # touches the latent z. This is the mechanism Set 3 mirrors: a shift in z before a linear
    # decoder is distributionally identical to z_shift and would test nothing different.
    # Writes uns["gene_shift_matrix"] (n_batches, n_genes) instead of uns["shift_matrix"], and
    # adds layers["counts_nobatch"] -- the perfect-correction ceiling.
    "gene_shift": partial(shift_magnitudes_multivariate_from_normal, batch_mode="genes"),
}

# The simulation whose generative process each model mirrors. Base Decipher has no batch term,
# so it is a baseline for whichever mechanism it is run on (use --sims to add more).
MODEL_SIMULATION = {
    "decipher": "z_shift",
    "decipher_vz": "z_shift",
    "decipher_vz2": "z_shift",
    "decipher_mf": "z_shift",
    "decipher_vz_add": "z_shift",
    "decipher_vz2_add": "z_shift",
    "decipher_mf_add": "z_shift",
    "decipher_zx": "gene_shift",
    "decipher_zx2": "gene_shift",
    "decipher_mf2": "gene_shift",
}
assert set(MODEL_SIMULATION) == set(PRESETS), "every preset needs a MODEL_SIMULATION entry"
assert set(MODEL_SIMULATION.values()) <= set(SIMULATIONS), "MODEL_SIMULATION names an unknown sim"

SWEEP_JOB_TYPE = "sweep_run"
SUMMARY_JOB_TYPE = "sweep_summary"

# the default comparison grid -- keep these fixed so new models stay comparable to old ones
SHIFT_SIGMAS = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
SEEDS = [3, 4, 5]
DECIPHER_SEEDS = [1, 2, 3]
# a run is only comparable to another if it was simulated with the same settings
SIM_SETTINGS = dict(n_batches=5, n_samples=500, n_genes=200, n_z_dims=3, biological_sigma=0.1)

METRICS = ["rho", "r2_overall", "r2_per_gene_median"]
METRIC_LABELS = {
    "rho": "rho  (|Spearman|, decipher_time vs latent_t)",
    "r2_overall": "overall R$^2$  (log1p reconstruction)",
    "r2_per_gene_median": "median per-gene R$^2$  (log1p)",
}
AGG_KEYS = ["model_tag", "shift_sigma"]


def train_and_compute_rho_r2_wandb(
    model_tag,
    decipher_seed,
    *,
    n_batches: int = 5,
    shift_sigma: float = 0.0,
    n_samples: int = 500,
    n_genes: int = 200,
    n_z_dims: int = 3,
    biological_sigma: float = 0.1,
    seed: int = 0,
    dim_z: int = None,
    output_root: str = None,
    wandb_project: str = WANDB_PROJECT,
    extra_config: dict = None,
    sim_mechanism: str = None,
):
    """
    Train one (model_tag, shift_sigma, seed, decipher_seed) combination with the
    unified decipher_models package, logging the run to Weights & Biases.

    `sim_mechanism` picks the data-generating process from SIMULATIONS; by default the one
    the model mirrors (MODEL_SIMULATION).

    `extra_config` is merged into the wandb config (e.g. the git commit), so runs from
    different sweeps can be told apart when they are later aggregated together.

    Returns
    -------
    dict with keys: rho, trained_h5ad, r2_overall, r2_per_gene_median,
    wandb_run_id, wandb_url, error (None on success).

    Side effects
    ------------
    Writes the trained adata and a reconstruction diagnostic figure under
    `output_root` (same layout as train_and_compute_rho_r2 in
    simulated_data_pipeline_aug1.py). Always opens and finishes exactly one
    wandb run.
    """
    if dim_z is None:
        dim_z = n_z_dims
    sim_mechanism = sim_mechanism or MODEL_SIMULATION[model_tag]
    simulate = SIMULATIONS[sim_mechanism]

    config = dc.tl.DecipherConfig(
        **PRESETS[model_tag],
        learning_rate=1e-3,
        seed=decipher_seed,
        dim_z=dim_z,
    )

    # every name and file carries the sim, so one model run on two mechanisms never collides
    run_tag = f"{model_tag}_{sim_mechanism}_sigma{shift_sigma}_seed{seed}"
    run_name = f"{run_tag}_dseed{decipher_seed}"
    wandb_config = {
        **config.to_dict(),
        "model_tag": model_tag,
        "display_name": DISPLAY_NAMES.get(model_tag, model_tag),
        "sim_mechanism": sim_mechanism,
        "shift_sigma": shift_sigma,
        # not "seed": DecipherConfig.to_dict() already has a "seed" (the training seed), and
        # the post-training config backfill below would silently overwrite it
        "sim_seed": seed,
        "decipher_seed": decipher_seed,
        "n_batches_sim": n_batches,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "n_z_dims": n_z_dims,
        "biological_sigma": biological_sigma,
        **(extra_config or {}),
    }
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=wandb_project,
        group=f"{model_tag}_{sim_mechanism}_sigma{shift_sigma}",
        job_type=SWEEP_JOB_TYPE,
        name=run_name,
        tags=[model_tag, sim_mechanism, f"sigma_{shift_sigma}"],
        config=wandb_config,
    )
    try:
        adata = simulate(
            n_batches=n_batches,
            shift_sigma=shift_sigma,
            n_samples=n_samples,
            n_genes=n_genes,
            n_z_dims=n_z_dims,
            biological_sigma=biological_sigma,
            seed=seed,
        )

        decipher, _ = dc.tl.decipher_train(
            adata,
            config,
            plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"},
        )
        # decipher_train calls config.initialize_from_adata internally, filling in
        # dim_genes/n_cells/n_batches/batch_key — backfill those into the wandb config.
        run.config.update(config.to_dict(), allow_val_change=True)

        for epoch, (train_elbo, val_nll) in enumerate(
            zip(decipher.train_losses_, decipher.val_losses_)
        ):
            run.log({"train_elbo": train_elbo, "val_nll": val_nll}, step=epoch)

        r2_result = dc.tl.reconstruction_r2_log1p(decipher, adata)
        r2_overall = r2_result["r2_overall"]
        r2_per_gene_median = float(np.nanmedian(r2_result["r2_per_gene"]))
        adata.uns["r2_overall"] = r2_overall
        adata.uns["r2_per_gene_median"] = r2_per_gene_median

        fig, axes = dc.pl.reconstruction_r2_log1p(decipher, adata, figsize=(21, 5))
        for ax in axes:
            ax.title.set_fontsize(10)
            ax.xaxis.label.set_fontsize(9)
            ax.yaxis.label.set_fontsize(9)
            ax.tick_params(labelsize=8)
            if ax.get_legend() is not None:
                for text in ax.get_legend().get_texts():
                    text.set_fontsize(8)
        fig.suptitle(
            f"{model_tag} | sim={sim_mechanism} | sigma={shift_sigma} | seed={seed} "
            f"| decipher seed={decipher_seed}",
            y=1.02,
            fontsize=11,
        )
        figs_dir = os.path.join(output_root, "figs")
        os.makedirs(figs_dir, exist_ok=True)
        file_stem = f"sigma{shift_sigma}_seed{seed}_{model_tag}_{sim_mechanism}_decipherseed_{decipher_seed}"
        fig_path = os.path.join(figs_dir, f"reconstruction_{file_stem}.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        run.log({"reconstruction_diagnostic": wandb.Image(fig)})
        plt.close(fig)

        dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
        filtered = adata.obs["decipher_clusters"].value_counts() > 10
        filtered_ids = set(filtered[filtered].index)
        ground_truths = (
            adata.obs.groupby("decipher_clusters")["latent_t"].mean().sort_values().index.to_list()
        )
        ground_truths = [c for c in ground_truths if c in filtered_ids]
        dc.tl.trajectories(adata, dc.tl.TConfig("trajectory", cluster_ids_list=ground_truths))
        dc.tl.decipher_rotate_space(adata)

        dc.tl.decipher_time(adata)
        m = adata.obs["decipher_time"].notna()
        rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
        rho = abs(rho)
        adata.uns["rho"] = rho

        # v space colored by batch / ground-truth pseudotime / recovered decipher time
        fig_v = dc.pl.decipher(
            adata,
            color=["batch", "latent_t", "decipher_time"],
            basis="decipher_v",
            ncols=3,
        )
        fig_v.suptitle(
            f"{model_tag} | sim={sim_mechanism} | sigma={shift_sigma} | seed={seed} "
            f"| decipher seed={decipher_seed}",
            y=1.05,
            fontsize=11,
        )
        v_fig_path = os.path.join(figs_dir, f"vspace_{file_stem}.png")
        fig_v.savefig(v_fig_path, dpi=120, bbox_inches="tight")
        run.log({"v_space": wandb.Image(fig_v)})
        plt.close(fig_v)

        run.log({"rho": rho, "r2_overall": r2_overall, "r2_per_gene_median": r2_per_gene_median})
        run.summary["rho"] = rho
        run.summary["r2_overall"] = r2_overall
        run.summary["r2_per_gene_median"] = r2_per_gene_median
        run.summary["status"] = "success"

        trained_dir = os.path.join(output_root, "trained")
        os.makedirs(trained_dir, exist_ok=True)
        trained_h5ad_path = os.path.join(trained_dir, f"{file_stem}.h5ad")
        adata.write(trained_h5ad_path)

        return {
            "rho": rho,
            "trained_h5ad": trained_h5ad_path,
            "r2_overall": r2_overall,
            "r2_per_gene_median": r2_per_gene_median,
            "wandb_run_id": run.id,
            "wandb_url": run.url,
            "error": None,
        }
    except Exception as e:
        run.summary["status"] = "failed"
        run.summary["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        # decipher_train ends by calling plot_decipher_v and discards the figure without
        # closing it, so every run leaks one until the whole sweep is done
        plt.close("all")
        run.finish()


def aggregate_runs(run_log):
    """Collapse raw runs in the order the nesting demands: decipher_seeds first, then seeds.

    Returns
    -------
    world : DataFrame
        One row per (model_tag, shift_sigma, seed) -- the per-seed mean over decipher_seeds,
        plus that seed's SD across its decipher_seeds (`<metric>_sd_train`).
    summary : DataFrame
        One row per (model_tag, shift_sigma) -- `<metric>_mean` (mean of the per-seed means),
        `<metric>_sd_between` (SD across the per-seed means; the error bar worth plotting),
        and `<metric>_sd_within` (training noise, pooled over seeds as sqrt of mean variance).
    """
    ok = run_log[run_log["error"].isna()] if "error" in run_log else run_log
    seed_keys = AGG_KEYS + ["seed"]

    per_seed_mean = ok.groupby(seed_keys, as_index=False)[METRICS].mean()
    per_seed_sd = ok.groupby(seed_keys, as_index=False)[METRICS].std(ddof=1)

    world = per_seed_mean.merge(
        per_seed_sd.rename(columns={m: f"{m}_sd_train" for m in METRICS}), on=seed_keys
    )

    # pooled within-seed spread is sqrt of the mean variance, not the mean of the SDs
    within_var = per_seed_sd.copy()
    within_var[METRICS] = within_var[METRICS] ** 2
    pooled = within_var.groupby(AGG_KEYS, as_index=False)[METRICS].mean()
    pooled[METRICS] = np.sqrt(pooled[METRICS])

    summary = (
        per_seed_mean.groupby(AGG_KEYS, as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_mean" for m in METRICS})
        .merge(
            per_seed_mean.groupby(AGG_KEYS, as_index=False)[METRICS]
            .std(ddof=1)
            .rename(columns={m: f"{m}_sd_between" for m in METRICS}),
            on=AGG_KEYS,
        )
        .merge(pooled.rename(columns={m: f"{m}_sd_within" for m in METRICS}), on=AGG_KEYS)
        .merge(ok.groupby(AGG_KEYS).size().rename("n_runs").reset_index(), on=AGG_KEYS)
        .merge(per_seed_mean.groupby(AGG_KEYS).size().rename("n_seeds").reset_index(), on=AGG_KEYS)
    )
    summary.insert(0, "model", summary["model_tag"].map(lambda t: DISPLAY_NAMES.get(t, t)))
    return world, summary


def models_in(df):
    """Model tags present in the data: registered presets in their fixed order, then any others."""
    present = set(df["model_tag"])
    return [t for t in PRESETS if t in present] + sorted(present - set(PRESETS))


def model_color(model_tag):
    if model_tag in MODEL_COLORS:
        return MODEL_COLORS[model_tag]
    # unregistered models get a stable colour that won't reuse a registered one
    extras = ["#5D4037", "#455A64", "#EF6C00", "#212121", "#9E9D24", "#4FC3F7"]
    return extras[sum(map(ord, model_tag)) % len(extras)]


def plot_aggregate(metric, run_log, world, summary, sim_mechanism=None):
    """One figure per metric: a panel per model showing nine runs collapsing to three world
    means and then to one banded point, plus an all-model comparison panel.

    The two spreads are drawn with different encodings on purpose -- caps on the open markers
    are training noise (across decipher_seeds, the thing being averaged away), the shaded band
    is the world-to-world SD (across seeds, the spread that should be believed).
    """
    ok = run_log[run_log["error"].isna()] if "error" in run_log else run_log
    jitter_rng = np.random.default_rng(0)
    models = models_in(summary)

    ncols = 4
    nrows = -(-(len(models) + 1) // ncols)  # one panel per model plus the comparison panel
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(21, 4.5 * nrows), sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()

    for i, model_tag in enumerate(models):
        ax = axes[i]
        color = model_color(model_tag)
        raw = ok[ok["model_tag"] == model_tag]
        wm = world[world["model_tag"] == model_tag]
        s = summary[summary["model_tag"] == model_tag].sort_values("shift_sigma")

        # the nine raw runs, jittered multiplicatively so they spread on a log x axis
        jitter = np.exp(jitter_rng.normal(0, 0.045, size=len(raw)))
        ax.scatter(
            raw["shift_sigma"] * jitter,
            raw[metric],
            s=13,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=2,
            label="individual runs",
        )
        # the three world means, capped by their own training noise
        ax.errorbar(
            wm["shift_sigma"],
            wm[metric],
            yerr=wm[f"{metric}_sd_train"],
            fmt="o",
            ms=5,
            mfc="none",
            mec=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=3,
            lw=0,
            alpha=0.9,
            zorder=3,
            label="world mean ± training noise",
        )
        # the plotted point: mean of the world means, banded by the between-world SD
        ax.fill_between(
            s["shift_sigma"],
            s[f"{metric}_mean"] - s[f"{metric}_sd_between"],
            s[f"{metric}_mean"] + s[f"{metric}_sd_between"],
            color=color,
            alpha=0.16,
            lw=0,
            zorder=1,
            label="mean ± world-to-world SD",
        )
        ax.plot(
            s["shift_sigma"],
            s[f"{metric}_mean"],
            color=color,
            lw=2,
            marker="o",
            ms=4.5,
            zorder=4,
        )

        ax.set_xscale("log")
        ax.set_title(DISPLAY_NAMES.get(model_tag, model_tag), fontsize=11)
        ax.grid(alpha=0.18, lw=0.6)
        if i == 0:
            ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

    ax = axes[len(models)]
    for model_tag in models:
        s = summary[summary["model_tag"] == model_tag].sort_values("shift_sigma")
        color = model_color(model_tag)
        ax.fill_between(
            s["shift_sigma"],
            s[f"{metric}_mean"] - s[f"{metric}_sd_between"],
            s[f"{metric}_mean"] + s[f"{metric}_sd_between"],
            color=color,
            alpha=0.10,
            lw=0,
        )
        ax.plot(
            s["shift_sigma"],
            s[f"{metric}_mean"],
            color=color,
            lw=1.8,
            marker="o",
            ms=3.5,
            label=DISPLAY_NAMES.get(model_tag, model_tag),
        )
    ax.set_xscale("log")
    ax.set_title("all models (mean ± world-to-world SD)", fontsize=11)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

    for ax in axes[len(models) + 1 :]:
        ax.set_visible(False)

    sigmas = sorted(summary["shift_sigma"].unique())
    n_used = len(models) + 1
    for i, ax in enumerate(axes[:n_used]):
        ax.xaxis.set_major_locator(FixedLocator(sigmas))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{s:g}" for s in sigmas]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=7.5, labelbottom=True)
        # label the lowest visible panel in each column, since hidden panels leave gaps
        if i + ncols >= n_used:
            ax.set_xlabel("shift_sigma (log scale)", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)

    n_seeds = int(summary["n_seeds"].max())
    n_runs = int(summary["n_runs"].max())
    sim_label = f"simulation: {sim_mechanism} — " if sim_mechanism else ""
    fig.suptitle(
        f"{metric} — {sim_label}{n_runs} runs per model/sigma "
        f"({n_seeds} simulated worlds × {n_runs // max(n_seeds, 1)} training seeds)",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fetch_sweep_runs(wandb_project=WANDB_PROJECT):
    """Every finished sweep run in the wandb project, from any sweep and any date, as one row
    per run. Only runs simulated with SIM_SETTINGS are kept, since nothing else is comparable.
    """
    api = wandb.Api()
    rows = []
    for r in api.runs(f"{WANDB_ENTITY}/{wandb_project}", per_page=500):
        cfg = r.config
        # runs without sim_seed predate the seed-key fix, and their stored seed is wrong
        if r.job_type != SWEEP_JOB_TYPE or "sim_seed" not in cfg:
            continue
        if any(cfg.get(k) != v for k, v in SIM_SETTINGS.items() if k != "n_batches"):
            continue
        ok = r.state == "finished" and r.summary.get("status") == "success"
        rows.append(
            {
                "model_tag": cfg["model_tag"],
                # runs logged before the simulation registry existed all used z_shift
                "sim_mechanism": cfg.get("sim_mechanism", "z_shift"),
                "shift_sigma": cfg["shift_sigma"],
                "seed": cfg["sim_seed"],
                "decipher_seed": cfg["decipher_seed"],
                **{m: (r.summary.get(m) if ok else np.nan) for m in METRICS},
                "error": None if ok else (r.summary.get("error") or f"run state: {r.state}"),
                "git_commit": cfg.get("git_commit"),
                "wandb_run_id": r.id,
                "created_at": pd.to_datetime(r.created_at, utc=True),
            }
        )
    return pd.DataFrame(rows)


def aggregate_sweep(output_root, wandb_project=WANDB_PROJECT, local_runs=None):
    """Aggregate every comparable run in the wandb project and publish one SUMMARY run.

    Models are only ever compared within one simulation mechanism: the SUMMARY run carries a
    separate set of figures and tables per mechanism, never a pooled one.

    `local_runs` are the records from a sweep that just finished; they take precedence over
    what the API returns, in case wandb hasn't finished indexing the newest runs.
    """
    runs = fetch_sweep_runs(wandb_project)
    if local_runs is not None and len(local_runs):
        local = local_runs.assign(created_at=pd.Timestamp.now(tz="UTC"))
        runs = pd.concat([runs, local], ignore_index=True)

    # a condition that was rerun counts once -- keep the newest attempt
    runs = (
        runs.sort_values("created_at")
        .drop_duplicates(
            ["model_tag", "sim_mechanism", "shift_sigma", "seed", "decipher_seed"], keep="last"
        )
        .reset_index(drop=True)
    )
    if runs.empty:
        raise RuntimeError(f"no comparable sweep runs found in wandb project {wandb_project!r}")

    commits = runs["git_commit"].dropna().unique()
    if len(commits) > 1:
        print(
            f"WARNING: summary mixes runs from {len(commits)} code versions: "
            + ", ".join(c[:10] for c in commits)
        )

    today = datetime.now().strftime("%m%d")
    agg_dir = os.path.join(output_root, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    runs.to_csv(os.path.join(agg_dir, "runs_used.csv"), index=False)
    sims = sorted(runs["sim_mechanism"].unique())
    per_sim = {}
    for sim in sims:
        sim_runs = runs[runs["sim_mechanism"] == sim]
        world, summary = aggregate_runs(sim_runs)
        summary.insert(1, "sim_mechanism", sim)
        per_sim[sim] = (sim_runs, world, summary)
        sim_dir = os.path.join(agg_dir, sim)
        os.makedirs(sim_dir, exist_ok=True)
        world.to_csv(os.path.join(sim_dir, "per_seed_means.csv"), index=False)
        summary.to_csv(os.path.join(sim_dir, "summary.csv"), index=False)

    n_failed = int(runs["error"].notna().sum())
    summary_run = wandb.init(
        entity=WANDB_ENTITY,
        project=wandb_project,
        job_type=SUMMARY_JOB_TYPE,
        name=f"SUMMARY_{today}",
        tags=["summary"],
        config={
            "n_runs_total": len(runs),
            "n_runs_failed": n_failed,
            "models": models_in(runs),
            "sim_mechanisms": sims,
            "models_per_sim": {s: models_in(per_sim[s][0]) for s in sims},
            "n_sigmas": runs["shift_sigma"].nunique(),
            "n_seeds": runs["seed"].nunique(),
            "n_decipher_seeds": runs["decipher_seed"].nunique(),
            "git_commits": list(commits),
            **SIM_SETTINGS,
        },
    )
    try:
        for sim, (sim_runs, world, summary) in per_sim.items():
            for metric in METRICS:
                fig = plot_aggregate(metric, sim_runs, world, summary, sim_mechanism=sim)
                fig.savefig(
                    os.path.join(agg_dir, sim, f"{metric}.png"), dpi=140, bbox_inches="tight"
                )
                summary_run.log({f"aggregate/{sim}/{metric}": wandb.Image(fig)})
                plt.close(fig)
            summary_run.log({f"summary_table/{sim}": wandb.Table(dataframe=summary)})
    finally:
        summary_run.finish()

    for sim in sims:
        sim_runs = per_sim[sim][0]
        print(
            f"Summary [{sim}]: {len(sim_runs)} runs from {sim_runs['model_tag'].nunique()} models "
            f"({int(sim_runs['error'].notna().sum())} failed, excluded)"
        )
    print(f"-> {agg_dir}")
    return {sim: per_sim[sim][2] for sim in sims}


def git_commit():
    """HEAD commit, suffixed '-dirty' when tracked or untracked changes exist."""
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def run_one(
    model_tag, sim_mechanism, shift_sigma, seed, decipher_seed, output_root, wandb_project, commit
):
    """Worker entry point: one run, always returning a record rather than raising."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    record = {
        "model": DISPLAY_NAMES.get(model_tag, model_tag),
        "model_tag": model_tag,
        "sim_mechanism": sim_mechanism,
        "shift_sigma": shift_sigma,
        "seed": seed,
        "decipher_seed": decipher_seed,
        "git_commit": commit,
    }
    started = time.time()
    try:
        record.update(
            train_and_compute_rho_r2_wandb(
                model_tag,
                decipher_seed,
                shift_sigma=shift_sigma,
                seed=seed,
                output_root=output_root,
                wandb_project=wandb_project,
                extra_config={"git_commit": commit},
                sim_mechanism=sim_mechanism,
                **SIM_SETTINGS,
            )
        )
    except Exception as e:
        record.update(
            {
                "rho": np.nan,
                "trained_h5ad": None,
                "r2_overall": np.nan,
                "r2_per_gene_median": np.nan,
                "wandb_run_id": None,
                "wandb_url": None,
                "error": f"{type(e).__name__}: {e}",
            }
        )
    record["runtime_s"] = round(time.time() - started, 1)
    return record


def parse_args():
    p = argparse.ArgumentParser(
        description="Sigma sweep over Decipher variants, logged to wandb and summarised from wandb.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(PRESETS),
        choices=list(PRESETS),
        help="model presets to train (default: all). Use to add new models later.",
    )
    p.add_argument(
        "--sims",
        nargs="+",
        default=None,
        choices=list(SIMULATIONS),
        help="simulation mechanisms to run every selected model on "
        "(default: each model's own entry in MODEL_SIMULATION)",
    )
    p.add_argument("--sigmas", nargs="+", type=float, default=SHIFT_SIGMAS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--decipher-seeds", nargs="+", type=int, default=DECIPHER_SEEDS)
    p.add_argument("--workers", type=int, default=8, help="runs trained at the same time")
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=4,
        help="CPU threads each run may use; workers x threads should be <= cores",
    )
    p.add_argument("--project", default=WANDB_PROJECT, help="wandb project")
    p.add_argument(
        "--output-root", default=None, help="default: Simulated Adata/wandb_sigma_sweep/<MMDD>"
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="skip training; rebuild the summary from every run already in wandb",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    today = datetime.now().strftime("%m%d")
    output_root = args.output_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "Simulated Adata",
        "wandb_sigma_sweep",
        today,
    )
    os.makedirs(output_root, exist_ok=True)

    if args.summary_only:
        aggregate_sweep(output_root, args.project)
        sys.exit(0)

    commit = git_commit()
    if commit is None or commit.endswith("-dirty"):
        print(f"WARNING: git commit is {commit!r}; these runs won't map to an exact code version")

    jobs = [
        (m, sim, sigma, sd, dsd)
        for m in args.models
        for sim in (args.sims or [MODEL_SIMULATION[m]])
        for sigma, sd, dsd in itertools.product(args.sigmas, args.seeds, args.decipher_seeds)
    ]
    print(
        f"{len(jobs)} runs on {args.workers} workers x {args.threads_per_worker} threads "
        f"-> wandb project {args.project!r}",
        flush=True,
    )

    # spawned workers inherit these before importing torch/numpy/numba, which is what caps
    # their thread pools -- otherwise each worker grabs every core and they thrash
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(args.threads_per_worker)
    os.environ["MPLBACKEND"] = "Agg"

    csv_path = os.path.join(output_root, "sweep_log.csv")
    run_log = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(run_one, *job, output_root, args.project, commit): job for job in jobs
        }
        for fut in as_completed(futures):
            m, sim, sigma, sd, dsd = futures[fut]
            try:
                record = fut.result()
            except Exception as e:  # the worker process itself died (e.g. killed, OOM)
                record = {
                    "model": DISPLAY_NAMES.get(m, m),
                    "model_tag": m,
                    "sim_mechanism": sim,
                    "shift_sigma": sigma,
                    "seed": sd,
                    "decipher_seed": dsd,
                    "git_commit": commit,
                    "rho": np.nan,
                    "r2_overall": np.nan,
                    "r2_per_gene_median": np.nan,
                    "error": f"worker crashed: {type(e).__name__}: {e}",
                }
            run_log.append(record)
            pd.DataFrame(run_log).to_csv(csv_path, index=False)
            status = "ok" if record.get("error") is None else f"FAILED ({record['error']})"
            print(
                f"[{len(run_log)}/{len(jobs)}] {m} [{sim}] sigma={sigma} seed={sd} dseed={dsd} "
                f"rho={record['rho']:.4f} {record.get('runtime_s', float('nan')):.0f}s {status}",
                flush=True,
            )

    print(f"all {len(jobs)} runs done in {(time.time() - started) / 60:.1f} min", flush=True)
    aggregate_sweep(output_root, args.project, local_runs=pd.DataFrame(run_log))
