"""Split-fix variant of `train_and_compute_rho_r2_bifurcation` (2026-10-08).

Why this file exists: the bifurcation simulator's `ad.concat` repeats cell names ('0'..'499' in
every batch), and `_make_train_val_split` picks validation cells by NAME with `.loc`, so each
pick marks one name in all batches (~58/42 split instead of 90/10). This module copies the
0918 training function and trains through `decipher_models2` (a copy of `decipher_models` with
only the `native` and `model5` presets), where the split is to be fixed in code. Differences
from the 0918 function:
  - the model is built from `decipher_models2` (`_build_model` below);
  - `uns["n_train"]`, `uns["n_val"]` record the split actually used;
  - `run_date` (a `%m%d` string) replaces the per-job `datetime.now()` for the output dir.
The 0918 modules are untouched; the simulator and every other helper are imported from them.
See Claude Files/plans/1007_splitfix-relay/.
"""

import importlib
import os
import subprocess
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from decipher_models2.presets import PRESETS

_bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")
shift_magnitudes_multivariate_bifurcation = _bif.shift_magnitudes_multivariate_bifurcation
_data_tag = _bif._data_tag
branch_cluster_orders = _bif.branch_cluster_orders
branch_silhouette = _bif.branch_silhouette


def _build_model(model: str):
    """Return (dc, DecipherConfig, preset_kwargs) for one model name, from `decipher_models2`."""
    if model not in PRESETS:
        raise ValueError(f"unknown model {model!r}. Expected one of: {', '.join(PRESETS)}")
    dc = importlib.import_module("decipher_models2")
    DecipherConfig = importlib.import_module("decipher_models2.tools._decipher").DecipherConfig
    return dc, DecipherConfig, PRESETS[model]


def _git_state() -> tuple:
    """(HEAD sha, dirty flag) of the repo this file lives in; ("unknown", False) outside git."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=here,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return sha, dirty


def train_and_compute_rho_r2_bifurcation(
    model,
    decipher_seed,
    n_batches: int = 5,
    shift_sigma: float = 0.0,
    n_samples: int = 500,
    n_genes: int = 200,
    n_z_dims: int = 3,
    biological_sigma: float = 0.1,
    seed: int = 0,
    dim_z: int = None,
    beta: float = 0.1,
    notebook_tag: str = None,
    batch_mode: str = "z",
    branching_t: float = 0.3,
    branch_prob: float = 0.5,
    branch_scale: float = 1.0,
    wandb_run=None,
    run_date: str = None,
):
    """Generate bifurcating data, train one variant, and score it.

    Same shape as `0801_simulated_data_pipeline.train_and_compute_rho_r2`, with two differences:
    the data has a fork in it, and pseudotime is scored along ONE TRAJECTORY PER ARM rather than a
    single spline that would have to zigzag across the fork.

    wandb_run : wandb Run or None
        ADDED 2026-09-18. When given, the per-epoch loss curves, the diagnostic figure and the
        final metrics are logged to it. Taken as a parameter rather than returning the decipher
        object because the losses and the figure only exist inside this function, and adding a
        return value would break every existing caller's unpacking.
        None keeps this function wandb-free, so the tests and the smoke script are unaffected.

    run_date : str or None
        `%m%d` output dir. None = today at call time; the sweep driver passes one value for the
        whole run so a sweep crossing midnight doesn't split across two dirs.

    Returns
    -------
    (rho, rho_plus, rho_minus, branch_asw, trained_h5ad_path, r2_overall, r2_per_gene_median, adata)
    """
    t_start = time.perf_counter()
    adata = shift_magnitudes_multivariate_bifurcation(
        n_batches=n_batches,
        shift_sigma=shift_sigma,
        n_samples=n_samples,
        n_genes=n_genes,
        n_z_dims=n_z_dims,
        biological_sigma=biological_sigma,
        seed=seed,
        batch_mode=batch_mode,
        branching_t=branching_t,
        branch_prob=branch_prob,
        branch_scale=branch_scale,
    )

    if dim_z is None:
        dim_z = n_z_dims

    t_simulated = time.perf_counter()
    dc, DecipherConfig, preset = _build_model(model)
    model_tag = model

    config = DecipherConfig(
        learning_rate=1e-3, seed=decipher_seed, dim_z=dim_z, beta=beta, **preset
    )
    decipher, _ = dc.tl.decipher_train(
        adata,
        config,
        plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"},
    )

    t_trained = time.perf_counter()
    if wandb_run is not None:
        # decipher_train fills in dim_genes/n_cells/n_batches/batch_key via
        # config.initialize_from_adata, so the config is only complete after training.
        wandb_run.config.update(config.to_dict(), allow_val_change=True)
        for epoch, (train_elbo, val_nll) in enumerate(
            zip(decipher.train_losses_, decipher.val_losses_)
        ):
            wandb_run.log({"train_elbo": train_elbo, "val_nll": val_nll}, step=epoch)

    # Reconstruction diagnostic -- computed right after training so we still have the in-memory
    # decipher object with train_losses_ / val_losses_.
    r2_result = dc.tl.reconstruction_r2_log1p(decipher, adata)
    r2_overall = r2_result["r2_overall"]
    r2_per_gene_median = float(np.nanmedian(r2_result["r2_per_gene"]))
    adata.uns["r2_overall"] = r2_overall
    adata.uns["r2_per_gene_median"] = r2_per_gene_median
    split = adata.obs["decipher_split"]
    adata.uns["n_train"] = int((split == "train").sum())
    adata.uns["n_val"] = int((split == "validation").sum())

    today = run_date or datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(
        script_dir, "..", "Sweeps and Results", "shift_sigma_sweep_bifurcation", today
    )

    # Save reconstruction diagnostic figure
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
        f"{model_tag} | sigma={shift_sigma} | decipher seed={decipher_seed}",
        y=1.02,
        fontsize=11,
    )
    figs_dir = os.path.join(out_root, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            figs_dir,
            f"reconstruction_sigma{shift_sigma}_seed{seed}_{model_tag}"
            f"_{_data_tag(branching_t, batch_mode)}_decipherseed_{decipher_seed}.png",
        ),
        dpi=120,
        bbox_inches="tight",
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log({"reconstruction_diagnostic": wandb.Image(fig)})
    plt.close(fig)
    t_recon = time.perf_counter()

    # Trajectories -- one per arm, sharing the trunk clusters.
    #
    # decipher_time (native-decipher/decipher/tools/trajectory_inference.py:402) loops over the
    # trajectories and writes each one's KNN-regressed time onto the cells in its clusters. The
    # trunk clusters belong to BOTH arms, so the second loop overwrites the first there -- that is
    # fine and not a bug: both arms share the same trunk geometry and both measure arc length from
    # the same root, so the two estimates agree and pseudotime stays globally consistent.
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    orders = branch_cluster_orders(adata)
    dc.tl.trajectories(
        adata, *[dc.tl.TConfig(name, cluster_ids_list=ids) for name, ids in orders.items()]
    )
    dc.tl.decipher_rotate_space(adata)
    dc.tl.decipher_time(adata)
    t_trajectories = time.perf_counter()

    # v-space figure (ported from wandb_sigma_sweep.py; previously made post-hoc by
    # 0918_postsweep_vspace_figures.py). Same filename scheme as the reconstruction figure.
    fig_v = dc.pl.decipher(
        adata, color=["batch", "latent_t", "decipher_time"], basis="decipher_v", ncols=3
    )
    fig_v.suptitle(
        f"{model_tag} | {'bif' if branching_t is not None else 'nobif'} | sigma={shift_sigma} "
        f"| seed={seed} | decipher seed={decipher_seed}",
        y=1.05,
        fontsize=11,
    )
    fig_v.savefig(
        os.path.join(
            figs_dir,
            f"vspace_sigma{shift_sigma}_seed{seed}_{model_tag}"
            f"_{_data_tag(branching_t, batch_mode)}_decipherseed_{decipher_seed}.png",
        ),
        dpi=120,
        bbox_inches="tight",
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log({"v_space": wandb.Image(fig_v)})
    plt.close(fig_v)
    t_vspace = time.perf_counter()

    # Metrics
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
    adata.uns["rho"] = abs(rho)

    # per-arm rho. Trunk cells (branch_id == 0) are in neither, but are already counted in pooled.
    # With branching_t=None there are no arms, so both are NaN -- meaningful, not missing.
    arm_rhos = {"rho_plus": np.nan, "rho_minus": np.nan}
    if "branch_id" in adata.obs.columns:
        for arm_name, arm_value in (("rho_plus", 1), ("rho_minus", -1)):
            sel = m & (adata.obs["branch_id"] == arm_value)
            if sel.sum() < 3:
                continue
            r, _ = spearmanr(adata.obs["decipher_time"][sel], adata.obs["latent_t"][sel])
            arm_rhos[arm_name] = abs(r)
    adata.uns.update(arm_rhos)

    branch_asw = branch_silhouette(adata, branching_t)
    adata.uns["branch_asw"] = branch_asw
    adata.uns["trajectory_names"] = list(orders.keys())

    # Run record (explains the long tail of run times). Wall-clock seconds per stage; the h5ad
    # write time is added to the returned in-memory adata only, since the file is already written.
    adata.uns["n_epochs"] = len(decipher.train_losses_)
    adata.uns["git_sha"], adata.uns["git_dirty"] = _git_state()
    adata.uns["stage_seconds"] = {
        "simulate": t_simulated - t_start,
        "train": t_trained - t_simulated,
        "r2_and_figure": t_recon - t_trained,
        "clustering_trajectories_rotate_time": t_trajectories - t_recon,
        "vspace_figure": t_vspace - t_trajectories,
    }

    if wandb_run is not None:
        # Both log() and summary: log() makes it plottable against the sweep axes, summary makes
        # it sortable in the runs table. NaN arm metrics on the no-bifurcation half are recorded
        # as-is -- they mean "no fork existed", not "measurement missing".
        final = {
            "rho": abs(rho),
            "rho_plus": arm_rhos["rho_plus"],
            "rho_minus": arm_rhos["rho_minus"],
            "branch_asw": branch_asw,
            "r2_overall": r2_overall,
            "r2_per_gene_median": r2_per_gene_median,
            "n_trajectories": len(orders),
        }
        wandb_run.log(final)
        for k, v in final.items():
            wandb_run.summary[k] = v
        wandb_run.summary["status"] = "success"

    # Save trained adata. As in aug1, two sweeps on the same calendar day that both leave
    # notebook_tag unset will overwrite each other's h5ads -- pass a distinct tag per notebook.
    trained_dir = os.path.join(out_root, "trained")
    os.makedirs(trained_dir, exist_ok=True)
    prefix = f"{notebook_tag}_" if notebook_tag else ""
    trained_h5ad_path = os.path.join(
        trained_dir,
        f"{prefix}sigma{shift_sigma}_seed{seed}_{model_tag}"
        f"_{_data_tag(branching_t, batch_mode)}_decipherseed_{decipher_seed}.h5ad",
    )
    t_before_write = time.perf_counter()
    adata.write(trained_h5ad_path)
    adata.uns["stage_seconds"] = {
        **adata.uns["stage_seconds"],
        "h5ad_write": time.perf_counter() - t_before_write,
    }

    return (
        abs(rho),
        arm_rhos["rho_plus"],
        arm_rhos["rho_minus"],
        branch_asw,
        trained_h5ad_path,
        r2_overall,
        r2_per_gene_median,
        adata,
    )
