r"""Post-hoc aggregate summary figures for the 504-run bifurcation sweep (tag `sweep0918_ds3`).

ADDED 2026-09-18. `train_and_compute_rho_r2_bifurcation`
(simulated_data_pipeline_bifurcation_sep18.py) never ported wandb_sigma_sweep.py's
`aggregate_runs`/`plot_aggregate` step, so this sweep's summary figures were never produced. This
script builds them post-hoc, purely from the sweep's CSV ledger -- no retraining, no h5ad reads,
no wandb API calls. It is a direct port of `aggregate_runs`/`plot_aggregate` from
wandb_sigma_sweep.py, adapted to this sweep's actual schema:

    model in {native, model2, model5} x batch_mode in {z, genes} x bifurcation in {bif, nobif}

collapsed into `arm` (a ready-made human label: "Base Decipher (z)", "Base Decipher (genes)",
"Set1 vz2", "Set3 zx2" -- 4 of them), which plays the role wandb_sigma_sweep.py's `model_tag` did.

`rho_plus`/`rho_minus`/`branch_asw` are only meaningful when bifurcation == "bif" (NaN otherwise
by design -- see the comment at simulated_data_pipeline_bifurcation_sep18.py:622), so this script
produces two separate figure sets rather than one, faceted by bifurcation state.
"""

import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(
    _SCRIPT_DIR,
    "..",
    "Simulated Adata",
    "shift_sigma_sweep_bifurcation",
    "0918",
    "sweep0918_ds3_sweep_log.csv",
)

AGG_KEYS = ["arm", "shift_sigma"]  # bifurcation is split into separate figure sets, not a facet

METRICS_BIF = ["rho", "rho_plus", "rho_minus", "branch_asw", "r2_overall", "r2_per_gene_median"]
METRICS_NOBIF = ["rho", "r2_overall", "r2_per_gene_median"]

METRIC_LABELS = {
    "rho": "rho (decipher_time vs latent_t)",
    "rho_plus": "rho, + arm",
    "rho_minus": "rho, - arm",
    "branch_asw": "branch silhouette (decipher_v)",
    "r2_overall": "R2 overall (reconstruction)",
    "r2_per_gene_median": "R2 per-gene median (reconstruction)",
}

# no fixed color table for `arm` values (unlike wandb_sigma_sweep.py's MODEL_COLORS). Colors are
# assigned by position in the (small, fixed) sorted arm list rather than a hash of the string --
# a hash risks collisions between similar labels (e.g. "Base Decipher (genes)" vs "Base Decipher
# (z)" differ only in a few characters and hashed to the same color in an earlier version of this
# script), which silently makes two different arms indistinguishable in the comparison panel.
_PALETTE = ["#C62828", "#2E7D32", "#1565C0", "#F9A825", "#00897B", "#7B1FA2", "#AD1457", "#EF6C00"]


def arm_colors(arms):
    return {arm: _PALETTE[i % len(_PALETTE)] for i, arm in enumerate(arms)}


def aggregate_runs(ok, metrics):
    """Collapse raw runs in the order the nesting demands: decipher_seeds first, then seeds.

    Direct port of wandb_sigma_sweep.py's `aggregate_runs`, with AGG_KEYS swapped to
    ["arm", "shift_sigma"]. `ok` must already be filtered to error-free rows.

    Returns
    -------
    world : DataFrame
        One row per (arm, shift_sigma, seed) -- the per-seed mean over decipher_seeds, plus that
        seed's SD across its decipher_seeds (`<metric>_sd_train`).
    summary : DataFrame
        One row per (arm, shift_sigma) -- `<metric>_mean` (mean of the per-seed means),
        `<metric>_sd_between` (SD across the per-seed means; the error bar worth plotting), and
        `<metric>_sd_within` (training noise, pooled over seeds as sqrt of mean variance).
    """
    seed_keys = AGG_KEYS + ["seed"]

    per_seed_mean = ok.groupby(seed_keys, as_index=False)[metrics].mean()
    per_seed_sd = ok.groupby(seed_keys, as_index=False)[metrics].std(ddof=1)

    world = per_seed_mean.merge(
        per_seed_sd.rename(columns={m: f"{m}_sd_train" for m in metrics}), on=seed_keys
    )

    # pooled within-seed spread is sqrt of the mean variance, not the mean of the SDs
    within_var = per_seed_sd.copy()
    within_var[metrics] = within_var[metrics] ** 2
    pooled = within_var.groupby(AGG_KEYS, as_index=False)[metrics].mean()
    pooled[metrics] = np.sqrt(pooled[metrics])

    summary = (
        per_seed_mean.groupby(AGG_KEYS, as_index=False)[metrics]
        .mean()
        .rename(columns={m: f"{m}_mean" for m in metrics})
        .merge(
            per_seed_mean.groupby(AGG_KEYS, as_index=False)[metrics]
            .std(ddof=1)
            .rename(columns={m: f"{m}_sd_between" for m in metrics}),
            on=AGG_KEYS,
        )
        .merge(pooled.rename(columns={m: f"{m}_sd_within" for m in metrics}), on=AGG_KEYS)
        .merge(ok.groupby(AGG_KEYS).size().rename("n_runs").reset_index(), on=AGG_KEYS)
        .merge(per_seed_mean.groupby(AGG_KEYS).size().rename("n_seeds").reset_index(), on=AGG_KEYS)
    )
    return world, summary


def plot_aggregate(metric, run_log, world, summary):
    """One figure per metric: two panels, one per `batch_mode`, each overlaying the two arms that
    share it -- "Base Decipher (genes)" with "Set3 zx2" (both batch_mode="genes"), and
    "Base Decipher (z)" with "Set1 vz2" (both batch_mode="z"). Grouping by batch_mode rather than
    a fixed pair of arm names means this stays correct if the sweep's arm set changes -- every
    arm's batch_mode is read straight from `run_log`, not hardcoded.

    Adapted from wandb_sigma_sweep.py's `plot_aggregate` (raw points, world-mean error bars,
    world-to-world SD band, all per arm), but replacing its "one panel per model + comparison
    panel" layout with this 2-column overlay, since here the comparison the user wants is
    specifically within each batch_mode pairing.
    """
    jitter_rng = np.random.default_rng(0)
    arms = sorted(summary["arm"].unique())
    colors = arm_colors(arms)

    arm_batch_mode = run_log.drop_duplicates("arm").set_index("arm")["batch_mode"].to_dict()
    groups = sorted({arm_batch_mode[a] for a in arms})

    fig, axes = plt.subplots(
        1, len(groups), figsize=(7 * len(groups), 4.8), sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()

    for gi, group in enumerate(groups):
        ax = axes[gi]
        group_arms = [a for a in arms if arm_batch_mode[a] == group]
        for arm in group_arms:
            color = colors[arm]
            raw = run_log[run_log["arm"] == arm].dropna(subset=[metric])
            wm = world[world["arm"] == arm].dropna(subset=[metric])
            s = summary[summary["arm"] == arm].dropna(subset=[f"{metric}_mean"]).sort_values(
                "shift_sigma"
            )

            jitter = np.exp(jitter_rng.normal(0, 0.045, size=len(raw)))
            ax.scatter(
                raw["shift_sigma"] * jitter,
                raw[metric],
                s=13,
                color=color,
                alpha=0.4,
                linewidths=0,
                zorder=2,
            )
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
            )
            ax.fill_between(
                s["shift_sigma"],
                s[f"{metric}_mean"] - s[f"{metric}_sd_between"],
                s[f"{metric}_mean"] + s[f"{metric}_sd_between"],
                color=color,
                alpha=0.15,
                lw=0,
                zorder=1,
            )
            ax.plot(
                s["shift_sigma"],
                s[f"{metric}_mean"],
                color=color,
                lw=2,
                marker="o",
                ms=4.5,
                zorder=4,
                label=f"{arm} (mean ± world-to-world SD)",
            )

        ax.set_xscale("log")
        ax.set_title(f"batch_mode = {group}", fontsize=11)
        ax.grid(alpha=0.18, lw=0.6)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    sigmas = sorted(summary["shift_sigma"].unique())
    for i, ax in enumerate(axes):
        ax.xaxis.set_major_locator(FixedLocator(sigmas))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{s:g}" for s in sigmas]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=7.5, labelbottom=True)
        ax.set_xlabel("shift_sigma (log scale)", fontsize=9)
        if i == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)

    n_seeds = int(summary["n_seeds"].max())
    n_runs = int(summary["n_runs"].max())
    fig.suptitle(
        f"{metric} -- {n_runs} runs per arm/sigma "
        f"({n_seeds} simulated worlds x {n_runs // max(n_seeds, 1)} training seeds)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="sweep resume-ledger CSV to aggregate")
    parser.add_argument(
        "--out-root",
        default=None,
        help="directory to write aggregate/<bif|nobif>/<metric>.png under (default: CSV's dir)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")]

    out_root = args.out_root or os.path.dirname(os.path.abspath(args.csv))
    agg_root = os.path.join(out_root, "aggregate")

    for bif_state, metrics in (("bif", METRICS_BIF), ("nobif", METRICS_NOBIF)):
        sub = df[df["bifurcation"] == bif_state]
        if sub.empty:
            print(f"no rows for bifurcation={bif_state!r}, skipping")
            continue
        world, summary = aggregate_runs(sub, metrics)
        out_dir = os.path.join(agg_root, bif_state)
        os.makedirs(out_dir, exist_ok=True)
        for metric in metrics:
            fig = plot_aggregate(metric, sub, world, summary)
            fig_path = os.path.join(out_dir, f"{metric}.png")
            fig.savefig(fig_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
