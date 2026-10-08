"""Compare the split-fix sweep (`sweep1008_splitfix`) against its control (`sweep1008_control`).

ADDED 2026-10-08 (split-fix relay, Step 6). Both sweeps run the same 126 jobs (native, model5;
genes; bif; 0918 sigmas; data seeds 3-5; decipher seeds 1-3) on the same machine type. The only
difference is the train/val split: the control goes through the unmodified 0918 path
(`decipher_models`, split by obs name, ~58/42), the split-fix sweep through `decipher_models2`
(positional split, 2250/250). Each pair is joined on `KEY`.

Caveat: the fix changes two things at once -- the training set (~1450 -> 2250 cells) and the
early-stopping validation set (~1050 -> 250 cells). This comparison cannot separate them.

Per pair it computes new - control for each metric. Per arm x sigma it reports the mean delta
over the 9 pairs (3 data seeds x 3 decipher seeds) against two noise scales:
- the paired SE and a one-sample t-test / Wilcoxon signed-rank on the 9 deltas;
- the decipher-seed noise SD: the SD across decipher seeds within a data seed, pooled over the
  3 data seeds and both sweeps. A difference between two independent trainings has SD
  sqrt(2) times that, so it is the floor a single-pair delta has to clear.
Control `n_epochs` (not in its CSV) comes from the wandb `train_elbo` history length; control
`n_train`/`n_val` (not in its CSV) from its h5ads' `obs["decipher_split"]`.

Usage (from this folder):
    python 1008_compare_splitfix.py               # reads wandb for control n_epochs
    python 1008_compare_splitfix.py --no-wandb    # control n_epochs left NaN
Writes `<out-dir>/{pairs.csv, by_arm_sigma.csv, by_arm.csv, delta_vs_sigma.png, summary.txt}`.
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

KEY = ["model", "batch_mode", "bifurcation", "shift_sigma", "seed", "decipher_seed"]
METRICS = ["rho", "rho_plus", "rho_minus", "branch_asw", "r2_overall"]
WANDB_PATH = "jpark-columbia/decipher-bc-sigma-sweep"
CONTROL_TAG = "sweep1008_control"
NEW_TAG = "sweep1008_splitfix"

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "..", "Sweeps and Results", "shift_sigma_sweep_bifurcation", "1008")


def pair_runs(control: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Join the two ledgers 1:1 on KEY; raise if any key is unmatched or duplicated."""
    for name, df in (("control", control), ("new", new)):
        dup = df.duplicated(KEY)
        if dup.any():
            raise ValueError(f"{name} ledger has {int(dup.sum())} duplicated KEY rows")
    merged = control.merge(new, on=KEY, how="outer", suffixes=("_control", "_new"), indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} unmatched keys:\n{unmatched[KEY + ['_merge']]}")
    return merged.drop(columns="_merge")


def add_deltas(pairs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add `d_<col>` = new - control for each column."""
    out = pairs.copy()
    for col in columns:
        out[f"d_{col}"] = out[f"{col}_new"] - out[f"{col}_control"]
    return out


def decipher_seed_noise_sd(pairs: pd.DataFrame, metric: str) -> pd.Series:
    """Pooled SD across decipher seeds within (model, sigma, data seed), over both sweeps."""
    long = pd.concat(
        [
            pairs[["model", "shift_sigma", "seed", f"{metric}_{s}"]].rename(
                columns={f"{metric}_{s}": "v"}
            )
            for s in ("control", "new")
        ],
        keys=["control", "new"],
        names=["sweep"],
    ).reset_index(level="sweep")
    var = long.groupby(["model", "shift_sigma", "sweep", "seed"])["v"].var(ddof=1)
    return np.sqrt(var.groupby(["model", "shift_sigma"]).mean())


def summarize(deltas: np.ndarray) -> dict:
    """Mean, SD, SE, t-test and Wilcoxon p, and number of positive deltas."""
    d = deltas[~np.isnan(deltas)]
    out = {"n": len(d), "mean": d.mean(), "sd": d.std(ddof=1), "n_pos": int((d > 0).sum())}
    out["se"] = out["sd"] / np.sqrt(len(d))
    out["p_t"] = stats.ttest_1samp(d, 0.0).pvalue
    out["p_wilcoxon"] = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return out


def by_arm_sigma(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (model, sigma, metric): delta summary plus the decipher-seed noise SD."""
    rows = []
    for metric in METRICS + ["n_epochs"]:
        noise = decipher_seed_noise_sd(pairs, metric) if metric in METRICS else None
        for (model, sigma), g in pairs.groupby(["model", "shift_sigma"]):
            row = {"model": model, "shift_sigma": sigma, "metric": metric}
            row["control_mean"] = g[f"{metric}_control"].mean()
            row["new_mean"] = g[f"{metric}_new"].mean()
            row.update(summarize(g[f"d_{metric}"].to_numpy(dtype=float)))
            row["noise_sd"] = np.nan if noise is None else noise.loc[(model, sigma)]
            rows.append(row)
    return pd.DataFrame(rows)


def by_arm(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (model, metric), all 63 pairs pooled over sigma."""
    rows = []
    for metric in METRICS + ["n_epochs"]:
        for model, g in pairs.groupby("model"):
            row = {"model": model, "metric": metric}
            row.update(summarize(g[f"d_{metric}"].to_numpy(dtype=float)))
            rows.append(row)
    return pd.DataFrame(rows)


def read_split_counts(h5ad_paths: pd.Series) -> pd.DataFrame:
    """n_train / n_val from each h5ad's obs['decipher_split']."""
    rows = []
    for path in h5ad_paths:
        counts = ad.read_h5ad(path, backed="r").obs["decipher_split"].value_counts()
        rows.append({"n_train": int(counts.get("train", 0)), "n_val": int(counts["validation"])})
    return pd.DataFrame(rows, index=h5ad_paths.index)


def fetch_wandb_epochs(run_ids: pd.Series) -> pd.Series:
    """Number of logged `train_elbo` steps per wandb run (= epochs trained); NaN without an id."""
    import wandb

    api = wandb.Api(timeout=60)
    out = {}
    for idx, run_id in run_ids.items():
        if pd.isna(run_id):
            out[idx] = np.nan
            continue
        run = api.run(f"{WANDB_PATH}/{run_id}")
        # scan_history(keys=...) can also return the post-training rows with train_elbo None.
        rows = run.scan_history(keys=["train_elbo"])
        out[idx] = sum(1 for row in rows if row.get("train_elbo") is not None)
    return pd.Series(out, dtype=float)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def plot_deltas(table: pd.DataFrame, out_png: str) -> None:
    """Mean delta +- 95% CI vs sigma, one panel per metric, one line per model."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.6), sharex=True)
    for ax, metric in zip(axes, METRICS):
        sub = table[table.metric == metric]
        for model, g in sub.groupby("model"):
            g = g.sort_values("shift_sigma")
            ci = stats.t.ppf(0.975, g["n"] - 1) * g["se"]
            ax.errorbar(g.shift_sigma, g["mean"], yerr=ci, marker="o", capsize=3, label=model)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"Δ{metric} (fixed − original)")
        ax.set_xlabel("shift_sigma")
    axes[0].legend()
    fig.suptitle(f"{NEW_TAG} − {CONTROL_TAG}: mean over 9 pairs, 95% CI", y=1.02)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--control", default=os.path.join(SWEEP_DIR, f"{CONTROL_TAG}_sweep_log.csv"))
    p.add_argument("--new", default=os.path.join(SWEEP_DIR, f"{NEW_TAG}_sweep_log.csv"))
    p.add_argument("--out-dir", default=os.path.join(SWEEP_DIR, "compare_splitfix"))
    p.add_argument("--no-wandb", action="store_true", help="skip control n_epochs from wandb")
    args = p.parse_args()

    control, new = pd.read_csv(args.control), pd.read_csv(args.new)
    for name, df in (("control", control), ("new", new)):
        if df["error"].notna().any():
            raise ValueError(f"{name} ledger has error rows")
    control[["n_train", "n_val"]] = read_split_counts(control["trained_h5ad"].map(resolve))
    control["n_epochs"] = np.nan if args.no_wandb else fetch_wandb_epochs(control["wandb_run_id"])
    pairs = add_deltas(pair_runs(control, new), METRICS + ["n_epochs"])
    assert len(pairs) == 126, len(pairs)

    os.makedirs(args.out_dir, exist_ok=True)
    keep = KEY + [
        f"{c}_{s}" for c in METRICS + ["n_epochs", "n_train", "n_val"] for s in ("control", "new")
    ]
    keep += [f"d_{c}" for c in METRICS + ["n_epochs"]]
    keep += ["wandb_run_id_control", "wandb_run_id_new", "git_sha"]
    pairs[keep].to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)
    table = by_arm_sigma(pairs)
    table.to_csv(os.path.join(args.out_dir, "by_arm_sigma.csv"), index=False)
    pooled = by_arm(pairs)
    pooled.to_csv(os.path.join(args.out_dir, "by_arm.csv"), index=False)
    plot_deltas(table, os.path.join(args.out_dir, "delta_vs_sigma.png"))

    splits = (
        pairs.groupby("decipher_seed")[
            ["n_train_control", "n_val_control", "n_train_new", "n_val_new"]
        ]
        .agg(lambda s: "/".join(map(str, sorted(s.unique()))))
        .to_string()
    )
    lines = [
        f"control: {CONTROL_TAG} ({os.path.basename(args.control)})",
        f"new:     {NEW_TAG} ({os.path.basename(args.new)})",
        f"pairs: {len(pairs)}; control rows without wandb id (n_epochs NaN): "
        f"{int(pairs['n_epochs_control'].isna().sum())}",
        "caveat: the fix changes training cells (~1450 -> 2250) AND the early-stopping validation"
        " set (~1050 -> 250) at once; this comparison cannot separate the two.",
        "",
        "split sizes by decipher seed:",
        splits,
        "",
        "pooled over sigma (63 pairs per model):",
        pooled.to_string(float_format=lambda x: f"{x:.4g}"),
        "",
        "by model x sigma (9 pairs each; noise_sd = decipher-seed SD):",
        table.to_string(float_format=lambda x: f"{x:.4g}"),
    ]
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(line.rstrip() for line in "\n".join(lines).split("\n")) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
