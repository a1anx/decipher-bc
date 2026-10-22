"""Compare the 1019 d3 sigma sweep (`decipher_models3`) across arms; no retraining.

Reads the sweep ledger CSV and the trained h5ads. Writes, under `--out`:
- `summary_by_arm_sigma.csv`: mean and sd over the runs of each (arm, sigma) cell, per metric;
- `paired_vs_native.csv`, `paired_vs_model5.csv`: per-pair deltas (arm - reference) joined on
  KEY minus `model`, summarized by arm x sigma x metric (mean, sd, n, frac_pos = share of pairs
  with delta > 0);
- `batch_asw_v_per_run.csv`: the cached per-h5ad batch ASW;
- `<metric>_vs_sigma.png`: one line per arm vs sigma (log x), band = mean +- sd over runs;
- with `--ledger-1008`: `side_by_side_1008_decipher_models2_vs_models3.csv`, native and model5 by
  sigma against the 1008 split-fix ledger. The 1008 numbers come from a DIFFERENT PACKAGE
  (`decipher_models2`), flagged in the file name and in the `package` column.

batch_asw_v, per h5ad: s_i = silhouette (euclidean) of cell i in `obsm["decipher_v"]` against
`obs["batch"]`, s = mean_i s_i in [-1, 1], and
    batch_asw_v = 1 - |s|
as in scIB: 1 = batches fully mixed (s = 0), 0 = batches fully separated (s = 1) or fully
anti-clustered (s = -1). NaN when there is a single batch. All cells (train and validation).

Usage (from this folder):
    python 1019_compare_d3_sweep.py --ledger <ledger.csv> --trained-dir <dir of h5ads> \
        [--out <dir>] [--ledger-1008 <1008 ledger.csv>]
Subset runs (added 2026-10-19 for the `sweep1019_d3ext` extension): `--ledger` may be repeated
(the ledgers are concatenated and must share no KEY), `--models` and `--sigma-min/--sigma-max`
(inclusive) keep only those runs, `--linear-x` plots sigma on a linear axis with one tick per
sigma, and `--references` names the arms paired against (default native and model5). E.g. the
4-arm sigma 7.5-11 extension:
    python 1019_compare_d3_sweep.py --ledger <1019>/sweep1019_d3_sweep_log.csv \
        --ledger <1019>/sweep1019_d3ext_sweep_log.csv --trained-dir <1019>/trained \
        --out <1019>/compare_d3_ext --models model5 model4r model5r model7 \
        --sigma-min 7.5 --sigma-max 11 --linear-x --references model5
h5ads are located as `--trained-dir` + basename of the ledger's `trained_h5ad`, since that
column can hold a pre-relocation path.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

KEY = ["model", "batch_mode", "bifurcation", "shift_sigma", "seed", "decipher_seed"]
PAIR_KEY = [c for c in KEY if c != "model"]
METRICS = [
    "rho",
    "rho_minus",
    "rho_plus",
    "branch_asw",
    "r2_overall",
    "r2_per_gene_median",
    "n_epochs",
    "best_epoch",
    "batch_asw_v",
]
# Categorical slots 1-5 of the dataviz reference palette (light), fixed order per arm.
ARM_COLOR = {
    "native": "#2a78d6",
    "model5": "#eb6834",
    "model4r": "#1baf7a",
    "model5r": "#eda100",
    "model7": "#e87ba4",
}
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(
    HERE, "..", "Sweeps and Results", "shift_sigma_sweep_bifurcation", "1019"
)


# ---------------------------------------------------------------- pure: batch ASW
def batch_asw(v: np.ndarray, batch: np.ndarray) -> float:
    """1 - |mean silhouette of `batch` in `v`|; NaN with fewer than 2 batches."""
    if len(np.unique(batch)) < 2:
        return float("nan")
    return float(1.0 - abs(silhouette_score(v, batch)))


# ---------------------------------------------------------------- pure: summary and pairing
def check_unique_key(ledger: pd.DataFrame) -> None:
    dup = ledger.duplicated(KEY)
    if dup.any():
        raise ValueError(f"ledger has {int(dup.sum())} duplicated KEY rows")


def concat_ledgers(ledgers: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack ledgers; raises if a KEY appears twice (within or across them)."""
    out = pd.concat(ledgers, ignore_index=True)
    check_unique_key(out)
    return out


def select_runs(
    ledger: pd.DataFrame,
    models: list[str] | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
) -> pd.DataFrame:
    """Rows of `models` (all if None) with sigma_min <= shift_sigma <= sigma_max (open if None)."""
    keep = pd.Series(True, index=ledger.index)
    if models is not None:
        absent = sorted(set(models) - set(ledger["model"]))
        if absent:
            raise ValueError(f"models not in ledger: {absent}")
        keep &= ledger["model"].isin(models)
    if sigma_min is not None:
        keep &= ledger["shift_sigma"] >= sigma_min
    if sigma_max is not None:
        keep &= ledger["shift_sigma"] <= sigma_max
    return ledger[keep]


def summarize_by_arm_sigma(ledger: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Mean and sd (ddof=1) over runs per (model, shift_sigma); columns `<metric>_mean/_sd`."""
    g = ledger.groupby(["model", "shift_sigma"])
    out = g[metrics].mean().add_suffix("_mean").join(g[metrics].std().add_suffix("_sd"))
    out.insert(0, "n", g.size())
    return out.reset_index()


def pair_deltas(ledger: pd.DataFrame, reference: str, metrics: list[str]) -> pd.DataFrame:
    """One row per non-reference run: KEY columns plus `metric` = arm - reference.

    Pairs on KEY minus `model`. Raises on a duplicated KEY, a missing reference model, or any
    arm whose set of pair keys differs from the reference's.
    """
    check_unique_key(ledger)
    ref = ledger[ledger["model"] == reference].set_index(PAIR_KEY)
    if ref.empty:
        raise ValueError(f"reference model {reference!r} not in ledger")
    parts = []
    for model, arm in ledger[ledger["model"] != reference].groupby("model"):
        arm = arm.set_index(PAIR_KEY)
        missing, extra = ref.index.difference(arm.index), arm.index.difference(ref.index)
        if len(missing) or len(extra):
            raise ValueError(
                f"{model} vs {reference}: {len(missing)} pairs missing, {len(extra)} unmatched"
            )
        delta = arm[metrics] - ref.loc[arm.index, metrics]
        delta.insert(0, "model", model)
        parts.append(delta.reset_index())
    return pd.concat(parts, ignore_index=True)


def summarize_deltas(deltas: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per (model, shift_sigma, metric): mean, sd, n, frac_pos of the pair deltas."""
    long = deltas.melt(
        id_vars=["model", "shift_sigma"], value_vars=metrics, var_name="metric", value_name="d"
    ).dropna(subset=["d"])
    g = long.groupby(["model", "shift_sigma", "metric"])["d"]
    out = pd.DataFrame(
        {
            "mean": g.mean(),
            "sd": g.std(),
            "n": g.size(),
            "frac_pos": g.apply(lambda s: (s > 0).mean()),
        }
    )
    return out.reset_index()


def side_by_side_1008(new: pd.DataFrame, old: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Native and model5 cell means by sigma: this sweep (decipher_models3) vs the 1008 ledger."""
    cols = [m for m in metrics if m in old.columns]
    frames = []
    for label, df in (("decipher_models3_1019", new), ("decipher_models2_1008", old)):
        sub = df[df["model"].isin(["native", "model5"])]
        grp = sub.groupby(["model", "shift_sigma"])
        t = grp[cols].mean()
        t.insert(0, "n", grp.size())
        t.insert(0, "package", label)
        frames.append(t.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["model", "shift_sigma", "package"])


# ---------------------------------------------------------------- I/O
def compute_batch_asw_per_run(ledger: pd.DataFrame, trained_dir: str) -> pd.Series:
    import anndata as ad

    vals = {}
    for idx, row in ledger.iterrows():
        path = os.path.join(trained_dir, os.path.basename(row["trained_h5ad"]))
        a = ad.read_h5ad(path)
        vals[idx] = batch_asw(np.asarray(a.obsm["decipher_v"]), a.obs["batch"].to_numpy())
    return pd.Series(vals, dtype=float)


def load_or_compute_batch_asw(ledger: pd.DataFrame, trained_dir: str, cache: str) -> pd.DataFrame:
    """Return ledger with `batch_asw_v`; per-run values are cached in `cache` (used if complete)."""
    if os.path.exists(cache):
        merged = ledger.merge(
            pd.read_csv(cache)[KEY + ["batch_asw_v"]], on=KEY, how="left", validate="1:1"
        )
        if merged["batch_asw_v"].notna().all():
            return merged
    out = ledger.copy()
    out["batch_asw_v"] = compute_batch_asw_per_run(ledger, trained_dir).values
    out[KEY + ["batch_asw_v"]].to_csv(cache, index=False)
    return out


def metric_figure(summary: pd.DataFrame, metric: str, log_x: bool = True) -> plt.Figure:
    """One line per arm vs sigma, band = mean +- sd; linear x gets one tick per sigma."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for model, g in summary.groupby("model"):
        g = g.sort_values("shift_sigma")
        color = ARM_COLOR.get(model, "#52514e")
        m, sd = g[f"{metric}_mean"].to_numpy(), g[f"{metric}_sd"].fillna(0).to_numpy()
        x = g["shift_sigma"].to_numpy()
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
        ax.plot(x, m, color=color, marker="o", ms=2.5, lw=1, mew=0, label=model)
    if log_x:
        ax.set_xscale("log")
    else:
        ax.set_xticks(sorted(summary["shift_sigma"].unique()))
    ax.set_xlabel("shift_sigma", color="#52514e")
    ax.set_ylabel(metric, color="#52514e")
    ax.set_title(
        f"{metric} vs shift_sigma (mean, band = sd over runs)",
        fontsize=10,
        color="#0b0b0b",
        loc="left",
    )
    ax.grid(color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_metric(summary: pd.DataFrame, metric: str, out_png: str, log_x: bool = True) -> None:
    fig = metric_figure(summary, metric, log_x)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ledger", required=True, action="append", help="repeatable")
    p.add_argument("--trained-dir", required=True)
    p.add_argument("--out", default=None, help="default: <1019 dir>/compare_d3")
    p.add_argument("--ledger-1008", default=None)
    p.add_argument("--models", nargs="+", default=None, help="keep only these arms")
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--linear-x", action="store_true", help="linear sigma axis (default log)")
    p.add_argument("--references", nargs="+", default=["native", "model5"])
    args = p.parse_args()
    out = args.out or os.path.join(DEFAULT_OUT_DIR, "compare_d3")
    os.makedirs(out, exist_ok=True)

    ledger = concat_ledgers([pd.read_csv(path) for path in args.ledger])
    ledger = select_runs(ledger, args.models, args.sigma_min, args.sigma_max)
    if ledger["error"].notna().any():
        raise ValueError("ledger has error rows")
    ledger = load_or_compute_batch_asw(
        ledger, args.trained_dir, os.path.join(out, "batch_asw_v_per_run.csv")
    )

    summary = summarize_by_arm_sigma(ledger, METRICS)
    summary.to_csv(os.path.join(out, "summary_by_arm_sigma.csv"), index=False)
    for ref in args.references:
        deltas = pair_deltas(ledger, ref, METRICS)
        summarize_deltas(deltas, METRICS).to_csv(
            os.path.join(out, f"paired_vs_{ref}.csv"), index=False
        )
    for metric in METRICS:
        plot_metric(summary, metric, os.path.join(out, f"{metric}_vs_sigma.png"), not args.linear_x)
    if args.ledger_1008:
        table = side_by_side_1008(ledger, pd.read_csv(args.ledger_1008), METRICS)
        table.to_csv(
            os.path.join(out, "side_by_side_1008_decipher_models2_vs_models3.csv"), index=False
        )
    print(f"wrote {sorted(os.listdir(out))} to {out}")


if __name__ == "__main__":
    main()
