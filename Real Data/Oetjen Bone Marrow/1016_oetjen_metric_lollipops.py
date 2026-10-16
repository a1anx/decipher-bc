"""Native vs model5 vs model5e metric lollipops for oetjen5 and oetjen6, one PNG per dataset.

Reads ``<run dir>/csv/seed<k>_metrics.csv`` (k = 0, 1, 2) from each dataset's native, m5 and m5e
folders. One panel per (space, metric, branch) row on v and z (decipher_time_conservation left
out). Each panel has its own y axis, shared with the same panel in the other dataset's PNG. A panel
has three vertical lollipops, native, model5 and model5e: the stem runs from the lowest to the highest
seed and the three seeds are 50%-transparent circles on it, all one colour, so overlapping seeds
show as a darker dot. Rows: v batch, z batch, erythroid branch, myeloid branch.

Run from the repo root:
    .venv/bin/python "Real Data/Oetjen Bone Marrow/1016_oetjen_metric_lollipops.py"
Writes ``1016_<dataset>_metric_lollipops.png`` next to this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = {
    "oetjen5": ("1016_oetjen5_native", "1016_oetjen5_m5", "1016_oetjen5_m5e"),
    "oetjen6": ("1015_oetjen6_native", "1015_oetjen6_m5", "1016_oetjen6_m5e"),
}
MODELS = ["native", "model5", "model5e"]
SEEDS = [0, 1, 2]
DOT = "#2a78d6"
STEM = "#8a8984"
TEXT = "#52514e"
# (space, metric, branch) per grid row
ROWS = [
    [("v", "batch_asw", "all"), ("v", "graph_ilisi", "all"), ("v", "graph_connectivity", "all")],
    [("z", "batch_asw", "all"), ("z", "graph_ilisi", "all"), ("z", "graph_connectivity", "all")],
    [
        ("v", "trajectory_conservation", "erythroid"),
        ("z", "trajectory_conservation", "erythroid"),
    ],
    [
        ("v", "trajectory_conservation", "myeloid"),
        ("z", "trajectory_conservation", "myeloid"),
    ],
]


def load(dataset: str) -> pd.DataFrame:
    """Long table of every metric row with model and seed columns."""
    frames = []
    for model, folder in zip(MODELS, DATASETS[dataset]):
        for seed in SEEDS:
            df = pd.read_csv(HERE / folder / "csv" / f"seed{seed}_metrics.csv")
            frames.append(df.assign(model=model, seed=seed))
    return pd.concat(frames, ignore_index=True)


def select(df: pd.DataFrame, space: str, metric: str, branch: str) -> pd.DataFrame:
    return df[(df.space == space) & (df.metric == metric) & (df.branch == branch)]


def ylims(frames: list[pd.DataFrame]) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Per panel key, one padded y range covering every dataset's values."""
    out = {}
    for key in (k for row in ROWS for k in row):
        values = pd.concat([select(df, *key).value for df in frames])
        lo, hi = values.min(), values.max()
        pad = max(hi - lo, 0.02) * 0.15
        out[key] = (lo - pad, hi + pad)
    return out


def panel(ax: plt.Axes, df: pd.DataFrame, key: tuple[str, str, str], ylim: tuple) -> None:
    space, metric, branch = key
    sel = select(df, *key)
    ax.set_ylim(*ylim)
    for x, model in enumerate(MODELS):
        m = sel[sel.model == model]
        ax.vlines(x, m.value.min(), m.value.max(), color=STEM, lw=2, zorder=2)
        ax.scatter([x] * len(m), m.value, s=60, color=DOT, alpha=0.5, linewidth=0, zorder=3)
    branch_txt = "" if branch == "all" else f"\n{branch} branch"
    ax.set_title(f"{metric} ({space}){branch_txt}", fontsize=10, color="#0b0b0b")
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def plot(dataset: str, df: pd.DataFrame, limits: dict) -> Path:
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(10, 13), constrained_layout=True)
    for row, keys in zip(axes, ROWS):
        for ax in row[len(keys) :]:
            ax.set_visible(False)
        for ax, key in zip(row, keys):
            panel(ax, df, key, limits[key])
    fig.suptitle(
        f"{dataset}: native vs model5 vs model5e, one dot per decipherseed (0, 1, 2)", fontsize=13
    )
    out = HERE / f"1016_{dataset}_metric_lollipops.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    frames = {dataset: load(dataset) for dataset in DATASETS}
    limits = ylims(list(frames.values()))
    for dataset, df in frames.items():
        print(plot(dataset, df, limits))


if __name__ == "__main__":
    main()
