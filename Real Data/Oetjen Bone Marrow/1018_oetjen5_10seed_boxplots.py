"""Boxplot grids of every suite metric for the 10-seed oetjen5 d3 arms (seeds 0-9).

Arms on the x axis: native (`1016_oetjen5_d3_native`), model5r, model7, model4r. Per-seed values
come from the suite table's `seed_row` (same metrics and stratified mixing as
`suite_per_seed.csv`, which only covers seeds 0-2). Each arm is compared with native over paired
seeds by the paired-test script's exact two-sided sign-flip p (no multiplicity correction).

Writes, next to this script:
  1016_oetjen5_suite/1018_10seed_per_seed.csv   (one row per arm x seed; reused unless --recompute)
  1018_oetjen5_10seed_boxplots_{training,mixing,conservation}.png

Run from the repo root: PYTHONPATH=. .venv/bin/python "Real Data/Oetjen Bone Marrow/1018_oetjen5_10seed_boxplots.py"
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paired = _load("oetjen5_paired", "1017_oetjen5_paired_tests.py")
suite = paired.suite

SEEDS = tuple(range(10))
# x-axis label -> run folder
ARMS = {
    "native": "1016_oetjen5_d3_native",
    "model5r": "1016_oetjen5_d3_model5r",
    "model7": "1016_oetjen5_d3_model7",
    "model4r": "1016_oetjen5_d3_model4r",
}
REF = "native"
COLORS = {"native": "#8a8a85", "model5r": "#2a78d6", "model7": "#eb6834", "model4r": "#1baf7a"}
CACHE = HERE / "1016_oetjen5_suite" / "1018_10seed_per_seed.csv"
OUT_STEM = "1018_oetjen5_10seed_boxplots"
# figure name -> panel layout (rows of suite columns; None leaves a cell empty)
FIGURES = {
    "training": [
        ["epochs_run", "best_epoch", "best_val_nll"],
        ["r2_overall", "r2_median_per_gene", None],
    ],
    "mixing": [
        [
            "v_graph_ilisi",
            "v_batch_asw",
            "v_strat_ilisi",
            "v_strat_batch_asw",
            "v_graph_connectivity",
        ],
        [
            "z_graph_ilisi",
            "z_batch_asw",
            "z_strat_ilisi",
            "z_strat_batch_asw",
            "z_graph_connectivity",
        ],
    ],
    "conservation": [
        ["v_traj_cons_erythroid", "v_traj_cons_myeloid"],
        ["z_traj_cons_erythroid", "z_traj_cons_myeloid"],
        ["decipher_time_cons_erythroid", "decipher_time_cons_myeloid"],
    ],
}
TITLES = {
    "training": "training and fit",
    "mixing": "batch mixing, incl. graph connectivity (scIB batch-correction metric); top v, bottom z",
    "conservation": "trajectory conservation (top v, middle z, bottom decipher time)",
}


def per_seed_table(recompute: bool) -> pd.DataFrame:
    """One row per arm x seed with every suite column; cached to CACHE."""
    if CACHE.exists() and not recompute:
        return pd.read_csv(CACHE)
    rows = []
    for arm, folder in ARMS.items():
        rows += [{"model": arm, **r} for r in suite.run_rows(HERE / folder, SEEDS, arm, [])]
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


def signflip_vs_ref(df: pd.DataFrame, arm: str, col: str) -> float:
    """Exact sign-flip p of arm - REF over seeds present in both."""
    a = df[df.model == arm].set_index("seed")[col]
    r = df[df.model == REF].set_index("seed")[col]
    common = a.index.intersection(r.index)
    return paired.sign_flip_p((a[common] - r[common]).to_numpy(dtype=float))


def draw_panel(ax, df: pd.DataFrame, col: str, rng: np.random.Generator) -> None:
    """Boxes, seed dots and sign-flip p vs REF for one suite column."""
    arms = list(ARMS)
    data = [df[df.model == a][col].to_numpy(dtype=float) for a in arms]
    bp = ax.boxplot(
        data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "black"}
    )
    for patch, arm in zip(bp["boxes"], arms):
        patch.set_facecolor(COLORS[arm])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[arm])
    for i, (vals, arm) in enumerate(zip(data, arms), start=1):
        x = i + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(x, vals, s=16, color=COLORS[arm], edgecolor="white", linewidth=0.6, zorder=3)
    lo = min(v.min() for v in data)
    hi = max(v.max() for v in data)
    pad = (hi - lo) * 0.18 or 0.05
    ax.set_ylim(lo - pad * 0.4, hi + pad)
    for i, arm in enumerate(arms[1:], start=2):
        p = signflip_vs_ref(df, arm, col)
        star = "*" if p < paired.ALPHA else ""
        ax.text(
            i,
            hi + pad * 0.45,
            f"p={p:.3g}{star}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold" if star else "normal",
            color="#222" if star else "#777",
        )
    ax.set_title(col, fontsize=10)
    ax.set_xticks(range(1, len(arms) + 1), arms, fontsize=8.5)
    ax.grid(axis="y", color="#e4e4e0", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(df: pd.DataFrame, name: str, layout: list[list[str | None]]) -> Path:
    nrow, ncol = len(layout), max(len(r) for r in layout)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow + 0.5), squeeze=False)
    rng = np.random.default_rng(0)
    for r, row in enumerate(layout):
        for c in range(ncol):
            col = row[c] if c < len(row) else None
            if col is None:
                axes[r, c].set_visible(False)
            else:
                draw_panel(axes[r, c], df, col, rng)
    fig.suptitle(
        f"oetjen5 d3 arms, seeds 0-9: {TITLES[name]}\none dot per seed; p = exact paired "
        "sign-flip vs native (two-sided, * p < 0.05, no multiplicity correction)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / f"{OUT_STEM}_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="ignore the per-seed cache")
    args = parser.parse_args()
    df = per_seed_table(args.recompute)
    counts = df.groupby("model").seed.nunique()
    print(counts.to_string())
    placed = {c for layout in FIGURES.values() for row in layout for c in row if c}
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    unplaced = numeric - placed - {"seed", "sanity_pass"}
    if unplaced:
        raise ValueError(f"suite columns in no figure: {sorted(unplaced)}")
    for name, layout in FIGURES.items():
        plot(df, name, layout)


if __name__ == "__main__":
    main()
