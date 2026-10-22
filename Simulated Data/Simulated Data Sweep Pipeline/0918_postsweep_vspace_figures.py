r"""Post-hoc v-space diagnostic figures for the 504-run bifurcation sweep (tag `sweep0918_ds3`).

ADDED 2026-09-18. `train_and_compute_rho_r2_bifurcation`
(0918_simulated_data_pipeline_bifurcation.py) only ports the reconstruction-figure block from
the earlier sigma sweep script -- its v-space figure block was never carried
over, so no run in this sweep has a `vspace_*.png`. This script fills that gap post-hoc, purely
from each run's saved `.h5ad` -- no retraining. `decipher_rotate_space`/`decipher_time` already
ran before the h5ad was written (0918_simulated_data_pipeline_bifurcation.py:613-614,665), so
`adata.obsm["decipher_v"]` and `adata.obs["decipher_time"]` are already there to plot.
"""

import argparse
import importlib
import os
import sys

import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt

# `decipher_models` is a directory at the repo root, not an installed package -- same sys.path
# fix as 0918_simulated_data_pipeline_bifurcation.py, needed here too since that module is only
# imported below, after this script's own top-level `decipher_models` import.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import decipher_models as dc  # noqa: E402

_data_tag = importlib.import_module("0918_simulated_data_pipeline_bifurcation")._data_tag

DEFAULT_CSV = os.path.join(
    _SCRIPT_DIR,
    "..",
    "Sweeps and Results",
    "shift_sigma_sweep_bifurcation",
    "0918",
    "sweep0918_ds3_sweep_log.csv",
)


def vspace_path(figs_dir, row):
    branching_t = 0.3 if row["bifurcation"] == "bif" else None
    stem = (
        f"sigma{row['shift_sigma']:g}_seed{int(row['seed'])}_{row['model']}"
        f"_{_data_tag(branching_t, row['batch_mode'])}_decipherseed_{int(row['decipher_seed'])}"
    )
    return os.path.join(figs_dir, f"vspace_{stem}.png")


def make_figure(adata, row):
    fig = dc.pl.decipher(
        adata,
        color=["batch", "latent_t", "decipher_time"],
        basis="decipher_v",
        ncols=3,
    )
    fig.suptitle(
        f"{row['arm']} | {row['bifurcation']} | sigma={row['shift_sigma']:g} "
        f"| seed={int(row['seed'])} | decipher seed={int(row['decipher_seed'])}",
        y=1.05,
        fontsize=11,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="sweep resume-ledger CSV to read")
    parser.add_argument(
        "--figs-dir",
        default=None,
        help="directory to write vspace_*.png under (default: CSV's dir / figs)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    figs_dir = args.figs_dir or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    n_written = 0
    for _, row in df.iterrows():
        out_path = vspace_path(figs_dir, row)
        adata = ad.read_h5ad(row["trained_h5ad"])
        fig = make_figure(adata, row)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        n_written += 1
        print(f"[{n_written}/{len(df)}] wrote {out_path}")

    print(f"done: {n_written} figures written to {figs_dir}")


if __name__ == "__main__":
    main()
