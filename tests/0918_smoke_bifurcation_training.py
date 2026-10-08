"""End-to-end smoke test for `train_and_compute_rho_r2_bifurcation`.

NOT part of the fast suite -- this one trains a model and takes minutes. Kept out of
`0918_test_bifurcation_pipeline.py` so that file stays runnable in seconds.

Run directly:

    python tests/0918_smoke_bifurcation_training.py [model_name]

`model_name` defaults to "decipher" (the baseline arm), which is the only variant importable
without the per-variant packages installed. Pass a variant name to exercise `_import_decipher`
against it before launching a sweep.
"""

import importlib
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Simulated Data", "Simulated Data Sweep Pipeline"))

bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")

warnings.filterwarnings("ignore")


def main(model="decipher"):
    rho, rho_plus, rho_minus, branch_asw, path, r2_overall, r2_med, adata = (
        bif.train_and_compute_rho_r2_bifurcation(
            model,
            decipher_seed=1,
            shift_sigma=0.5,
            seed=3,
            n_batches=3,
            n_samples=300,
            n_genes=100,
            n_z_dims=3,
            branching_t=0.3,
            notebook_tag="verify0918",
        )
    )

    n_nan = adata.obs["decipher_time"].isna().sum()
    print("\n" + "=" * 60)
    print(f"  model              : {model}")
    print(f"  trajectories built : {adata.uns['trajectory_names']}")
    print(f"  rho       (pooled) : {rho:.3f}")
    print(f"  rho_plus           : {rho_plus:.3f}")
    print(f"  rho_minus          : {rho_minus:.3f}")
    print(f"  branch_asw         : {branch_asw:.3f}")
    print(f"  r2_overall         : {r2_overall:.3f}")
    print(f"  r2_per_gene_median : {r2_med:.3f}")
    print(f"  decipher_time NaNs : {n_nan} / {adata.n_obs}")
    print(f"  wrote              : {os.path.relpath(path, REPO_ROOT)}")

    fails = []
    if len(adata.uns["trajectory_names"]) != 2:
        fails.append("expected 2 trajectories")
    if not all(np.isfinite([rho, rho_plus, rho_minus, branch_asw, r2_overall, r2_med])):
        fails.append("a metric came back non-finite")
    if n_nan > 0.05 * adata.n_obs:
        fails.append(f"{n_nan} NaN decipher_time values")
    if not os.path.exists(path):
        fails.append("h5ad not written")

    print("=" * 60)
    print("SMOKE TEST PASSED" if not fails else "SMOKE TEST FAILURES: " + "; ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "decipher"))
