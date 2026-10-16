"""End-to-end smoke test for the seven `decipher_models3` presets.

NOT collected by pytest (the name has no `test_` prefix): it trains seven tiny models on CPU,
about 30 s in total. Run it directly from the repo root:

    .venv/bin/python tests/1016_smoke_decipher_models3.py [preset ...]

With no arguments it runs native and model1-model6. For each preset it trains 3 epochs on up to
1000 cells of a local simulated bifurcation h5ad (raw counts only; trained obsm/uns are stripped),
then runs `trajectories`, `decipher_time` and `gene_patterns` (batch-averaged, and for batch arms
also one named batch). It prints one line per preset and exits non-zero on any failure.
Checkpoints go to a temp directory, never into the repo.
"""

import os
import sys
import tempfile
import time
import traceback
import warnings

import anndata as ad
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import decipher_models3 as dc  # noqa: E402
from decipher_models3.presets import PRESETS  # noqa: E402
from decipher_models3.utils import DECIPHER_GLOBALS  # noqa: E402

warnings.filterwarnings("ignore")

DATA = os.path.join(
    REPO_ROOT,
    "Simulated Data",
    "Sweeps and Results",
    "shift_sigma_sweep_bifurcation",
    "1008",
    "trained",
    "sweep1008_splitfix_sigma1.0_seed4_model5_bif_genes_decipherseed_2.h5ad",
)
BATCH_KEY = "batch"
N_EPOCHS = 3
MAX_CELLS = 1000


def load_counts(path: str = DATA, max_cells: int = MAX_CELLS) -> ad.AnnData:
    """Raw counts and the batch / latent_t columns only, subsampled and with unique names."""
    full = ad.read_h5ad(path)
    keep = np.sort(np.random.default_rng(0).permutation(full.n_obs)[:max_cells])
    adata = ad.AnnData(
        X=full.layers["counts"][keep].copy(), obs=full.obs.iloc[keep][[BATCH_KEY, "latent_t"]]
    )
    adata.obs_names_make_unique()
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
    return adata


def run_preset(name: str, adata: ad.AnnData) -> str:
    """Train, infer trajectories and gene patterns for one preset; return the summary line."""
    adata = adata.copy()
    n_batches = adata.obs[BATCH_KEY].cat.categories.size
    config = dc.tl.DecipherConfig(**PRESETS[name], n_batches=n_batches, seed=0, n_epochs=N_EPOCHS)
    # Pin torch's thread count before every training, as the sweep drivers do: the thread count
    # sets the float reduction order, and the previous preset's `cell_clusters` (numba, via
    # pynndescent) raises torch's shared OpenMP count from the physical-core default to every
    # logical CPU, so otherwise val NLL depends on which presets ran before.
    torch.set_num_threads(1)
    t0 = time.time()
    model, val_losses = dc.tl.decipher_train(
        adata, config, batch_key=BATCH_KEY, plot_kwargs={"color": BATCH_KEY}
    )
    t_train = time.time() - t0

    assert model.best_epoch_ is not None, "best_epoch_ not set"
    assert len(val_losses) == N_EPOCHS and np.isfinite(val_losses).all(), "bad val losses"
    for key in ("decipher_v", "decipher_z"):
        assert np.isfinite(adata.obsm[key]).all(), f"non-finite {key}"

    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    # `trajectories` drops clusters of 10 cells or fewer, so keep only larger ones.
    clusters = adata.obs["decipher_clusters"]
    big = clusters.value_counts()[lambda counts: counts > 10].index
    by_time = adata.obs[clusters.isin(big)].groupby(clusters, observed=True)["latent_t"].mean()
    order = by_time.sort_values().index
    assert len(order) >= 2, "fewer than two clusters"
    dc.tl.trajectories(adata, dc.tl.TConfig("traj", cluster_ids_list=[str(c) for c in order]))
    dc.tl.decipher_time(adata)
    assert np.isfinite(adata.obs["decipher_time"].dropna()).all(), "non-finite decipher_time"

    batch_names = [None] if name == "native" else [None, adata.obs[BATCH_KEY].cat.categories[0]]
    for batch in batch_names:
        dc.tl.gene_patterns(adata, n_samples=20, batch=batch)
        gp = adata.uns["decipher"]["gene_patterns"]["traj"]
        for key in ("mean", "q25", "q75", "mean2"):
            assert np.isfinite(gp[key]).all(), f"non-finite gene_patterns {key} (batch={batch})"
        assert (gp["q25"] <= gp["q75"]).all(), f"q25 > q75 (batch={batch})"

    return (
        f"{name:7s} best_epoch={model.best_epoch_} val_nll={min(val_losses):.2f} "
        f"clusters={len(order)} train={t_train:.1f}s total={time.time() - t0:.1f}s"
    )


def main(presets: list) -> int:
    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
        print(f"unknown preset(s) {unknown}; choose from {sorted(PRESETS)}")
        return 2
    adata = load_counts()
    n_failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        DECIPHER_GLOBALS["save_folder"] = tmp + os.sep
        for name in presets:
            try:
                print(f"OK   {run_preset(name, adata)}", flush=True)
            except Exception:
                n_failed += 1
                print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print("SMOKE TEST PASSED" if not n_failed else f"SMOKE TEST FAILED: {n_failed} preset(s)")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:] or ["native", "model1", "model2", "model3", "model4", "model5", "model6"])
    )
