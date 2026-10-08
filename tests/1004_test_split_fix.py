"""No-training checks for the train/val split fix (Claude Files/plans/1007_splitfix-rerun-plan.md).

The bifurcation simulator's `ad.concat` repeats cell names across batches, and
`_make_train_val_split` picks validation cells by name, so every pick marks one name in all
batches. `obs_names_make_unique()` restores a true 90/10 split without touching model code; the
1008 sweep instead trains through `decipher_models2`, where the split is fixed in code.
"""

import importlib
import os
import sys

import numpy as np

from decipher_models.tools.decipher import _make_train_val_split

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Simulated Data",
        "Simulated Data Sweep Pipeline",
    ),
)

bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")
sweep = importlib.import_module("1008_sweep_splitfix_bifurcation")
splitfix = importlib.import_module("1008_simulated_data_pipeline_splitfix")

SIM = dict(n_batches=5, n_samples=500, n_genes=50, n_z_dims=3, shift_sigma=1.0, seed=3)


def _split_counts(adata):
    _make_train_val_split(adata, 0.1, 0)
    vc = adata.obs["decipher_split"].value_counts()
    return int(vc["train"]), int(vc["validation"])


def test_default_names_repeat_and_split_is_wrong():
    adata = bif.shift_magnitudes_multivariate_bifurcation(**SIM, branching_t=0.3)
    assert not adata.obs_names.is_unique
    n_train, n_val = _split_counts(adata)
    assert n_val > 250 and n_train + n_val == 2500


def test_unique_names_give_exact_split_and_same_counts():
    old = bif.shift_magnitudes_multivariate_bifurcation(**SIM, branching_t=0.3)
    new = old.copy()
    new.obs_names_make_unique()
    assert new.obs_names.is_unique
    assert _split_counts(new) == (2250, 250)
    assert np.array_equal(np.asarray(old.X), np.asarray(new.X))


def test_splitfix_function_is_the_0918_one_plus_run_date():
    import inspect

    old = inspect.signature(bif.train_and_compute_rho_r2_bifurcation).parameters
    new = inspect.signature(splitfix.train_and_compute_rho_r2_bifurcation).parameters
    assert set(new) - set(old) == {"run_date", "arm"}
    assert set(old) <= set(new)


def test_sweep_grid_is_the_0918_bif_genes_subset_for_native_and_model5():
    js = sweep.jobs(run_date="1008")
    assert sweep.SHIFT_SIGMAS == [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
    assert len(js) == 2 * 7 * 3 * 3 == 126
    assert all(j["run_date"] == "1008" for j in js)
    assert {j["bifurcation"] for j in js} == {"bif"}
    assert {(j["model"], j["batch_mode"]) for j in js} == {("native", "genes"), ("model5", "genes")}
    # each job's KEY also exists in the 0918 grid, so the two ledgers join 1:1
    old = importlib.import_module("0918_sweep_3models_bifurcation")
    old_keys = {tuple(str(j[k]) for k in old.KEY) for j in old.jobs()}
    assert {tuple(str(j[k]) for k in sweep.KEY) for j in js} <= old_keys
