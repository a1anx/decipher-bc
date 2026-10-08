"""Checks for the 1008 split-fix sweep and `decipher_models2` (Claude Files/plans/1007_splitfix-relay/).

Figure layout: the 1008 pipeline writes figures into the same relative tree the user hand-sorted
`0918/figs/` into, so a 1008 run and its 0918 twin pair up path for path.

Split fix: `decipher_models2` assigns the train/val split by position. The only training here is
a 3-epoch run per preset (~2 s each), which the relay plan asks for as the equivalence check.
"""

import importlib
import os
import sys

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt

import decipher_models
import decipher_models2
from decipher_models.presets import PRESETS as PRESETS_1
from decipher_models.tools.decipher import _make_train_val_split as split_by_name
from decipher_models2.presets import PRESETS as PRESETS_2
from decipher_models2.tools.decipher import _make_train_val_split as split_by_position

matplotlib.use("Agg")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "Simulated Data", "Simulated Data Sweep Pipeline")
FIGS_0918 = os.path.join(
    REPO, "Simulated Data", "Sweeps and Results", "shift_sigma_sweep_bifurcation", "0918", "figs"
)
sys.path.insert(0, PIPELINE)

splitfix = importlib.import_module("1008_simulated_data_pipeline_splitfix")
bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")


@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("kind", ["reconstruction", "vspace"])
@pytest.mark.parametrize("model", ["native", "model5"])
def test_fig_path_exists_in_the_sorted_0918_tree(tmp_path, model, kind, sigma):
    path = splitfix._fig_path(str(tmp_path), kind, model, "genes", 0.3, sigma, 3, 1)
    relative = os.path.relpath(path, tmp_path)
    assert os.path.isfile(os.path.join(FIGS_0918, relative)), relative


def test_fig_path_mixes_sigma_formats_like_0918(tmp_path):
    recon = splitfix._fig_path(str(tmp_path), "reconstruction", "model5", "genes", 0.3, 10.0, 3, 2)
    vspace = splitfix._fig_path(str(tmp_path), "vspace", "native", "genes", 0.3, 10.0, 4, 1)
    assert os.path.relpath(recon, tmp_path) == os.path.join(
        "model5", "bif", "sigma10.0", "reconstruction",
        "reconstruction_sigma10.0_seed3_model5_bif_genes_decipherseed_2.png",
    )  # fmt: skip
    assert os.path.relpath(vspace, tmp_path) == os.path.join(
        "native_genes", "bif", "sigma10.0", "vspace",
        "vspace_sigma10_seed4_native_bif_genes_decipherseed_1.png",
    )  # fmt: skip


# --- Step 4: positional train/val split in decipher_models2 -----------------------------------
# The bifurcation simulator repeats cell names ('0'..'499' in each of 5 batches). decipher_models
# picks validation cells by name (the documented bug); decipher_models2 picks them by position.
# Importing the private `_make_train_val_split` follows tests/1004_test_split_fix.py: the split is
# the unit under test and has no public entry point short of a full training run.


SIM = dict(n_batches=5, n_samples=500, n_genes=50, n_z_dims=3, shift_sigma=1.0, seed=3)


def _duplicate_name_data():
    adata = bif.shift_magnitudes_multivariate_bifurcation(**SIM, branching_t=0.3)
    assert not adata.obs_names.is_unique
    return adata


def _unique_name_copy(adata):
    unique = adata.copy()
    unique.obs_names_make_unique()
    return unique


def _split(split_fn, adata, seed=0):
    split_fn(adata, 0.1, seed)
    return adata.obs["decipher_split"]


def test_presets_are_native_and_model5_with_decipher_models_flags():
    assert PRESETS_2 == {name: PRESETS_1[name] for name in ("native", "model5")}


def test_positional_split_on_duplicate_names_is_exactly_90_10():
    split = _split(split_by_position, _duplicate_name_data())
    assert list(split.cat.categories) == ["train", "validation"]
    assert split.value_counts().to_dict() == {"train": 2250, "validation": 250}


def test_by_name_split_on_duplicate_names_is_the_documented_bug():
    split = _split(split_by_name, _duplicate_name_data())
    assert split.value_counts().to_dict() == {"train": 1490, "validation": 1010}


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_positional_split_picks_the_cells_by_name_split_picks_on_unique_names(seed):
    raw = _duplicate_name_data()
    unique = _unique_name_copy(raw)
    fixed = _split(split_by_position, raw, seed)
    reference = _split(split_by_name, unique, seed)
    assert np.array_equal(fixed.to_numpy(), reference.to_numpy())
    assert list(fixed.cat.categories) == list(reference.cat.categories)


def _train_losses(package, preset, adata, save_folder, monkeypatch):
    monkeypatch.setitem(package.utils.DECIPHER_GLOBALS, "save_folder", str(save_folder))
    presets = PRESETS_2 if package is decipher_models2 else PRESETS_1
    config = package.tools._decipher.DecipherConfig(**presets[preset], seed=1, n_epochs=3)
    decipher, val_losses = package.tl.decipher_train(adata, config, plot_kwargs={"color": "batch"})
    plt.close("all")
    return np.array(decipher.train_losses_), np.array(val_losses)


@pytest.mark.parametrize("preset", ["native", "model5"])
def test_short_training_on_raw_data_matches_decipher_models_on_unique_names(
    tmp_path, monkeypatch, preset
):
    raw = _duplicate_name_data()
    unique = _unique_name_copy(raw)
    new = _train_losses(decipher_models2, preset, raw, tmp_path / "new", monkeypatch)
    old = _train_losses(decipher_models, preset, unique, tmp_path / "old", monkeypatch)
    np.testing.assert_allclose(new[0], old[0], rtol=1e-6)
    np.testing.assert_allclose(new[1], old[1], rtol=1e-6)
