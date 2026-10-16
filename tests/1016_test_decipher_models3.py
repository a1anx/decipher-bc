"""Checks for `decipher_models3`: the seven presets and the positional train/val split.

Sections are appended piece by piece (`# --- P01: ...`, `# --- P02: ...`); each is self-contained.
"""

import importlib
import os
import sys

import numpy as np
import pytest

from decipher_models.presets import PRESETS as PRESETS_1
from decipher_models.tools.decipher import _make_train_val_split as split_by_name
from decipher_models3.presets import PRESETS as PRESETS_3
from decipher_models3.tools.decipher import _make_train_val_split as split_by_position

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Simulated Data", "Simulated Data Sweep Pipeline"))
_p01_bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")

# --- P01: presets and split --------------------------------------------------------------------
# The bifurcation simulator repeats cell names ('0'..'499' in each of 5 batches), so the split
# must be positional. The private `_make_train_val_split` is the unit under test, as in
# tests/1008_test_decipher_models2.py.

_P01_SIM = dict(n_batches=5, n_samples=500, n_genes=50, n_z_dims=3, shift_sigma=1.0, seed=3)


def _p01_duplicate_name_data():
    adata = _p01_bif.shift_magnitudes_multivariate_bifurcation(**_P01_SIM, branching_t=0.3)
    assert not adata.obs_names.is_unique
    return adata


def _p01_split(split_fn, adata, seed=0):
    split_fn(adata, 0.1, seed)
    return adata.obs["decipher_split"]


def test_presets_are_the_seven_decipher_models_presets():
    assert sorted(PRESETS_3) == [
        "model1",
        "model2",
        "model3",
        "model4",
        "model5",
        "model6",
        "native",
    ]
    assert PRESETS_3 == PRESETS_1


@pytest.mark.parametrize("name", sorted(PRESETS_1))
def test_preset_flags_equal_decipher_models(name):
    assert PRESETS_3[name] == PRESETS_1[name]


def test_positional_split_on_duplicate_names_is_exactly_90_10():
    split = _p01_split(split_by_position, _p01_duplicate_name_data())
    assert list(split.cat.categories) == ["train", "validation"]
    assert split.value_counts().to_dict() == {"train": 2250, "validation": 250}


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_positional_split_picks_the_cells_by_name_split_picks_on_unique_names(seed):
    raw = _p01_duplicate_name_data()
    unique = raw.copy()
    unique.obs_names_make_unique()
    fixed = _p01_split(split_by_position, raw, seed)
    reference = _p01_split(split_by_name, unique, seed)
    assert np.array_equal(fixed.to_numpy(), reference.to_numpy())


# --- P03: best-epoch restore -------------------------------------------------------------------
# `decipher_train` returns the weights of the lowest-val-NLL epoch, not the last. Arm-agnostic
# port of tests/1015_test_m5_restore_best.py: a full run must equal a run truncated right after its
# best epoch (same seed, no early stopping), compared on the whole state_dict and theta without
# naming any parameter. Run on one concat_z arm (model2) and one concat_x arm (model5).

_P03_EPOCHS = 30
_P03_BATCH_KEY = "batch"


def _p03_make_adata(n_cells=200, n_genes=50, n_batches=3, seed=0):
    """Synthetic integer counts with a categorical batch column."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=n_genes), size=(n_cells, n_genes))
    obs = pd.DataFrame(
        {_P03_BATCH_KEY: pd.Categorical([f"b{i % n_batches}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


def _p03_train(preset, n_epochs, save_folder):
    """Train `preset` without early stopping; return (state_dict, theta, val_losses, best_epoch)."""
    import matplotlib
    import pyro

    import decipher_models3 as dc
    from decipher_models3.tools._decipher import DecipherConfig
    from decipher_models3.utils import DECIPHER_GLOBALS

    matplotlib.use("Agg")
    DECIPHER_GLOBALS["save_folder"] = str(save_folder)
    config = DecipherConfig(
        **PRESETS_3[preset],
        seed=1,
        n_epochs=n_epochs,
        learning_rate=5e-2,
        early_stopping_patience=None,
    )
    model, val_losses = dc.tl.decipher_train(
        _p03_make_adata(),
        config,
        batch_key=_P03_BATCH_KEY,
        plot_every_k_epochs=-2,
        plot_kwargs={"color": _P03_BATCH_KEY},
    )
    theta = pyro.param("theta").detach().clone()
    return model.state_dict(), theta, val_losses, model.best_epoch_


@pytest.fixture(scope="module", params=["model2", "model5"])
def _p03_full_and_truncated(request, tmp_path_factory):
    """(full run, run truncated after the full run's best epoch) for one arm."""
    folder = tmp_path_factory.mktemp("ckpt")
    full = _p03_train(request.param, _P03_EPOCHS, folder)
    return full, _p03_train(request.param, full[3] + 1, folder)


def test_best_epoch_is_the_argmin_of_val_nll_and_not_the_last_epoch(_p03_full_and_truncated):
    (_, _, val_losses, best_epoch), _ = _p03_full_and_truncated
    assert len(val_losses) == _P03_EPOCHS
    assert best_epoch == int(np.argmin(val_losses))
    assert best_epoch < _P03_EPOCHS - 1  # the last epoch is worse, so the restore does something


def test_restored_weights_equal_run_stopped_at_best_epoch(_p03_full_and_truncated):
    import torch

    (full_sd, _, _, _), (trunc_sd, _, _, _) = _p03_full_and_truncated
    assert full_sd.keys() == trunc_sd.keys()
    for key in full_sd:
        torch.testing.assert_close(full_sd[key], trunc_sd[key], rtol=0, atol=0, msg=key)


def test_restored_theta_equals_run_stopped_at_best_epoch(_p03_full_and_truncated):
    import torch

    (_, full_theta, _, _), (_, trunc_theta, _, _) = _p03_full_and_truncated
    torch.testing.assert_close(full_theta, trunc_theta, rtol=0, atol=0)
