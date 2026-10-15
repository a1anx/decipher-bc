"""decipher_m5 `decipher_train` returns the weights of the lowest-val-NLL epoch, not the last.

With this data and seed (lr 5e-2, 30 epochs, early stopping off) val NLL is lowest at epoch 28 and
rises at epoch 29, so the restored model must equal a run truncated after epoch 28.
"""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyro
import pytest
import torch

import decipher_m5 as dc
from decipher_m5.presets import PRESETS
from decipher_m5.tools._decipher import DecipherConfig
from decipher_m5.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")

BATCH_KEY = "batch"
BEST_EPOCH = 28


def make_adata(n_cells: int = 200, n_genes: int = 50, n_batches: int = 3, seed: int = 0):
    """Synthetic integer counts with a categorical batch column."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=n_genes), size=(n_cells, n_genes))
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([f"b{i % n_batches}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


def train(n_epochs: int, save_folder) -> tuple:
    """Train model5 without early stopping; return (state_dict, theta, val_losses, best_epoch)."""
    DECIPHER_GLOBALS["save_folder"] = str(save_folder)
    config = DecipherConfig(
        **PRESETS["model5"],
        seed=1,
        n_epochs=n_epochs,
        learning_rate=5e-2,
        early_stopping_patience=None,
    )
    model, val_losses = dc.tl.decipher_train(
        make_adata(),
        config,
        batch_key=BATCH_KEY,
        plot_every_k_epochs=-2,
        plot_kwargs={"color": BATCH_KEY},
    )
    theta = pyro.param("theta").detach().clone()
    return model.state_dict(), theta, val_losses, model.best_epoch_


@pytest.fixture(scope="module")
def full_and_truncated(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ckpt")
    return train(30, folder), train(BEST_EPOCH + 1, folder)


def test_last_epoch_is_worse_than_best_so_the_scenario_is_exercised(full_and_truncated):
    (_, _, val_losses, best_epoch), _ = full_and_truncated
    assert (best_epoch, int(np.argmin(val_losses))) == (BEST_EPOCH, BEST_EPOCH)
    assert val_losses[-1] > val_losses[BEST_EPOCH]


def test_restored_weights_equal_run_stopped_at_best_epoch(full_and_truncated):
    (full_sd, _, _, _), (trunc_sd, _, _, _) = full_and_truncated
    assert full_sd.keys() == trunc_sd.keys()
    for key in full_sd:
        torch.testing.assert_close(full_sd[key], trunc_sd[key], rtol=0, atol=0, msg=key)


def test_restored_theta_equals_run_stopped_at_best_epoch(full_and_truncated):
    (_, full_theta, _, _), (_, trunc_theta, _, _) = full_and_truncated
    torch.testing.assert_close(full_theta, trunc_theta, rtol=0, atol=0)
