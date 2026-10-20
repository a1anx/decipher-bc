"""`decipher_train(run_dir=...)`: per-epoch CSV, run summary, kill-and-resume.

Tiny model5 runs (60 cells, up to 4 epochs).
"""

import csv
import json

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

import decipher_models3 as dc
from decipher_models3.presets import PRESETS
from decipher_models3.run_logging import checkpoint_path
from decipher_models3.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")


def make_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(60, 12)).astype(np.float32)
    obs = pd.DataFrame({"batch": pd.Categorical([f"b{i % 3}" for i in range(60)])})
    obs.index = [f"cell{i}" for i in range(60)]
    return ad.AnnData(X=counts, obs=obs)


@pytest.fixture(autouse=True)
def checkpoints_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setitem(DECIPHER_GLOBALS, "save_folder", str(tmp_path / "models"))
    torch.set_num_threads(1)


def train(run_dir, n_epochs):
    config = dc.tl.DecipherConfig(**PRESETS["model5"], n_epochs=n_epochs, batch_size=16)
    decipher, val_losses = dc.tl.decipher_train(
        make_adata(), config, plot_every_k_epochs=-2, run_dir=str(run_dir)
    )
    return decipher, val_losses


def csv_epochs(run_dir):
    with open(run_dir / "metrics.csv", newline="") as f:
        return [int(r["epoch"]) for r in csv.DictReader(f)]


def test_two_epoch_run_writes_the_csv_and_summary(tmp_path):
    decipher, val_losses = train(tmp_path, 2)
    with open(tmp_path / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [list(r) for r in rows] == [["epoch", "train_elbo", "val_nll", "lr"]] * 2
    assert [float(r["val_nll"]) for r in rows] == pytest.approx(val_losses)
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert (summary["best_epoch"], summary["n_epochs_trained"]) == (decipher.best_epoch_, 2)


def test_killed_and_resumed_run_equals_an_uninterrupted_run(tmp_path):
    _, straight = train(tmp_path / "straight", 4)
    train(tmp_path / "resumed", 2)
    _, resumed = train(tmp_path / "resumed", 4)
    assert resumed == straight
    assert csv_epochs(tmp_path / "resumed") == [0, 1, 2, 3]


def test_resume_drops_an_epoch_logged_after_the_last_checkpoint(tmp_path):
    train(tmp_path, 2)
    with open(tmp_path / "metrics.csv", "a") as f:  # epoch 2 logged, then killed before saving
        f.write("2,1.0,1.0,0.001\n")
    train(tmp_path, 3)
    assert csv_epochs(tmp_path) == [0, 1, 2]


def test_resume_with_a_different_config_raises(tmp_path):
    train(tmp_path, 1)
    config = dc.tl.DecipherConfig(**PRESETS["model5"], n_epochs=2, batch_size=16, seed=7)
    with pytest.raises(ValueError, match=r"different config \(\['seed'\]\)"):
        dc.tl.decipher_train(make_adata(), config, plot_every_k_epochs=-2, run_dir=str(tmp_path))


def test_resume_from_a_checkpoint_with_an_extra_legacy_key_equals_an_uninterrupted_run(tmp_path):
    # Checkpoints written by older versions of decipher_train carry extra keys; resume ignores them.
    _, straight = train(tmp_path / "straight", 2)
    train(tmp_path / "legacy", 1)
    path = checkpoint_path(str(tmp_path / "legacy"))
    torch.save({**torch.load(path, weights_only=False), "legacy_run_id": "abc"}, path)
    _, resumed = train(tmp_path / "legacy", 2)
    assert resumed == straight
