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
