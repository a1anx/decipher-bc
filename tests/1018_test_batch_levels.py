"""Batch codes in `decipher_models3` are built against the full training AnnData's levels.

Regression for the nut50k bug: AnnData subsetting drops unused categories, so a split missing a
batch level had its `.cat.codes` shifted and its cells were scored with a neighbouring batch's
embedding. Only one tiny training run (60 cells, 1 epoch); everything else uses untrained models.
"""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyro
import pytest
import torch

import decipher_models3 as dc
from decipher_models3.presets import PRESETS
from decipher_models3.tools import gene_patterns, get_batch_idx
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.tools._decipher.data import (
    batch_codes,
    decipher_load_model,
    decipher_save_model,
    make_data_loader_from_adata,
)
from decipher_models3.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")

KEY = "batch"
N_GENES = 12
LEVELS = ["b0", "b1", "b2"]
# For 60 cells, val_frac 0.1 and seed 0 the positional split puts these cells in validation.
VAL_POSITIONS = [15, 29, 31, 33, 41, 56]
TRAIN_ONLY = "a_train_only"  # sorts first, so dropping it shifts every other subset code
VAL_ONLY = "z_val_only"
POINTS = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0]])


def make_adata(labels) -> ad.AnnData:
    """Integer counts with a categorical batch column holding `labels`."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(len(labels), N_GENES)).astype(np.float32)
    obs = pd.DataFrame(
        {KEY: pd.Categorical(labels)}, index=[f"cell{i}" for i in range(len(labels))]
    )
    return ad.AnnData(X=counts, obs=obs)


def three_batch_adata() -> ad.AnnData:
    return make_adata([LEVELS[i % 3] for i in range(30)])


def rare_level_adata() -> ad.AnnData:
    """60 cells: TRAIN_ONLY on two train cells, VAL_ONLY on two val cells, b0/b1 elsewhere."""
    labels = [f"b{i % 2}" for i in range(60)]
    labels[0] = labels[1] = TRAIN_ONLY
    labels[VAL_POSITIONS[0]] = labels[VAL_POSITIONS[1]] = VAL_ONLY
    return make_adata(labels)


def full_codes(adata, mask) -> np.ndarray:
    return adata.obs[KEY].cat.codes.to_numpy()[np.asarray(mask)].astype(np.int64)


def build_model(adata, preset="model5") -> Decipher:
    """Untrained eval model on `adata`'s levels; the batch table gets random nonzero rows."""
    config = DecipherConfig(**PRESETS[preset])
    config.initialize_from_adata(adata, batch_key=KEY)
    pyro.clear_param_store()
    torch.manual_seed(0)
    model = Decipher(config).eval()
    with torch.no_grad():
        model.batch_emb.weight.normal_()
    return model


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """model, adata -> adata with that model saved (checkpoints in tmp_path), one trajectory."""
    monkeypatch.setitem(DECIPHER_GLOBALS, "save_folder", str(tmp_path))

    def _saved(model, adata):
        decipher_save_model(adata, model)
        adata.uns["decipher"]["trajectories"] = {
            "t": {"points": POINTS, "times": np.arange(len(POINTS), dtype=float)}
        }
        return adata

    return _saved


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """(adata, config) after a 1-epoch model5 run on `rare_level_adata`."""
    adata = rare_level_adata()
    config = DecipherConfig(**PRESETS["model5"], n_epochs=1, batch_size=16)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(DECIPHER_GLOBALS, "save_folder", str(tmp_path_factory.mktemp("ckpt")))
        dc.tl.decipher_train(adata, config, batch_key=KEY, plot_every_k_epochs=-2)
    return adata, config


# --- (a) training: levels from the full adata ---------------------------------------------------


def test_split_precondition_rare_levels_sit_on_one_side(trained):
    adata, _ = trained
    split = adata.obs["decipher_split"].astype(str).to_numpy()
    by_level = adata.obs.groupby(KEY, observed=True)["decipher_split"]
    assert list(np.flatnonzero(split == "validation")) == VAL_POSITIONS
    assert set(by_level.get_group(TRAIN_ONLY).astype(str)) == {"train"}
    assert set(by_level.get_group(VAL_ONLY).astype(str)) == {"validation"}


def test_training_records_every_level_of_the_full_adata(trained):
    adata, config = trained
    assert config.batch_levels == [TRAIN_ONLY, "b0", "b1", VAL_ONLY]
    assert config.n_batches == 4


def test_val_loader_codes_equal_full_adata_codes(trained):
    adata, config = trained
    is_val = (adata.obs["decipher_split"] == "validation").to_numpy()
    loader = make_data_loader_from_adata(
        adata[is_val], 16, context_levels={KEY: config.batch_levels}
    )
    np.testing.assert_array_equal(loader.dataset.tensors[1].numpy(), full_codes(adata, is_val))


# --- (b) get_batch_idx on a subset --------------------------------------------------------------


def test_get_batch_idx_on_subset_missing_a_level_returns_full_codes():
    adata = three_batch_adata()
    config = build_model(adata).config
    no_b0 = (adata.obs[KEY] != "b0").to_numpy()
    codes = get_batch_idx(adata[no_b0], config)
    np.testing.assert_array_equal(codes.numpy(), full_codes(adata, no_b0))


# --- (c) unknown labels -------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["b9", np.nan], ids=["unknown", "nan"])
def test_batch_codes_raises_on_a_label_outside_the_levels(bad):
    with pytest.raises(ValueError, match=f"{KEY!r}.*not training batch levels"):
        batch_codes(pd.Series(["b0", bad, "b1"], dtype=object), LEVELS, KEY)


def test_batch_codes_maps_labels_to_level_positions_as_int64():
    codes = batch_codes(["b2", "b0", "b2"], LEVELS)
    np.testing.assert_array_equal(codes, np.array([2, 0, 2], dtype=np.int64))
    assert codes.dtype == np.int64


# --- (d) gene_patterns on a subset --------------------------------------------------------------


def test_gene_patterns_by_name_on_subset_missing_a_level_matches_full(saved):
    adata = three_batch_adata()
    full = saved(build_model(adata), adata)
    subset = full[(full.obs[KEY] != "b0").to_numpy()].copy()

    gene_patterns(full, n_samples=2, batch="b2")
    gene_patterns(subset, n_samples=2, batch="b2")

    np.testing.assert_array_equal(
        subset.uns["decipher"]["gene_patterns"]["t"]["mean"],
        full.uns["decipher"]["gene_patterns"]["t"]["mean"],
    )


# --- (e) checkpoints saved before the fix -------------------------------------------------------


def old_checkpoint(saved) -> ad.AnnData:
    """A saved model whose config has no `batch_levels`, as before the fix."""
    adata = three_batch_adata()
    saved(build_model(adata), adata)
    del adata.uns["decipher"]["config"]["batch_levels"]
    return adata


def test_load_old_checkpoint_without_levels_raises_clear_error(saved):
    with pytest.raises(ValueError, match="predates the batch-levels fix.*cat.categories"):
        decipher_load_model(old_checkpoint(saved))


def test_load_old_checkpoint_with_explicit_levels_stores_and_uses_them(saved):
    adata = old_checkpoint(saved)
    model = decipher_load_model(adata, batch_levels=LEVELS)
    assert model.config.batch_levels == LEVELS
    assert decipher_load_model(adata).config.batch_levels == LEVELS


def test_load_with_levels_of_wrong_length_raises(saved):
    with pytest.raises(ValueError, match="2 levels.*n_batches=3"):
        decipher_load_model(old_checkpoint(saved), batch_levels=["b0", "b1"])


# --- (f) round trips ----------------------------------------------------------------------------


def test_batch_levels_round_trip_through_to_dict():
    config = build_model(three_batch_adata()).config
    assert DecipherConfig(**config.to_dict()).batch_levels == LEVELS


def test_batch_levels_round_trip_through_save_h5ad_and_load(saved, tmp_path):
    adata = three_batch_adata()
    saved(build_model(adata), adata).write_h5ad(tmp_path / "saved.h5ad")
    loaded = decipher_load_model(ad.read_h5ad(tmp_path / "saved.h5ad"))
    assert loaded.config.batch_levels == LEVELS
