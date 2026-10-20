"""Checks for `decipher_models3`: the seven presets and the positional train/val split.

Sections are appended piece by piece (`# --- P01: ...`, `# --- P02: ...`); each is self-contained.
"""

import importlib
import os
import sys

import numpy as np
import pytest

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


def test_presets_are_the_seven_decipher_models_presets_plus_the_new_arms():
    # model4r/model5r: recon_batch="residual" (batch-invariance plan 1b); native_mmd/model2_mmd/
    # model5_mmd: v_invariance="mmd" (plan 1a); model7/model7i: prior_batch="residual" (plan
    # model7/model7i); model8: cycle_v=True (cycle penalty on v); model8v: model8 +
    # prior="vamp"; model5rz/model5r0/model7iw/model7im/model5r_mmd/model8r: the 1017 controls
    # (tests/1017_test_decipher_models3_controls.py). Only in decipher_models3.
    assert sorted(PRESETS_3) == [
        "model1",
        "model2",
        "model2_mmd",
        "model3",
        "model4",
        "model4r",
        "model5",
        "model5_mmd",
        "model5r",
        "model5r0",
        "model5r_mmd",
        "model5rz",
        "model6",
        "model7",
        "model7i",
        "model7im",
        "model7iw",
        "model8",
        "model8r",
        "model8v",
        "native",
        "native_mmd",
    ]


def test_positional_split_on_duplicate_names_is_exactly_90_10():
    split = _p01_split(split_by_position, _p01_duplicate_name_data())
    assert list(split.cat.categories) == ["train", "validation"]
    assert split.value_counts().to_dict() == {"train": 2250, "validation": 250}


# --- P03: best-epoch restore -------------------------------------------------------------------
# `decipher_train` returns the weights of the lowest-val-NLL epoch, not the last. Arm-agnostic
# port of tests/1015_test_m5_restore_best.py: a full run must equal a run truncated right after its
# best epoch (same seed, no early stopping), compared on the whole state_dict and theta without
# naming any parameter. Run on one concat_z arm (model2) and one concat_x arm (model5).
# Training is pinned to 1 torch thread: the thread count sets the float reduction order, so the
# val NLL curve (and whether the last epoch is the best) changes with it; numba calls elsewhere
# in the session raise torch's count. At 1 thread over 40 epochs the best epoch is 38 (model2)
# and 34 (model5).

_P03_EPOCHS = 40
_P03_THREADS = 1
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
    import torch

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
    threads = torch.get_num_threads()
    torch.set_num_threads(_P03_THREADS)
    try:
        model, val_losses = dc.tl.decipher_train(
            _p03_make_adata(),
            config,
            batch_key=_P03_BATCH_KEY,
            plot_every_k_epochs=-2,
            plot_kwargs={"color": _P03_BATCH_KEY},
        )
    finally:
        torch.set_num_threads(threads)
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


# --- P02: model math, loading -----------------------------------------------------
# Untrained eval-mode models on a small synthetic AnnData, no training beyond single SVI steps.
# Eval mode, Pyro seed fixed before every sampling call.

import anndata as ad  # noqa: E402
import pandas as pd  # noqa: E402
import pyro  # noqa: E402
import torch  # noqa: E402
from pyro.infer import SVI, Trace_ELBO  # noqa: E402
from pyro.optim import Adam  # noqa: E402

import decipher_models3.tools._decipher.data as _p02_data_3  # noqa: E402
from decipher_models3.tools._decipher import Decipher, DecipherConfig  # noqa: E402
from decipher_models3.utils import DECIPHER_GLOBALS as _P02_GLOBALS_3  # noqa: E402

_P02_BATCH_KEY = "batch"
_P02_N_BATCHES = 3
_P02_N_GENES = 50
_P02_D_E = 8  # default dim_batch_embedding
_P02_CONCAT_X = ["model4", "model5", "model6"]
_P02_BATCH_ARMS = ["model1", "model2", "model3", *_P02_CONCAT_X]
_P02_OUTPUTS = ["v", "z", "z_raw", "impute", "model log-prob"]


def _p02_make_adata(n_cells=120, seed=0):
    """Synthetic integer counts with a categorical batch column."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=_P02_N_GENES), size=(n_cells, _P02_N_GENES))
    obs = pd.DataFrame(
        {_P02_BATCH_KEY: pd.Categorical([f"b{i % _P02_N_BATCHES}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


_P02_ADATA = _p02_make_adata()
_P02_X = torch.tensor(_P02_ADATA.X)
_P02_BATCH_IDX = torch.tensor(_P02_ADATA.obs[_P02_BATCH_KEY].cat.codes.values).long()


def _p02_build(preset, package="decipher_models3", seed=0, batch_key=_P02_BATCH_KEY):
    """Untrained eval-mode Decipher of `package`, built with Pyro seed `seed`."""
    tools = importlib.import_module(f"{package}.tools._decipher")
    presets = importlib.import_module(f"{package}.presets").PRESETS
    config = tools.DecipherConfig(**presets[preset])
    config.initialize_from_adata(_P02_ADATA, batch_key=batch_key)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(seed)
    return tools.Decipher(config).eval()


def _p02_randomize_emb(model, seed=1):
    """Give `batch_emb` nonzero rows (a zero table makes every batch path trivially equal)."""
    with torch.no_grad():
        weight = model.batch_emb.weight
        weight.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(seed)))
    return model


def _p02_outputs(model, x=_P02_X, batch_idx=_P02_BATCH_IDX):
    """v, z, z_raw, imputed counts and the model trace log-prob under a replayed guide."""
    v, z, z_raw = map(torch.as_tensor, model.compute_v_z_numpy(x, batch_idx))
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    impute = torch.as_tensor(model.impute_gene_expression_numpy(x, batch_idx))
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = pyro.poutine.trace(model.guide).get_trace(x, batch_idx)
        replayed = pyro.poutine.replay(model.model, trace=guide_trace)
        log_prob = pyro.poutine.trace(replayed).get_trace(x, batch_idx).log_prob_sum()
    return {"v": v, "z": z, "z_raw": z_raw, "impute": impute, "model log-prob": log_prob}


def _p02_widths(model):
    """Input widths of (encoder_x_to_z, encoder_zx_to_v, decoder_v_to_z, decoder_z_to_x)."""
    nets = (model.encoder_x_to_z, model.encoder_zx_to_v, model.decoder_v_to_z, model.decoder_z_to_x)
    return tuple(net.layers[0].in_features for net in nets)


def test_additive_batch_embedding_mode_raises():
    with pytest.raises(ValueError, match="batch_embedding_mode='additive'"):
        DecipherConfig(batch_conditioning="decoder_only", batch_embedding_mode="additive")


@pytest.mark.parametrize("arm", _P02_CONCAT_X)
def test_concat_x_batch_emb_is_zero_initialized(arm):
    model = _p02_build(arm)
    assert torch.equal(model.batch_emb.weight, torch.zeros(_P02_N_BATCHES, _P02_D_E))


@pytest.mark.parametrize("arm", _P02_CONCAT_X)
def test_untrained_concat_x_model_is_batch_blind(arm):
    _, z, z_raw = _p02_build(arm).compute_v_z_numpy(_P02_X, _P02_BATCH_IDX)
    np.testing.assert_allclose(z, z_raw, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("arm", _P02_CONCAT_X)
def test_one_svi_step_moves_concat_x_embedding_off_zero(arm):
    model = _p02_build(arm).train()
    svi = SVI(model.model, model.guide, Adam({"lr": 1e-2}), Trace_ELBO())
    svi.step(_P02_X, _P02_BATCH_IDX)
    assert model.batch_emb.weight.abs().sum() > 0


def test_model4_guide_ignores_the_batch():
    model = _p02_randomize_emb(_p02_build("model4"))
    as_b0 = model.compute_v_z_numpy(_P02_X, torch.zeros(len(_P02_X), dtype=torch.long))
    as_b1 = model.compute_v_z_numpy(_P02_X, torch.ones(len(_P02_X), dtype=torch.long))
    for a, b in zip(as_b0, as_b1):  # v, z, z_raw
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("arm", ["model1", "model4"])
def test_decoder_only_generative_model_depends_on_the_batch(arm):
    model = _p02_randomize_emb(_p02_build(arm))
    as_b0 = _p02_outputs(model, batch_idx=torch.zeros(len(_P02_X), dtype=torch.long))
    as_b1 = _p02_outputs(model, batch_idx=torch.ones(len(_P02_X), dtype=torch.long))
    assert not torch.allclose(as_b0["model log-prob"], as_b1["model log-prob"])


_P02_G, _P02_DZ, _P02_DV, _P02_E = _P02_N_GENES, 10, 2, _P02_D_E
# (encoder_x_to_z, encoder_zx_to_v, decoder_v_to_z, decoder_z_to_x) input widths per arm.
_P02_EXPECTED_WIDTHS = {
    "native": (_P02_G, _P02_G + _P02_DZ, _P02_DV, _P02_DZ),
    "model1": (_P02_G, _P02_G + _P02_DZ, _P02_DV + _P02_E, _P02_DZ),
    "model2": (_P02_G + _P02_E, _P02_G + _P02_DZ, _P02_DV + _P02_E, _P02_DZ),
    "model3": (_P02_G + _P02_E, _P02_G, _P02_DV + _P02_E, _P02_DZ),
    "model4": (_P02_G, _P02_G + _P02_DZ, _P02_DV, _P02_DZ + _P02_E),
    "model5": (_P02_G + _P02_E, _P02_G + _P02_DZ, _P02_DV, _P02_DZ + _P02_E),
    "model6": (_P02_G + _P02_E, _P02_G, _P02_DV, _P02_DZ + _P02_E),
}


@pytest.mark.parametrize("arm", sorted(_P02_EXPECTED_WIDTHS))
def test_network_input_widths_follow_the_arm(arm):
    assert _p02_widths(_p02_build(arm)) == _P02_EXPECTED_WIDTHS[arm]


@pytest.mark.parametrize("arm", _P02_BATCH_ARMS)
def test_zero_batches_builds_no_table_and_native_widths(arm):
    model = _p02_build(arm, batch_key="no_such_column")
    assert model.batch_emb is None
    v_encoder = _P02_G if model.config.mean_field_v else _P02_G + _P02_DZ
    assert _p02_widths(model) == (_P02_G, v_encoder, _P02_DV, _P02_DZ)


@pytest.mark.parametrize("arm", ["model1", "model2", "model5"])
def test_batch_idx_none_equals_batch_zero(arm):
    model = _p02_randomize_emb(_p02_build(arm))
    from_none = _p02_outputs(model, batch_idx=None)
    from_zeros = _p02_outputs(model, batch_idx=torch.zeros(len(_P02_X), dtype=torch.long))
    for name in _P02_OUTPUTS:
        torch.testing.assert_close(from_none[name], from_zeros[name], rtol=0, atol=0)


@pytest.mark.parametrize("arm", sorted(_P02_EXPECTED_WIDTHS))
def test_save_load_roundtrip_reproduces_the_model(arm, tmp_path, monkeypatch):
    monkeypatch.setitem(_P02_GLOBALS_3, "save_folder", str(tmp_path))
    model = _p02_build(arm)
    if model.batch_emb is not None:
        _p02_randomize_emb(model)
    saved = _P02_ADATA.copy()
    _p02_data_3.decipher_save_model(saved, model)
    saved.write_h5ad(tmp_path / "saved.h5ad")
    loaded = _p02_data_3.decipher_load_model(ad.read_h5ad(tmp_path / "saved.h5ad"))

    assert isinstance(loaded, Decipher)
    for a, b in zip(
        loaded.compute_v_z_numpy(_P02_X, _P02_BATCH_IDX),
        model.compute_v_z_numpy(_P02_X, _P02_BATCH_IDX),
    ):
        np.testing.assert_array_equal(a, b)


# --- P04: gene patterns ------------------------------------------------------------------------
# `gene_patterns(adata, ..., batch=None)` for every arm, ported from
# the old decipher_m5 tests. Trajectory points have no batch: the default averages
# softmax(decode) uniformly over batches, `batch=` pins one batch or a subset. concat_z arms take
# the batch in the prior v -> z, concat_x arms in the decoder z -> x. Untrained eval-mode models
# with a randomized `batch_emb` (concat_x is zero-init), saved to tmp_path. No training.

from decipher_models3.tools import gene_patterns  # noqa: E402

_P04_BATCHES = ["b0", "b1", "b2"]
_P04_L_SCALE = 10_000
_P04_POINTS = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0], [-1.0, 0.5]])
_P04_RTOL, _P04_ATOL = 1e-5, 1e-6
_P04_BATCH_ARMS = ["model1", "model2", "model3", "model4", "model5", "model6"]


def _p04_build(preset):
    """Untrained eval model of `preset`; a batch table, if any, gets random nonzero rows."""
    config = DecipherConfig(**PRESETS_3[preset])
    config.initialize_from_adata(_P02_ADATA, batch_key=_P02_BATCH_KEY)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    model = Decipher(config).eval()
    if model.batch_emb is not None:
        with torch.no_grad():
            weight = model.batch_emb.weight
            weight.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(1)))
    return model


@pytest.fixture
def _p04_saved(tmp_path, monkeypatch):
    """Return a function model -> AnnData with that model saved and one trajectory 't'."""
    monkeypatch.setitem(_P02_GLOBALS_3, "save_folder", str(tmp_path))

    def _saved(model):
        adata = _P02_ADATA.copy()
        _p02_data_3.decipher_save_model(adata, model)
        adata.uns["decipher"]["trajectories"] = {
            "t": {"points": _P04_POINTS, "times": np.arange(len(_P04_POINTS), dtype=float)}
        }
        return adata

    return _saved


def _p04_decode_mean(model, code):
    """Expected `mean` for one batch code (None = batch-blind), from the networks themselves:
    e_b goes into decoder_v_to_z under concat_z and into decoder_z_to_x under concat_x."""
    v = torch.FloatTensor(_P04_POINTS)
    with torch.no_grad():
        if code is None:
            z_loc, _ = model.decoder_v_to_z(v)
            return torch.softmax(model.decoder_z_to_x(z_loc), -1).numpy() * _P04_L_SCALE
        e = model.batch_emb(torch.full((len(v),), code, dtype=torch.long))
        if model.config.batch_embedding_mode == "concat_z":
            z_loc, _ = model.decoder_v_to_z(torch.cat([v, e], -1))
            logits = model.decoder_z_to_x(z_loc)
        else:
            z_loc, _ = model.decoder_v_to_z(v)
            logits = model.decoder_z_to_x(torch.cat([z_loc, e], -1))
        return torch.softmax(logits, -1).numpy() * _P04_L_SCALE


def _p04_pattern(adata, batch=None, n_samples=5):
    gene_patterns(adata, l_scale=_P04_L_SCALE, n_samples=n_samples, batch=batch)
    return adata.uns["decipher"]["gene_patterns"]["t"]


@pytest.mark.parametrize("arm", _P04_BATCH_ARMS)
def test_gene_patterns_default_mean_is_uniform_average_over_every_batch(_p04_saved, arm):
    model = _p04_build(arm)
    mean = _p04_pattern(_p04_saved(model))["mean"]
    expected = np.mean([_p04_decode_mean(model, b) for b in range(len(_P04_BATCHES))], axis=0)
    np.testing.assert_allclose(mean, expected, rtol=_P04_RTOL, atol=_P04_ATOL)


@pytest.mark.parametrize("arm", _P04_BATCH_ARMS)
def test_gene_patterns_single_batch_name_uses_that_batch_only(_p04_saved, arm):
    model = _p04_build(arm)
    mean = _p04_pattern(_p04_saved(model), batch="b2")["mean"]
    np.testing.assert_allclose(mean, _p04_decode_mean(model, 2), rtol=_P04_RTOL, atol=_P04_ATOL)


@pytest.mark.parametrize("arm", _P04_BATCH_ARMS)
def test_gene_patterns_batch_list_averages_over_listed_batches(_p04_saved, arm):
    model = _p04_build(arm)
    mean = _p04_pattern(_p04_saved(model), batch=["b0", "b2"])["mean"]
    expected = (_p04_decode_mean(model, 0) + _p04_decode_mean(model, 2)) / 2
    np.testing.assert_allclose(mean, expected, rtol=_P04_RTOL, atol=_P04_ATOL)


@pytest.mark.parametrize("arm", _P04_BATCH_ARMS)
def test_gene_patterns_differ_between_batches(_p04_saved, arm):
    adata = _p04_saved(_p04_build(arm))
    as_b0 = _p04_pattern(adata, batch="b0")["mean"].copy()
    as_b2 = _p04_pattern(adata, batch="b2")["mean"]
    assert not np.allclose(as_b0, as_b2, rtol=_P04_RTOL, atol=_P04_ATOL)


@pytest.mark.parametrize("arm", _P04_BATCH_ARMS)
def test_gene_patterns_q_band_collapses_onto_mean_when_prior_scale_is_zero(_p04_saved, arm):
    """With softplus(z_scale) ~ 0 every z sample is the per-batch prior mean, so each sample's
    batch-averaged expression -- and so q25 and q75 -- must equal `mean`."""
    model = _p04_build(arm)
    head, d_z = model.decoder_v_to_z.layers[-1], model.config.dim_z
    with torch.no_grad():
        head.weight[d_z:] = 0.0
        head.bias[d_z:] = -50.0
    pattern = _p04_pattern(_p04_saved(model))
    for q in ("q25", "q75"):
        np.testing.assert_allclose(pattern[q], pattern["mean"], rtol=_P04_RTOL, atol=_P04_ATOL)


@pytest.mark.parametrize("arm", _P04_BATCH_ARMS)
def test_gene_patterns_q_band_is_one_row_per_point_with_q25_at_most_q75(_p04_saved, arm):
    pattern = _p04_pattern(_p04_saved(_p04_build(arm)), n_samples=20)
    assert pattern["q25"].shape == pattern["q75"].shape == (len(_P04_POINTS), _P02_N_GENES)
    assert np.all(pattern["q25"] <= pattern["q75"])


def test_gene_patterns_native_mean_is_the_batch_blind_decode(_p04_saved):
    model = _p04_build("native")
    mean = _p04_pattern(_p04_saved(model))["mean"]
    np.testing.assert_allclose(mean, _p04_decode_mean(model, None), rtol=_P04_RTOL, atol=_P04_ATOL)


@pytest.mark.parametrize("batch", ["nope", ["b0", "nope"], []])
def test_gene_patterns_unknown_or_empty_batch_raises(_p04_saved, batch):
    adata = _p04_saved(_p04_build("model2"))
    with pytest.raises(ValueError):
        _p04_pattern(adata, batch=batch)


def test_gene_patterns_batch_on_native_model_raises(_p04_saved):
    adata = _p04_saved(_p04_build("native"))
    with pytest.raises(ValueError, match="not batch-conditioned"):
        _p04_pattern(adata, batch="b0")
