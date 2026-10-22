"""`decipher_models3` `recon_batch="residual"` (presets model4r, model5r; batch-invariance plan 1b).

logits = W z + c + r([z; e_b]): W z + c = `decoder_z_to_x(z)` with no e_b columns, r =
`decoder_residual` (L + D -> 64 -> G) with its output layer zero at init, and an N(0, 1)
`batch_emb` built after the networks. Checked here: the seven existing presets are unchanged
(state_dict and model trace log-prob fingerprints taken at base commit 991ff31, before the field
existed), the init is batch-blind in the decoder but not dead, the widths, the config error, and
`gene_patterns`. Untrained models on a small synthetic AnnData; at most one SVI step.
"""

import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import decipher_models3.tools._decipher.data as data
from decipher_models3.presets import PRESETS
from decipher_models3.tools import gene_patterns
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.utils import DECIPHER_GLOBALS

BATCH_KEY = "batch"
BATCHES = ["b0", "b1", "b2"]
N_GENES, DIM_Z = 50, 10
RESIDUAL = ["model4r", "model5r"]
POINTS = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0], [-1.0, 0.5]])
L_SCALE = 10_000

# preset -> (sha256 of the seed-0 state_dict, model trace log-prob with a randomized table), both
# computed at 991ff31 with the builders below.
BASE_FINGERPRINTS = {
    "native": ("903ee70064c4678d729eccf52fa75a357191071b0e8d1403bbce13560ba69b52", -17732.296875),
    "model1": ("319d89316c83faa015ce264485f396d0db46a6a88d6f55f134bda2647ff217bc", -18126.5703125),
    "model2": ("3b39716648759c892885588823774f8c904299472456b9f2be9d3716738a3023", -17053.6484375),
    "model3": ("3f17d50bc76d82ca948528f5fb27dcad3c84e356c282033f3b1c6d615d607fc3", -16940.74609375),
    "model4": ("a9ba24eca9dc531770d92b74e1bee08723888bdb94b7a9f043a6830da51dc5c1", -17715.37109375),
    "model5": ("e07c0da42de5661ab93798ed9cbf1ff4bceee060d8578484249c3869c56652ed", -17130.6484375),
    "model6": (
        "0f823beffd9949970d7fc687279f8bdadcaf5c2cf5bbc4c82fbe801979de6871",
        -17201.962890625,
    ),
}
# preset -> (sha256 of the seed-0 state_dict, model trace log-prob with the table as built), both
# computed at e38f1bc. Pins the residual arms' construction order (networks, r, then the table)
# against later edits of `Decipher.__init__`.
RESIDUAL_FINGERPRINTS = {
    "model4r": ("60726b8b1c154a8ddbf06ee4c3e83191a9c3cb9cdbe6e2ec1b748fd2751e93df", -17732.296875),
    "model5r": ("911fbf90769f73ec1252264609bf8f4bf4ecfa6835d182ca3ab937b55238f9f9", -16405.5078125),
}
# The field left at its default, and set explicitly to the default.
FLAGS_OFF = [{}, {"recon_batch": "concat"}]


def _make_adata(n_cells=120, seed=0):
    """Synthetic integer counts with a categorical batch column (as tests/1016_test_decipher_models3.py)."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=N_GENES), size=(n_cells, N_GENES))
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([BATCHES[i % len(BATCHES)] for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


ADATA = _make_adata()
X = torch.tensor(ADATA.X)
BATCH_IDX = torch.tensor(ADATA.obs[BATCH_KEY].cat.codes.values).long()


def _build(preset, seed=0, batch_key=BATCH_KEY, **overrides):
    """Untrained eval-mode Decipher of `preset` (plus config overrides), Pyro seed `seed`."""
    config = DecipherConfig(**{**PRESETS[preset], **overrides})
    config.initialize_from_adata(ADATA, batch_key=batch_key)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(seed)
    return Decipher(config).eval()


def _randomize(param, seed=1):
    """Overwrite `param` with N(0, 1) draws from a fixed generator."""
    with torch.no_grad():
        param.copy_(torch.randn(param.shape, generator=torch.Generator().manual_seed(seed)))


def _state_dict_sha256(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def _model_log_prob(model):
    """Model trace log-prob under a replayed guide trace, Pyro seed 0, real batch codes."""
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = pyro.poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = pyro.poutine.replay(model.model, trace=guide_trace)
        return pyro.poutine.trace(replayed).get_trace(X, BATCH_IDX).log_prob_sum().item()


def _decode_mean(model, code):
    """softmax(W z_loc + c + r([z_loc; e_b])) * L_SCALE at POINTS for batch `code`, from the
    networks themselves (the prior is batch-blind under concat_x)."""
    v = torch.FloatTensor(POINTS)
    with torch.no_grad():
        z_loc, _ = model.decoder_v_to_z(v)
        e = model.batch_emb(torch.full((len(v),), code, dtype=torch.long))
        logits = model.decoder_z_to_x(z_loc) + model.decoder_residual(torch.cat([z_loc, e], -1))
        return torch.softmax(logits, -1).numpy() * L_SCALE


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """Return a function model -> AnnData with that model saved and one trajectory 't'."""
    monkeypatch.setitem(DECIPHER_GLOBALS, "save_folder", str(tmp_path))

    def _saved(model):
        adata = ADATA.copy()
        data.decipher_save_model(adata, model)
        adata.uns["decipher"]["trajectories"] = {
            "t": {"points": POINTS, "times": np.arange(len(POINTS), dtype=float)}
        }
        return adata

    return _saved


def _pattern_mean(adata, batch=None):
    gene_patterns(adata, l_scale=L_SCALE, n_samples=5, batch=batch)
    return adata.uns["decipher"]["gene_patterns"]["t"]["mean"].copy()


# --- flags off: the seven existing presets are what they were at 991ff31 ------------------------


@pytest.mark.parametrize("overrides", FLAGS_OFF, ids=["default", "explicit_concat"])
@pytest.mark.parametrize("preset", sorted(BASE_FINGERPRINTS))
def test_flags_off_state_dict_equals_base_commit(preset, overrides):
    assert _state_dict_sha256(_build(preset, **overrides)) == BASE_FINGERPRINTS[preset][0]


@pytest.mark.parametrize("overrides", FLAGS_OFF, ids=["default", "explicit_concat"])
@pytest.mark.parametrize("preset", sorted(BASE_FINGERPRINTS))
def test_flags_off_model_log_prob_equals_base_commit(preset, overrides):
    model = _build(preset, **overrides)
    if model.batch_emb is not None:
        _randomize(model.batch_emb.weight)
    assert _model_log_prob(model) == pytest.approx(BASE_FINGERPRINTS[preset][1], rel=1e-6)


# --- residual: construction ---------------------------------------------------------------------


@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_state_dict_equals_e38f1bc_fingerprint(preset):
    assert _state_dict_sha256(_build(preset)) == RESIDUAL_FINGERPRINTS[preset][0]


@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_model_log_prob_equals_e38f1bc_fingerprint(preset):
    assert _model_log_prob(_build(preset)) == pytest.approx(
        RESIDUAL_FINGERPRINTS[preset][1], rel=1e-6
    )


@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_W_has_input_width_dim_z(preset):
    assert _build(preset).decoder_z_to_x.layers[0].in_features == DIM_Z


@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_batch_emb_is_not_zero_initialized(preset):
    assert _build(preset).batch_emb.weight.abs().min() > 0


@pytest.mark.parametrize("preset, base", [("model4r", "model4"), ("model5r", "model5")])
@pytest.mark.parametrize("net", ["encoder_x_to_z", "encoder_zx_to_v", "decoder_v_to_z"])
def test_residual_keeps_the_base_arms_seed_draws_for_encoders_and_prior(preset, base, net):
    residual, reference = getattr(_build(preset), net), getattr(_build(base), net)
    for (key, a), (_, b) in zip(residual.state_dict().items(), reference.state_dict().items()):
        assert torch.equal(a, b), key


@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_with_zero_batches_builds_no_table_and_no_r(preset):
    model = _build(preset, batch_key="no_such_column")
    assert (model.batch_emb, model.decoder_residual) == (None, None)


@pytest.mark.parametrize(
    "flags",
    [
        dict(batch_conditioning="decoder_only", batch_embedding_mode="concat_z"),
        dict(batch_conditioning="decoder_encoder", batch_embedding_mode="concat_z"),
        dict(batch_conditioning="none", batch_embedding_mode="concat_x"),
        dict(batch_conditioning="none", batch_embedding_mode="concat_z"),
    ],
)
def test_residual_without_a_concat_x_batch_decoder_raises(flags):
    with pytest.raises(ValueError, match="recon_batch='residual'"):
        DecipherConfig(**flags, recon_batch="residual")


# --- residual: init and gradients ---------------------------------------------------------------


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_decoder_at_init_is_the_batch_blind_W_z_plus_c(saved, preset, batch):
    model = _build(preset)
    v = torch.FloatTensor(POINTS)
    with torch.no_grad():
        expected = torch.softmax(model.decoder_z_to_x(model.decoder_v_to_z(v)[0]), -1)
    mean = _pattern_mean(saved(model), batch=batch)
    np.testing.assert_allclose(mean, expected.numpy() * L_SCALE, rtol=1e-6, atol=1e-6)


GRAD_PARAMS = {
    "r_output_layer": lambda model: model.decoder_residual[-1].weight,
    "batch_emb": lambda model: model.batch_emb.weight,
}


@pytest.mark.parametrize("param", sorted(GRAD_PARAMS))
@pytest.mark.parametrize("preset", RESIDUAL)
def test_residual_param_has_nonzero_gradient_after_one_svi_step(preset, param):
    model = _build(preset).train()
    SVI(model.model, model.guide, Adam({"lr": 1e-2}), Trace_ELBO()).step(X, BATCH_IDX)
    Trace_ELBO().differentiable_loss(model.model, model.guide, X, BATCH_IDX).backward()
    assert GRAD_PARAMS[param](model).grad.abs().sum() > 0


# --- residual: gene patterns after the weights are perturbed ------------------------------------


def _perturbed(preset):
    """`preset` with r's zero output layer replaced by random weights."""
    model = _build(preset)
    _randomize(model.decoder_residual[-1].weight, seed=2)
    _randomize(model.decoder_residual[-1].bias, seed=3)
    return model


def test_model5r_gene_patterns_default_mean_is_uniform_average_over_every_batch(saved):
    model = _perturbed("model5r")
    expected = np.mean([_decode_mean(model, b) for b in range(len(BATCHES))], axis=0)
    np.testing.assert_allclose(_pattern_mean(saved(model)), expected, rtol=1e-5, atol=1e-6)


def test_model5r_gene_patterns_differ_between_batches_after_perturbation(saved):
    adata = saved(_perturbed("model5r"))
    assert not np.allclose(_pattern_mean(adata, "b0"), _pattern_mean(adata, "b2"), rtol=1e-5)
