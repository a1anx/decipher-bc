"""`decipher_models3` model7 and model7i (batch-invariance plan): the anchored residual prior
`prior_batch="residual"` and the in-span decoder `recon_batch="residual_in_span"`.

model7:  z_loc = f_mean(v) + t([v; e_b]), z-scale from f_var(v); logits = W z + c + r([z; e_b]);
         guide d_z([log1p x; e_b]); training penalty "prior_residual" = mean_cells ||t||^2
         (lambda_r).
model7i: as model7 with logits = W(z + ρ([z; e_b])) + c.

Checked here: every existing preset keeps its state_dict and model log-prob (fingerprints taken
at base 0fa879a) and its `prior_batch` mapping; the config errors; model7/model7i's widths, seed
draws and pins; at init the prior is f_mean(v) and the logits W z + c for every batch; every
parameter gradient of the ELBO and of the penalty equals a written-out reference (float64); the
z-scale ignores the batch; gene_patterns uses the residual prior and decoder per batch; the
penalty weight, its absence from val NLL, and the unweighted history at lambda_r = 10.
"""

import hashlib

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO
from torch.nn.functional import softmax, softplus

import decipher_models3 as dc
import decipher_models3.tools._decipher.data as data
import decipher_models3.tools.decipher as dtool
from decipher_models3.presets import PRESETS
from decipher_models3.tools import gene_patterns
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.tools._decipher.data import make_data_loader_from_adata
from decipher_models3.tools.decipher import penalized_step, predictive_log_likelihood
from decipher_models3.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")

BATCH_KEY = "batch"
BATCHES = ["b0", "b1", "b2"]
N_GENES, DIM_Z, DIM_V, D_E = 50, 10, 2, 8
ANCHORED = ["model7", "model7i"]
POINTS = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0], [-1.0, 0.5]])
L_SCALE = 10_000
EPS = 1e-5  # Decipher's epsilon in the guide scales and the NB parameters
BETA = 0.1  # DecipherConfig.beta default

# Every preset that existed before model7 -> its prior wiring.
PRIOR_BATCH = {
    "native": "none",
    "model1": "concat",
    "model2": "concat",
    "model3": "concat",
    "model4": "none",
    "model5": "none",
    "model6": "none",
    "model4r": "none",
    "model5r": "none",
    "native_mmd": "none",
    "model2_mmd": "concat",
    "model5_mmd": "none",
    "model8": "none",
    "model8v": "none",
}
# preset -> (sha256 of the seed-0 state_dict, model log-prob with a randomized table), computed at
# base 0fa879a (before prior_batch / lambda_r existed) with the builders below.
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
    "model4r": ("60726b8b1c154a8ddbf06ee4c3e83191a9c3cb9cdbe6e2ec1b748fd2751e93df", -17732.296875),
    "model5r": (
        "911fbf90769f73ec1252264609bf8f4bf4ecfa6835d182ca3ab937b55238f9f9",
        -16421.857421875,
    ),
    "native_mmd": (
        "903ee70064c4678d729eccf52fa75a357191071b0e8d1403bbce13560ba69b52",
        -17732.296875,
    ),
    "model2_mmd": (
        "3b39716648759c892885588823774f8c904299472456b9f2be9d3716738a3023",
        -17053.6484375,
    ),
    "model5_mmd": (
        "e07c0da42de5661ab93798ed9cbf1ff4bceee060d8578484249c3869c56652ed",
        -17130.6484375,
    ),
    # computed at methods 60577d0 (model8 = model5's networks + the cycle penalty)
    "model8": (
        "e07c0da42de5661ab93798ed9cbf1ff4bceee060d8578484249c3869c56652ed",
        -17130.6484375,
    ),
    # computed at methods fd8a37a (model8v = model8 + VampPrior pseudo-inputs, built last)
    "model8v": ("f2f813831266e35931cf07d4fad26f3275b4c2b53637854af3bb065ddd081f59", -17129.0),
}
# preset -> (sha256 of the seed-0 state_dict, model log-prob with the table as built). Pins the
# construction order (networks, r or ρ, t, then the N(0, 1) table) against later edits.
ANCHORED_FINGERPRINTS = {
    "model7": ("2913558dc3e48bdb3a5cd9f7069ec6a53acc3c921e75249105858dd3d8df2338", -16428.55859375),
    "model7i": (
        "15b2e1f38e7d5825f40813fb0027e9b87169cb5d0ec86577b02efb5f2a73db8b",
        -16353.306640625,
    ),
}
# Each existing preset's own penalty weights (none have prior_residual).
OWN_PENALTIES = {"native_mmd": {"mmd_v": 1.0}, "model2_mmd": {"mmd_v": 1.0}}
OWN_PENALTIES.update({"model5_mmd": {"mmd_v": 1.0}, "model8": {"cycle_v": 1.0}})
OWN_PENALTIES["model8v"] = {"cycle_v": 1.0}


def _make_adata(n_cells: int = 120, seed: int = 0) -> ad.AnnData:
    """Synthetic integer counts with a categorical batch column (as the residual tests)."""
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


def _build(preset: str, seed: int = 0, batch_key: str = BATCH_KEY, **overrides) -> Decipher:
    """Untrained eval-mode Decipher of `preset` (plus config overrides), Pyro seed `seed`."""
    config = DecipherConfig(**{**PRESETS[preset], **overrides})
    config.initialize_from_adata(ADATA, batch_key=batch_key)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(seed)
    return Decipher(config).eval()


def _randomize(param: torch.Tensor, seed: int = 1, scale: float = 1.0) -> None:
    """Overwrite `param` with scale * N(0, 1) draws from a fixed generator."""
    with torch.no_grad():
        draws = torch.randn(param.shape, generator=torch.Generator().manual_seed(seed))
        param.copy_(scale * draws.to(param.dtype))


def _r(model: Decipher) -> torch.nn.Module:
    """The decoder's batch network: r (model7) or ρ (model7i)."""
    return model.decoder_residual or model.decoder_residual_in_span


def _perturbed(preset: str, **overrides) -> Decipher:
    """`preset` with t's and r's (or ρ's) zero output layers replaced by small random weights."""
    model = _build(preset, **overrides)
    for seed, param in enumerate(
        [model.prior_residual[-1].weight, model.prior_residual[-1].bias]
        + [_r(model)[-1].weight, _r(model)[-1].bias]
    ):
        _randomize(param, seed=10 + seed, scale=0.3)
    return model


def _state_dict_sha256(model: Decipher) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def _model_log_prob(model: Decipher) -> float:
    """Model trace log-prob under a replayed guide trace, Pyro seed 0, real batch codes."""
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = poutine.replay(model.model, trace=guide_trace)
        return poutine.trace(replayed).get_trace(X, BATCH_IDX).log_prob_sum().item()


def _model_sites(model: Decipher, code: int):
    """(v, z, z-site Normal, x-site NB) of the model run with every cell in batch `code`, on one
    fixed guide sample (real batch codes, Pyro seed 0), so v and z are the same for every code."""
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        codes = torch.full_like(BATCH_IDX, code)
        replayed = poutine.replay(model.model, trace=guide_trace)
        nodes = poutine.trace(replayed).get_trace(X, codes).nodes
    v, z = guide_trace.nodes["v"]["value"], guide_trace.nodes["z"]["value"]
    return v, z, nodes["z"]["fn"].base_dist, nodes["x"]["fn"].base_dist


# --- flags off: every existing preset is what it was at 0fa879a ---------------------------------


def _explicit(preset: str) -> dict:
    """The new fields set explicitly: the preset's derived prior wiring and a non-default lambda_r."""
    return {"prior_batch": PRIOR_BATCH[preset], "lambda_r": 7.0}


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
@pytest.mark.parametrize("preset", sorted(BASE_FINGERPRINTS))
def test_existing_preset_state_dict_equals_base_fingerprint(preset, explicit):
    model = _build(preset, **(_explicit(preset) if explicit else {}))
    assert _state_dict_sha256(model) == BASE_FINGERPRINTS[preset][0]


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
@pytest.mark.parametrize("preset", sorted(BASE_FINGERPRINTS))
def test_existing_preset_model_log_prob_equals_base_fingerprint(preset, explicit):
    model = _build(preset, **(_explicit(preset) if explicit else {}))
    if model.batch_emb is not None:
        _randomize(model.batch_emb.weight)
    assert _model_log_prob(model) == pytest.approx(BASE_FINGERPRINTS[preset][1], rel=1e-6)


@pytest.mark.parametrize("preset", sorted(BASE_FINGERPRINTS))
def test_existing_preset_has_no_prior_residual_penalty(preset):
    expected = OWN_PENALTIES.get(preset, {})
    assert _build(preset, lambda_r=7.0).penalty_weights() == expected


@pytest.mark.parametrize("preset", sorted(PRIOR_BATCH))
def test_existing_preset_maps_onto_its_prior_wiring(preset):
    assert DecipherConfig(**PRESETS[preset]).prior_batch == PRIOR_BATCH[preset]


def test_new_fields_default_to_a_batch_blind_prior_and_unit_lambda_r():
    config = DecipherConfig()
    assert (config.prior_batch, config.lambda_r) == ("none", 1.0)


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_preset_is_model5_with_a_residual_prior_and_unit_lambda_r(preset):
    recon = {"model7": "residual", "model7i": "residual_in_span"}[preset]
    expected = {**PRESETS["model5"], "recon_batch": recon, "prior_batch": "residual"}
    assert PRESETS[preset] == {**expected, "lambda_r": 1.0}


# --- config errors ------------------------------------------------------------------------------

_NOT_CONCAT_X_BATCH = [
    dict(batch_conditioning="decoder_only", batch_embedding_mode="concat_z"),
    dict(batch_conditioning="decoder_encoder", batch_embedding_mode="concat_z"),
    dict(batch_conditioning="none", batch_embedding_mode="concat_x"),
    dict(batch_conditioning="none", batch_embedding_mode="concat_z"),
]


@pytest.mark.parametrize("flags", _NOT_CONCAT_X_BATCH)
def test_residual_prior_without_a_concat_x_batch_arm_raises(flags):
    with pytest.raises(ValueError, match="prior_batch='residual'"):
        DecipherConfig(**flags, prior_batch="residual")


@pytest.mark.parametrize("flags", _NOT_CONCAT_X_BATCH)
def test_in_span_decoder_without_a_concat_x_batch_arm_raises(flags):
    with pytest.raises(ValueError, match="recon_batch='residual_in_span'"):
        DecipherConfig(**flags, recon_batch="residual_in_span")


@pytest.mark.parametrize(
    "preset, prior_batch", [("model5", "concat"), ("model2", "none"), ("native", "concat")]
)
def test_prior_wiring_that_contradicts_the_mode_raises(preset, prior_batch):
    with pytest.raises(ValueError, match="contradicts"):
        DecipherConfig(**PRESETS[preset], prior_batch=prior_batch)


def test_unknown_prior_batch_raises():
    with pytest.raises(ValueError, match="prior_batch='additive' is not one of"):
        DecipherConfig(**PRESETS["model5"], prior_batch="additive")


def test_negative_lambda_r_raises():
    with pytest.raises(ValueError, match="lambda_r"):
        DecipherConfig(**{**PRESETS["model7"], "lambda_r": -0.1})


def test_zero_lambda_r_is_allowed():
    assert DecipherConfig(**{**PRESETS["model7"], "lambda_r": 0.0}).lambda_r == 0.0


# --- construction -------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_state_dict_equals_its_fingerprint(preset):
    assert _state_dict_sha256(_build(preset)) == ANCHORED_FINGERPRINTS[preset][0]


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_model_log_prob_equals_its_fingerprint(preset):
    assert _model_log_prob(_build(preset)) == pytest.approx(
        ANCHORED_FINGERPRINTS[preset][1], rel=1e-6
    )


_G_SHAPES = {
    "prior_residual.0.weight": (32, DIM_V + D_E),
    "prior_residual.0.bias": (32,),
    "prior_residual.2.weight": (DIM_Z, 32),
    "prior_residual.2.bias": (DIM_Z,),
}
NEW_SHAPES = {
    "model7": {
        **_G_SHAPES,
        "decoder_residual.0.weight": (64, DIM_Z + D_E),
        "decoder_residual.0.bias": (64,),
        "decoder_residual.2.weight": (N_GENES, 64),
        "decoder_residual.2.bias": (N_GENES,),
    },
    "model7i": {
        **_G_SHAPES,
        "decoder_residual_in_span.0.weight": (32, DIM_Z + D_E),
        "decoder_residual_in_span.0.bias": (32,),
        "decoder_residual_in_span.2.weight": (DIM_Z, 32),
        "decoder_residual_in_span.2.bias": (DIM_Z,),
    },
}


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_t_and_r_have_the_planned_widths(preset):
    state = _build(preset).state_dict()
    model5_keys = set(_build("model5").state_dict())
    new = {key: tuple(value.shape) for key, value in state.items() if key not in model5_keys}
    assert new == NEW_SHAPES[preset]


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_prior_and_W_are_batch_blind_networks(preset):
    model = _build(preset)
    widths = (
        model.decoder_v_to_z.layers[0].in_features,
        model.decoder_z_to_x.layers[0].in_features,
    )
    assert widths == (DIM_V, DIM_Z)


@pytest.mark.parametrize(
    "preset, overrides",
    [
        ("model7", {}),
        ("model7i", {}),
        ("model5", {"prior_batch": "residual"}),
        ("model5", {"recon_batch": "residual_in_span"}),
    ],
    ids=["model7", "model7i", "prior_residual_only", "in_span_only"],
)
def test_anchored_batch_emb_has_the_normal_init(preset, overrides):
    assert _build(preset, **overrides).batch_emb.weight.abs().min() > 0


_SHARED_NETS = {
    "model7": [
        "encoder_x_to_z",
        "encoder_zx_to_v",
        "decoder_v_to_z",
        "decoder_z_to_x",
        "decoder_residual",
    ],
    "model7i": ["encoder_x_to_z", "encoder_zx_to_v", "decoder_v_to_z", "decoder_z_to_x"],
}


@pytest.mark.parametrize(
    "preset, net", [(p, net) for p, nets in _SHARED_NETS.items() for net in nets]
)
def test_anchored_keeps_model5r_seed_draws_for_the_networks_built_before_t(preset, net):
    anchored, reference = getattr(_build(preset), net), getattr(_build("model5r"), net)
    for (key, a), (_, b) in zip(anchored.state_dict().items(), reference.state_dict().items()):
        assert torch.equal(a, b), key


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_with_zero_batches_builds_no_table_t_or_r_and_has_no_penalty(preset):
    model = _build(preset, batch_key="no_such_column")
    built = (model.batch_emb, model.prior_residual, _r(model))
    assert (built, model.penalty_weights()) == ((None, None, None), {})


# --- with model8's cycle penalty -------------------------------------------------------------


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_with_the_cycle_flag_explicitly_off_keeps_its_fingerprint(preset):
    model = _build(preset, cycle_v=False, lambda_c=7.0)
    assert _state_dict_sha256(model) == ANCHORED_FINGERPRINTS[preset][0]


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_with_cycle_v_has_both_penalty_weights(preset):
    model = _build(preset, cycle_v=True, lambda_c=3.0, lambda_r=10.0)
    assert model.penalty_weights() == {"prior_residual": 10.0, "cycle_v": 3.0}


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_with_cycle_v_keeps_the_prior_residual_penalty_value(preset):
    model = _perturbed(preset, cycle_v=True, lambda_r=10.0)
    pyro.util.set_rng_seed(7)
    guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
    values = model.penalties(guide_trace, X, BATCH_IDX)
    reference = _reference_penalty(model, guide_trace.nodes["v"]["value"], BATCH_IDX)
    assert values["prior_residual"].item() == pytest.approx(reference.item(), rel=1e-6)
    assert values["cycle_v"].item() > 0


# --- with model8v's VampPrior on v ---------------------------------------------------------------


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_with_vamp_keeps_every_anchored_weight_and_adds_only_pseudo_inputs(preset):
    vamp, plain = _build(preset, prior="vamp").state_dict(), _build(preset).state_dict()
    assert set(vamp) - set(plain) == {"pseudo_inputs_raw"}
    for key in plain:
        assert torch.equal(vamp[key], plain[key]), key


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_with_vamp_has_the_anchored_prior_on_z_and_a_finite_step(preset):
    model = _perturbed(preset, prior="vamp", lambda_r=10.0)
    v, _, z_dist, _ = _model_sites(model, 1)
    e = model.batch_emb.weight[torch.full((len(v),), 1)]
    with torch.no_grad():
        expected_loc, _ = _reference_prior(model, v, e)
    torch.testing.assert_close(z_dist.loc, expected_loc, rtol=1e-6, atol=1e-6)
    model.train()
    loss, values = penalized_step(model, Trace_ELBO(), _optimizer(), X, BATCH_IDX)
    assert np.isfinite(loss) and list(values) == ["prior_residual"]


# --- init: the prior is f_mean(v) and the logits W z + c, for every batch -------------------------


@pytest.mark.parametrize("code", range(len(BATCHES)))
@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_prior_at_init_is_f_mean_for_every_batch(preset, code):
    model = _build(preset)
    v, _, z_dist, _ = _model_sites(model, code)
    with torch.no_grad():
        f_loc, f_scale = model.decoder_v_to_z(v)
    assert torch.equal(z_dist.loc, f_loc) and torch.equal(z_dist.scale, softplus(f_scale))


def _nb_logits(model: Decipher, logits: torch.Tensor) -> torch.Tensor:
    """The x-site NB logits Decipher.model builds from reconstruction logits."""
    mu = softmax(logits, dim=-1)
    theta = pyro.param("theta")
    return torch.log(X.sum(-1, keepdim=True) * mu + EPS) - torch.log(theta + EPS)


@pytest.mark.parametrize("code", range(len(BATCHES)))
@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_logits_at_init_are_W_z_plus_c_for_every_batch(preset, code):
    model = _build(preset)
    _, z, _, x_dist = _model_sites(model, code)
    with torch.no_grad():
        expected = _nb_logits(model, model.decoder_z_to_x(z))
    torch.testing.assert_close(x_dist.logits, expected, rtol=1e-6, atol=1e-6)


# --- z-scale ------------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ANCHORED)
def test_anchored_z_scale_is_f_var_for_every_batch(preset):
    model = _perturbed(preset)
    v = _model_sites(model, 0)[0]
    with torch.no_grad():
        f_scale = softplus(model.decoder_v_to_z(v)[1])
    for code in range(len(BATCHES)):
        assert torch.equal(_model_sites(model, code)[2].scale, f_scale), code


@pytest.mark.parametrize("preset", ANCHORED)
def test_perturbed_anchored_z_loc_depends_on_the_batch(preset):
    model = _perturbed(preset)
    locs = [_model_sites(model, code)[2].loc for code in range(len(BATCHES))]
    assert not torch.allclose(locs[0], locs[1]) and not torch.allclose(locs[1], locs[2])


# --- gradients: every parameter against a written-out reference (float64) -----------------------


def _reference_decoder(model: Decipher, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """W z + c + r([z; e]) (model7) or W(z + ρ([z; e])) + c (model7i), by hand."""
    ze = torch.cat([z, e], dim=-1)
    if model.config.recon_batch == "residual":
        return model.decoder_z_to_x(z) + model.decoder_residual(ze)
    return model.decoder_z_to_x(z + model.decoder_residual_in_span(ze))


def _reference_prior(model: Decipher, v: torch.Tensor, e: torch.Tensor):
    """(z_loc, z_scale) = (f_mean(v) + t([v; e]), softplus(f~(v))) by hand; f~ = the raw scale
    head."""
    f_loc, f_scale = model.decoder_v_to_z(v)
    return f_loc + model.prior_residual(torch.cat([v, e], dim=-1)), softplus(f_scale)


def _reference_loss(model: Decipher, x, codes, eps_z, eps_v) -> torch.Tensor:
    """-ELBO of one reparameterized sample (noise eps_z, eps_v), written out term by term."""
    normal = torch.distributions.Normal
    xl, e = torch.log1p(x), model.batch_emb.weight[codes]
    z_loc, z_raw = model.encoder_x_to_z(torch.cat([xl, e], dim=-1))
    z_scale = softplus(z_raw) + EPS
    z = z_loc + z_scale * eps_z
    v_loc, v_raw = model.encoder_zx_to_v(torch.cat([z, xl], dim=-1))
    v_scale = softplus(v_raw) + EPS
    v = v_loc + v_scale * eps_v
    log_q = (
        normal(z_loc, z_scale).log_prob(z).sum() + BETA * normal(v_loc, v_scale).log_prob(v).sum()
    )
    log_pv = BETA * normal(0.0, 1.0).log_prob(v).sum()
    log_pz = normal(*_reference_prior(model, v, e)).log_prob(z).sum()
    theta = pyro.param("theta")
    nb_logits = torch.log(
        x.sum(-1, keepdim=True) * softmax(_reference_decoder(model, z, e), -1) + EPS
    )
    nb = torch.distributions.NegativeBinomial(
        theta + EPS, logits=nb_logits - torch.log(theta + EPS)
    )
    return -(log_pv + log_pz + nb.log_prob(x).sum() - log_q)


def _guide_noise(model: Decipher, guide_trace, x, codes):
    """The standard-normal noise (eps_z, eps_v) behind a guide trace's z and v samples."""
    with torch.no_grad():
        xl, e = torch.log1p(x), model.batch_emb.weight[codes]
        z_loc, z_raw = model.encoder_x_to_z(torch.cat([xl, e], dim=-1))
        z = guide_trace.nodes["z"]["value"]
        v_loc, v_raw = model.encoder_zx_to_v(torch.cat([z, xl], dim=-1))
        eps_z = (z - z_loc) / (softplus(z_raw) + EPS)
        eps_v = (guide_trace.nodes["v"]["value"] - v_loc) / (softplus(v_raw) + EPS)
    return eps_z, eps_v


def _grads(model: Decipher, loss: torch.Tensor) -> dict:
    params = dict(model.named_parameters())
    return dict(
        zip(
            params,
            torch.autograd.grad(loss, list(params.values()), allow_unused=True, retain_graph=True),
        )
    )


def _double_perturbed(preset: str, **overrides) -> Decipher:
    model = _perturbed(preset, **overrides).double()
    pyro.clear_param_store()
    return model


X64 = X.double()


@pytest.fixture(scope="module", params=ANCHORED)
def _elbo_grads(request):
    """(pyro's -ELBO gradients, the reference's) for every parameter, on one seeded sample."""
    model = _double_perturbed(request.param)
    pyro.util.set_rng_seed(7)
    pyro_loss = Trace_ELBO().differentiable_loss(model.model, model.guide, X64, BATCH_IDX)
    pyro.util.set_rng_seed(7)
    guide_trace = poutine.trace(model.guide).get_trace(X64, BATCH_IDX)
    eps_z, eps_v = _guide_noise(model, guide_trace, X64, BATCH_IDX)
    reference = _reference_loss(model, X64, BATCH_IDX, eps_z, eps_v)
    return request.param, _grads(model, pyro_loss), _grads(model, reference)


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got - want).norm() / want.norm()).item()


GRAD_PATHS = {
    "model7": ["prior_residual", "decoder_residual", "batch_emb"],
    "model7i": ["prior_residual", "decoder_residual_in_span", "batch_emb"],
}


def test_elbo_gradient_of_t_r_and_batch_emb_matches_the_reference(_elbo_grads):
    preset, got, want = _elbo_grads
    for key in [k for k in want if k.split(".")[0] in GRAD_PATHS[preset]]:
        assert want[key].norm() > 0, f"{key}: dead path in the reference"
        assert _rel_err(got[key], want[key]) < 1e-5, key


def test_elbo_gradient_of_every_other_parameter_matches_the_reference(_elbo_grads):
    preset, got, want = _elbo_grads
    for key in [k for k in want if k.split(".")[0] not in GRAD_PATHS[preset]]:
        if want[key] is None or want[key].norm() == 0:
            assert got[key] is None or got[key].norm() == 0, key
        else:
            assert _rel_err(got[key], want[key]) < 1e-5, key


def _reference_penalty(model: Decipher, v: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """mean over cells of ||t([v_i; e_{b_i}])||^2, written out cell by cell."""
    e = model.batch_emb.weight[codes]
    t = model.prior_residual(torch.cat([v, e], dim=-1))
    return sum((t[i] ** 2).sum() for i in range(len(v))) / len(v)


@pytest.fixture(scope="module", params=ANCHORED)
def _penalty_and_reference(request):
    """(model at lambda_r = 10, penalties()'s value, the reference on the same guide sample)."""
    model = _double_perturbed(request.param, lambda_r=10.0)
    pyro.util.set_rng_seed(7)
    guide_trace = poutine.trace(model.guide).get_trace(X64, BATCH_IDX)
    value = model.penalties(guide_trace, X64, BATCH_IDX)["prior_residual"]
    reference = _reference_penalty(model, guide_trace.nodes["v"]["value"], BATCH_IDX)
    return model, value, reference


def test_penalty_is_the_unweighted_mean_squared_norm_of_t(_penalty_and_reference):
    _, value, reference = _penalty_and_reference
    assert reference.item() > 0.01
    assert value.item() == pytest.approx(reference.item(), rel=1e-10)


def test_penalty_gradient_of_every_parameter_matches_the_reference(_penalty_and_reference):
    model, value, reference = _penalty_and_reference
    got, want = _grads(model, value), _grads(model, reference)
    reached = {key.split(".")[0] for key in want if want[key] is not None and want[key].norm() > 0}
    assert {"prior_residual", "batch_emb", "encoder_x_to_z", "encoder_zx_to_v"} <= reached
    for key in want:
        if want[key] is None:
            assert got[key] is None, key
        else:
            assert _rel_err(got[key], want[key]) < 1e-5, key


# --- penalty weight, the training step and val NLL ----------------------------------------------


@pytest.mark.parametrize("lambda_r", [0.0, 0.1, 10.0])
@pytest.mark.parametrize("preset", ANCHORED)
def test_penalty_weight_is_lambda_r(preset, lambda_r):
    assert _build(preset, lambda_r=lambda_r).penalty_weights() == {"prior_residual": lambda_r}


def _optimizer() -> pyro.optim.PyroOptim:
    return pyro.optim.ClippedAdam({"lr": 5e-2, "weight_decay": 1e-4})


@pytest.mark.parametrize("preset", ANCHORED)
def test_penalized_step_scores_the_elbo_and_penalty_on_one_guide_sample(preset):
    model = _perturbed(preset, lambda_r=10.0)  # != 1, so a weighted value would differ
    pyro.util.set_rng_seed(7)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = poutine.replay(model.model, trace=guide_trace)
        model_trace = poutine.trace(replayed).get_trace(X, BATCH_IDX)
        elbo_loss = (guide_trace.log_prob_sum() - model_trace.log_prob_sum()).item()
        penalty = _reference_penalty(model, guide_trace.nodes["v"]["value"], BATCH_IDX).item()
    pyro.util.set_rng_seed(7)
    loss, values = penalized_step(model, Trace_ELBO(), _optimizer(), X, BATCH_IDX)
    assert (loss, values) == (
        pytest.approx(elbo_loss, rel=1e-5),
        {"prior_residual": pytest.approx(penalty, rel=1e-5)},
    )


_G_KEY = "prior_residual.0.weight"


def _sgd_update(preset: str, lambda_r: float) -> torch.Tensor:
    """Change of t's first-layer weight after one plain-SGD (lr 1) penalized step, seeded."""
    model = _perturbed(preset, lambda_r=lambda_r)
    before = model.state_dict()[_G_KEY].clone()
    pyro.util.set_rng_seed(7)
    penalized_step(model, Trace_ELBO(), pyro.optim.SGD({"lr": 1.0}), X, BATCH_IDX)
    return model.state_dict()[_G_KEY] - before


@pytest.mark.parametrize("preset", ANCHORED)
def test_penalized_step_adds_lambda_r_times_the_penalty_gradient(preset):
    """With SGD, update(10) - update(0) = -10 dP/dw on the step's own sample (relative norm, as
    the MMD test: the ELBO gradient's float32 rounding does not cancel exactly)."""
    model = _perturbed(preset)
    pyro.util.set_rng_seed(7)
    guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
    penalty = model.penalties(guide_trace, X, BATCH_IDX)["prior_residual"]
    (grad,) = torch.autograd.grad(penalty, model.prior_residual[0].weight)
    diff = _sgd_update(preset, 10.0) - _sgd_update(preset, 0.0)
    assert ((diff + 10.0 * grad).norm() / (10.0 * grad).norm()).item() < 1e-2


def _val_nll(preset: str, state_dict: dict, **overrides) -> float:
    model = _build(preset, **overrides)
    model.load_state_dict(state_dict)
    pyro.util.set_rng_seed(0)
    loader = make_data_loader_from_adata(
        ADATA, 64, context_levels={BATCH_KEY: list(ADATA.obs[BATCH_KEY].cat.categories)}
    )
    return predictive_log_likelihood(model, loader, n_samples=2)


@pytest.mark.parametrize("preset", ANCHORED)
def test_val_nll_is_the_same_for_zero_and_positive_lambda_r(preset):
    weights = _perturbed(preset).state_dict()  # t != 0, so the penalty is not 0
    assert _val_nll(preset, weights, lambda_r=0.0) == _val_nll(preset, weights, lambda_r=10.0)


# --- gene_patterns: the residual prior and the decoder variant, per batch -----------------------


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


def _pattern_mean(adata, batch=None) -> np.ndarray:
    gene_patterns(adata, l_scale=L_SCALE, n_samples=5, batch=batch)
    return adata.uns["decipher"]["gene_patterns"]["t"]["mean"].copy()


def _reference_pattern_mean(model: Decipher, code: int) -> np.ndarray:
    """softmax(decoder(z_loc_b, e_b)) * L_SCALE at POINTS, z_loc_b = f_mean(v) + t([v; e_b])."""
    v = torch.FloatTensor(POINTS)
    with torch.no_grad():
        e = model.batch_emb.weight[torch.full((len(v),), code)]
        z_loc, _ = _reference_prior(model, v, e)
        return softmax(_reference_decoder(model, z_loc, e), -1).numpy() * L_SCALE


@pytest.mark.parametrize("code", range(len(BATCHES)))
@pytest.mark.parametrize("preset", ANCHORED)
def test_gene_patterns_for_one_batch_use_its_residual_prior_and_decoder(saved, preset, code):
    model = _perturbed(preset)
    got = _pattern_mean(saved(model), BATCHES[code])
    np.testing.assert_allclose(got, _reference_pattern_mean(model, code), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("preset", ANCHORED)
def test_gene_patterns_default_mean_is_the_uniform_average_over_batches(saved, preset):
    model = _perturbed(preset)
    expected = np.mean([_reference_pattern_mean(model, b) for b in range(len(BATCHES))], axis=0)
    np.testing.assert_allclose(_pattern_mean(saved(model)), expected, rtol=1e-5, atol=1e-6)


# --- decipher_train at lambda_r = 10: history and train ELBO stay unweighted ---------------------

_EPOCHS = 3
_STEPS_PER_EPOCH = 3  # 231 train cells of _make_adata(256), batch size 64, drop_last
_OBS_PER_EPOCH = _STEPS_PER_EPOCH * 64


@pytest.fixture(scope="module", params=ANCHORED)
def _recorded_lambda10(request, tmp_path_factory):
    """`preset` trained at lambda_r = 10 with every penalized_step's reported (ELBO loss,
    unweighted prior_residual) recorded, shaped (epochs, steps)."""
    recorded = []

    def recording_step(*args):
        loss, values = penalized_step(*args)
        recorded.append((loss, values["prior_residual"]))
        return loss, values

    DECIPHER_GLOBALS["save_folder"] = str(tmp_path_factory.mktemp("rec")) + "/"
    config = DecipherConfig(
        **{**PRESETS[request.param], "lambda_r": 10.0},
        seed=1,
        n_epochs=_EPOCHS,
        early_stopping_patience=None,
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dtool, "penalized_step", recording_step)
            model, _ = dc.tl.decipher_train(
                _make_adata(n_cells=256),
                config,
                batch_key=BATCH_KEY,
                plot_every_k_epochs=-2,
                plot_kwargs={"color": BATCH_KEY},
            )
    finally:
        torch.set_num_threads(threads)
    steps = np.array(recorded).reshape(_EPOCHS, _STEPS_PER_EPOCH, 2)
    return model, steps[..., 0], steps[..., 1]


def test_penalty_history_is_the_per_epoch_mean_of_the_unweighted_step_values(_recorded_lambda10):
    model, _, penalty = _recorded_lambda10
    assert penalty[1:].min() > 0, "t must have moved off zero for this test to mean anything"
    expected = penalty.mean(axis=1).tolist()
    assert model.penalty_history_ == {"prior_residual": pytest.approx(expected, rel=1e-12)}


def test_train_elbo_of_a_model7_run_excludes_the_penalty(_recorded_lambda10):
    model, elbo, _ = _recorded_lambda10
    expected = (elbo.sum(axis=1) / _OBS_PER_EPOCH).tolist()
    assert model.train_losses_ == pytest.approx(expected, rel=1e-12)
