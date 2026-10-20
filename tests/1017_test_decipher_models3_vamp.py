"""`decipher_models3` model8v of the batch-invariance plan: model8 with a VampPrior on v
(`prior="vamp"`, `n_pseudo_inputs` = K), p(v) = (1/K) sum_k q(v | u_k, b_k).

Checks: the mixture density matches a numpy logsumexp; the model's v site is that mixture of the
guide's own v distribution at the pseudo-inputs (z-encoder loc, softplus v scale + eps, b_k round
robin); pseudo-inputs are >= 0, seeded by config.seed without touching the global RNG, and get
gradients through log p(v); the pseudo-input passes leave BatchNorm running statistics alone;
flags off builds the same model for every existing preset; model8v is model8 plus the
pseudo-inputs (fingerprint pinned); val NLL scores v under the VampPrior and excludes the cycle
penalty; best-epoch restore and checkpoints carry the pseudo-inputs; K <= 0 raises.
"""

import copy
import hashlib

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
import torch.nn as nn
from pyro import poutine
from pyro.infer import Trace_ELBO
from torch.nn.functional import softplus

import decipher_models3 as dc
import decipher_models3.tools._decipher.data as data
from decipher_models3.presets import PRESETS
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.tools._decipher.decipher import round_robin_batches, vamp_mixture
from decipher_models3.tools.decipher import penalized_step, predictive_log_likelihood
from decipher_models3.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")

# Every preset other than model8v.
EXISTING = [
    "native",
    "model1",
    "model2",
    "model3",
    "model4",
    "model5",
    "model6",
    "model4r",
    "model5r",
    "native_mmd",
    "model2_mmd",
    "model5_mmd",
    "model8",
]
BATCH_KEY = "batch"
N_GENES = 30
N_BATCHES = 3
EPS = 1e-5  # the guide's floor on the v scale, written here so a changed constant fails

# (sha256 of the seed-0 state_dict, model trace log-prob with a randomized table) of model8v,
# taken at the build commit with the builders below. Pins the construction (model8's networks and
# table, then K x G pseudo-inputs from a generator seeded with config.seed) and the prior's math.
MODEL8V_FINGERPRINT = (
    "427a3f69ed9162df401e7c4041b10189fd8b309f2f99be92c976d496055f8160",
    -11498.744140625,
)


def _adata(n_cells: int = 128, n_batches: int = N_BATCHES, seed: int = 0) -> ad.AnnData:
    """Synthetic integer counts with a categorical batch column."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=N_GENES), size=(n_cells, N_GENES))
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([f"b{i % n_batches}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


ADATA = _adata()
X = torch.tensor(ADATA.X)
BATCH_IDX = torch.tensor(ADATA.obs[BATCH_KEY].cat.codes.values).long()
# Fixed v values (one per cell) at which the prior density is evaluated.
V = torch.randn(len(X), 2, generator=torch.Generator().manual_seed(11))


def _build(
    preset: str = "model8v",
    pyro_seed: int = 0,
    adata: ad.AnnData = ADATA,
    batch_key: str = BATCH_KEY,
    **overrides,
) -> Decipher:
    """Untrained Decipher of `preset` (plus config overrides), built with Pyro seed `pyro_seed`."""
    config = DecipherConfig(**{**PRESETS[preset], **overrides})
    config.initialize_from_adata(adata, batch_key=batch_key)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(pyro_seed)
    return Decipher(config)


def _randomize(param: torch.Tensor, seed: int = 1) -> None:
    """Overwrite `param` with N(0, 1) draws from a fixed generator."""
    with torch.no_grad():
        param.copy_(torch.randn(param.shape, generator=torch.Generator().manual_seed(seed)))


def _model8v(**overrides) -> Decipher:
    """model8v with a nonzero batch table, so b_k reaches the z-encoder."""
    model = _build("model8v", **overrides)
    _randomize(model.batch_emb.weight)
    return model


def _v_site_log_prob(model: Decipher, v: torch.Tensor = V) -> torch.Tensor:
    """Per-cell unscaled log p(v) of the model's "v" site, with v fixed to `v`."""
    trace = poutine.trace(poutine.condition(model.model, data={"v": v})).get_trace(X, BATCH_IDX)
    trace.compute_log_prob()
    return trace.nodes["v"]["unscaled_log_prob"]


def _model_log_prob(model: Decipher) -> float:
    """Model trace log-prob under a replayed guide trace, Pyro seed 0, real batch codes."""
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = poutine.replay(model.model, trace=guide_trace)
        return poutine.trace(replayed).get_trace(X, BATCH_IDX).log_prob_sum().item()


def _state_dict_sha256(model: Decipher) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def _np_vamp_log_prob(v, loc, scale) -> np.ndarray:
    """log (1/K) sum_k prod_d N(v_d; loc_kd, scale_kd), per row of v, in float64 numpy."""
    v, loc, scale = (np.asarray(t, dtype=np.float64) for t in (v, loc, scale))
    z = (v[:, None, :] - loc[None]) / scale[None]
    comp = (-0.5 * np.log(2 * np.pi) - np.log(scale)[None] - 0.5 * z**2).sum(-1)
    top = comp.max(1, keepdims=True)
    return (top + np.log(np.exp(comp - top).sum(1, keepdims=True)))[:, 0] - np.log(len(loc))


def _components_reference(model: Decipher) -> tuple[torch.Tensor, torch.Tensor]:
    """q(v | u_k, b_k) written out with model5's networks (inputs [log1p x; e_b] and [z; log1p x]):
    u = softplus(raw) replaces x (so log1p u), b_k = k mod B, z = z-encoder loc,
    scale = softplus + EPS."""
    x_log = torch.log1p(softplus(model.pseudo_inputs_raw))
    b = torch.arange(model.config.n_pseudo_inputs) % model.config.n_batches
    z_loc, _ = model.encoder_x_to_z(torch.cat([x_log, model.batch_emb(b)], dim=1))
    v_loc, v_scale = model.encoder_zx_to_v(torch.cat([z_loc, x_log], dim=1))
    return v_loc, softplus(v_scale) + EPS


def _reference_log_prob(model: Decipher, v: torch.Tensor = V) -> torch.Tensor:
    """logsumexp_k log N(v; loc_k, scale_k) - log K from `_components_reference`, in torch."""
    loc, scale = _components_reference(model)
    comp = torch.distributions.Normal(loc, scale).log_prob(v[:, None, :]).sum(-1)
    return torch.logsumexp(comp, dim=1) - np.log(model.config.n_pseudo_inputs)


# --- the mixture density -------------------------------------------------------------------------

_LOC = torch.tensor([[0.0, 0.0], [1.0, -1.0], [3.0, 2.0]])
_SCALE = torch.tensor([[1.0, 1.0], [0.5, 2.0], [1.5, 0.3]])
_V_HAND = torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.0, 1.0], [-1.0, 3.0]])


def test_vamp_mixture_log_prob_matches_a_numpy_logsumexp():
    got = vamp_mixture(_LOC, _SCALE).log_prob(_V_HAND).numpy()
    np.testing.assert_allclose(got, _np_vamp_log_prob(_V_HAND, _LOC, _SCALE), rtol=1e-6)


def test_vamp_mixture_of_one_component_is_that_gaussian():
    got = vamp_mixture(_LOC[1:2], _SCALE[1:2]).log_prob(_V_HAND)
    expected = torch.distributions.Normal(_LOC[1], _SCALE[1]).log_prob(_V_HAND).sum(-1)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_model_v_site_is_the_vamp_mixture_of_the_components(mode):
    model = _model8v().train(mode == "train")
    with torch.no_grad():
        loc, scale = model.vamp_components()
        got = _v_site_log_prob(model).numpy()
    np.testing.assert_allclose(got, _np_vamp_log_prob(V, loc, scale), rtol=1e-5)


# --- the components: the guide's own v distribution at the pseudo-inputs -------------------------


def test_vamp_components_follow_the_guides_path_in_eval_mode():
    model = _model8v().eval()
    with torch.no_grad():
        got, expected = model.vamp_components(), _components_reference(model)
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e)


def test_vamp_components_in_train_mode_normalize_with_the_pseudo_inputs_own_statistics():
    """Train mode: BatchNorm normalizes the K pseudo-inputs with their own statistics, as a
    plain train-mode pass does (the reference runs unfrozen on a copy)."""
    model = _model8v().train()
    with torch.no_grad():
        got = model.vamp_components()
        expected = _components_reference(copy.deepcopy(model))
    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e)


def test_log_p_v_matches_the_reference():
    model = _model8v().train()
    with torch.no_grad():
        torch.testing.assert_close(_v_site_log_prob(model), _reference_log_prob(model))


_NETWORKS = ["pseudo_inputs_raw", "batch_emb", "encoder_x_to_z", "encoder_zx_to_v"]


@pytest.fixture(scope="module")
def _prior_and_reference_grads() -> tuple[dict, dict]:
    """Parameter gradients of sum_i log p(v_i) at fixed v: through the model's v site, and
    through the reference, each on a twin model8v (same weights)."""
    grads = []
    for log_prob in (_v_site_log_prob, _reference_log_prob):
        model = _model8v().train()
        log_prob(model).sum().backward()
        grads.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
    return grads[0], grads[1]


def test_log_p_v_gradient_reaches_only_the_pseudo_inputs_table_and_encoders(
    _prior_and_reference_grads,
):
    got, expected = _prior_and_reference_grads
    assert sorted(got) == sorted(expected)
    assert {name.split(".")[0] for name in got} == set(_NETWORKS)


@pytest.mark.parametrize("network", _NETWORKS)
def test_log_p_v_gradient_matches_the_reference_on_every_network(
    _prior_and_reference_grads, network
):
    got, expected = _prior_and_reference_grads
    names = sorted(k for k in expected if k.split(".")[0] == network)
    # The mixture and the reference sum in a different order. Entries that are 0 in exact
    # arithmetic (biases right before a train-mode BatchNorm) differ by float32 rounding, so atol
    # is relative to the network's largest gradient.
    largest = max(got[name].abs().max().item() for name in names)
    assert largest > 0
    for name in names:
        torch.testing.assert_close(got[name], expected[name], rtol=1e-4, atol=1e-5 * largest)


def test_one_training_step_moves_the_pseudo_inputs():
    model = _build("model8v")
    before = model.pseudo_inputs_raw.detach().clone()
    pyro.util.set_rng_seed(3)
    optimizer = pyro.optim.ClippedAdam({"lr": 1e-2, "weight_decay": 1e-4})
    penalized_step(model, Trace_ELBO(), optimizer, X, BATCH_IDX)
    assert not torch.equal(model.pseudo_inputs_raw.detach(), before)


# --- pseudo-inputs and their batches -------------------------------------------------------------


def test_pseudo_inputs_are_non_negative_for_negative_raw_values():
    model = _build("model8v")
    with torch.no_grad():
        model.pseudo_inputs_raw.fill_(-30.0)
        model.pseudo_inputs_raw[0, :5] = torch.tensor([-1.0, -5.0, 0.0, 2.0, -100.0])
    u = model.pseudo_inputs()
    assert u.shape == (100, N_GENES) and (u >= 0).all()


def test_pseudo_inputs_at_init_are_positive_counts():
    u = _build("model8v").pseudo_inputs()
    assert (u > 0).all()


def test_pseudo_input_init_is_set_by_config_seed():
    same = _build("model8v", pyro_seed=5).pseudo_inputs_raw
    assert torch.equal(_build("model8v", pyro_seed=9).pseudo_inputs_raw, same)
    assert not torch.equal(_build("model8v", pyro_seed=9, seed=1).pseudo_inputs_raw, same)


def test_pseudo_input_init_is_standard_normal_before_softplus():
    raw = _build("model8v", n_pseudo_inputs=400).pseudo_inputs_raw.detach()
    assert abs(raw.mean().item()) < 0.02 and abs(raw.std().item() - 1.0) < 0.02


@pytest.mark.parametrize("k, n_batches", [(6, 3), (7, 3), (100, 5), (100, 3), (2, 5), (1, 1)])
def test_round_robin_batches_are_k_mod_b(k, n_batches):
    assert round_robin_batches(k, n_batches).tolist() == [i % n_batches for i in range(k)]


@pytest.mark.parametrize("k, n_batches", [(7, 3), (100, 3), (100, 5), (2, 5)])
def test_round_robin_batches_are_balanced(k, n_batches):
    counts = np.bincount(round_robin_batches(k, n_batches).numpy(), minlength=n_batches)
    expected = [k // n_batches + (b < k % n_batches) for b in range(n_batches)]
    assert counts.tolist() == expected


def test_model_assigns_pseudo_inputs_round_robin_when_k_is_not_divisible_by_b():
    model = _build("model8v", n_pseudo_inputs=7)
    assert model.pseudo_input_batches().tolist() == [0, 1, 2, 0, 1, 2, 0]


def test_vamp_on_native_has_no_pseudo_input_batches_and_a_finite_log_p_v():
    model = _build("native", batch_key="no_such_column", prior="vamp", n_pseudo_inputs=5)
    with torch.no_grad():
        log_p = _v_site_log_prob(model)
    assert model.pseudo_input_batches() is None and torch.isfinite(log_p).all()


# --- BatchNorm ------------------------------------------------------------------------------------


def _batch_norms(model: Decipher) -> list:
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def test_pseudo_input_passes_leave_encoder_batch_norm_statistics_alone():
    """The model touches the encoders only through the pseudo-inputs, in train mode."""
    model = _model8v().train()
    before = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("encoder_")}
    with torch.no_grad():
        _v_site_log_prob(model)
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_pseudo_input_passes_restore_batch_norm_tracking():
    model = _model8v().train()
    with torch.no_grad():
        model.vamp_components()
    assert [m.track_running_stats for m in _batch_norms(model)] == [True] * 3


# --- flags off and the preset --------------------------------------------------------------------


@pytest.mark.parametrize("preset", EXISTING)
def test_flags_off_builds_the_same_state_dict_as_default_fields(preset):
    default = _build(preset).state_dict()
    explicit = _build(preset, prior="normal", n_pseudo_inputs=7).state_dict()
    assert default.keys() == explicit.keys() and "pseudo_inputs_raw" not in default
    for key in default:
        assert torch.equal(default[key], explicit[key]), key


@pytest.mark.parametrize("preset", EXISTING)
def test_flags_off_gives_the_same_model_log_prob_as_default_fields(preset):
    default = _build(preset).eval()
    explicit = _build(preset, prior="normal", n_pseudo_inputs=7).eval()
    assert _model_log_prob(default) == _model_log_prob(explicit)


@pytest.mark.parametrize("preset", EXISTING)
def test_flags_off_ignores_an_invalid_n_pseudo_inputs(preset):
    assert DecipherConfig(**{**PRESETS[preset], "n_pseudo_inputs": 0}).n_pseudo_inputs == 0


def test_model8v_preset_is_model8_with_the_vamp_prior():
    assert PRESETS["model8v"] == {**PRESETS["model8"], "prior": "vamp", "n_pseudo_inputs": 100}


def test_model8v_is_model8_plus_the_pseudo_inputs():
    v8, base = _build("model8v").state_dict(), _build("model8").state_dict()
    assert set(v8) - set(base) == {"pseudo_inputs_raw"} and set(base) <= set(v8)
    assert v8["pseudo_inputs_raw"].shape == (100, N_GENES)
    for key in base:
        assert torch.equal(v8[key], base[key]), key


def test_model8v_init_leaves_the_global_rng_where_model8_leaves_it():
    """No reseed after building: the first guide sample is model8's, so the init drew nothing
    from the global RNG."""
    samples = []
    for preset in ("model8v", "model8"):
        model = _build(preset)
        samples.append(poutine.trace(model.guide).get_trace(X, BATCH_IDX).nodes["z"]["value"])
    assert torch.equal(samples[0], samples[1])


def test_model8v_state_dict_equals_fingerprint():
    assert _state_dict_sha256(_build("model8v")) == MODEL8V_FINGERPRINT[0]


def test_model8v_model_log_prob_equals_fingerprint():
    model = _model8v().eval()
    assert _model_log_prob(model) == pytest.approx(MODEL8V_FINGERPRINT[1], rel=1e-6)


def test_model8v_cycle_penalty_is_model8s():
    """The guide still returns (z_loc, v_loc, ...), so the cycle reads the same v_loc."""
    values = []
    for preset in ("model8v", "model8"):
        model = _build(preset)
        _randomize(model.batch_emb.weight)
        pyro.util.set_rng_seed(0)
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        pyro.util.set_rng_seed(1)
        values.append(model.penalties(guide_trace, X, BATCH_IDX)["cycle_v"])
    assert torch.equal(values[0], values[1])


@pytest.mark.parametrize("n_pseudo_inputs", [0, -3])
def test_vamp_with_no_pseudo_inputs_raises(n_pseudo_inputs):
    with pytest.raises(
        ValueError, match=f"n_pseudo_inputs >= 1; got n_pseudo_inputs={n_pseudo_inputs}"
    ):
        DecipherConfig(**{**PRESETS["model8v"], "n_pseudo_inputs": n_pseudo_inputs})


# --- val NLL ---------------------------------------------------------------------------------------


def _val_nll(model: Decipher) -> float:
    """predictive_log_likelihood on one minibatch of all cells, one sample, Pyro seed 0."""
    model.eval()
    pyro.util.set_rng_seed(0)
    return predictive_log_likelihood(model, [(X, BATCH_IDX)], n_samples=1)


def _loaded(preset: str, state_dict: dict, **overrides) -> Decipher:
    model = _build(preset, **overrides)
    model.load_state_dict(state_dict)
    return model


def test_val_nll_excludes_the_cycle_penalty():
    weights = _model8v(pyro_seed=4).state_dict()
    nll = [_val_nll(_loaded("model8v", weights, lambda_c=lam)) for lam in (0.0, 10.0)]
    assert nll[0] == nll[1]


def test_val_nll_scores_v_under_the_vamp_prior_at_weight_one():
    """Same shared weights and guide sample: val NLL of model8v minus that of model8 is
    sum_i [log p_vamp(v_i) - log N(v_i; 0, I)], the eval-mode prior at beta = 1."""
    model = _model8v()
    weights = model.state_dict()
    shared = {k: v for k, v in weights.items() if k != "pseudo_inputs_raw"}
    gap = _val_nll(model) - _val_nll(_loaded("model8", shared))
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        v = poutine.trace(model.guide).get_trace(X, BATCH_IDX).nodes["v"]["value"]
        loc, scale = model.vamp_components()
    normal = (-0.5 * np.log(2 * np.pi) - 0.5 * v.double().numpy() ** 2).sum(-1)
    expected = (_np_vamp_log_prob(v, loc, scale) - normal).sum()
    assert gap == pytest.approx(expected, rel=1e-4)


# --- best-epoch restore and checkpoints ------------------------------------------------------------
# Seed 0, lr 5e-2, K = 7 on 200 cells x 30 genes: at 1 thread val NLL is lowest at epoch 1 of 3,
# so the full run restores epoch 1 and must equal the run stopped after epoch 1.

_TRAIN = dict(seed=0, learning_rate=5e-2, early_stopping_patience=None, n_pseudo_inputs=7)


def _train(n_epochs: int, save_folder) -> tuple[dict, list, int]:
    DECIPHER_GLOBALS["save_folder"] = str(save_folder) + "/"
    config = DecipherConfig(**{**PRESETS["model8v"], **_TRAIN}, n_epochs=n_epochs)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, val_losses = dc.tl.decipher_train(
            _adata(n_cells=200),
            config,
            batch_key=BATCH_KEY,
            plot_every_k_epochs=-2,
            plot_kwargs={"color": BATCH_KEY},
        )
    finally:
        torch.set_num_threads(threads)
    return copy.deepcopy(model.state_dict()), val_losses, model.best_epoch_


@pytest.fixture(scope="module")
def _full_and_truncated(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ckpt")
    full = _train(3, folder)
    return full, _train(full[2] + 1, folder)


def test_best_epoch_is_before_the_last(_full_and_truncated):
    (_, val_losses, best_epoch), _ = _full_and_truncated
    assert np.isfinite(val_losses).all() and best_epoch == int(np.argmin(val_losses)) == 1


def test_best_epoch_restore_brings_back_the_pseudo_inputs(_full_and_truncated):
    (full, _, _), (truncated, _, _) = _full_and_truncated
    init = _build("model8v", **_TRAIN, adata=_adata(n_cells=200)).pseudo_inputs_raw.detach()
    assert not torch.equal(truncated["pseudo_inputs_raw"], init)  # they trained
    assert torch.equal(full["pseudo_inputs_raw"], truncated["pseudo_inputs_raw"])


def test_checkpoint_round_trip_keeps_the_pseudo_inputs(tmp_path, monkeypatch):
    monkeypatch.setitem(DECIPHER_GLOBALS, "save_folder", str(tmp_path))
    model = _build("model8v")
    _randomize(model.pseudo_inputs_raw, seed=3)
    adata = ADATA.copy()
    data.decipher_save_model(adata, model)
    loaded = data.decipher_load_model(adata)
    assert torch.equal(loaded.pseudo_inputs_raw, model.pseudo_inputs_raw)
