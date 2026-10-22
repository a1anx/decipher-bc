"""`decipher_models3` model8 of the batch-invariance plan: latent cycle-consistency of v under a
swapped batch (sysVI) on model5 (`cycle_v=True`, weight `lambda_c`), through the penalty
interface of `Decipher` (`penalty_weights` / `penalties`).

Checks: flags off builds the same model as before for every existing preset; `model8` builds
model5; the config raises off model5's wiring and the model raises with fewer than 2 batches; the
swapped batch b' is never the cell's own and is uniform over the others; the penalty is finite,
reaches every network on both sides of the cycle, leaves BatchNorm running statistics alone, does
not depend on b' when nothing sees the batch, is 0 when v is constant, and stays out of val NLL;
`penalized_step` scores it on the ELBO's guide sample; training logs it per epoch.
"""

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

import decipher_models3 as dc
from decipher_models3.presets import PRESETS
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.tools._decipher.data import make_data_loader_from_adata
from decipher_models3.tools._decipher.decipher import draw_other_batch
from decipher_models3.tools.decipher import penalized_step, predictive_log_likelihood
from decipher_models3.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")

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
]
BATCH_KEY = "batch"
N_GENES = 30


def _adata(n_cells: int = 128, n_batches: int = 3, seed: int = 0) -> ad.AnnData:
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


def _build(preset: str, seed: int = 0, adata: ad.AnnData = ADATA, **overrides) -> Decipher:
    """Untrained Decipher of `preset` (plus config overrides), built with Pyro seed `seed`."""
    config = DecipherConfig(**{**PRESETS[preset], **overrides})
    config.initialize_from_adata(adata, batch_key=BATCH_KEY)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(seed)
    return Decipher(config)


def _guide_trace(model: Decipher, seed: int = 0) -> poutine.Trace:
    pyro.util.set_rng_seed(seed)
    return poutine.trace(model.guide).get_trace(X, BATCH_IDX)


def _cycle(model: Decipher, guide_trace: poutine.Trace, seed: int = 1) -> torch.Tensor:
    """The cycle penalty on `guide_trace`, with the b' draw seeded by `seed`."""
    pyro.util.set_rng_seed(seed)
    return model.penalties(guide_trace, X, BATCH_IDX)["cycle_v"]


def _model_log_prob(model: Decipher) -> torch.Tensor:
    """Model trace log-prob under a replayed guide, seeded."""
    with torch.no_grad():
        guide_trace = _guide_trace(model)
        replayed = poutine.replay(model.model, trace=guide_trace)
        return poutine.trace(replayed).get_trace(X, BATCH_IDX).log_prob_sum()


def _assert_same_state(a: Decipher, b: Decipher) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


# --- flags off: unchanged models; model8 is model5's network ------------------------------------


@pytest.mark.parametrize("preset", EXISTING)
def test_flags_off_builds_the_same_state_dict_as_default_fields(preset):
    _assert_same_state(_build(preset), _build(preset, cycle_v=False, lambda_c=7.0))


@pytest.mark.parametrize("preset", EXISTING)
def test_flags_off_gives_the_same_model_log_prob_as_default_fields(preset):
    default = _build(preset).eval()
    explicit = _build(preset, cycle_v=False, lambda_c=7.0).eval()
    assert torch.equal(_model_log_prob(default), _model_log_prob(explicit))


@pytest.mark.parametrize("preset", EXISTING)
def test_flags_off_has_no_cycle_penalty(preset):
    model = _build(preset, cycle_v=False, lambda_c=7.0)
    values = model.penalties(_guide_trace(model), X, BATCH_IDX)
    assert "cycle_v" not in model.penalty_weights() and "cycle_v" not in values


def test_model8_builds_the_state_dict_of_model5():
    _assert_same_state(_build("model8"), _build("model5"))


def test_model8_gives_the_model_log_prob_of_model5():
    base = _model_log_prob(_build("model5").eval())
    assert torch.equal(_model_log_prob(_build("model8").eval()), base)


@pytest.mark.parametrize("lambda_c", [0.0, 1.0, 10.0])
def test_cycle_penalty_weight_is_lambda_c(lambda_c):
    assert _build("model8", lambda_c=lambda_c).penalty_weights() == {"cycle_v": lambda_c}


# --- config and data errors ---------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["native", "model1", "model2", "model3"])
def test_cycle_with_concat_z_raises(preset):
    with pytest.raises(ValueError, match="cycle_v=True needs batch_embedding_mode='concat_x'"):
        DecipherConfig(**{**PRESETS[preset], "cycle_v": True})


def test_cycle_with_decoder_only_raises():
    with pytest.raises(ValueError, match="batch_conditioning='decoder_only'"):
        DecipherConfig(**{**PRESETS["model4"], "cycle_v": True})


def test_cycle_with_model6_wiring_builds():
    assert DecipherConfig(**{**PRESETS["model6"], "cycle_v": True}).cycle_v


def test_negative_lambda_c_raises():
    with pytest.raises(ValueError, match="lambda_c=-0.1 must be >= 0"):
        DecipherConfig(**{**PRESETS["model8"], "lambda_c": -0.1})


@pytest.mark.parametrize("n_batches", [0, 1])
def test_cycle_model_with_fewer_than_two_batches_raises(n_batches):
    adata = _adata(n_batches=1)
    key = BATCH_KEY if n_batches else "no_such_column"
    config = DecipherConfig(**PRESETS["model8"])
    config.initialize_from_adata(adata, batch_key=key)
    with pytest.raises(ValueError, match=f"at least 2 batches.*n_batches={n_batches}"):
        Decipher(config)


def test_cycle_without_batch_codes_raises():
    model = _build("model8")
    with pytest.raises(ValueError, match="batch_idx=None"):
        model.penalties(_guide_trace(model), X, None)


# --- the swapped batch b' ------------------------------------------------------------------------

_A = torch.arange(4).repeat(6000)  # 6000 cells in each of 4 batches


def _draw(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return draw_other_batch(_A, 4)


def test_other_batch_is_never_the_cells_own():
    assert not (_draw() == _A).any()


def test_other_batch_is_uniform_over_the_other_batches():
    other = _draw()
    freq = np.zeros((4, 4))
    for a in range(4):
        freq[a] = np.bincount(other[_A == a].numpy(), minlength=4) / 6000
    expected = (np.ones((4, 4)) - np.eye(4)) / 3
    np.testing.assert_allclose(freq, expected, atol=0.02)


def test_other_batch_draw_is_seeded():
    assert torch.equal(_draw(seed=5), _draw(seed=5))


def test_other_batch_with_one_batch_raises():
    with pytest.raises(ValueError, match="n_batches >= 2"):
        draw_other_batch(torch.zeros(3, dtype=torch.long), 1)


# --- the cycle penalty ---------------------------------------------------------------------------


def _cycle_reference(model: Decipher, guide_trace: poutine.Trace, seed: int = 1) -> torch.Tensor:
    """The cycle written out with model5's networks (inputs [z; e_b], [log1p x; e_b],
    [z; log1p x]), with its graph: z and v_loc from the guide trace, b' drawn after reseeding
    with `seed`."""
    pyro.util.set_rng_seed(seed)
    other = draw_other_batch(BATCH_IDX, model.config.n_batches)
    z, v_loc = guide_trace.nodes["z"]["value"], guide_trace.nodes["_RETURN"]["value"][1]
    e = model.batch_emb(other)
    logits = model.decoder_z_to_x(torch.cat([z, e], dim=1))
    x_log = torch.log1p(X.sum(1, keepdim=True) * torch.softmax(logits, dim=1))
    z_loc, _ = model.encoder_x_to_z(torch.cat([x_log, e], dim=1))
    v_cycle, _ = model.encoder_zx_to_v(torch.cat([z_loc, x_log], dim=1))
    return ((v_loc - v_cycle) ** 2).sum(1).mean()


def _model8_with_a_nonzero_table() -> Decipher:
    model = _build("model8")
    with torch.no_grad():
        nn.init.normal_(model.batch_emb.weight)
    return model


def test_cycle_penalty_matches_a_step_by_step_reference():
    model = _model8_with_a_nonzero_table()
    guide_trace = _guide_trace(model)
    expected = _cycle_reference(model, guide_trace).item()
    assert _cycle(model, guide_trace).item() == pytest.approx(expected, rel=1e-5)


_NETWORKS = ["batch_emb", "decoder_z_to_x", "encoder_x_to_z", "encoder_zx_to_v"]


@pytest.fixture(scope="module")
def _penalty_and_reference_grads() -> tuple[dict, dict]:
    """Parameter gradients of the cycle penalty and of the reference, each on a twin model8
    (same weights, same guide seed, same b' seed), so every path of both sides is compared."""
    grads = []
    for penalty in (_cycle, _cycle_reference):
        model = _model8_with_a_nonzero_table()
        penalty(model, _guide_trace(model)).backward()
        grads.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
    return grads[0], grads[1]


@pytest.mark.parametrize("network", _NETWORKS)
def test_cycle_gradient_matches_the_reference_on_every_network(
    _penalty_and_reference_grads, network
):
    got, expected = _penalty_and_reference_grads
    names = sorted(k for k in expected if k.startswith(network + "."))
    assert names and sorted(k for k in got if k.startswith(network + ".")) == names
    for name in names:
        torch.testing.assert_close(got[name], expected[name], rtol=1e-5, atol=1e-7)


def test_cycle_penalty_value_is_unweighted():
    """penalties() returns P itself; lambda_c only weights it inside penalized_step."""
    values = []
    for lambda_c in (1.0, 10.0):
        model = _build("model8", lambda_c=lambda_c)
        with torch.no_grad():
            nn.init.normal_(model.batch_emb.weight, generator=torch.Generator().manual_seed(0))
        values.append(_cycle(model, _guide_trace(model)).item())
    assert values[0] == values[1]


def test_cycle_penalty_is_finite_and_positive():
    model = _build("model8")
    value = _cycle(model, _guide_trace(model)).item()
    assert np.isfinite(value) and value > 0


@pytest.mark.parametrize(
    "param", ["batch_emb", "decoder_z_to_x", "encoder_x_to_z", "encoder_zx_to_v"]
)
def test_cycle_gradient_reaches_every_network(param):
    model = _build("model8")
    _cycle(model, _guide_trace(model)).backward()
    module = getattr(model, param)
    grad = (module if param == "batch_emb" else module.layers[0]).weight.grad
    assert grad is not None and grad.abs().sum().item() > 0


def test_cycle_gradient_flows_into_the_guides_v_loc():
    model = _build("model8")
    guide_trace = _guide_trace(model)
    v_loc = guide_trace.nodes["_RETURN"]["value"][1]
    (grad,) = torch.autograd.grad(_cycle(model, guide_trace), v_loc)
    assert grad.abs().sum().item() > 0


def _batch_norms(model: Decipher) -> list:
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]


def test_cycle_restores_batch_norm_tracking():
    model = _build("model8")
    _cycle(model, _guide_trace(model))
    assert [m.track_running_stats for m in _batch_norms(model)] == [True] * 3


def test_guide_pass_after_the_cycle_updates_the_running_stats():
    model = _build("model8")
    _cycle(model, _guide_trace(model))
    before = model.encoder_x_to_z.batch_norms[0].running_mean.clone()
    _guide_trace(model, seed=5)
    assert not torch.equal(model.encoder_x_to_z.batch_norms[0].running_mean, before)


def test_cycle_penalty_leaves_batch_norm_running_stats_alone():
    model = _build("model8")
    guide_trace = _guide_trace(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _cycle(model, guide_trace)
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_cycle_penalty_does_not_depend_on_b_prime_when_nothing_sees_the_batch():
    model = _build("model8")  # concat_x: zero table at init, so decoder and encoder ignore b
    guide_trace = _guide_trace(model)
    assert torch.equal(_cycle(model, guide_trace, seed=1), _cycle(model, guide_trace, seed=2))


@pytest.mark.parametrize(
    "seeing, blind", [("decoder_z_to_x", "encoder_x_to_z"), ("encoder_x_to_z", "decoder_z_to_x")]
)
def test_cycle_penalty_depends_on_b_prime_through_each_network(seeing, blind):
    """Nonzero table; `blind`'s e_b input columns (the last D) are zeroed, so b' can change the
    penalty only through `seeing`: the decoder decodes, and the encoder re-encodes, under b'."""
    model = _build("model8")
    with torch.no_grad():
        nn.init.normal_(model.batch_emb.weight)
        getattr(model, blind).layers[0].weight[:, -model.config.dim_batch_embedding :] = 0.0
    guide_trace = _guide_trace(model)
    assert not torch.equal(_cycle(model, guide_trace, seed=1), _cycle(model, guide_trace, seed=2))


def test_cycle_penalty_is_zero_when_v_is_constant():
    model = _build("model8")
    with torch.no_grad():  # zero first layer: hidden 0 -> BatchNorm 0 -> v_loc = output bias
        model.encoder_zx_to_v.layers[0].weight.zero_()
        model.encoder_zx_to_v.layers[0].bias.zero_()
    assert _cycle(model, _guide_trace(model)).item() == 0.0


def test_cycle_with_the_residual_decoder_decodes_through_r():
    """cycle_v on model5r's wiring (recon_batch="residual"): b' reaches the decode through r, so
    r's zero-initialized output layer gets a gradient from the cycle alone."""
    model = _build("model5r", cycle_v=True)
    value = _cycle(model, _guide_trace(model))
    value.backward()
    grad = model.decoder_residual[-1].weight.grad
    assert np.isfinite(value.item()) and grad is not None and grad.abs().sum().item() > 0


# --- val NLL excludes the penalty ----------------------------------------------------------------


def _val_nll(preset: str, state_dict: dict, **overrides) -> float:
    model = _build(preset, **overrides)
    model.load_state_dict(state_dict)
    model.eval()
    pyro.util.set_rng_seed(0)
    loader = make_data_loader_from_adata(
        ADATA, 64, context_levels={BATCH_KEY: list(ADATA.obs[BATCH_KEY].cat.categories)}
    )
    return predictive_log_likelihood(model, loader, n_samples=2)


def test_val_nll_is_the_same_for_zero_and_positive_lambda_c():
    weights = _build("model8", seed=4).state_dict()
    nll = [_val_nll("model8", weights, lambda_c=lam) for lam in (0.0, 10.0)]
    assert nll[0] == nll[1] == _val_nll("model5", weights)


# --- penalized_step ------------------------------------------------------------------------------


def _optimizer() -> pyro.optim.PyroOptim:
    return pyro.optim.ClippedAdam({"lr": 5e-2, "weight_decay": 1e-4})


def test_penalized_step_scores_the_cycle_on_the_elbo_guide_sample():
    """With one seed, the step's cycle penalty equals that of the guide sample and the b' the
    seed draws, in that order (the replayed model draws nothing in between)."""
    reference = _build("model8")
    pyro.util.set_rng_seed(7)
    with torch.no_grad():
        guide_trace = poutine.trace(reference.guide).get_trace(X, BATCH_IDX)
        expected = reference.penalties(guide_trace, X, BATCH_IDX)["cycle_v"].item()
    model = _build("model8")
    pyro.util.set_rng_seed(7)
    _, values = penalized_step(model, Trace_ELBO(), _optimizer(), X, BATCH_IDX)
    assert values["cycle_v"] == pytest.approx(expected, rel=1e-5)


def test_positive_lambda_c_changes_the_weights_of_the_step():
    states = []
    for lam in (0.0, 10.0):
        model = _build("model8", lambda_c=lam)
        pyro.util.set_rng_seed(3)
        penalized_step(model, Trace_ELBO(), _optimizer(), X, BATCH_IDX)
        states.append(model.state_dict()["decoder_z_to_x.layers.0.weight"].clone())
    assert not torch.equal(states[0], states[1])


# --- decipher_train: penalty_history_ ------------------------------------------------------------

_EPOCHS = 3


@pytest.fixture(scope="module")
def _trained_model8(tmp_path_factory):
    DECIPHER_GLOBALS["save_folder"] = str(tmp_path_factory.mktemp("models")) + "/"
    config = DecipherConfig(**PRESETS["model8"], seed=1, n_epochs=_EPOCHS)
    config.early_stopping_patience = None
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, _ = dc.tl.decipher_train(
            _adata(n_cells=256),
            config,
            batch_key=BATCH_KEY,
            plot_every_k_epochs=-2,
            plot_kwargs={"color": BATCH_KEY},
        )
    finally:
        torch.set_num_threads(threads)
    return model


def test_training_logs_one_finite_cycle_mean_per_epoch(_trained_model8):
    history = _trained_model8.penalty_history_
    assert list(history) == ["cycle_v"]
    assert len(history["cycle_v"]) == _EPOCHS and np.isfinite(history["cycle_v"]).all()
