"""`decipher_models3` training penalties, unit 1a of the batch-invariance plan: the MMD invariance
penalty on v (`v_invariance="mmd"`, weight `lambda_inv`), the penalty interface on `Decipher`
(`penalty_weights` / `penalties`), `penalized_step` and `decipher.penalty_history_`.

Checks: flags off builds the same model as before; the `*_mmd` presets build their base arm; the
MMD matches a loop-by-loop numpy reference, is 0 for one batch, > 0 for shifted batches,
invariant to relabelling and has gradients reaching both encoders; the penalty is not in val NLL;
`penalized_step` with lambda_inv = 0 is `svi.step`, and uses the ELBO's own guide sample; the
training loop fills `penalty_history_` per epoch.
"""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO

import decipher_models3 as dc
import decipher_models3.tools.decipher as dtool
from decipher_models3.presets import PRESETS
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.tools._decipher.data import make_data_loader_from_adata
from decipher_models3.tools._decipher.decipher import mmd_v_penalty
from decipher_models3.tools.decipher import penalized_step, predictive_log_likelihood
from decipher_models3.utils import DECIPHER_GLOBALS

matplotlib.use("Agg")

SEVEN = ["native", "model1", "model2", "model3", "model4", "model5", "model6"]
# Every preset without a penalty: the seven plus 1b's residual-decoder arms.
NO_PENALTY = [*SEVEN, "model4r", "model5r"]
MMD_PRESETS = {"native_mmd": "native", "model2_mmd": "model2", "model5_mmd": "model5"}
BATCH_KEY = "batch"
N_GENES = 30
# The plan's bandwidths, as multiples of the median pairwise distance. Written here as a literal
# (not imported) so a changed constant in the code fails the reference tests.
SCALES = (0.1, 0.5, 1.0, 2.0, 5.0)


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


def _build(preset: str, seed: int = 0, **overrides) -> Decipher:
    """Untrained Decipher of `preset` (plus config overrides), built with Pyro seed `seed`."""
    config = DecipherConfig(**{**PRESETS[preset], **overrides})
    config.initialize_from_adata(ADATA, batch_key=BATCH_KEY)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(seed)
    return Decipher(config)


def _model_log_prob(model: Decipher) -> torch.Tensor:
    """Model trace log-prob under a replayed guide, seeded."""
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = poutine.replay(model.model, trace=guide_trace)
        return poutine.trace(replayed).get_trace(X, BATCH_IDX).log_prob_sum()


def _assert_same_state(a: Decipher, b: Decipher) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


def _mmd_reference(v: np.ndarray, labels: np.ndarray) -> float:
    """The penalty written out pair by pair: sum over batches with >= 2 cells of the U-statistic
    MMD^2(batch b, all), Gaussian kernel summed over the scales x the median pairwise distance."""
    n = len(v)
    dist = np.array([[np.linalg.norm(v[i] - v[j]) for j in range(n)] for i in range(n)])
    upper = dist[np.triu_indices(n, 1)]
    assert len(upper) % 2 == 1, "use an odd pair count: torch takes the lower median"
    median = np.median(upper)
    k = sum(np.exp(-(dist**2) / (2 * (s * median) ** 2)) for s in SCALES)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    within_all = np.mean([k[i, j] for i, j in pairs])
    total = 0.0
    for b in np.unique(labels):
        members = np.flatnonzero(labels == b)
        if len(members) < 2:
            continue
        within_b = np.mean([k[i, j] for i, j in pairs if i in members and j in members])
        cross = np.mean([k[i, j] for i, j in pairs if i in members])
        total += within_b + within_all - 2 * cross
    return total


# --- flags off: unchanged models ----------------------------------------------------------------


@pytest.mark.parametrize("preset", NO_PENALTY)
def test_flags_off_builds_the_same_state_dict_as_default_fields(preset):
    _assert_same_state(_build(preset), _build(preset, v_invariance="none", lambda_inv=7.0))


@pytest.mark.parametrize("preset", NO_PENALTY)
def test_flags_off_gives_the_same_model_log_prob_as_default_fields(preset):
    default = _build(preset).eval()
    explicit = _build(preset, v_invariance="none", lambda_inv=7.0).eval()
    assert torch.equal(_model_log_prob(default), _model_log_prob(explicit))


@pytest.mark.parametrize("preset", NO_PENALTY)
def test_flags_off_has_no_penalties(preset):
    model = _build(preset)
    guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
    assert (model.penalty_weights(), model.penalties(guide_trace, X, BATCH_IDX)) == ({}, {})


@pytest.mark.parametrize("preset", sorted(MMD_PRESETS))
def test_mmd_preset_builds_the_state_dict_of_its_base_arm(preset):
    _assert_same_state(_build(preset), _build(MMD_PRESETS[preset]))


@pytest.mark.parametrize("preset", sorted(MMD_PRESETS))
def test_mmd_preset_gives_the_model_log_prob_of_its_base_arm(preset):
    base = _model_log_prob(_build(MMD_PRESETS[preset]).eval())
    assert torch.equal(_model_log_prob(_build(preset).eval()), base)


@pytest.mark.parametrize("lambda_inv", [0.0, 0.1, 10.0])
def test_mmd_penalty_weight_is_lambda_inv(lambda_inv):
    assert _build("model5_mmd", lambda_inv=lambda_inv).penalty_weights() == {"mmd_v": lambda_inv}


def test_penalty_fields_default_to_off_with_unit_weight():
    config = DecipherConfig()
    assert (config.v_invariance, config.lambda_inv) == ("none", 1.0)


@pytest.mark.parametrize("preset", sorted(MMD_PRESETS))
def test_mmd_preset_is_its_base_arm_plus_mmd_at_unit_weight(preset):
    expected = {**PRESETS[MMD_PRESETS[preset]], "v_invariance": "mmd", "lambda_inv": 1.0}
    assert PRESETS[preset] == expected


# --- the MMD penalty ----------------------------------------------------------------------------

_RNG = np.random.default_rng(3)
_V = _RNG.normal(size=(11, 2)).astype(np.float32)
# batch 2 has exactly 2 cells (kept), batches 3 and 4 have one cell each (skipped)
_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4])


def test_mmd_matches_the_pairwise_numpy_reference():
    got = mmd_v_penalty(torch.tensor(_V), torch.tensor(_LABELS)).item()
    assert got == pytest.approx(_mmd_reference(_V.astype(np.float64), _LABELS), rel=1e-5)


def _mmd_with_constant_median(v: torch.Tensor, labels: torch.Tensor, median: float):
    """The same U-statistic as the penalty, with the bandwidth median passed in as a constant."""
    n = len(v)
    sq_dist = (v[:, None, :] - v[None, :, :]).pow(2).sum(-1)
    k = sum(torch.exp(-sq_dist / (2 * (s * median) ** 2)) for s in SCALES)
    k = k.masked_fill(torch.eye(n, dtype=torch.bool), 0.0)
    total = 0.0
    for b in labels.unique():
        in_b, n_b = labels == b, int((labels == b).sum())
        if n_b >= 2:
            total = total + k[in_b][:, in_b].sum() / (n_b * (n_b - 1)) + k.sum() / (n * (n - 1))
            total = total - 2 * k[in_b].sum() / (n_b * (n - 1))
    return total


def test_mmd_gradient_treats_the_median_bandwidth_as_a_constant():
    v = torch.tensor(_V, requires_grad=True)
    labels = torch.tensor(_LABELS)
    upper = torch.pdist(v.detach())  # 55 pairs: odd, so the median is unique
    median = float(upper.median())
    (got,) = torch.autograd.grad(mmd_v_penalty(v, labels), v)
    (want,) = torch.autograd.grad(_mmd_with_constant_median(v, labels, median), v)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)


def test_mmd_is_zero_for_one_batch():
    assert mmd_v_penalty(torch.tensor(_V), torch.zeros(len(_V), dtype=torch.long)).item() == 0.0


def test_mmd_is_positive_for_two_batches_with_shifted_v():
    labels = torch.tensor([0] * 6 + [1] * 5)
    v = torch.tensor(_V) + 3.0 * labels[:, None]
    assert mmd_v_penalty(v, labels).item() > 0.5


def test_mmd_is_invariant_to_batch_relabelling():
    relabelled = np.array([2, 4, 0, 1, 3])[_LABELS]
    v = torch.tensor(_V)
    assert mmd_v_penalty(v, torch.tensor(relabelled)).item() == pytest.approx(
        mmd_v_penalty(v, torch.tensor(_LABELS)).item(), rel=1e-6
    )


def test_mmd_is_zero_when_every_batch_has_one_cell():
    assert mmd_v_penalty(torch.tensor(_V[:3]), torch.tensor([0, 1, 2])).item() == 0.0


def test_mmd_without_batch_codes_raises():
    with pytest.raises(ValueError, match="batch_idx=None"):
        mmd_v_penalty(torch.tensor(_V), None)


def test_mmd_model_without_batches_raises():
    config = DecipherConfig(**PRESETS["native_mmd"])
    config.initialize_from_adata(ADATA, batch_key="no_such_column")
    with pytest.raises(ValueError, match="n_batches=0"):
        Decipher(config)


def test_mmd_model_with_one_batch_builds_and_has_zero_penalty():
    config = DecipherConfig(**PRESETS["model5_mmd"])
    config.initialize_from_adata(_adata(n_batches=1), batch_key=BATCH_KEY)
    model = Decipher(config)
    guide_trace = poutine.trace(model.guide).get_trace(X, torch.zeros(len(X), dtype=torch.long))
    assert model.penalties(guide_trace, X, torch.zeros(len(X), dtype=torch.long)) == {"mmd_v": 0.0}


def test_mmd_is_zero_not_nan_when_all_v_coincide():
    assert mmd_v_penalty(torch.zeros(6, 2), torch.tensor([0, 0, 0, 1, 1, 1])).item() == 0.0


@pytest.mark.parametrize("encoder", ["encoder_zx_to_v", "encoder_x_to_z"])
def test_mmd_penalty_gradient_reaches_both_encoders(encoder):
    model = _build("model5_mmd")
    guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
    model.penalties(guide_trace, X, BATCH_IDX)["mmd_v"].backward()
    grad = getattr(model, encoder).layers[0].weight.grad
    assert grad is not None and grad.abs().sum().item() > 0


# --- val NLL excludes penalties -----------------------------------------------------------------


def _val_nll(preset: str, state_dict: dict, **overrides) -> float:
    model = _build(preset, **overrides)
    model.load_state_dict(state_dict)
    model.eval()
    pyro.util.set_rng_seed(0)
    loader = make_data_loader_from_adata(
        ADATA, 64, context_levels={BATCH_KEY: list(ADATA.obs[BATCH_KEY].cat.categories)}
    )
    return predictive_log_likelihood(model, loader, n_samples=2)


def test_val_nll_is_the_same_for_zero_and_positive_lambda_inv():
    weights = _build("model5_mmd", seed=4).state_dict()
    nll = [_val_nll("model5_mmd", weights, lambda_inv=lam) for lam in (0.0, 10.0)]
    assert nll[0] == nll[1] == _val_nll("model5", weights)


# --- penalized_step -----------------------------------------------------------------------------

_K_STEPS = 4
_MINIBATCHES = [(X[i::_K_STEPS], BATCH_IDX[i::_K_STEPS]) for i in range(_K_STEPS)]


def _optimizer() -> pyro.optim.PyroOptim:
    return pyro.optim.ClippedAdam({"lr": 5e-2, "weight_decay": 1e-4})


def _after_k_svi_steps(preset: str) -> tuple[dict, torch.Tensor, list]:
    model = _build(preset)
    svi = SVI(model.model, model.guide, _optimizer(), Trace_ELBO())
    losses = [svi.step(*xc) for xc in _MINIBATCHES]
    return model.state_dict(), pyro.param("theta").detach().clone(), losses


def _after_k_penalized_steps(preset: str, **overrides) -> tuple[dict, torch.Tensor, list]:
    model = _build(preset, **overrides)
    elbo, optimizer = Trace_ELBO(), _optimizer()
    losses = [penalized_step(model, elbo, optimizer, *xc)[0] for xc in _MINIBATCHES]
    return model.state_dict(), pyro.param("theta").detach().clone(), losses


@pytest.fixture(scope="module", params=["native", "model2", "model5"])
def _svi_and_penalized_runs(request):
    base = request.param
    return _after_k_svi_steps(base), _after_k_penalized_steps(f"{base}_mmd", lambda_inv=0.0)


def test_penalized_step_with_zero_lambda_gives_the_svi_step_weights(_svi_and_penalized_runs):
    (svi_state, _, _), (pen_state, _, _) = _svi_and_penalized_runs
    assert svi_state.keys() == pen_state.keys()
    for key in svi_state:
        assert torch.equal(svi_state[key], pen_state[key]), key


def test_penalized_step_with_zero_lambda_gives_the_svi_step_theta(_svi_and_penalized_runs):
    (_, svi_theta, _), (_, pen_theta, _) = _svi_and_penalized_runs
    assert torch.equal(svi_theta, pen_theta)


def test_penalized_step_reports_the_svi_step_elbo_loss(_svi_and_penalized_runs):
    (_, _, svi_losses), (_, _, pen_losses) = _svi_and_penalized_runs
    assert pen_losses == pytest.approx(svi_losses, rel=1e-6)


def test_positive_lambda_changes_the_weights_of_the_step():
    zero = _after_k_penalized_steps("model5_mmd", lambda_inv=0.0)[0]
    ten = _after_k_penalized_steps("model5_mmd", lambda_inv=10.0)[0]
    key = "encoder_zx_to_v.layers.0.weight"
    assert not torch.equal(zero[key], ten[key])


def test_penalized_step_scores_the_elbo_and_penalty_on_one_guide_sample():
    """The step's ELBO and MMD equal those of the guide sample the same seed draws."""
    model = _build("model5_mmd", lambda_inv=10.0)  # != 1, so a weighted value would differ
    pyro.util.set_rng_seed(7)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = poutine.replay(model.model, trace=guide_trace)
        model_trace = poutine.trace(replayed).get_trace(X, BATCH_IDX)
        elbo_loss = (guide_trace.log_prob_sum() - model_trace.log_prob_sum()).item()
        mmd = mmd_v_penalty(guide_trace.nodes["v"]["value"], BATCH_IDX).item()
    pyro.util.set_rng_seed(7)
    loss, values = penalized_step(model, Trace_ELBO(), _optimizer(), X, BATCH_IDX)
    assert (loss, values["mmd_v"]) == (pytest.approx(elbo_loss, rel=1e-5), pytest.approx(mmd))


def test_penalized_step_reports_the_elbo_loss_without_the_penalty():
    """Same init and sample: the reported loss at lambda_inv = 1e3 (lambda * P is about 20, far
    above float32 resolution of the ELBO) equals the one at lambda_inv = 0."""
    losses = {}
    for lambda_inv in (0.0, 1e3):
        model = _build("model5_mmd", lambda_inv=lambda_inv)
        pyro.util.set_rng_seed(7)
        losses[lambda_inv] = penalized_step(model, Trace_ELBO(), _optimizer(), X, BATCH_IDX)[0]
    assert losses[1e3] == losses[0.0]


def test_negative_lambda_inv_raises():
    with pytest.raises(ValueError, match="lambda_inv"):
        DecipherConfig(**{**PRESETS["model5_mmd"], "lambda_inv": -1.0})


def test_penalized_step_raises_when_penalties_and_weights_differ(monkeypatch):
    monkeypatch.setattr(Decipher, "penalties", lambda self, guide_trace, x, batch_idx=None: {})
    with pytest.raises(ValueError, match="do not match"):
        penalized_step(_build("model5_mmd"), Trace_ELBO(), _optimizer(), X, BATCH_IDX)


_KEY = "encoder_zx_to_v.layers.0.weight"


def _sgd_update(lambda_inv: float) -> torch.Tensor:
    """Change of one weight after one plain-SGD (lr 1) penalized step, seeded sample."""
    model = _build("model5_mmd", lambda_inv=lambda_inv)
    before = model.state_dict()[_KEY].clone()
    pyro.util.set_rng_seed(7)
    penalized_step(model, Trace_ELBO(), pyro.optim.SGD({"lr": 1.0}), X, BATCH_IDX)
    return model.state_dict()[_KEY] - before


def test_penalized_step_adds_lambda_times_the_penalty_gradient():
    """With SGD, update(lambda) - update(0) = -lambda * dP/dw on the step's own sample.

    Compared by relative norm: the two updates also carry the much larger ELBO gradient, whose
    float32 rounding (~5e-5 absolute) does not cancel exactly. A wrong sign, weight or sample
    gives a relative error near 1 or above."""
    model = _build("model5_mmd")
    pyro.util.set_rng_seed(7)
    guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
    penalty = model.penalties(guide_trace, X, BATCH_IDX)["mmd_v"]
    (grad,) = torch.autograd.grad(penalty, model.encoder_zx_to_v.layers[0].weight)
    diff = _sgd_update(10.0) - _sgd_update(0.0)
    assert ((diff + 10.0 * grad).norm() / (10.0 * grad).norm()).item() < 1e-2


# --- decipher_train: penalty_history_ -----------------------------------------------------------

_EPOCHS = 3


def _train(preset: str, save_folder, **overrides) -> tuple[Decipher, list]:
    DECIPHER_GLOBALS["save_folder"] = str(save_folder) + "/"
    defaults = dict(seed=1, n_epochs=_EPOCHS, early_stopping_patience=None)
    config = DecipherConfig(**{**PRESETS[preset], **defaults, **overrides})
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, val_losses = dc.tl.decipher_train(
            _adata(n_cells=256),
            config,
            batch_key=BATCH_KEY,
            plot_every_k_epochs=-2,
            plot_kwargs={"color": BATCH_KEY},
        )
    finally:
        torch.set_num_threads(threads)
    return model, val_losses


@pytest.fixture(scope="module")
def _trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    return {
        "model5": _train("model5", folder)[0],
        "model5_mmd": _train("model5_mmd", folder)[0],
        "model5_mmd_l0": _train("model5_mmd", folder, lambda_inv=0.0)[0],
    }


def test_penalty_history_has_one_finite_mean_per_epoch(_trained):
    history = _trained["model5_mmd"].penalty_history_
    assert list(history) == ["mmd_v"]
    assert len(history["mmd_v"]) == _EPOCHS and np.isfinite(history["mmd_v"]).all()


def test_penalty_history_is_empty_without_a_penalty(_trained):
    assert _trained["model5"].penalty_history_ == {}


def test_training_with_zero_lambda_gives_the_unpenalized_weights(_trained):
    _assert_same_state(_trained["model5_mmd_l0"], _trained["model5"])


def test_an_error_inside_a_penalty_propagates_out_of_training(monkeypatch, tmp_path):
    def broken_penalties(self, guide_trace, x, batch_idx=None):
        raise RuntimeError("broken penalty")

    monkeypatch.setattr(Decipher, "penalties", broken_penalties)
    with pytest.raises(RuntimeError, match="broken penalty"):
        _train("model5_mmd", tmp_path)


_STEPS_PER_EPOCH = 3  # 231 train cells of _adata(256), batch size 64, drop_last
_OBS_PER_EPOCH = _STEPS_PER_EPOCH * 64


@pytest.fixture(scope="module")
def _recorded_lambda10(tmp_path_factory):
    """model5_mmd trained at lambda_inv = 10 (!= 1, so weighted and unweighted values differ),
    with every penalized_step's reported (ELBO loss, unweighted mmd_v) recorded, (epochs, steps)."""
    recorded = []

    def recording_step(*args):
        loss, values = penalized_step(*args)
        recorded.append((loss, values["mmd_v"]))
        return loss, values

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dtool, "penalized_step", recording_step)
        model = _train("model5_mmd", tmp_path_factory.mktemp("rec"), lambda_inv=10.0)[0]
    steps = np.array(recorded).reshape(_EPOCHS, _STEPS_PER_EPOCH, 2)
    return model, steps[..., 0], steps[..., 1]


def test_penalty_history_is_the_per_epoch_mean_of_the_unweighted_step_values(_recorded_lambda10):
    model, _, mmd = _recorded_lambda10
    assert model.penalty_history_["mmd_v"] == pytest.approx(mmd.mean(axis=1).tolist(), rel=1e-12)


def test_train_elbo_of_a_penalized_run_excludes_the_penalty(_recorded_lambda10):
    model, elbo, _ = _recorded_lambda10
    expected = (elbo.sum(axis=1) / _OBS_PER_EPOCH).tolist()
    assert model.train_losses_ == pytest.approx(expected, rel=1e-12)


def test_early_stopped_run_has_one_penalty_mean_per_epoch_run(tmp_path):
    model = _train(
        "model5_mmd", tmp_path, n_epochs=40, early_stopping_patience=1, learning_rate=5e-2
    )[0]
    assert len(model.train_losses_) < 40, "the run must stop early for this test to mean anything"
    assert len(model.penalty_history_["mmd_v"]) == len(model.train_losses_)


def test_training_without_a_penalty_never_calls_penalized_step(monkeypatch, tmp_path):
    def unexpected(*args):
        raise AssertionError("penalized_step called for an arm without a penalty")

    monkeypatch.setattr(dtool, "penalized_step", unexpected)
    model, val_losses = _train("model5", tmp_path)
    assert len(val_losses) == _EPOCHS


def test_an_error_in_an_unpenalized_step_keeps_the_upstream_print_and_return(monkeypatch, tmp_path):
    def broken_step(self, *args):
        raise RuntimeError("broken svi step")

    monkeypatch.setattr(dtool.SVI, "step", broken_step)
    model, val_losses = dc.tl.decipher_train(
        _adata(n_cells=256),
        DecipherConfig(**PRESETS["model5"], n_epochs=_EPOCHS),
        batch_key=BATCH_KEY,
        plot_every_k_epochs=-2,
    )
    assert (val_losses, model.best_epoch_) == ([], None)
