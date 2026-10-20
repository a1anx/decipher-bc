"""`decipher_models3` controls on the residual result (batch-invariance follow-up, A1 and B1).

New config fields: `recon_batch="concat_plus_residual_z"` (logits = h([z; e_b]) + r_z(z), r_z =
`decoder_residual_z`, dim_z -> 64 -> G, zero output layer), `batch_emb_init` ("auto" | "zero" |
"normal", applied after construction) and `residual_hidden` (width of r / r_z / ρ). New presets:
model5rz, model5r0, model7iw, model7im, model5r_mmd, model8r.

Checked here: every existing preset keeps its pinned state_dict with the new fields set to their
defaults (and with overrides that restate its own rule); each new preset's widths, parameter
counts (at G = 2000, oetjen5's HVG count) and seed draws; model5rz's likelihood is NB with mean
softmax(h([z; e_b]) + r_z(z)) for every batch; the config errors; model5r_mmd and model8r take a
penalized step, and model8r's cycle penalty alone sends a gradient into every guide network.
"""

import hashlib
import importlib

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO
from torch.nn.functional import softmax

from decipher_models3.presets import LEGACY_PACKAGES, PRESETS
from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.tools.decipher import penalized_step

# Pins and data of the model7 tests (fingerprints at base 0fa879a and of model7/model7i).
_m7 = importlib.import_module("1017_test_decipher_models3_model7")
ADATA, BATCH_KEY = _m7.ADATA, _m7.BATCH_KEY
X = torch.tensor(ADATA.X)
BATCH_IDX = torch.tensor(ADATA.obs[BATCH_KEY].cat.codes.values).long()
N_GENES, DIM_Z, D_E, N_BATCHES = 50, 10, 8, 3
PINS = {
    **{name: pin[0] for name, pin in _m7.BASE_FINGERPRINTS.items()},
    **{name: pin[0] for name, pin in _m7.ANCHORED_FINGERPRINTS.items()},
}
NEW_PRESETS = ["model5rz", "model5r0", "model7iw", "model7im", "model5r_mmd", "model8r"]
EPS = 1e-5  # Decipher's epsilon in the NB parameters
OETJEN5_GENES = 2000  # hvg_2000 of oetjen5_shared.h5ad, the dataset these presets run on


def _adata(n_genes: int) -> ad.AnnData:
    """Six cells of ones over `n_genes` genes in three batches (shapes only)."""
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical(["b0", "b1", "b2"] * 2)}, index=[f"c{i}" for i in range(6)]
    )
    return ad.AnnData(X=np.ones((6, n_genes), dtype=np.float32), obs=obs)


def _build(
    preset: str, seed: int = 0, adata: ad.AnnData = ADATA, batch_key: str = BATCH_KEY, **overrides
) -> Decipher:
    """Untrained eval-mode Decipher of `preset` (plus config overrides), Pyro seed `seed`."""
    config = DecipherConfig(**{**PRESETS[preset], **overrides})
    config.initialize_from_adata(adata, batch_key=batch_key)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(seed)
    return Decipher(config).eval()


def _sha256(model: Decipher) -> str:
    """sha256 of the state_dict, keys and bytes in order (as the model7 tests' pins)."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def _n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _shapes(model: Decipher) -> dict:
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _randomize(param: torch.Tensor, seed: int, scale: float = 0.3) -> None:
    with torch.no_grad():
        param.copy_(scale * torch.randn(param.shape, generator=torch.Generator().manual_seed(seed)))


# --- existing presets: the new fields at their defaults change nothing --------------------------


@pytest.mark.parametrize("preset", sorted(PINS))
def test_existing_preset_with_new_fields_explicitly_default_keeps_its_fingerprint(preset):
    model = _build(preset, batch_emb_init="auto", residual_hidden=None)
    assert _sha256(model) == PINS[preset]


@pytest.mark.parametrize(
    "preset, overrides",
    [
        ("model5", {"batch_emb_init": "zero"}),
        ("model2", {"batch_emb_init": "normal"}),
        ("model5r", {"batch_emb_init": "normal", "residual_hidden": 64}),
        ("model7i", {"batch_emb_init": "normal", "residual_hidden": 32}),
    ],
)
def test_overrides_that_restate_a_presets_own_rule_keep_its_fingerprint(preset, overrides):
    assert _sha256(_build(preset, **overrides)) == PINS[preset]


def test_new_fields_default_to_the_auto_init_and_the_module_widths():
    config = DecipherConfig()
    assert (config.batch_emb_init, config.residual_hidden) == ("auto", None)


def test_new_presets_are_their_parent_plus_the_planned_flags():
    expected = {
        "model5rz": {
            **PRESETS["model5"],
            "recon_batch": "concat_plus_residual_z",
            "batch_emb_init": "normal",
        },
        "model5r0": {**PRESETS["model5r"], "batch_emb_init": "zero"},
        "model7iw": {**PRESETS["model7i"], "residual_hidden": 4500},
        "model7im": {**PRESETS["model7i"], "residual_hidden": 512},
        "model5r_mmd": {**PRESETS["model5r"], "v_invariance": "mmd", "lambda_inv": 1.0},
        "model8r": {**PRESETS["model5r"], "cycle_v": True, "lambda_c": 1.0},
    }
    assert {name: PRESETS[name] for name in NEW_PRESETS} == expected


def test_new_presets_have_no_legacy_package():
    assert {name: LEGACY_PACKAGES[name] for name in NEW_PRESETS} == dict.fromkeys(NEW_PRESETS)


# --- widths and parameter counts -----------------------------------------------------------------

# preset -> (parent, state_dict entries added or reshaped relative to the parent at G = N_GENES,
# entries removed). Every other entry has the parent's shape.
SHAPE_DIFF = {
    "model5rz": (
        "model5",
        {
            "decoder_residual_z.0.weight": (64, DIM_Z),
            "decoder_residual_z.0.bias": (64,),
            "decoder_residual_z.2.weight": (N_GENES, 64),
            "decoder_residual_z.2.bias": (N_GENES,),
        },
    ),
    "model5r0": ("model5r", {}),
    "model7iw": (
        "model7i",
        {
            "decoder_residual_in_span.0.weight": (4500, DIM_Z + D_E),
            "decoder_residual_in_span.0.bias": (4500,),
            "decoder_residual_in_span.2.weight": (DIM_Z, 4500),
        },
    ),
    "model7im": (
        "model7i",
        {
            "decoder_residual_in_span.0.weight": (512, DIM_Z + D_E),
            "decoder_residual_in_span.0.bias": (512,),
            "decoder_residual_in_span.2.weight": (DIM_Z, 512),
        },
    ),
    "model5r_mmd": ("model5r", {}),
    "model8r": ("model5r", {}),
}


@pytest.mark.parametrize("preset", NEW_PRESETS)
def test_new_preset_has_its_parents_shapes_plus_the_planned_entries(preset):
    parent, diff = SHAPE_DIFF[preset]
    assert _shapes(_build(preset)) == {**_shapes(_build(parent)), **diff}


def test_model5rz_keeps_model5s_decoder_with_its_e_b_columns():
    assert _shapes(_build("model5rz"))["decoder_z_to_x.layers.0.weight"] == (N_GENES, DIM_Z + D_E)


# Parameter count of r (model5r) or r_z (model5rz) or ρ (model7i, -iw, -im) at G = 2000.
R_PARAMS_AT_OETJEN5 = {
    "model5r": (DIM_Z + D_E) * 64 + 64 + 64 * 2000 + 2000,  # 131216
    "model5rz": DIM_Z * 64 + 64 + 64 * 2000 + 2000,  # 130704
    "model7i": 29 * 32 + 10,  # 938
    "model7iw": 29 * 4500 + 10,  # 130510
    "model7im": 29 * 512 + 10,  # 14858
}


def _r_at_oetjen5(preset: str) -> torch.nn.Module:
    model = _build(preset, adata=_adata(OETJEN5_GENES))
    return model.decoder_residual or model.decoder_residual_z or model.decoder_residual_in_span


@pytest.mark.parametrize("preset", sorted(R_PARAMS_AT_OETJEN5))
def test_decoder_batch_network_has_the_planned_parameter_count_at_oetjen5(preset):
    assert _n_params(_r_at_oetjen5(preset)) == R_PARAMS_AT_OETJEN5[preset]


def test_model7iw_rho_is_within_5_percent_of_model5r_r_at_oetjen5():
    wide, r = (_n_params(_r_at_oetjen5(name)) for name in ("model7iw", "model5r"))
    assert abs(wide - r) / r < 0.05


@pytest.mark.parametrize("preset", ["model7iw", "model7im"])
def test_widened_rho_leaves_t_at_width_32(preset):
    assert _shapes(_build(preset))["prior_residual.0.weight"] == (32, 2 + D_E)


def test_residual_hidden_sets_r_width_on_model5r():
    assert _shapes(_build("model5r", residual_hidden=16))["decoder_residual.0.weight"] == (
        16,
        DIM_Z + D_E,
    )


# --- seed draws and the table init ---------------------------------------------------------------


def _all_but_table(model: Decipher) -> dict:
    return {k: v for k, v in model.state_dict().items() if k != "batch_emb.weight"}


def _assert_state_equal(got: dict, want: dict) -> None:
    assert got.keys() == want.keys()
    for key in want:
        assert torch.equal(got[key], want[key]), key


@pytest.mark.parametrize("seed", [0, 5])
def test_model5r0_table_is_zero(seed):
    table = _build("model5r0", seed=seed).batch_emb.weight
    assert torch.equal(table, torch.zeros(N_BATCHES, D_E))


@pytest.mark.parametrize("seed", [0, 5])
def test_model5r0_other_weights_are_bit_equal_to_model5r(seed):
    _assert_state_equal(
        _all_but_table(_build("model5r0", seed=seed)), _all_but_table(_build("model5r", seed=seed))
    )


@pytest.mark.parametrize("seed", [0, 5])
def test_normal_init_on_model5_keeps_every_other_weight(seed):
    _assert_state_equal(
        _all_but_table(_build("model5", seed=seed, batch_emb_init="normal")),
        _all_but_table(_build("model5", seed=seed)),
    )


def test_normal_init_on_model5_gives_a_nonzero_table():
    assert (_build("model5", batch_emb_init="normal").batch_emb.weight != 0).all()


def test_model5rz_table_is_the_normal_draw_not_zero():
    assert (_build("model5rz").batch_emb.weight != 0).all()


def test_concat_plus_residual_z_with_auto_init_has_a_zero_table():
    model = _build("model5rz", batch_emb_init="auto")
    assert torch.equal(model.batch_emb.weight, torch.zeros(N_BATCHES, D_E))


@pytest.mark.parametrize("seed", [0, 5])
def test_model5rz_keeps_model5s_seed_draws_for_every_network(seed):
    rz = _build("model5rz", seed=seed).state_dict()
    rz = {k: v for k, v in rz.items() if not k.startswith(("decoder_residual_z.", "batch_emb."))}
    _assert_state_equal(rz, _all_but_table(_build("model5", seed=seed)))


def test_model5rz_r_z_output_layer_starts_at_zero():
    r_z = _build("model5rz").decoder_residual_z
    assert torch.equal(r_z[-1].weight, torch.zeros(N_GENES, 64))
    assert torch.equal(r_z[-1].bias, torch.zeros(N_GENES))


# --- model5rz: the likelihood is NB(softmax(h([z; e_b]) + r_z(z))) for every batch -------------


def _x_site_logits(model: Decipher, code: int):
    """(z, NB logits of the x site) for every cell decoded under batch `code`, one guide sample."""
    pyro.util.set_rng_seed(0)
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
        replayed = poutine.replay(model.model, trace=guide_trace)
        nodes = poutine.trace(replayed).get_trace(X, torch.full_like(BATCH_IDX, code)).nodes
    return guide_trace.nodes["z"]["value"], nodes["x"]["fn"].base_dist.logits


def _perturbed_model5rz() -> Decipher:
    """model5rz with r_z's zero output layer replaced by small random weights."""
    model = _build("model5rz")
    _randomize(model.decoder_residual_z[-1].weight, seed=11)
    _randomize(model.decoder_residual_z[-1].bias, seed=12)
    return model


@pytest.mark.parametrize("code", range(N_BATCHES))
def test_model5rz_likelihood_mean_is_softmax_of_h_plus_batch_blind_r_z(code):
    model = _perturbed_model5rz()
    z, logits = _x_site_logits(model, code)
    with torch.no_grad():
        e = model.batch_emb.weight[code].expand(len(z), -1)
        mean = softmax(
            model.decoder_z_to_x(torch.cat([z, e], -1)) + model.decoder_residual_z(z), -1
        )
        expected = torch.log(X.sum(-1, keepdim=True) * mean + EPS) - torch.log(model.theta + EPS)
    torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-5)


def test_model5rz_r_z_takes_z_alone():
    r_z = _perturbed_model5rz().decoder_residual_z
    with pytest.raises(RuntimeError):
        r_z(torch.zeros(4, DIM_Z + D_E))


# --- config errors ---------------------------------------------------------------------------------

NO_H = ["native", "model1", "model2", "model3", "model4", "model5", "model6", "model8"]


@pytest.mark.parametrize("preset", NO_H)
def test_residual_hidden_on_an_arm_without_r_or_rho_raises(preset):
    with pytest.raises(ValueError, match="residual_hidden=16 needs a decoder batch network"):
        DecipherConfig(**PRESETS[preset], residual_hidden=16)


@pytest.mark.parametrize("width", [0, -3])
def test_nonpositive_residual_hidden_raises(width):
    with pytest.raises(ValueError, match=f"residual_hidden={width} must be >= 1"):
        DecipherConfig(**PRESETS["model5r"], residual_hidden=width)


@pytest.mark.parametrize("init", ["zero", "normal"])
def test_forced_table_init_on_native_raises(init):
    with pytest.raises(ValueError, match=f"batch_emb_init={init!r} needs a batch arm"):
        DecipherConfig(**PRESETS["native"], batch_emb_init=init)


def test_unknown_batch_emb_init_raises():
    with pytest.raises(ValueError, match="batch_emb_init='uniform' is not one of"):
        DecipherConfig(**PRESETS["model5"], batch_emb_init="uniform")


@pytest.mark.parametrize("preset", ["native", "model2"])
def test_concat_plus_residual_z_without_a_concat_x_batch_arm_raises(preset):
    with pytest.raises(ValueError, match="recon_batch='concat_plus_residual_z' needs a batch"):
        DecipherConfig(**PRESETS[preset], recon_batch="concat_plus_residual_z")


def test_concat_plus_residual_z_with_zero_batches_builds_no_table_or_r_z():
    model = _build("model5rz", batch_key="absent")
    assert (model.batch_emb, model.decoder_residual_z) == (None, None)


# --- the penalty presets on model5r --------------------------------------------------------------


@pytest.mark.parametrize(
    "preset, weights", [("model5r_mmd", {"mmd_v": 1.0}), ("model8r", {"cycle_v": 1.0})]
)
def test_penalty_preset_on_model5r_has_its_weight_and_model5rs_networks(preset, weights):
    model = _build(preset)
    assert (model.penalty_weights(), _sha256(model)) == (weights, PINS["model5r"])


@pytest.mark.parametrize("preset", ["model5r_mmd", "model8r"])
def test_penalty_preset_on_model5r_takes_a_finite_step_that_moves_r(preset):
    model = _build(preset).train()
    before = model.decoder_residual[-1].weight.detach().clone()
    optimizer = pyro.optim.ClippedAdam({"lr": 1e-3})
    loss, values = penalized_step(model, Trace_ELBO(), optimizer, X, BATCH_IDX)
    assert np.isfinite([loss, *values.values()]).all()
    assert not torch.equal(model.decoder_residual[-1].weight, before)


GUIDE_NETS = ["encoder_x_to_z", "encoder_zx_to_v"]


@pytest.mark.parametrize("net", GUIDE_NETS)
def test_model8r_cycle_penalty_alone_has_a_nonzero_gradient_on_every_guide_weight(net):
    model = _build("model8r").train()
    _randomize(model.decoder_residual[-1].weight, seed=21)
    pyro.util.set_rng_seed(3)
    guide_trace = poutine.trace(model.guide).get_trace(X, BATCH_IDX)
    model.zero_grad()
    model.penalties(guide_trace, X, BATCH_IDX)["cycle_v"].backward()
    # Weights only: a bias before BatchNorm cancels, and v_loc's output bias cancels in v - v'.
    grads = {
        name: p.grad.abs().sum().item()
        for name, p in getattr(model, net).named_parameters()
        if name.startswith("layers.") and name.endswith(".weight")
    }
    assert grads.keys() == {"layers.0.weight", "layers.1.weight"}
    assert min(grads.values()) > 0
