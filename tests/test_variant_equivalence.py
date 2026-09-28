import dataclasses
import importlib

import pyro
import pyro.poutine as poutine
import pytest
import torch

from decipher_models.presets import LEGACY_PACKAGES as OLD_PACKAGES
from decipher_models.presets import PRESETS
from decipher_models.tools._decipher.decipher import Decipher, DecipherConfig

DIM_GENES = 20
DIM_Z = 6
DIM_V = 2
N_BATCHES = 3
DIM_BATCH_EMBEDDING = 4
N_CELLS = 16


# model6 renamed `encoder_zx_to_v` -> `encoder_x_to_v` when it went mean-field.
# The unified package keeps the original name for every variant (the network's shape already
# says whether it consumes z), so its state_dict keys need remapping before they will load.
STATE_DICT_RENAMES = {
    "model6": ("encoder_x_to_v.", "encoder_zx_to_v."),
}


def make_config(preset_kwargs):
    config = DecipherConfig(
        dim_z=DIM_Z,
        dim_v=DIM_V,
        n_batches=N_BATCHES,
        dim_batch_embedding=DIM_BATCH_EMBEDDING,
        **preset_kwargs,
    )
    config.dim_genes = DIM_GENES
    config.n_cells = N_CELLS
    config._initialized_from_adata = True
    return config


def make_old_config(OldDecipherConfig):
    field_names = {f.name for f in dataclasses.fields(OldDecipherConfig)}
    kwargs = dict(dim_z=DIM_Z, dim_v=DIM_V)
    if "n_batches" in field_names:
        kwargs["n_batches"] = N_BATCHES
    if "dim_batch_embedding" in field_names:
        kwargs["dim_batch_embedding"] = DIM_BATCH_EMBEDDING
    config = OldDecipherConfig(**kwargs)
    config.dim_genes = DIM_GENES
    config.n_cells = N_CELLS
    config._initialized_from_adata = True
    return config


def make_synthetic_counts(n_cells=N_CELLS, dim_genes=DIM_GENES, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.poisson(torch.ones(n_cells, dim_genes) * 5, generator=generator)


def make_batch_idx(n_cells=N_CELLS, n_batches=N_BATCHES, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, n_batches, (n_cells,), generator=generator)


@pytest.mark.parametrize("preset_name", PRESETS.keys())
def test_old_package_is_importable(preset_name):
    """Guard for the `pythonpath` list in pyproject.toml.

    test_numeric_equivalence_with_old_package uses importorskip, so a legacy package that has
    moved or been renamed turns the strongest check in this file into a silent skip. This test
    turns that into one explicit failure naming the package.
    """
    old_pkg = OLD_PACKAGES[preset_name]
    importlib.import_module(f"{old_pkg}.tools._decipher.decipher")


@pytest.mark.parametrize("preset_name", PRESETS.keys())
def test_shapes(preset_name):
    config = make_config(PRESETS[preset_name])
    model = Decipher(config)

    batch_on = config.batch_conditioning != "none"
    encoder_on = config.batch_conditioning == "decoder_encoder"
    mode = config.batch_embedding_mode

    decoder_extra = DIM_BATCH_EMBEDDING if (batch_on and mode == "concat_z") else 0
    encoder_extra = DIM_BATCH_EMBEDDING if (encoder_on and mode == "concat_z") else 0
    encoder_context = N_BATCHES if (encoder_on and mode == "concat_x") else 0
    recon_context = N_BATCHES if (batch_on and mode == "concat_x") else 0

    assert model.decoder_v_to_z.input_dim == DIM_V + decoder_extra
    assert model.encoder_x_to_z.input_dim == DIM_GENES + encoder_extra
    expected_zx_input = DIM_GENES if config.mean_field_v else DIM_GENES + DIM_Z
    assert model.encoder_zx_to_v.input_dim == expected_zx_input
    assert model.decoder_z_to_x.input_dim == DIM_Z

    # the one-hot context is what "concat_x" uses instead of widening input_dim
    assert model.encoder_x_to_z.context_dim == encoder_context
    assert model.decoder_z_to_x.context_dim == recon_context
    assert model.decoder_v_to_z.context_dim == 0
    assert model.encoder_zx_to_v.context_dim == 0

    tables = ("batch_emb", "batch_shift", "batch_shift_prior", "batch_shift_post")
    if not batch_on:
        expected_tables = ()
    elif mode == "concat_z":
        expected_tables = ("batch_emb",)
    else:  # concat_x learns no table at all -- the context is a fixed one-hot
        expected_tables = ()

    for name in tables:
        assert hasattr(model, name) == (name in expected_tables), name


@pytest.mark.parametrize("preset_name", PRESETS.keys())
def test_forward_pass_smoke(preset_name):
    config = make_config(PRESETS[preset_name])
    model = Decipher(config)
    model.eval()

    x = make_synthetic_counts()
    batch_idx = make_batch_idx() if config.batch_conditioning != "none" else None

    pyro.clear_param_store()
    guide_trace = poutine.trace(model.guide).get_trace(x, batch_idx)
    model_trace = poutine.trace(poutine.replay(model.model, trace=guide_trace)).get_trace(
        x, batch_idx
    )

    assert guide_trace.nodes["z"]["value"].shape == (N_CELLS, DIM_Z)
    assert guide_trace.nodes["v"]["value"].shape == (N_CELLS, DIM_V)
    assert model_trace.nodes["x"]["value"].shape == (N_CELLS, DIM_GENES)


def _normal_params(fn):
    base = getattr(fn, "base_dist", fn)
    return base.loc, base.scale


def _nb_logits(fn):
    base = getattr(fn, "base_dist", fn)
    return base.logits


@pytest.mark.parametrize("preset_name", PRESETS.keys())
def test_numeric_equivalence_with_old_package(preset_name):
    """Strongest check: an old variant's trained weights, loaded into the unified model with
    the matching preset config, must reproduce the old variant's outputs exactly.
    Skipped when the old package isn't importable in this environment.
    """
    old_pkg = OLD_PACKAGES[preset_name]
    old_decipher_mod = pytest.importorskip(f"{old_pkg}.tools._decipher.decipher")
    OldDecipher = old_decipher_mod.Decipher
    OldDecipherConfig = old_decipher_mod.DecipherConfig

    old_config = make_old_config(OldDecipherConfig)
    old_model = OldDecipher(old_config)
    old_model.eval()

    new_config = make_config(PRESETS[preset_name])
    new_model = Decipher(new_config)
    new_model.eval()

    state = old_model.state_dict()
    rename = STATE_DICT_RENAMES.get(preset_name)
    if rename is not None:
        old_prefix, new_prefix = rename
        state = {
            (new_prefix + k[len(old_prefix) :] if k.startswith(old_prefix) else k): v
            for k, v in state.items()
        }
    new_model.load_state_dict(state)

    x = make_synthetic_counts()
    batch_idx = make_batch_idx() if PRESETS[preset_name]["batch_conditioning"] != "none" else None

    # --- encoder side: guide() returns (z_loc, v_loc, z_scale, v_scale). z_loc/z_scale/v_scale
    # are pure functions of the weights and x, but v_loc is computed from the *sampled* z (not
    # z_loc), so the RNG must be reseeded identically before each call for v_loc to match too.
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    old_z_loc, old_v_loc, old_z_scale, old_v_scale = old_model.guide(x, batch_idx)
    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    new_z_loc, new_v_loc, new_z_scale, new_v_scale = new_model.guide(x, batch_idx)

    assert torch.allclose(old_z_loc, new_z_loc, atol=1e-5)
    assert torch.allclose(old_v_loc, new_v_loc, atol=1e-5)
    assert torch.allclose(old_z_scale, new_z_scale, atol=1e-5)
    assert torch.allclose(old_v_scale, new_v_scale, atol=1e-5)

    # --- decoder side: condition v and z to fixed values, compare the deterministic
    # z-prior (decoder_v_to_z) and reconstruction (decoder_z_to_x) parameters ---
    v_fixed = torch.randn(N_CELLS, DIM_V)
    z_fixed = torch.randn(N_CELLS, DIM_Z)

    pyro.clear_param_store()
    old_trace = poutine.trace(
        poutine.condition(old_model.model, data={"v": v_fixed, "z": z_fixed})
    ).get_trace(x, batch_idx)
    pyro.clear_param_store()
    new_trace = poutine.trace(
        poutine.condition(new_model.model, data={"v": v_fixed, "z": z_fixed})
    ).get_trace(x, batch_idx)

    old_z_prior_loc, old_z_prior_scale = _normal_params(old_trace.nodes["z"]["fn"])
    new_z_prior_loc, new_z_prior_scale = _normal_params(new_trace.nodes["z"]["fn"])
    assert torch.allclose(old_z_prior_loc, new_z_prior_loc, atol=1e-5)
    assert torch.allclose(old_z_prior_scale, new_z_prior_scale, atol=1e-5)

    old_logits = _nb_logits(old_trace.nodes["x"]["fn"])
    new_logits = _nb_logits(new_trace.nodes["x"]["fn"])
    assert torch.allclose(old_logits, new_logits, atol=1e-5)
