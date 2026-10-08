"""decipher_m5: the table-based model5 computes the same function as the one-hot model5.

Untrained models on a small synthetic AnnData, eval mode, no training. The reference is
`decipher_models2` (one-hot batch columns). float32 agreement is checked at rtol 1e-5 / atol 1e-6,
which holds for untrained weights; trained checkpoints differ by a few ulp, so none are tested.
"""

import importlib
import math

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro.infer import Trace_ELBO
from torch.nn.functional import one_hot

from decipher_m5.presets import PRESETS as M5_PRESETS
from decipher_m5.tools._decipher import (
    Decipher,
    DecipherConfig,
    decipher_load_model,
    remap_model5_state_dict,
)

RTOL, ATOL = 1e-5, 1e-6
REF = "decipher_models2"
NEW = "decipher_m5"
BATCH_KEY = "batch"
N_BATCHES = 3
OUTPUTS = ["guide z_loc", "guide z_scale", "decoder_z_to_x", "impute", "v", "z", "z_raw", "elbo"]


def make_adata(n_cells: int = 200, n_genes: int = 50, n_batches: int = N_BATCHES, seed: int = 0):
    """Synthetic integer counts with a categorical batch column and unique obs names."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=n_genes), size=(n_cells, n_genes))
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([f"b{i % n_batches}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


def build_model(package: str, preset: str, adata, seed: int = 0):
    """Untrained eval-mode Decipher built as `decipher_train` does: (model, x, batch_idx)."""
    tools = importlib.import_module(f"{package}.tools.decipher")
    data = importlib.import_module(f"{package}.tools._decipher.data")
    presets = importlib.import_module(f"{package}.presets").PRESETS
    cls = importlib.import_module(f"{package}.tools._decipher")

    adata = adata.copy()
    config = cls.DecipherConfig(**presets[preset], seed=seed)
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
    tools._make_train_val_split(adata, config.val_frac, config.seed)
    adata_train = adata[adata.obs["decipher_split"] == "train", :]
    config.initialize_from_adata(adata_train, batch_key=BATCH_KEY)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(config.seed)
    model = cls.Decipher(config=config).eval()

    x = torch.tensor(data.get_dense_X(adata_train), dtype=torch.float32)
    return model, x, data.get_batch_idx(adata_train, config)


def decode_logits(model, z, batch_idx):
    """decoder_z_to_x logits through public attributes, for either batch layout."""
    if getattr(model, "batch_ctx_dec", None) is not None:  # decipher_m5 model5
        return model.decoder_z_to_x(z, offset=model.batch_ctx_dec(batch_idx))
    if model.decoder_z_to_x.context_dim > 0:  # one-hot model5
        return model.decoder_z_to_x(z, context=one_hot(batch_idx, model.config.n_batches).float())
    return model.decoder_z_to_x(z)  # native


def model_outputs(model, x, batch_idx, seed: int = 0) -> dict:
    """Every eval-mode output compared between packages, keyed by name."""
    out = {}
    with torch.no_grad():
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        z_loc, _, z_scale, _ = model.guide(x, batch_idx)
        out["guide z_loc"], out["guide z_scale"] = z_loc, z_scale
        out["decoder_z_to_x"] = decode_logits(model, z_loc, batch_idx)
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        out["impute"] = torch.as_tensor(model.impute_gene_expression_numpy(x, batch_idx))
        v, z, z_raw = model.compute_v_z_numpy(x, batch_idx)
        out["v"], out["z"], out["z_raw"] = map(torch.as_tensor, (v, z, z_raw))
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        out["elbo"] = torch.tensor(Trace_ELBO().loss(model.model, model.guide, x, batch_idx))
    return out


def remapped_model5(adata):
    """(reference model5, decipher_m5 model5 loaded with its remapped weights, x, batch_idx)."""
    m_ref, x, batch_idx = build_model(REF, "model5", adata)
    m_new, _, _ = build_model(NEW, "model5", adata)
    m_new.load_state_dict(remap_model5_state_dict(m_ref.state_dict()))  # strict
    return m_ref, m_new, x, batch_idx


@pytest.fixture(scope="module")
def adata():
    return make_adata()


@pytest.fixture(scope="module")
def model5_pair(adata):
    return remapped_model5(adata)


@pytest.fixture(scope="module")
def model5_outputs(model5_pair):
    m_ref, m_new, x, batch_idx = model5_pair
    return model_outputs(m_ref, x, batch_idx), model_outputs(m_new, x, batch_idx)


@pytest.fixture(scope="module")
def native_outputs(adata):
    m_ref, x, batch_idx = build_model(REF, "native", adata)
    m_new, _, _ = build_model(NEW, "native", adata)
    return model_outputs(m_ref, x, batch_idx), model_outputs(m_new, x, batch_idx)


@pytest.fixture(scope="module")
def roundtrip_outputs(adata, tmp_path_factory):
    """Outputs of a fresh reference state_dict saved with torch.save, loaded, remapped into m5."""
    m_ref, x, batch_idx = build_model(REF, "model5", adata)
    path = tmp_path_factory.mktemp("roundtrip") / "model5.pt"
    torch.save(m_ref.state_dict(), path)
    m_new, _, _ = build_model(NEW, "model5", adata)
    m_new.load_state_dict(remap_model5_state_dict(torch.load(path)), strict=True)
    return model_outputs(m_ref, x, batch_idx), model_outputs(m_new, x, batch_idx)


@pytest.fixture(scope="module")
def loaded_checkpoint_outputs(adata, tmp_path_factory):
    """Outputs of a `decipher_models2` checkpoint loaded by `decipher_m5`'s `decipher_load_model`."""
    folder = tmp_path_factory.mktemp("checkpoint")
    ref_globals = importlib.import_module(f"{REF}.utils").DECIPHER_GLOBALS
    new_globals = importlib.import_module(f"{NEW}.utils").DECIPHER_GLOBALS
    old = ref_globals["save_folder"], new_globals["save_folder"]
    ref_globals["save_folder"] = new_globals["save_folder"] = str(folder)
    try:
        m_ref, x, batch_idx = build_model(REF, "model5", adata)
        saved = adata.copy()
        importlib.import_module(f"{REF}.tools._decipher.data").decipher_save_model(saved, m_ref)
        saved.write_h5ad(folder / "saved.h5ad")
        loaded = decipher_load_model(ad.read_h5ad(folder / "saved.h5ad")).eval()
    finally:
        ref_globals["save_folder"], new_globals["save_folder"] = old
    assert loaded.batch_ctx_enc is not None, "old checkpoint loaded without batch tables"
    return model_outputs(m_ref, x, batch_idx), model_outputs(loaded, x, batch_idx)


@pytest.mark.parametrize("name", OUTPUTS)
def test_remapped_model5_output_matches_one_hot_reference(model5_outputs, name):
    out_ref, out_new = model5_outputs
    torch.testing.assert_close(out_new[name], out_ref[name], rtol=RTOL, atol=ATOL)


def test_remapped_model5_z_differs_from_z_raw(model5_outputs):
    """The z comparison is not trivial: batch correction actually moves z."""
    _, out_new = model5_outputs
    assert not torch.allclose(out_new["z"], out_new["z_raw"], rtol=RTOL, atol=ATOL)


def test_native_state_dict_is_identical_to_reference_for_same_seed(adata):
    m_ref, _, _ = build_model(REF, "native", adata)
    m_new, _, _ = build_model(NEW, "native", adata)
    sd_ref, sd_new = m_ref.state_dict(), m_new.state_dict()
    assert list(sd_new) == list(sd_ref)
    assert all(torch.equal(sd_new[k], sd_ref[k]) for k in sd_ref)


@pytest.mark.parametrize("name", OUTPUTS)
def test_native_output_matches_reference_for_same_seed(native_outputs, name):
    out_ref, out_new = native_outputs
    torch.testing.assert_close(out_new[name], out_ref[name], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("side", ["enc", "dec"])
@pytest.mark.parametrize("param", ["table", "weight", "bias"])
def test_model5_init_within_one_over_sqrt_batches_plus_input_dim(adata, side, param):
    model, _, _ = build_model(NEW, "model5", adata)
    table, net = {
        "enc": (model.batch_ctx_enc, model.encoder_x_to_z),
        "dec": (model.batch_ctx_dec, model.decoder_z_to_x),
    }[side]
    first = net.layers[0]
    tensor = {"table": table.weight, "weight": first.weight, "bias": first.bias}[param]
    bound = 1 / math.sqrt(N_BATCHES + first.in_features)
    assert 0.5 * bound < tensor.abs().max().item() <= bound


def test_remap_gives_tables_equal_to_old_one_hot_weight_columns(model5_pair):
    m_ref, m_new, _, _ = model5_pair
    dec_columns = m_ref.decoder_z_to_x.layers[0].weight[:, :N_BATCHES]
    enc_columns = m_ref.encoder_x_to_z.layers[0].weight[:, :N_BATCHES]
    assert torch.equal(m_new.batch_ctx_dec.weight.T, dec_columns)
    assert torch.equal(m_new.batch_ctx_enc.weight.T, enc_columns)


@pytest.mark.parametrize("name", OUTPUTS)
def test_saved_state_dict_roundtrip_remaps_into_m5_with_matching_output(roundtrip_outputs, name):
    out_ref, out_new = roundtrip_outputs
    torch.testing.assert_close(out_new[name], out_ref[name], rtol=RTOL, atol=ATOL)


def test_zero_batches_builds_no_tables_and_is_batch_blind(adata):
    config = DecipherConfig(**M5_PRESETS["model5"])
    config.initialize_from_adata(adata, batch_key="no_such_column")  # -> n_batches = 0
    model = Decipher(config).eval()
    assert model.batch_ctx_enc is None and model.batch_ctx_dec is None
    assert not any(k.startswith("batch_ctx") for k in model.state_dict())
    _, z, z_raw = model.compute_v_z_numpy(adata.X)
    assert np.array_equal(z, z_raw)


def test_batch_idx_none_equals_all_zeros(model5_pair):
    _, m_new, x, _ = model5_pair
    zeros = torch.zeros(len(x), dtype=torch.long)
    with torch.no_grad():
        from_none = m_new.compute_v_z_numpy(x, None)
        from_zeros = m_new.compute_v_z_numpy(x, zeros)
    for a, b in zip(from_none, from_zeros):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", OUTPUTS)
def test_old_model5_checkpoint_loads_into_m5_with_matching_output(loaded_checkpoint_outputs, name):
    out_ref, out_new = loaded_checkpoint_outputs
    torch.testing.assert_close(out_new[name], out_ref[name], rtol=RTOL, atol=ATOL)


def test_remap_rejects_native_state_dict(adata):
    m_native, _, _ = build_model(NEW, "native", adata)
    with pytest.raises(ValueError):
        remap_model5_state_dict(m_native.state_dict())


def test_remap_rejects_state_dict_already_in_table_layout(model5_pair):
    _, m_new, _, _ = model5_pair
    with pytest.raises(ValueError):
        remap_model5_state_dict(m_new.state_dict())
