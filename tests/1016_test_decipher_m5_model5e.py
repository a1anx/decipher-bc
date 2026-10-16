"""decipher_m5 model5e: model5's graph with model2's shared learned batch embedding.

Guide on z as model2 in `decipher_models` (g_xz([log1p x; e_b])), reconstruction f_zx([z; e_b]),
prior batch-blind. One zero-initialized table `batch_emb` (n_batches x dim_batch_embedding).
Untrained eval-mode models on a small synthetic AnnData.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from decipher_m5.presets import PRESETS
from decipher_m5.tools._decipher import Decipher, DecipherConfig, decipher_load_model
from decipher_m5.tools._decipher.data import decipher_save_model, get_batch_idx
from decipher_m5.utils import DECIPHER_GLOBALS
from decipher_models.presets import PRESETS as MODELS_PRESETS
from decipher_models.tools._decipher import Decipher as Model2Decipher
from decipher_models.tools._decipher import DecipherConfig as Model2Config

BATCH_KEY = "batch"
N_BATCHES = 3
N_GENES = 50


def make_adata(n_cells: int = 120, seed: int = 0) -> ad.AnnData:
    """Synthetic integer counts with a categorical batch column."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=N_GENES), size=(n_cells, N_GENES))
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([f"b{i % N_BATCHES}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


def build(adata, batch_key: str = BATCH_KEY, **overrides) -> Decipher:
    """Untrained eval-mode model5e."""
    config = DecipherConfig(**{**PRESETS["model5e"], **overrides})
    config.initialize_from_adata(adata, batch_key=batch_key)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    return Decipher(config).eval()


@pytest.fixture(scope="module")
def adata():
    return make_adata()


@pytest.fixture(scope="module")
def data(adata):
    x = torch.tensor(adata.X)
    return x, get_batch_idx(adata, build(adata).config)


def randomize_embedding(model: Decipher, seed: int = 1) -> Decipher:
    with torch.no_grad():
        model.batch_emb.weight.copy_(
            torch.randn(model.batch_emb.weight.shape, generator=torch.Generator().manual_seed(seed))
        )
    return model


def test_model5e_has_one_zero_initialized_batch_table(adata):
    model = build(adata)
    batch_keys = [k for k in model.state_dict() if k.startswith("batch_")]
    assert batch_keys == ["batch_emb.weight"]
    assert torch.equal(model.batch_emb.weight, torch.zeros(N_BATCHES, 8))


def test_model5e_prior_takes_only_v(adata):
    model = build(adata)
    assert model.decoder_v_to_z.layers[0].in_features == model.config.dim_v


def test_model5e_guide_matches_decipher_models_model2_with_same_weights(adata, data):
    x, batch_idx = data
    cfg2 = Model2Config(**MODELS_PRESETS["model2"])
    cfg2.initialize_from_adata(adata, batch_key=BATCH_KEY)
    model2 = Model2Decipher(cfg2).eval()
    model5e = build(adata)
    shared = {
        k: v
        for k, v in model2.state_dict().items()
        if k.startswith(("encoder_x_to_z.", "encoder_zx_to_v.", "batch_emb."))
    }
    model5e.load_state_dict(shared, strict=False)

    out2 = model2.compute_v_z_numpy(x, batch_idx)
    out5e = model5e.compute_v_z_numpy(x, batch_idx)

    for a, b in zip(out5e, out2):  # v, z, z_raw
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_model5e_at_zero_init_is_batch_blind(adata, data):
    x, batch_idx = data
    model = build(adata)
    _, z, z_raw = model.compute_v_z_numpy(x, batch_idx)
    np.testing.assert_allclose(z, z_raw, rtol=1e-6, atol=1e-6)


def test_model5e_reconstruction_shift_between_batches_is_decoder_weight_times_embedding(adata):
    model = randomize_embedding(build(adata))
    z = torch.randn(4, model.config.dim_z, generator=torch.Generator().manual_seed(2))
    b0, b1 = torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)
    with torch.no_grad():
        logits0 = model.decoder_z_to_x(torch.cat([z, model.batch_emb(b0)], -1))
        logits1 = model.decoder_z_to_x(torch.cat([z, model.batch_emb(b1)], -1))
        w_e = model.decoder_z_to_x.layers[0].weight[:, model.config.dim_z :]
        expected = w_e @ (model.batch_emb.weight[1] - model.batch_emb.weight[0])

    torch.testing.assert_close(logits1 - logits0, expected.expand(4, -1), rtol=1e-5, atol=1e-5)


def test_model5e_imputation_depends_on_batch_once_embedding_is_learned(adata, data):
    x, _ = data
    model = randomize_embedding(build(adata))
    as_b0 = model.impute_gene_expression_numpy(x, torch.zeros(len(x), dtype=torch.long))
    as_b1 = model.impute_gene_expression_numpy(x, torch.ones(len(x), dtype=torch.long))
    assert not np.allclose(as_b0, as_b1)


def test_one_svi_step_moves_zero_embedding_off_zero(adata, data):
    x, batch_idx = data
    model = build(adata).train()
    svi = SVI(model.model, model.guide, Adam({"lr": 1e-2}), Trace_ELBO())
    svi.step(x, batch_idx)
    assert model.batch_emb.weight.abs().sum() > 0


def test_model5e_checkpoint_roundtrip_keeps_embedding_width(adata, data, tmp_path):
    x, batch_idx = data
    model = randomize_embedding(build(adata, dim_batch_embedding=4))
    old_folder = DECIPHER_GLOBALS["save_folder"]
    DECIPHER_GLOBALS["save_folder"] = str(tmp_path)
    try:
        saved = adata.copy()
        decipher_save_model(saved, model)
        saved.write_h5ad(tmp_path / "saved.h5ad")
        loaded = decipher_load_model(ad.read_h5ad(tmp_path / "saved.h5ad"))
    finally:
        DECIPHER_GLOBALS["save_folder"] = old_folder

    assert loaded.batch_emb.weight.shape == (N_BATCHES, 4)
    for a, b in zip(loaded.compute_v_z_numpy(x, batch_idx), model.compute_v_z_numpy(x, batch_idx)):
        np.testing.assert_array_equal(a, b)


def test_model5e_with_zero_batches_builds_no_table_and_native_widths(adata):
    model = build(adata, batch_key="no_such_column")
    assert model.batch_emb is None
    assert model.encoder_x_to_z.layers[0].in_features == N_GENES
    assert model.decoder_z_to_x.layers[0].in_features == model.config.dim_z


def test_unknown_batch_context_raises():
    with pytest.raises(ValueError):
        DecipherConfig(batch_context="concat_z")


def test_model5e_generative_likelihood_of_x_depends_on_batch_given_z(adata, data):
    x, _ = data
    model = randomize_embedding(build(adata))
    fixed = {"v": torch.zeros(len(x), 2), "z": torch.zeros(len(x), model.config.dim_z)}

    def x_mean(b: int) -> torch.Tensor:
        batch_idx = torch.full((len(x),), b, dtype=torch.long)
        trace = pyro.poutine.trace(pyro.poutine.condition(model.model, fixed)).get_trace(
            x, batch_idx
        )
        return trace.nodes["x"]["fn"].mean

    with torch.no_grad():
        assert not torch.allclose(x_mean(0), x_mean(1))
