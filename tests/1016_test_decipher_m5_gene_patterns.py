"""decipher_m5 `gene_patterns` under batch-conditioned models (scVI-style `batch=` argument).

Trajectory points have no batch, so the default decodes once per batch and averages
softmax(logits) uniformly; `batch=` pins one batch or a subset. Native keeps the single
batch-blind pass. Untrained eval-mode models with randomized batch tables, saved to a tmp folder.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import pytest
import torch

from decipher_m5.presets import PRESETS
from decipher_m5.tools import gene_patterns
from decipher_m5.tools._decipher import Decipher, DecipherConfig
from decipher_m5.tools._decipher.data import decipher_save_model
from decipher_m5.utils import DECIPHER_GLOBALS

BATCH_KEY = "batch"
BATCHES = ["b0", "b1", "b2"]
N_GENES = 30
L_SCALE = 10_000
POINTS = np.array([[0.0, 0.0], [1.0, -0.5], [2.0, 1.0], [-1.0, 0.5]])


def make_adata(n_cells: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(n_cells, N_GENES)).astype(np.float32)
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([BATCHES[i % len(BATCHES)] for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts, obs=obs)


def build(preset: str) -> Decipher:
    """Untrained eval model with every batch table randomized, so batches decode differently."""
    adata = make_adata()
    config = DecipherConfig(**PRESETS[preset])
    config.initialize_from_adata(adata, batch_key=BATCH_KEY)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(0)
    model = Decipher(config).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for table in (model.batch_emb, model.batch_ctx_dec, model.batch_ctx_enc):
            if table is not None:
                table.weight.copy_(torch.randn(table.weight.shape, generator=generator))
    return model


@pytest.fixture
def saved(tmp_path):
    """Return a function preset -> (adata with a saved model and one trajectory, model)."""
    old_folder = DECIPHER_GLOBALS["save_folder"]
    DECIPHER_GLOBALS["save_folder"] = str(tmp_path)

    def _saved(preset: str):
        model = build(preset)
        adata = make_adata()
        decipher_save_model(adata, model)
        adata.uns["decipher"]["trajectories"] = {
            "t": {"points": POINTS, "times": np.arange(len(POINTS), dtype=float)}
        }
        return adata, model

    yield _saved
    DECIPHER_GLOBALS["save_folder"] = old_folder


def decode_mean(model: Decipher, batch_code) -> np.ndarray:
    """Expected `mean` pattern for one batch code (None = batch-blind), via the public layers."""
    with torch.no_grad():
        z_mean, _ = model.decoder_v_to_z(torch.FloatTensor(POINTS))
        b = torch.full((len(POINTS),), batch_code if batch_code is not None else 0)
        if model.batch_emb is not None:
            logits = model.decoder_z_to_x(torch.cat([z_mean, model.batch_emb(b)], -1))
        elif model.batch_ctx_dec is not None:
            logits = model.decoder_z_to_x(z_mean, offset=model.batch_ctx_dec(b))
        else:
            logits = model.decoder_z_to_x(z_mean)
        return torch.softmax(logits, -1).numpy() * L_SCALE


def mean_pattern(adata) -> np.ndarray:
    return adata.uns["decipher"]["gene_patterns"]["t"]["mean"]


@pytest.mark.parametrize("preset", ["model5", "model5e"])
def test_default_mean_is_uniform_average_over_every_batch(saved, preset):
    adata, model = saved(preset)
    gene_patterns(adata, l_scale=L_SCALE, n_samples=5)
    expected = np.mean([decode_mean(model, b) for b in range(len(BATCHES))], axis=0)
    np.testing.assert_allclose(mean_pattern(adata), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("preset", ["model5", "model5e"])
def test_single_batch_name_decodes_with_that_batch_only(saved, preset):
    adata, model = saved(preset)
    gene_patterns(adata, l_scale=L_SCALE, n_samples=5, batch="b2")
    np.testing.assert_allclose(mean_pattern(adata), decode_mean(model, 2), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("preset", ["model5", "model5e"])
def test_batch_list_averages_over_listed_batches(saved, preset):
    adata, model = saved(preset)
    gene_patterns(adata, l_scale=L_SCALE, n_samples=5, batch=["b0", "b2"])
    expected = (decode_mean(model, 0) + decode_mean(model, 2)) / 2
    np.testing.assert_allclose(mean_pattern(adata), expected, rtol=1e-5, atol=1e-6)


def test_native_mean_is_the_batch_blind_decode(saved):
    adata, model = saved("native")
    gene_patterns(adata, l_scale=L_SCALE, n_samples=5)
    np.testing.assert_allclose(mean_pattern(adata), decode_mean(model, None), rtol=1e-5, atol=1e-6)


def test_quantile_band_is_one_row_per_point_with_q25_at_most_q75(saved):
    adata, _ = saved("model5e")
    gene_patterns(adata, l_scale=L_SCALE, n_samples=20)
    pattern = adata.uns["decipher"]["gene_patterns"]["t"]
    assert pattern["q25"].shape == pattern["q75"].shape == (len(POINTS), N_GENES)
    assert np.all(pattern["q25"] <= pattern["q75"])


@pytest.mark.parametrize("batch", ["nope", ["b0", "nope"], []])
def test_unknown_or_empty_batch_raises(saved, batch):
    adata, _ = saved("model5e")
    with pytest.raises(ValueError):
        gene_patterns(adata, l_scale=L_SCALE, n_samples=5, batch=batch)


def test_batch_on_native_model_raises(saved):
    adata, _ = saved("native")
    with pytest.raises(ValueError):
        gene_patterns(adata, l_scale=L_SCALE, n_samples=5, batch="b0")
