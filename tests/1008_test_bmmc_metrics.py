"""Known-answer tests for the BMMC batch/biology metrics module."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Healthy Human Bone Marrow Mononuclear Cells"
    / "1008_bmmc_metrics.py"
)
_spec = importlib.util.spec_from_file_location("bmmc_metrics", _PATH)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

N_PER_BATCH = 150
N_BATCHES = 3


def _batches() -> np.ndarray:
    return np.repeat(np.arange(N_BATCHES), N_PER_BATCH)


def _mixed() -> tuple[np.ndarray, np.ndarray]:
    """One Gaussian cloud, batches drawn from the same distribution."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(N_BATCHES * N_PER_BATCH, 2)), _batches()


def _separated() -> tuple[np.ndarray, np.ndarray]:
    """Each batch its own cloud, 100 units apart."""
    emb, batch = _mixed()
    return emb + 100.0 * batch[:, None], batch


def test_ilisi_balanced_neighbourhoods_is_n_batches():
    # 15 sites 1 unit apart, 10 cells of each batch stacked on each site: every cell's
    # 29 neighbours are its own site, so the batch mix is 9/10/10 (LISI 2.99).
    sites = np.repeat(np.arange(15.0), N_BATCHES * 10)[:, None]
    batch = np.tile(np.repeat(np.arange(N_BATCHES), 10), 15)
    lisi = metrics.ilisi(sites, batch, perplexity=10)
    np.testing.assert_allclose(lisi, N_BATCHES, atol=0.02)


def test_ilisi_random_mixing_is_well_above_one():
    emb, batch = _mixed()
    assert np.median(metrics.ilisi(emb, batch, perplexity=10)) > 2.0


def test_ilisi_separated_is_one():
    emb, batch = _separated()
    np.testing.assert_allclose(metrics.ilisi(emb, batch, perplexity=10), 1.0)


def test_ilisi_within_returns_per_group_arrays():
    emb, batch = _separated()
    groups = np.tile(["a", "b"], len(batch) // 2)
    out = metrics.ilisi(emb, batch, perplexity=10, within=groups)
    assert sorted(out) == ["a", "b"]
    np.testing.assert_allclose(np.concatenate(list(out.values())), 1.0)


def test_clisi_pure_celltypes_is_one():
    emb, celltype = _separated()
    np.testing.assert_allclose(metrics.clisi(emb, celltype, perplexity=10), 1.0)


def test_lisi_median_normalizes_to_unit_interval():
    lisi = np.array([1.0, 2.0, 3.0])
    assert metrics.lisi_median(lisi, n_labels=3) == {"median": 2.0, "median_normalized": 0.5}


def test_normalize_lisi_single_label_raises():
    with pytest.raises(ValueError, match="at least 2 labels"):
        metrics.normalize_lisi(np.ones(5), n_labels=1)


def test_kbet_mixed_mostly_accepts():
    emb, batch = _mixed()
    assert metrics.kbet_acceptance(emb, batch, k=30) > 0.85


def test_kbet_separated_rejects_all():
    emb, batch = _separated()
    assert metrics.kbet_acceptance(emb, batch, k=30) == 0.0


def test_kbet_within_single_batch_group_is_nan():
    emb, batch = _separated()
    out = metrics.kbet_acceptance(emb, batch, k=30, within=batch)
    assert list(out) == [0, 1, 2] and all(np.isnan(v) for v in out.values())


def test_kbet_single_batch_raises():
    emb, _ = _mixed()
    with pytest.raises(ValueError, match="at least 2 batches"):
        metrics.kbet_acceptance(emb, np.zeros(len(emb)), k=30)


def test_silhouette_batch_mixed_is_near_one():
    emb, batch = _mixed()
    celltype = np.zeros(len(batch))
    assert metrics.silhouette_batch(emb, batch, celltype) == pytest.approx(1.0, abs=0.05)


def test_silhouette_batch_separated_is_near_zero():
    emb, batch = _separated()
    celltype = np.zeros(len(batch))
    assert metrics.silhouette_batch(emb, batch, celltype) == pytest.approx(0.0, abs=0.05)


def test_silhouette_label_separated_is_near_one():
    emb, labels = _separated()
    assert metrics.silhouette_label(emb, labels) == pytest.approx(1.0, abs=0.05)


def test_silhouette_label_singleton_label_raises():
    emb, _ = _mixed()
    with pytest.raises(ValueError, match="2..n_cells-1 labels"):
        metrics.silhouette_label(emb, np.zeros(len(emb)))


def test_empty_embedding_raises():
    with pytest.raises(ValueError, match="at least 2 cells"):
        metrics.ilisi(np.empty((0, 2)), np.empty(0))


def test_pc_regression_generating_and_independent_pcs():
    rng = np.random.default_rng(0)
    cov = np.repeat(["x", "y"], 500)
    pcs = np.column_stack([np.where(cov == "x", -1.0, 1.0), rng.normal(size=1000)])
    r2, total = metrics.pc_regression(pcs, np.array([0.6, 0.4]), cov)
    np.testing.assert_allclose(r2, [1.0, 0.0], atol=0.01)
    assert total == pytest.approx(0.6, abs=0.01)


def test_trajectory_corr_flips_sign():
    ref = np.arange(10.0)
    out = metrics.trajectory_corr(-3.0 * ref + 1.0, ref)
    assert out == metrics.TrajectoryCorr(
        pearson=pytest.approx(1.0), r2=pytest.approx(1.0), spearman=pytest.approx(1.0), sign=-1
    )
