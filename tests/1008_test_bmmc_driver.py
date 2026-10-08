"""Tests for the BMMC training driver's job grid and subsampling (no training)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Healthy Human Bone Marrow Mononuclear Cells"
    / "1008_bmmc_run_decipher.py"
)
_spec = importlib.util.spec_from_file_location("bmmc_run_decipher", _PATH)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)

SHARED = dict(tag="t", use_wandb=False, max_cells=None, max_epochs=3, out_dir="x")


def _ids(tier):
    return [j["run_id"] for j in driver.make_jobs(tier=tier, **SHARED)]


@pytest.mark.parametrize("tier,n", [("T1", 12), ("T2", 8), ("T3", 6)])
def test_tier_has_expected_run_count(tier, n):
    assert len(_ids(tier)) == n


def test_tiers_do_not_overlap_and_total_26_unique_runs():
    ids = _ids("T1") + _ids("T2") + _ids("T3")
    assert len(ids) == 26
    assert len(set(ids)) == 26


def test_t3_is_native_and_donor_at_5000_hvg_seeds_1_to_3():
    jobs = driver.make_jobs(tier="T3", **SHARED)
    assert {(j["config"], j["hvg"], j["seed"]) for j in jobs} == {
        (c, 5000, s) for c in ("native", "BC-donor") for s in (1, 2, 3)
    }


def test_comma_tier_concatenates_tiers():
    assert len(driver.make_jobs(tier="T1,T2", **SHARED)) == 20


@pytest.mark.parametrize(
    "config,preset,col",
    [
        ("native", "native", "none"),
        ("BC-donor", "model5", "donor"),
        ("BC-site", "model5", "site"),
        ("BC-sample", "model5", "sample"),
    ],
)
def test_config_maps_to_preset_and_batch_variable(config, preset, col):
    (job,) = driver.make_jobs(configs=[config], seeds=[1], hvgs=[2000], **SHARED)
    assert (job["preset"], job["conditioned_on"]) == (preset, col)


def test_explicit_grid_is_configs_times_seeds_times_hvgs():
    jobs = driver.make_jobs(configs=["native"], seeds=[1, 2], hvgs=[2000, 5000], **SHARED)
    assert len(jobs) == 4


def test_unknown_config_raises():
    with pytest.raises(KeyError):
        driver.make_jobs(configs=["BC-nope"], **SHARED)


def test_stratified_subsample_keeps_every_sample_including_tiny_ones():
    labels = np.array(["a"] * 1000 + ["b"] * 500 + ["c"] * 3)
    idx = driver.stratified_subsample(labels, 100)
    assert set(labels[idx]) == {"a", "b", "c"}
    assert 95 <= len(idx) <= 105


def test_stratified_subsample_is_deterministic_and_sorted():
    labels = np.repeat(["a", "b"], 100)
    first = driver.stratified_subsample(labels, 40)
    assert np.array_equal(first, driver.stratified_subsample(labels, 40))
    assert np.array_equal(first, np.sort(first))


def test_stratified_subsample_returns_everything_when_max_cells_exceeds_n():
    assert np.array_equal(driver.stratified_subsample(np.array(["a"] * 5), 50), np.arange(5))
