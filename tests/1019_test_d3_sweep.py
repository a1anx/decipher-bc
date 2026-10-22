"""No-training checks for the 1019 decipher_models3 sweep driver and the 1008 pipeline's d3 path."""

import importlib
import inspect
import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Simulated Data",
        "Simulated Data Sweep Pipeline",
    ),
)

d3 = importlib.import_module("1019_sweep_d3_bifurcation")
pipeline = importlib.import_module("1008_simulated_data_pipeline_splitfix")

from decipher_models3.presets import PRESETS  # noqa: E402
import decipher_models3 as dc  # noqa: E402

FIVE = ["native", "model5", "model4r", "model5r", "model7"]


def _ledger_row(job, error):
    return {**{k: job[k] for k in d3.KEY}, "error": error}


def test_grid_has_315_jobs():
    assert len(d3.jobs()) == 315


def test_arms_are_the_five_presets_on_genes_data():
    assert d3.ARMS == [(p, p, "genes") for p in FIVE]


def test_every_job_carries_tag_package_and_run_date():
    jobs = d3.jobs(notebook_tag="t", run_date="1019")
    assert {(j["notebook_tag"], j["package"], j["run_date"]) for j in jobs} == {
        ("t", "decipher_models3", "1019")
    }


def test_driver_defaults_to_the_d3_tag_and_package():
    assert (d3.NOTEBOOK_TAG, d3.PACKAGE) == ("sweep1019_d3", "decipher_models3")


@pytest.mark.parametrize("preset", FIVE)
def test_preset_builds_a_decipher_config(preset):
    config = dc.tl.DecipherConfig(learning_rate=1e-3, seed=1, dim_z=3, beta=0.1, **PRESETS[preset])
    assert config.seed == 1


def test_resume_skips_completed_rows_and_keeps_errored_rows(tmp_path):
    done_job, failed_job, _ = d3.jobs()[:3]
    csv = tmp_path / "ledger.csv"
    pd.DataFrame([_ledger_row(done_job, None), _ledger_row(failed_job, "RuntimeError: x")]).to_csv(
        csv, index=False
    )

    done, prior = d3.completed_keys(str(csv))

    assert done == {tuple(str(done_job[k]) for k in d3.KEY)}
    assert len(prior) == 2


def test_resume_on_a_missing_ledger_is_empty(tmp_path):
    assert d3.completed_keys(str(tmp_path / "none.csv")) == (set(), None)


def test_pipeline_accepts_decipher_models3():
    assert "decipher_models3" in pipeline.PACKAGES


def test_pipeline_default_package_is_still_decipher_models2():
    default = (
        inspect.signature(pipeline.train_and_compute_rho_r2_bifurcation)
        .parameters["package"]
        .default
    )
    assert default == "decipher_models2"


# ---- pinned run date and per-job run dirs (2026-10-19) ----


class _FakePool:
    """Replaces the spawn pool: records the jobs it is given and runs none of them."""

    def __init__(self):
        self.jobs = []
        self.kwargs = {}

    def Pool(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def imap_unordered(self, _fn, todo):
        self.jobs = list(todo)
        return iter(())


def _run_main(monkeypatch, tmp_path, argv):
    return _run_main_pool(monkeypatch, tmp_path, argv).jobs


def _run_main_pool(monkeypatch, tmp_path, argv):
    """main() with its out-dir root moved under tmp_path and the pool faked."""
    pool = _FakePool()
    monkeypatch.setattr(d3, "_HERE", str(tmp_path / "pipeline"))
    monkeypatch.setattr(d3, "get_context", lambda _method: pool)
    monkeypatch.setattr(sys, "argv", ["driver", *argv])
    assert d3.main() == 0
    return pool


def test_each_pool_worker_runs_one_job(monkeypatch, tmp_path):
    pool = _run_main_pool(monkeypatch, tmp_path, ["--models", "native", "--workers", "3"])

    assert (pool.kwargs["processes"], pool.kwargs["maxtasksperchild"]) == (3, 1)


def test_run_date_pins_out_dir_and_every_job(monkeypatch, tmp_path):
    jobs = _run_main(monkeypatch, tmp_path, ["--run-date", "0101", "--models", "native"])

    out_dir = tmp_path / "Sweeps and Results" / "shift_sigma_sweep_bifurcation" / "0101"
    assert out_dir.is_dir()
    assert {j["run_date"] for j in jobs} == {"0101"} and len(jobs) == 63


def test_every_job_gets_a_run_dir_under_the_out_dir(monkeypatch, tmp_path):
    jobs = _run_main(monkeypatch, tmp_path, ["--run-date", "0101", "--models", "native"])

    runs = tmp_path / "pipeline" / ".." / "Sweeps and Results" / "shift_sigma_sweep_bifurcation"
    assert d3.job_run_dir(jobs[0]) == os.path.join(
        str(runs / "0101" / "runs"), "native_bif_genes_sigma0.1_seed3_decipherseed1"
    )
