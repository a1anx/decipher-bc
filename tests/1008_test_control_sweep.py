"""No-training checks that the control driver runs the same 126 jobs as the split-fix sweep."""

import importlib
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Simulated Data",
        "Simulated Data Sweep Pipeline",
    ),
)

control = importlib.import_module("1008_sweep_control_bifurcation")
driver = importlib.import_module("0918_sweep_3models_bifurcation")
splitfix = importlib.import_module("1008_sweep_splitfix_bifurcation")

FULL_0918_GRID = driver.jobs()  # before any narrowing
_SAVED = (list(driver.ARMS), dict(driver.BIFURCATIONS), driver.NOTEBOOK_TAG)


def _control_jobs():
    control.narrow_driver_grid()
    try:
        return driver.jobs(notebook_tag=control.CONTROL_TAG)
    finally:
        driver.ARMS, driver.BIFURCATIONS, driver.NOTEBOOK_TAG = _SAVED


def test_control_has_126_jobs():
    assert len(_control_jobs()) == 126


def test_control_jobs_equal_the_matching_0918_jobs_except_the_tag():
    wanted = [
        {**j, "notebook_tag": control.CONTROL_TAG}
        for j in FULL_0918_GRID
        if j["batch_mode"] == "genes"
        and j["model"] in ("native", "model5")
        and j["bifurcation"] == "bif"
    ]
    assert _control_jobs() == wanted


def test_control_keys_equal_the_splitfix_keys():
    key = driver.KEY
    assert key == splitfix.KEY
    assert {tuple(j[k] for k in key) for j in _control_jobs()} == {
        tuple(j[k] for k in key) for j in splitfix.jobs()
    }


def test_driver_grid_is_restored_after_narrowing():
    _control_jobs()
    assert len(driver.jobs()) == 504
