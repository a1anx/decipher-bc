"""No-training checks for the split-fix vs control comparison (`1008_compare_splitfix.py`)."""

import importlib
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

compare = importlib.import_module("1008_compare_splitfix")


def _ledger(sigmas, rho):
    return pd.DataFrame(
        {
            "model": "native",
            "batch_mode": "genes",
            "bifurcation": "bif",
            "shift_sigma": sigmas,
            "seed": 3,
            "decipher_seed": 1,
            "rho": rho,
        }
    )


def test_pair_runs_and_add_deltas_give_new_minus_control_per_key():
    control = _ledger([0.1, 10.0], [0.9, 0.5])
    new = _ledger([10.0, 0.1], [0.4, 0.95])

    out = compare.add_deltas(compare.pair_runs(control, new), ["rho"])

    got = out.set_index("shift_sigma")["d_rho"].sort_index()
    pd.testing.assert_series_equal(
        got, pd.Series([0.05, -0.1], index=pd.Index([0.1, 10.0], name="shift_sigma"), name="d_rho")
    )


def test_pair_runs_raises_on_unmatched_key():
    with pytest.raises(ValueError, match="1 unmatched keys"):
        compare.pair_runs(_ledger([0.1, 10.0], [0.9, 0.5]), _ledger([0.1], [0.9]))


def test_pair_runs_raises_on_duplicated_key():
    with pytest.raises(ValueError, match="duplicated KEY"):
        compare.pair_runs(_ledger([0.1, 0.1], [0.9, 0.8]), _ledger([0.1], [0.9]))
