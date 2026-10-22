"""No-training checks for the d3 sigma-sweep comparison (`1019_compare_d3_sweep.py`)."""

import importlib
import os
import sys

import numpy as np
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

compare = importlib.import_module("1019_compare_d3_sweep")


def _ledger(models=("native", "model5", "model7"), sigmas=(0.1, 1.0), seeds=(3, 4)):
    """Synthetic ledger: rho = base[model] + 0.1 * sigma_index + 0.01 * seed."""
    base = {"native": 0.5, "model5": 0.6, "model7": 0.4}
    rows = []
    for model in models:
        for i, sigma in enumerate(sigmas):
            for seed in seeds:
                rows.append(
                    {
                        "model": model,
                        "batch_mode": "genes",
                        "bifurcation": "bif",
                        "shift_sigma": sigma,
                        "seed": seed,
                        "decipher_seed": 1,
                        "rho": base[model] + 0.1 * i + 0.01 * seed,
                        "n_epochs": 100 + seed,
                    }
                )
    return pd.DataFrame(rows)


def test_pair_deltas_gives_one_row_per_key_minus_model():
    out = compare.pair_deltas(_ledger(), "native", ["rho"])

    expected = pd.DataFrame(
        {
            "model": ["model5"] * 4 + ["model7"] * 4,
            "batch_mode": "genes",
            "bifurcation": "bif",
            "shift_sigma": [0.1, 0.1, 1.0, 1.0] * 2,
            "seed": [3, 4] * 4,
            "decipher_seed": 1,
            "rho": [0.1] * 4 + [-0.1] * 4,
        }
    )
    pd.testing.assert_frame_equal(
        out[expected.columns].reset_index(drop=True), expected, check_exact=False
    )


def test_summarize_deltas_matches_hand_computed_values():
    led = _ledger(models=("native", "model5"))
    led.loc[(led.model == "model5") & (led.seed == 3), "rho"] -= 0.3  # deltas -0.2 and +0.1

    out = compare.summarize_deltas(compare.pair_deltas(led, "native", ["rho"]), ["rho"])

    sd = np.std([-0.2, 0.1], ddof=1)
    expected = pd.DataFrame(
        {
            "model": "model5",
            "shift_sigma": [0.1, 1.0],
            "metric": "rho",
            "mean": [-0.05, -0.05],
            "sd": [sd, sd],
            "n": [2, 2],
            "frac_pos": [0.5, 0.5],
        }
    )
    pd.testing.assert_frame_equal(out, expected, check_exact=False)


def test_summarize_by_arm_sigma_matches_hand_computed_values():
    out = compare.summarize_by_arm_sigma(_ledger(models=("native",)), ["n_epochs"])

    expected = pd.DataFrame(
        {
            "model": "native",
            "shift_sigma": [0.1, 1.0],
            "n": [2, 2],
            "n_epochs_mean": [103.5, 103.5],
            "n_epochs_sd": [np.std([103, 104], ddof=1)] * 2,
        }
    )
    pd.testing.assert_frame_equal(out, expected, check_exact=False)


def test_pair_deltas_raises_on_missing_pair():
    led = _ledger()
    drop = (led.model == "model7") & (led.seed == 4) & (led.shift_sigma == 1.0)
    with pytest.raises(ValueError, match="1 pairs missing"):
        compare.pair_deltas(led[~drop], "native", ["rho"])


def test_pair_deltas_raises_on_duplicated_key():
    led = _ledger()
    with pytest.raises(ValueError, match="duplicated KEY"):
        compare.pair_deltas(pd.concat([led, led.iloc[[0]]]), "native", ["rho"])


def test_batch_asw_is_one_when_mixed_and_zero_when_separated():
    batch = np.array(["a"] * 40 + ["b"] * 40)
    at_p_or_q = np.tile([[0.0, 0.0]] * 20 + [[1.0, 0.0]] * 20, (2, 1))  # both batches on P and Q
    apart = at_p_or_q + np.where(batch[:, None] == "a", 0.0, 100.0)  # batch b shifted far away

    assert compare.batch_asw(at_p_or_q, batch) == pytest.approx(1.0, abs=0.05)
    assert compare.batch_asw(apart, batch) == pytest.approx(0.0, abs=0.05)


def test_select_runs_keeps_named_models_within_sigma_bounds_inclusive():
    led = _ledger(sigmas=(0.1, 1.0, 5.0))

    out = compare.select_runs(led, models=["model5", "model7"], sigma_min=1.0, sigma_max=5.0)

    assert sorted(set(zip(out.model, out.shift_sigma))) == [
        ("model5", 1.0),
        ("model5", 5.0),
        ("model7", 1.0),
        ("model7", 5.0),
    ]


def test_select_runs_without_filters_returns_ledger_unchanged():
    led = _ledger()

    pd.testing.assert_frame_equal(compare.select_runs(led), led)


def test_select_runs_raises_on_model_not_in_ledger():
    with pytest.raises(ValueError, match="model9"):
        compare.select_runs(_ledger(), models=["model5", "model9"])


def test_concat_ledgers_joins_disjoint_ledgers():
    a, b = _ledger(seeds=(3,)), _ledger(seeds=(4,))

    out = compare.concat_ledgers([a, b])

    pd.testing.assert_frame_equal(out, pd.concat([a, b], ignore_index=True))


def test_concat_ledgers_raises_on_key_in_both():
    led = _ledger()
    with pytest.raises(ValueError, match="duplicated KEY"):
        compare.concat_ledgers([led, led.iloc[[0]]])


@pytest.mark.parametrize("log_x, scale", [(True, "log"), (False, "linear")])
def test_metric_figure_x_scale_follows_log_x(log_x, scale):
    summary = compare.summarize_by_arm_sigma(_ledger(sigmas=(7.5, 8.0, 11.0)), ["rho"])

    ax = compare.metric_figure(summary, "rho", log_x=log_x).axes[0]

    assert ax.get_xscale() == scale


def test_metric_figure_linear_ticks_are_the_sigmas():
    summary = compare.summarize_by_arm_sigma(_ledger(sigmas=(7.5, 8.0, 11.0)), ["rho"])

    ax = compare.metric_figure(summary, "rho", log_x=False).axes[0]

    assert list(ax.get_xticks()) == [7.5, 8.0, 11.0]
