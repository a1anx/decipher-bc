"""Criteria logic and seed aggregation of the BMMC evaluation (hand-made tables, no data)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Healthy Human Bone Marrow Mononuclear Cells"
    / "1008_bmmc_evaluate.py"
)
_spec = importlib.util.spec_from_file_location("bmmc_evaluate", _PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def per_run(values: dict) -> pd.DataFrame:
    """Per-run rows from {(config, embedding, scope, metric, variable): [seed values]}."""
    rows = [
        dict(config=c, embedding=e, scope=s, metric=m, variable=v, seed=i + 1, value=x)
        for (c, e, s, m, v), xs in values.items()
        for i, x in enumerate(xs)
    ]
    return pd.DataFrame(rows)


def criteria_table(
    bc_ilisi: list[float], bc_ery: list[float], bc_mye=(0.9, 0.9, 0.9)
) -> pd.DataFrame:
    """Native vs BC-donor for one variable; native iLISI sd = 0.1, trajectory sd = 0.05."""
    nat_i, nat_t = [1.9, 2.0, 2.1], [0.85, 0.90, 0.95]
    vals = {}
    for config, ilisi, ery, mye in [
        ("native", nat_i, nat_t, [0.9, 0.9, 0.9]),
        ("BC-donor", bc_ilisi, bc_ery, list(bc_mye)),
    ]:
        for var in ev.BATCH_VARS:
            vals[(config, "v", "global", "ilisi_median", var)] = ilisi
            vals[(config, "v", "global", "ilisi_median_norm", var)] = [(x - 1) / 8 for x in ilisi]
        vals[(config, "v", "set_erythroid", "traj_spearman", "erythroid")] = ery
        vals[(config, "v", "set_myeloid", "traj_spearman", "myeloid")] = mye
    for config in ("BC-site", "BC-sample"):
        for key, xs in list(vals.items()):
            if key[0] == "native":
                vals[(config, *key[1:])] = xs
    return ev.aggregate_seeds(per_run(vals))


NO_PATHS = pd.DataFrame(columns=["run_id", "config", "trajectory", "passes_progenitor"])


def donor_row(agg: pd.DataFrame, paths: pd.DataFrame = NO_PATHS) -> pd.Series:
    crit = ev.build_criteria(agg, paths)
    return crit[crit.variable == "donor"].iloc[0]


def test_aggregate_seeds_gives_mean_sample_sd_and_n():
    agg = ev.aggregate_seeds(
        per_run({("native", "v", "global", "ilisi_median", "donor"): [1, 2, 3]})
    )
    row = agg.iloc[0]
    assert (row["mean"], row["sd"], row["n"]) == (2.0, 1.0, 3)


def test_matrix_wide_formats_mean_pm_sd_and_min_n():
    agg = ev.aggregate_seeds(
        per_run(
            {
                ("native", "v", "global", "ilisi_median", "donor"): [1, 2, 3],
                ("BC-donor", "v", "global", "ilisi_median", "donor"): [3, 3, 3],
            }
        )
    )
    wide = ev.matrix_wide(agg, ["native", "BC-donor"])
    col = "v|global|ilisi_median|donor"
    assert wide[col].to_dict() == {"native": "2.000 ± 1.000", "BC-donor": "3.000 ± 0.000"}
    assert wide["n_seeds"].to_dict() == {"native": 3, "BC-donor": 3}


def test_ilisi_improves_when_gain_exceeds_native_sd():
    row = donor_row(criteria_table([3.0, 3.0, 3.0], [0.9] * 3))
    assert (row["ilisi_diff"], row["ilisi_verdict"], row["ilisi_exceeds_native_sd"]) == (
        pytest.approx(1.0),
        "improves",
        True,
    )


def test_ilisi_within_noise_when_gain_below_native_sd():
    row = donor_row(criteria_table([2.05, 2.05, 2.05], [0.9] * 3))
    assert row["ilisi_verdict"] == "within noise"
    assert not row["ilisi_exceeds_native_sd"]
    assert not row["passes_all"]


def test_ilisi_worse_when_drop_exceeds_native_sd():
    row = donor_row(criteria_table([1.5, 1.5, 1.5], [0.9] * 3))
    assert row["ilisi_verdict"] == "worse"


@pytest.mark.parametrize(
    "bc_ery, ok",
    [
        ([0.85, 0.85, 0.85], True),  # 0.05 below native mean 0.90 = exactly one sd: allowed
        ([0.84, 0.84, 0.84], False),  # more than one sd below
        ([0.99, 0.99, 0.99], True),  # higher is fine
    ],
)
def test_trajectory_not_worse_allows_one_native_sd(bc_ery, ok):
    row = donor_row(criteria_table([3.0] * 3, bc_ery))
    assert row["erythroid_not_worse"] == ok
    assert row["passes_all"] == ok


def test_passes_all_needs_improvement_and_both_trajectories():
    good = donor_row(criteria_table([3.0] * 3, [0.9] * 3))
    bad_myeloid = donor_row(criteria_table([3.0] * 3, [0.9] * 3, bc_mye=(0.5, 0.5, 0.5)))
    assert good["passes_all"]
    assert not bad_myeloid["passes_all"] and not bad_myeloid["myeloid_not_worse"]


def test_ilisi_verdict_is_undetermined_without_seed_sd():
    assert ev.ilisi_verdict(2.0, float("nan"), 3.0) == "undetermined"


def test_criteria_has_one_row_per_conditioned_variable_and_lists_path_flags():
    paths = pd.DataFrame(
        [
            dict(
                run_id="BC-donor_hvg2000_ds2",
                config="BC-donor",
                trajectory="erythroid",
                passes_progenitor=False,
            ),
            dict(
                run_id="BC-donor_hvg2000_ds1",
                config="BC-donor",
                trajectory="myeloid",
                passes_progenitor=True,
            ),
        ]
    )
    crit = ev.build_criteria(criteria_table([3.0] * 3, [0.9] * 3), paths)
    assert crit["variable"].tolist() == ["donor", "site", "sample"]
    assert (
        crit.set_index("variable").loc["donor", "path_flags_bc"] == "BC-donor_hvg2000_ds2:erythroid"
    )


def test_pick_best_config_prefers_passing_then_larger_normalised_gain():
    crit = pd.DataFrame(
        {
            "config": ["BC-donor", "BC-site", "BC-sample"],
            "passes_all": [False, True, True],
            "ilisi_norm_diff": [0.9, 0.1, 0.3],
        }
    )
    assert ev.pick_best_config(crit) == "BC-sample"


def test_path_record_flags_path_missing_progenitor():
    cell_type = pd.Series(["HSC"] * 3 + ["MK/E prog"] * 3 + ["Reticulocyte"] * 3)
    clusters = pd.Series(["0"] * 3 + ["1"] * 3 + ["2"] * 3)
    in_set = np.ones(9, bool)
    skip = ev.path_record("erythroid", ["0", "2"], "0", "2", cell_type, clusters, in_set)
    full = ev.path_record("erythroid", ["0", "1", "2"], "0", "2", cell_type, clusters, in_set)
    assert (skip["passes_progenitor"], skip["progenitor_cluster"]) == (False, "1")
    assert skip["majority_types"] == "HSC > Reticulocyte"
    assert full["passes_progenitor"] and full["coverage"] == 1.0
    assert skip["coverage"] == pytest.approx(6 / 9)


def test_set_masks_cover_hsc_plus_each_branch():
    masks = ev.set_masks(np.array(["HSC", "erythroid", "myeloid", "myeloid"]))
    assert masks["set_erythroid"].tolist() == [True, True, False, False]
    assert masks["set_myeloid"].tolist() == [True, False, True, True]


def test_baseline_to_rows_splits_perm_null_from_real_value():
    base = pd.DataFrame(
        [
            dict(scope="global", group="all", metric="ilisi_median", variable="donor", value=1.9),
            dict(
                scope="global", group="all", metric="ilisi_median_perm", variable="donor", value=3.6
            ),
            dict(scope="lineage", group="HSC", metric="ilisi_median", variable="donor", value=9.0),
        ]
    )
    out = ev.baseline_to_rows(base)
    assert out.to_dict("records") == [
        dict(scope="global", metric="ilisi_median", variable="donor", pca_value=1.9, pca_perm=3.6)
    ]


def test_trajectory_rows_skip_cells_without_reference_but_count_coverage_on_all():
    time = np.array([0.0, 1.0, 2.0, 3.0, 9.0])
    ref = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    on_path = np.array([True, True, True, False, False])
    rows = {
        r["metric"]: r for r in ev.trajectory_rows("erythroid", time, ref, on_path, "set_erythroid")
    }
    assert rows["traj_spearman"]["value"] == pytest.approx(1.0)
    assert rows["traj_spearman"]["n_cells"] == 4
    assert rows["traj_coverage"]["value"] == pytest.approx(3 / 5)
