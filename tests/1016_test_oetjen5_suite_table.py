"""Pure helpers of `Real Data/Oetjen Bone Marrow/1016_oetjen5_suite_table.py`: stratified (per
cell type) graph iLISI / batch ASW, skipped cell types, cell weighting, scan-folder discovery,
`choose_lambda` and `beats_native`, and the beats_native columns of the mean table. No training;
the stratified metrics build small kNN graphs with scanpy."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1016_oetjen5_suite_table.py"
)
_spec = importlib.util.spec_from_file_location("oetjen5_suite_table", SCRIPT)
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)

TRAJ = suite.TRAJ_CRITERIA
CRITERIA_COLS = [c for c, _, _ in suite.CRITERIA]


def _mixed_type(n: int, offset: float, batches: tuple[str, str], label: str, seed: int):
    """`n` cells on a jittered line at height `offset`, alternating between two batches."""
    rng = np.random.default_rng(seed)
    emb = np.column_stack([np.arange(n, dtype=float), np.full(n, offset)])
    emb += rng.normal(scale=0.05, size=emb.shape)
    batch = np.array([batches[i % 2] for i in range(n)])
    return emb, batch, np.full(n, label)


def _two_types():
    """Types A (batches b1/b2) and B (b3/b4), each mixed within, far apart: globally each
    neighbourhood holds only 2 of the 4 batches."""
    parts = [_mixed_type(60, 0.0, ("b1", "b2"), "A", 0), _mixed_type(60, 1e3, ("b3", "b4"), "B", 1)]
    return tuple(np.concatenate(x) for x in zip(*parts))


def _vspace(emb: np.ndarray, batch: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """A v-space table with v = `emb` and z = `emb` padded with zeros to 10 columns."""
    z = np.column_stack([emb, np.zeros((len(emb), 8))])
    cols = {"sample": batch, "cell_type": labels, "v1": emb[:, 0], "v2": emb[:, 1]}
    cols |= {f"z{i + 1}": z[:, i] for i in range(10)}
    return pd.DataFrame(cols)


def test_stratified_ilisi_high_when_mixed_within_types_but_separated_globally():
    emb, batch, labels = _two_types()
    distances, _ = suite.native.knn_graph(emb)
    global_ilisi = suite.native.graph_ilisi(distances, batch)

    row, skipped = suite.stratified_metrics(_vspace(emb, batch, labels))

    assert global_ilisi < 0.4
    assert sorted(row) == [
        "v_strat_batch_asw",
        "v_strat_ilisi",
        "z_strat_batch_asw",
        "z_strat_ilisi",
    ]
    assert row["v_strat_ilisi"] > 0.9
    assert row["z_strat_ilisi"] == pytest.approx(row["v_strat_ilisi"])
    assert skipped == {}


def test_single_batch_type_is_skipped_and_leaves_the_others_unchanged():
    emb, batch, labels = _two_types()
    single = _mixed_type(30, 2e3, ("b1", "b1"), "C", 2)
    with_c = [np.concatenate([a, b]) for a, b in zip((emb, batch, labels), single)]

    table, skipped = suite.per_type_mixing(*with_c)

    expected, _ = suite.per_type_mixing(emb, batch, labels)
    pd.testing.assert_frame_equal(table, expected)
    assert skipped == {"C": "1 batch"}


@pytest.mark.parametrize(
    ("n_cells", "expected_skipped"),
    [(15, {"D": "15 cells <= k=15"}), (16, {})],
)
def test_type_with_at_most_k_cells_is_skipped(n_cells, expected_skipped):
    batch = np.array([f"b{i % 2}" for i in range(n_cells)])

    kept, skipped = suite.split_types(batch, np.full(n_cells, "D"))

    assert skipped == expected_skipped
    assert kept == ([] if expected_skipped else ["D"])


def test_cell_weighted_mean_weights_each_type_by_its_cells():
    per_type = pd.DataFrame(
        {"cell_type": ["A", "B"], "n_cells": [30, 10], "graph_ilisi": [1.0, 0.0]}
    )

    assert suite.cell_weighted_mean(per_type, "graph_ilisi") == pytest.approx(0.75)


def test_cell_weighted_mean_raises_without_usable_types():
    empty = pd.DataFrame(columns=["cell_type", "n_cells", "graph_ilisi"])

    with pytest.raises(ValueError, match="no cell type"):
        suite.cell_weighted_mean(empty, "graph_ilisi")


def test_scan_dirs_finds_scan_folders_by_name_and_preset(tmp_path):
    for name in [
        "1016_oetjen5_d3_native_mmd_li1",
        "1016_oetjen5_d3_native_mmd_li0p1",
        "1016_oetjen5_d3_model8_lc10",
        "1016_oetjen5_d3_model2",
        "1016_oetjen5_d3_native_mmd",
    ]:
        (tmp_path / name).mkdir()

    found = [(p, t, lam, d.name) for p, t, lam, d in suite.scan_dirs(tmp_path)]
    only_mmd = [t for _, t, _, _ in suite.scan_dirs(tmp_path, "native_mmd")]

    assert found == [
        ("model8", "lc10", 10.0, "1016_oetjen5_d3_model8_lc10"),
        ("native_mmd", "li0p1", 0.1, "1016_oetjen5_d3_native_mmd_li0p1"),
        ("native_mmd", "li1", 1.0, "1016_oetjen5_d3_native_mmd_li1"),
    ]
    assert only_mmd == ["li0p1", "li1"]


# native_d3 bounds: v_strat_ilisi max 0.24, trajectory columns min 0.5, best val NLL max 1004
NATIVE = pd.DataFrame(
    {
        "v_strat_ilisi": [0.20, 0.22, 0.24],
        **{col: [0.5, 0.75, 1.0] for col in TRAJ},
        "best_val_nll": [1000.0, 1002.0, 1004.0],
    }
)


def _scan_row(tag: str, lam: float, **overrides: float) -> dict:
    """A seed-0 scan row passing every criterion unless overridden."""
    row = {"tag": tag, "lambda": lam, "v_strat_ilisi": 0.3, **{c: 0.75 for c in TRAJ}}
    return row | {"best_val_nll": 1000.0} | overrides


def test_choose_lambda_picks_highest_ilisi_among_passing():
    scan = pd.DataFrame(
        [
            _scan_row("li0p1", 0.1, v_strat_ilisi=0.30),
            # trajectory and NLL exactly on native's bounds still pass
            _scan_row(
                "li1", 1.0, v_strat_ilisi=0.35, v_traj_cons_erythroid=0.5, best_val_nll=1004.0
            ),
            _scan_row("li10", 10.0, v_strat_ilisi=0.50, best_val_nll=1010.0),
        ]
    )

    _, tag, flag = suite.choose_lambda(scan, NATIVE)

    assert (tag, flag) == ("li1", "pass")


def test_choose_lambda_ties_go_to_the_smaller_lambda():
    scan = pd.DataFrame([_scan_row("li1", 1.0), _scan_row("li0p1", 0.1)])

    _, tag, flag = suite.choose_lambda(scan, NATIVE)

    assert (tag, flag) == ("li0p1", "pass")


def test_choose_lambda_none_pass_picks_largest_lambda_keeping_trajectory():
    no_mix = {"v_strat_ilisi": 0.2}
    scan = pd.DataFrame(
        [
            _scan_row("li0p1", 0.1, **no_mix),
            _scan_row("li1", 1.0, **no_mix),
            _scan_row("li10", 10.0, **no_mix, decipher_time_cons_myeloid=0.25),
        ]
    )

    _, tag, flag = suite.choose_lambda(scan, NATIVE)

    assert (tag, flag) == ("li1", "none-pass-traj-kept")


def test_choose_lambda_none_keeping_trajectory_picks_smallest_lambda():
    scan = pd.DataFrame(
        [
            _scan_row("li10", 10.0, v_traj_cons_erythroid=0.25),
            _scan_row("li0p1", 0.1, v_traj_cons_myeloid=0.25),
            _scan_row("li1", 1.0, decipher_time_cons_erythroid=0.25),
        ]
    )

    _, tag, flag = suite.choose_lambda(scan, NATIVE)

    assert (tag, flag) == ("li0p1", "none-pass-smallest")


def test_choose_lambda_table_lists_each_criterion_value_bound_and_pass():
    scan = pd.DataFrame([_scan_row("li1", 1.0, best_val_nll=1005.0)])

    table, _, _ = suite.choose_lambda(scan, NATIVE)

    expected = pd.DataFrame(
        {
            "tag": "li1",
            "lambda": 1.0,
            "criterion": CRITERIA_COLS,
            "value": [0.3, 0.75, 0.75, 0.75, 0.75, 1005.0],
            "rule": ["> native max", *[">= native min"] * 4, "<= native max"],
            "bound": [0.24, 0.5, 0.5, 0.5, 0.5, 1004.0],
            "pass": [True, True, True, True, True, False],
        }
    )
    pd.testing.assert_frame_equal(table, expected)


def test_choose_lambda_rejects_duplicate_tags():
    scan = pd.DataFrame([_scan_row("li1", 1.0), _scan_row("li1", 1.0)])

    with pytest.raises(ValueError, match="one row per tag"):
        suite.choose_lambda(scan, NATIVE)


def _arm(**overrides: list[float]) -> pd.DataFrame:
    """Three seed rows of an arm passing every beats_native criterion unless overridden; the
    trajectory and NLL means sit exactly on native's bounds (min 0.5, max 1004)."""
    cols = {"v_strat_ilisi": [0.25, 0.3, 0.35], **{c: [0.25, 0.5, 0.75] for c in TRAJ}}
    return pd.DataFrame(cols | {"best_val_nll": [1002.0, 1004.0, 1006.0]} | overrides)


def test_beats_native_when_every_criterion_holds():
    assert suite.beats_native(_arm(), NATIVE) == {
        **{f"beats_{c}": True for c in CRITERIA_COLS},
        "beats_native": True,
    }


@pytest.mark.parametrize(
    ("column", "values"),
    [
        # mean 0.41 is high, but the lowest seed does not exceed native max 0.24
        ("v_strat_ilisi", [0.24, 0.5, 0.5]),
        # best seed 0.75 is above native min 0.5, but the mean 0.458 is not
        *[(col, [0.25, 0.375, 0.75]) for col in TRAJ],
        # best seed 1000 is below native max 1004, but the mean 1004.33 is not
        ("best_val_nll", [1000.0, 1000.0, 1013.0]),
    ],
)
def test_beats_native_fails_on_a_single_criterion(column, values):
    result = suite.beats_native(_arm(**{column: values}), NATIVE)

    assert result == {
        **{f"beats_{c}": c != column for c in CRITERIA_COLS},
        "beats_native": False,
    }


def test_build_mean_adds_beats_native_per_model_when_native_d3_present():
    def rows(model: str, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(model=model, seed=[0, 1, 2], sanity_pass=True)

    per_seed = pd.concat([rows(suite.NATIVE_REF, NATIVE), rows("model2", _arm())])

    mean = suite.build_mean(per_seed)

    assert mean["model"].tolist() == [suite.NATIVE_REF, "model2"]
    assert mean["beats_native"].tolist() == [False, True]
    assert mean["beats_v_strat_ilisi"].tolist() == [False, True]


# ---- batch-fu P02: new arms, tagged finals, --min-lambda ----


def test_new_final_arms_are_in_the_model_order_with_their_folders():
    new = ["model5rz", "model5r0", "model7iw", "model7im", "model5r_mmd", "model8r"]
    assert [m in suite.MODEL_ORDER for m in new] == [True] * 6
    assert [suite.SOURCES[m] for m in new] == [f"1016_oetjen5_d3_{m}" for m in new]


def test_tagged_final_dirs_name_arms_preset_underscore_tag_and_are_not_scans(tmp_path):
    names = [
        "1016_oetjen5_d3_model5r_mmd_li100_final",
        "1016_oetjen5_d3_model8r_lc1000_final",
        "1016_oetjen5_d3_model8r_lc10",
        "1016_oetjen5_d3_model8r",
    ]
    for name in names:
        (tmp_path / name).mkdir()

    tagged = suite.tagged_final_dirs(tmp_path)
    scans = [(p, t) for p, t, _, _ in suite.scan_dirs(tmp_path)]

    assert tagged == {
        "model5r_mmd_li100": "1016_oetjen5_d3_model5r_mmd_li100_final",
        "model8r_lc1000": "1016_oetjen5_d3_model8r_lc1000_final",
    }
    assert scans == [("model8r", "lc10")]


def test_scan_dirs_min_lambda_keeps_only_lambdas_at_or_above_it(tmp_path):
    for tag in ["li1", "li10", "li100", "li1000"]:
        (tmp_path / f"1016_oetjen5_d3_model5r_mmd_{tag}").mkdir()

    kept = [t for _, t, _, _ in suite.scan_dirs(tmp_path, "model5r_mmd", min_lambda=100)]
    everything = [t for _, t, _, _ in suite.scan_dirs(tmp_path, "model5r_mmd")]

    assert (kept, everything) == (["li100", "li1000"], ["li1", "li10", "li100", "li1000"])


def test_model5_d3_is_a_final_arm_next_to_the_unchanged_m5e_model5():
    assert (
        suite.SOURCES["model5_d3"],
        suite.SOURCES["model5"],
        "model5_d3" in suite.MODEL_ORDER,
    ) == (
        "1016_oetjen5_d3_model5",
        "1016_oetjen5_m5e",
        True,
    )
