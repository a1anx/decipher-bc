"""`Real Data/Oetjen Bone Marrow/1017_oetjen5_paired_tests.py`: exact sign-flip and sign tests,
seed pairing (unpaired seeds listed, < 3 paired refused), the mean-vs-mean gate at δ, and arm
folder discovery. No training."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1017_oetjen5_paired_tests.py"
)
_spec = importlib.util.spec_from_file_location("oetjen5_paired_tests", SCRIPT)
paired = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paired)

TRAJ = paired.suite.TRAJ_CRITERIA


def _rows(seeds: list[int], value: float = 0.5, **cols: list[float]) -> pd.DataFrame:
    """Per-seed table with every METRICS column at `value`, overridden per column by `cols`."""
    data = {"seed": seeds, **{m: [value] * len(seeds) for m in paired.METRICS}}
    return pd.DataFrame({**data, **cols})


def test_sign_flip_p_on_hand_checked_vector():
    # |sum| of the 16 flips of (1, 2, 3, -1) is >= 5 for 7, 5, 5 and their negatives: 6/16
    assert paired.sign_flip_p(np.array([1.0, 2.0, 3.0, -1.0])) == 6 / 16


def test_sign_flip_p_all_positive_n10_is_two_over_1024():
    assert paired.sign_flip_p(np.arange(1.0, 11.0)) == 2 / 1024


def test_sign_flip_p_counts_float_ties_as_extreme():
    # |sum| >= 0.2 for ±(0.1+0.2+0.1), ±(0.1+0.2-0.1), ±(-0.1+0.2+0.1); the last two tie only
    # up to rounding, so an exact >= would give 4/8
    assert paired.sign_flip_p(np.array([0.1, 0.2, -0.1])) == 6 / 8


def test_sign_test_p_drops_zeros_on_hand_case():
    # nonzero n = 6, one negative: 2 * (C(6,0) + C(6,1)) / 2**6
    assert paired.sign_test_p(np.array([1.0, 2.0, 3.0, 4.0, 0.0, -1.0, 5.0])) == 14 / 64


def test_sign_test_p_all_zero_is_one():
    assert paired.sign_test_p(np.zeros(4)) == 1.0


def test_compare_pair_refuses_fewer_than_three_paired_seeds_and_lists_them():
    with pytest.raises(ValueError, match=r"paired \[0, 1\], arm only \[5\], ref only \[2\]"):
        paired.compare_pair(_rows([0, 1, 5]), _rows([0, 1, 2]), ["v_strat_ilisi"])


def test_compare_pair_lists_unpaired_seeds_and_stats():
    arm = _rows([0, 1, 2, 3], v_strat_ilisi=[0.6, 0.7, 0.4, 9.0])

    table = paired.compare_pair(arm, _rows([0, 1, 2]), ["v_strat_ilisi"])

    expected = {
        "metric": "v_strat_ilisi",
        "n": 3,
        "n_pos": 2,
        "n_neg": 1,
        "paired_seeds": "0 1 2",
        "arm_only_seeds": "3",
        "ref_only_seeds": "",
    }
    assert table.iloc[0][list(expected)].to_dict() == expected
    assert table.iloc[0][["mean_d", "min_d", "max_d"]].tolist() == pytest.approx(
        [0.2 / 3, -0.1, 0.2]
    )


def test_run_tests_lists_underpaired_pair_as_skipped():
    rows = {paired.NATIVE_REF: _rows([0, 1, 2]), "model5r": _rows([0, 7, 8])}

    results, gates, skipped = paired.run_tests(rows, [("model5r", paired.NATIVE_REF)])

    assert results.empty and gates.empty
    assert skipped == [
        "model5r vs native_d3: 1 paired seeds < 3 (paired [0], arm only [7, 8], ref only [1, 2])"
    ]


@pytest.mark.parametrize("arm_value, passes", [(0.45, True), (0.4499, False)])
def test_mean_rule_at_delta_boundary(arm_value, passes):
    arm = _rows([0, 1, 2], **{TRAJ[0]: [arm_value] * 3})

    out = paired.mean_rule(arm, _rows([0, 1, 2], value=0.5))

    assert out == {f"mean_{c}": (passes if c == TRAJ[0] else True) for c in TRAJ} | {
        "mean_rule": passes
    }


def test_arm_folders_finds_tagged_finals_not_scans_and_lists_absent(tmp_path):
    for name in ("1016_oetjen5_d3_native", "1016_oetjen5_d3_model5r_mmd_li1000_final"):
        (tmp_path / name).mkdir()
    (tmp_path / "1016_oetjen5_d3_model8r_lc100").mkdir()

    present, absent = paired.arm_folders(tmp_path)

    assert sorted(present) == ["model5r_mmd_li1000", "native_d3"]
    assert "model8r" in absent and "native_d3" not in absent


def test_seeds_present_reads_meta_files(tmp_path):
    for seed in (0, 2, 10):
        (tmp_path / f"seed{seed}_meta.json").write_text("{}")
    (tmp_path / "seed1_loss.csv").write_text("")

    assert paired.seeds_present(tmp_path) == (0, 2, 10)
