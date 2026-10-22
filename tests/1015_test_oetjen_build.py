"""Tests for the Oetjen GEO parsing helpers (no download)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1015_build_oetjen_h5ad.py"
)
_spec = importlib.util.spec_from_file_location("oetjen_build", _PATH)
oetjen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oetjen)

GEO_TEXT = """^SAMPLE = GSM3396161
!Sample_title = Bone marrow A
!Sample_characteristics_ch1 = age: 59 years old
^SAMPLE = GSM3396180
!Sample_title = Bone marrow Sk1
!Sample_characteristics_ch1 = age: 55 years old
"""


@pytest.mark.parametrize(
    "sample,donor", [("A", "A"), ("C1", "C"), ("Ck", "C"), ("Sk2", "S"), ("W", "W")]
)
def test_donor_of_strips_replicate_suffix(sample, donor):
    assert oetjen.donor_of(sample) == donor


@pytest.mark.parametrize("sample", ["", "a", "AB", "C12"])
def test_donor_of_rejects_unknown_ids(sample):
    with pytest.raises(ValueError):
        oetjen.donor_of(sample)


def test_parse_geo_samples_reads_gsm_title_and_age():
    expected = pd.DataFrame(
        {"gsm": ["GSM3396161", "GSM3396180"], "sample": ["A", "Sk1"], "age": [59, 55]}
    )
    pd.testing.assert_frame_equal(oetjen.parse_geo_samples(GEO_TEXT), expected)


def test_parse_geo_samples_raises_when_age_missing():
    with pytest.raises(ValueError, match="GSM1"):
        oetjen.parse_geo_samples("^SAMPLE = GSM1\n!Sample_title = Bone marrow A\n")


_INSPECT = _PATH.with_name("1015_oetjen_inspect.py")
_spec2 = importlib.util.spec_from_file_location("oetjen_inspect", _INSPECT)
inspect_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(inspect_mod)


def _two_blobs(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(0, 0.1, (n, 2)), rng.normal(10, 0.1, (n, 2))])


def test_transfer_labels_takes_majority_of_nearest_reference():
    ref = _two_blobs()
    labels = np.array(["x"] * 40 + ["y"] * 40)
    pred, conf = inspect_mod.transfer_labels(ref, labels, np.array([[0.0, 0.0], [10.0, 10.0]]), k=5)
    np.testing.assert_array_equal(pred, ["x", "y"])
    np.testing.assert_array_equal(conf, [1.0, 1.0])


def test_knn_mixing_flags_separated_samples():
    emb = _two_blobs()
    sample = np.array(["s1"] * 40 + ["s2"] * 40)
    mix = inspect_mod.knn_mixing(emb, sample, within=np.zeros(80), k=5)
    expected = pd.DataFrame([[2.0, 0.0], [0.0, 2.0]], index=["s1", "s2"], columns=["s1", "s2"])
    pd.testing.assert_frame_equal(mix, expected)


def test_knn_mixing_within_groups_ignores_composition():
    # blobs are cell types; both samples are evenly mixed inside each type
    emb = _two_blobs()
    sample = np.tile(["s1", "s2"], 40)
    celltype = np.array(["a"] * 40 + ["b"] * 40)
    mix = inspect_mod.knn_mixing(emb, sample, within=celltype, k=5)
    assert np.allclose(mix.values, 1.0, atol=0.35)


def test_knn_mixing_rejects_length_mismatch():
    with pytest.raises(ValueError):
        inspect_mod.knn_mixing(np.zeros((4, 2)), np.array(["a"] * 3), np.zeros(4), k=1)
