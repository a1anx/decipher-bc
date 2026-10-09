"""`rotate_v` in the BMMC v-space deck script: one orientation for every run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Healthy Human Bone Marrow Mononuclear Cells"
    / "1008_bmmc_vspace_deck.py"
)
_spec = importlib.util.spec_from_file_location("bmmc_vspace_deck", SCRIPT)
deck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deck)

MYELOID = ["HSC", "G/M prog", "CD14+ Mono", "CD16+ Mono"]


def make_y(theta: float, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    """Myeloid arm along +x, erythroid arm along +y, then rotated by theta (and mirrored)."""
    rng = np.random.default_rng(0)
    myeloid = np.repeat(np.arange(4.0), 50)
    v_my = np.c_[myeloid, np.zeros_like(myeloid)]
    v_ery = np.c_[np.zeros(50), np.linspace(1, 4, 50)]
    v = np.vstack([v_my, v_ery]) + rng.normal(scale=0.05, size=(250, 2))
    cell_type = np.array([MYELOID[int(i)] for i in myeloid] + ["Erythroblast"] * 50)
    c, s = np.cos(theta), np.sin(theta)
    v = v @ np.array([[c, s], [-s, c]])
    if mirror:
        v = v * np.array([1.0, -1.0])
    return v, cell_type


@pytest.mark.parametrize("theta, mirror", [(0.0, False), (2.0, False), (4.0, True)])
def test_rotate_v_puts_myeloid_on_v1_and_erythroid_up(theta, mirror):
    v, cell_type = make_y(theta, mirror)
    w = v @ deck.rotate_v(v, cell_type)
    my = cell_type != "Erythroblast"
    rank = np.array([MYELOID.index(c) for c in cell_type[my]])
    assert np.corrcoef(w[my, 0], rank)[1, 0] > 0.99
    assert w[~my, 1].mean() > 1.0


def test_rotate_v_is_orthogonal():
    v, cell_type = make_y(1.0, True)
    r = deck.rotate_v(v, cell_type)
    np.testing.assert_allclose(r @ r.T, np.eye(2), atol=1e-12)


def test_rotate_v_raises_without_two_myeloid_types():
    v = np.zeros((3, 2))
    with pytest.raises(ValueError, match="at least two"):
        deck.rotate_v(v, np.array(["HSC", "HSC", "Erythroblast"]))
