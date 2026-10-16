"""Pure helpers of `Real Data/Oetjen Bone Marrow/1015_oetjen6_native_decipher.py`: branch columns,
per-branch decipher time, the scIB metrics (batch ASW, graph connectivity, graph iLISI, DPT root,
trajectory conservation), reconstruction R^2, trajectory sanity check. No training (the script
imports decipher/wandb/scanpy only inside functions not tested here)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1015_oetjen6_native_decipher.py"
)
_spec = importlib.util.spec_from_file_location("oetjen6_native", SCRIPT)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)

PASSING_MAJORITIES = [
    "HSPCs",
    "Erythrocytes",
    "CD14+ Monocytes",
]  # majorities: start, ery end, myeloid end


def _counts(majorities: list[str]) -> pd.DataFrame:
    """Cluster x cell_type counts: cluster str(i) has 10 cells of majorities[i], 1 of others."""
    types = ["HSPCs", "Erythroid progenitors", "Erythrocytes", "CD14+ Monocytes", "CD16+ Monocytes"]
    rows = {str(i): {t: 10 if t == m else 1 for t in types} for i, m in enumerate(majorities)}
    return pd.DataFrame.from_dict(rows, orient="index")


PATHS = {"erythroid": ["0", "1"], "myeloid": ["0", "2"]}


def test_branch_columns_keep_hspc_cluster_in_both_and_exclude_other_branch():
    obs = pd.DataFrame(
        {
            "lineage": ["HSC"] * 4 + ["erythroid"] * 3 + ["myeloid"] * 3,
            "cluster": ["hspc"] * 4 + ["ery"] * 3 + ["mye"] * 3,
        }
    )
    branches = native.branch_columns(obs["lineage"])
    share = branches.groupby(obs["cluster"]).mean()
    expected = pd.DataFrame(
        {"erythroid_branch": [1.0, 1.0, 0.0], "myeloid_branch": [0.0, 1.0, 1.0]},
        index=pd.Index(["ery", "hspc", "mye"], name="cluster"),
    )
    pd.testing.assert_frame_equal(share, expected)


def test_reconstruction_r2_is_one_for_perfect_reconstruction():
    x = np.array([[0, 1, 5], [2, 3, 0], [4, 0, 1]], dtype=float)
    r2 = native.reconstruction_r2(x, x.copy())
    assert r2["r2_overall"] == pytest.approx(1.0)
    np.testing.assert_allclose(r2["r2_per_gene"], [1.0, 1.0, 1.0])


def test_reconstruction_r2_zero_variance_gene_is_nan():
    x = np.array([[0, 2, 5], [2, 2, 0], [4, 2, 1]], dtype=float)
    r2 = native.reconstruction_r2(x, x + 1)
    assert np.isnan(r2["r2_per_gene"][1])
    assert np.isfinite(r2["r2_per_gene"][[0, 2]]).all()


def test_trajectory_sanity_passes_on_expected_majorities():
    result = native.trajectory_sanity(PATHS, _counts(PASSING_MAJORITIES))
    assert result == {
        "erythroid_start": {"cluster": "0", "majority": "HSPCs", "pass": True},
        "erythroid_end": {"cluster": "1", "majority": "Erythrocytes", "pass": True},
        "myeloid_start": {"cluster": "0", "majority": "HSPCs", "pass": True},
        "myeloid_end": {"cluster": "2", "majority": "CD14+ Monocytes", "pass": True},
        "pass": True,
    }


@pytest.mark.parametrize(
    "majorities, failing",
    [
        (["Erythroid progenitors", "Erythrocytes", "CD14+ Monocytes"], "erythroid_start"),
        (["HSPCs", "CD16+ Monocytes", "CD14+ Monocytes"], "erythroid_end"),
        (["HSPCs", "Erythrocytes", "Erythroid progenitors"], "myeloid_end"),
    ],
)
def test_trajectory_sanity_fails_on_wrong_majority(majorities, failing):
    result = native.trajectory_sanity(PATHS, _counts(majorities))
    assert result["pass"] is False
    assert result[failing]["pass"] is False


def _graph(n: int, edges: list[tuple[int, int, float]]) -> sparse.csr_matrix:
    """Symmetric weighted graph on n nodes from (i, j, weight) edges."""
    i, j, w = zip(*edges)
    g = sparse.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    return g.maximum(g.T)


def test_branch_time_regresses_branch_cells_and_leaves_others_nan():
    points = np.column_stack([np.arange(20.0), np.zeros(20)])
    v = np.array([[9.5, 0.0], [3.0, 0.0]])
    out = native.branch_time(v, points, np.arange(20.0), np.array([True, False]))
    np.testing.assert_array_equal(out, [9.5, np.nan])  # mean time of points 5..14


def test_batch_asw_is_zero_for_separated_batches():
    emb = np.array([[0.0], [0.1], [10.0], [10.1]])
    assert native.batch_asw(emb, np.array([0, 0, 1, 1]), np.zeros(4)) == pytest.approx(
        0.0, abs=0.02
    )


def test_batch_asw_is_one_minus_one_over_n_when_batches_coincide():
    n = 5
    emb = np.tile(np.arange(n, dtype=float), 2)[:, None]
    batch = np.repeat([0, 1], n)
    assert native.batch_asw(emb, batch, np.zeros(2 * n)) == pytest.approx(1 - 1 / n)


def test_batch_asw_raises_for_label_with_one_batch():
    with pytest.raises(ValueError, match="batches"):
        native.batch_asw(np.arange(4.0)[:, None], np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]))


def test_graph_connectivity_averages_largest_component_share_per_label():
    # label a: 0-1-2-3 connected; label b: 4-5 and 6-7 apart (the 3-4 edge crosses labels)
    graph = _graph(8, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (6, 7, 1)])
    labels = np.array(list("aaaabbbb"))
    assert native.graph_connectivity(graph, labels) == pytest.approx((1 + 0.5) / 2)


def test_graph_neighbors_orders_by_shortest_path_and_pads_unreachable():
    graph = _graph(5, [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)])  # node 4 isolated
    dist, idx = native.graph_neighbors(graph, n_neighbors=2)
    np.testing.assert_array_equal(dist[[0, 3]], [[1, 3], [3, 5]])
    np.testing.assert_array_equal(idx[[0, 3]], [[1, 2], [2, 1]])
    np.testing.assert_array_equal(idx[4], [-1, -1])


@pytest.mark.parametrize("labels, expected", [([0, 0, 1], 2.0), ([0, 0, 0], 1.0)])
def test_lisi_counts_effective_labels_among_equidistant_neighbours(labels, expected):
    # cell 0's two neighbours (cells 1, 2) are equidistant, so their weights are equal
    dist, idx = np.array([[1.0, 1.0]]), np.array([[1, 2]])
    out = native.lisi(dist, idx, np.array(labels), perplexity=2)
    np.testing.assert_allclose(out, [expected])


def test_lisi_raises_for_cell_without_neighbours():
    with pytest.raises(ValueError, match="no neighbours"):
        native.lisi(np.array([[np.inf]]), np.array([[-1]]), np.array([0, 1]), perplexity=2)


def test_graph_ilisi_is_zero_when_each_batch_is_its_own_component():
    graph = _graph(6, [(0, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 1)])
    assert native.graph_ilisi(graph, np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(0.0)


def test_graph_ilisi_raises_for_one_batch():
    with pytest.raises(ValueError, match="2 batches"):
        native.graph_ilisi(_graph(2, [(0, 1, 1)]), np.zeros(2))


@pytest.mark.parametrize("sign", [1, -1])
def test_dpt_root_picks_extreme_root_cell_on_most_separating_component(sign):
    # component 1 puts the root cells (0-2) at one end (+ or -); cell 2 is the most extreme there.
    # Component 0 barely separates them.
    dcs = np.array([[0.1, 5.0], [0.0, 6.0], [0.2, 7.0], [0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]) * [
        1,
        sign,
    ]
    is_root = np.array([True, True, True, False, False, False])
    assert native.dpt_root(dcs, is_root) == 2


def test_dpt_root_raises_without_root_cells():
    with pytest.raises(ValueError, match="root type"):
        native.dpt_root(np.zeros((3, 2)), np.zeros(3, dtype=bool))


@pytest.mark.parametrize(
    "pseudotime, expected",
    [
        ([1.0, 2.0, 3.0, 4.0], 1.0),
        ([4.0, 3.0, 2.0, 1.0], 0.0),
        ([1.0, np.nan, np.nan, 4.0], 0.0),  # fewer than 3 shared cells: not computable
    ],
)
def test_trajectory_conservation_rescales_spearman(pseudotime, expected):
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    assert native.trajectory_conservation(np.array(pseudotime), reference) == pytest.approx(
        expected
    )
