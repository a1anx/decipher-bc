"""Invariants of the bifurcating simulated-data pipeline.

Covers `Simulated Data/Simulated Data Sweep Pipeline/0918_simulated_data_pipeline_bifurcation.py`.
No training -- every test here runs in seconds. The end-to-end training smoke test
(`train_and_compute_rho_r2_bifurcation`) is deliberately NOT here; it takes minutes and needs the
model packages installed.

The two invariants worth protecting:

1. `branching_t=None` reproduces `0801_simulated_data_pipeline` byte-for-byte. This is what makes
   the bifurcation sweep comparable to the existing straight-line sweeps -- if an edit to the
   generator perturbs the batch machinery, this catches it.
2. The batch shift is a rigid translation of the whole Y, identical on both arms. If that ever
   stops holding, "bifurcation only" is no longer true and the sweep is confounded.
"""

import importlib
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Simulated Data",
        "Simulated Data Sweep Pipeline",
    ),
)

aug1 = importlib.import_module("0801_simulated_data_pipeline")
bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")

BRANCHING_T = 0.3
BASE_KW = dict(
    n_batches=3,
    n_samples=200,
    n_genes=50,
    n_z_dims=3,
    biological_sigma=0.1,
    seed=3,
)


def _latent_z(adata, n_z_dims=3):
    return np.column_stack([adata.obs[f"latent_z{i}"].values for i in range(n_z_dims)])


def _gt_branch_asw(adata, n_z_dims=3, center_per_batch=False):
    """Ground-truth branch silhouette in latent_z, post-fork cells only."""
    Z = _latent_z(adata, n_z_dims)
    if center_per_batch:
        Z = Z.copy()
        for b in adata.obs["batch"].unique():
            m = (adata.obs["batch"] == b).values
            Z[m] -= Z[m].mean(axis=0)
    post = (adata.obs["latent_t"] > BRANCHING_T).values
    return float(silhouette_score(Z[post], adata.obs["branch_id"].values[post]))


# --------------------------------------------------------------------------- 1. control path


@pytest.fixture(scope="module", params=["z", "genes"])
def control_pair(request):
    """(aug1 output, sep18 output with branching off) at identical arguments."""
    kw = dict(BASE_KW, shift_sigma=0.5, batch_mode=request.param)
    return (
        aug1.shift_magnitudes_multivariate_from_normal(**kw),
        bif.shift_magnitudes_multivariate_bifurcation(**kw, branching_t=None),
        request.param,
    )


def test_control_path_matrix_and_obs_are_identical(control_pair):
    a, b, _ = control_pair
    assert np.array_equal(np.asarray(a.X), np.asarray(b.X))
    assert np.array_equal(a.obs["latent_t"].values, b.obs["latent_t"].values)
    assert np.array_equal(a.obs["batch"].values.astype(str), b.obs["batch"].values.astype(str))
    for i in range(BASE_KW["n_z_dims"]):
        assert np.array_equal(a.obs[f"latent_z{i}"].values, b.obs[f"latent_z{i}"].values)


def test_control_path_batch_uns_is_identical(control_pair):
    a, b, batch_mode = control_pair
    key = "shift_matrix" if batch_mode == "z" else "gene_shift_matrix"
    assert np.array_equal(a.uns[key], b.uns[key])
    if batch_mode == "genes":
        assert np.array_equal(
            np.asarray(a.layers["counts_nobatch"]), np.asarray(b.layers["counts_nobatch"])
        )


def test_control_path_has_no_branch_fields(control_pair):
    _, b, _ = control_pair
    assert not any(k in b.uns for k in ("w_branch", "branching_t", "branch_prob", "branch_scale"))
    assert not any(c in b.obs.columns for c in ("branch_id", "branch_coord"))


def test_concat_uns_keys_dropped_by_aug1_are_restored(control_pair):
    """ad.concat(merge="same") drops array/list uns keys when n_batches > 1.

    This is a pre-existing aug1 bug, left unpatched there on purpose (see
    design-decisions/0918_bifurcation-design-decisions.md Q4). The sep18 file restores both keys,
    so its uns is a strict SUPERSET of aug1's -- the one intentional difference in the control
    path. If aug1 is ever fixed too, this test is what will tell you.
    """
    a, b, _ = control_pair
    assert "w_bio" not in a.uns and "latent_z_names" not in a.uns
    assert set(b.uns) - set(a.uns) == {"w_bio", "latent_z_names"}

    w_bio_used = aug1.simulate_multivariate(
        n_samples=10,
        n_genes=BASE_KW["n_genes"],
        n_z_dims=BASE_KW["n_z_dims"],
        seed=BASE_KW["seed"],
        cell_seed=1,
    ).uns["w_bio"]
    assert np.array_equal(b.uns["w_bio"], w_bio_used)
    # and it stays recoverable from the seed alone, which is why aug1 needs no fix
    assert np.array_equal(
        w_bio_used,
        np.random.default_rng(BASE_KW["seed"]).standard_normal((1, BASE_KW["n_z_dims"])),
    )


# --------------------------------------------------------------------------- 2. the Y exists


@pytest.fixture(scope="module")
def forked():
    return bif.shift_magnitudes_multivariate_bifurcation(
        n_batches=1,
        shift_sigma=0.0,
        n_samples=500,
        n_genes=200,
        n_z_dims=3,
        biological_sigma=0.1,
        seed=3,
        branching_t=BRANCHING_T,
        branch_prob=0.5,
    )


def test_trunk_is_shared_and_arms_are_populated(forked):
    t = forked.obs["latent_t"].values
    b_id = forked.obs["branch_id"].values
    assert (b_id[t <= BRANCHING_T] == 0).all()
    assert (b_id[t > BRANCHING_T] != 0).all()
    assert abs((b_id[t > BRANCHING_T] == 1).mean() - 0.5) < 0.06


def test_branch_coord_is_signed_by_arm_and_zero_on_the_trunk(forked):
    t = forked.obs["latent_t"].values
    c = forked.obs["branch_coord"].values
    b_id = forked.obs["branch_id"].values
    assert (c[t <= BRANCHING_T] == 0).all()
    assert (np.sign(c[t > BRANCHING_T]) == b_id[t > BRANCHING_T]).all()


def test_true_v_is_two_dimensional(forked):
    """Matches DecipherConfig.dim_v = 2, so true v and decipher_v are directly comparable."""
    assert forked.obsm["latent_v"].shape == (forked.n_obs, 2)


def test_branch_direction_is_orthogonal_to_the_trunk(forked):
    """Without this, a bad seed collapses the Y into a line and the fork silently vanishes."""
    w_bio, w_branch = forked.uns["w_bio"], forked.uns["w_branch"]
    cos = (w_bio @ w_branch.T).item() / (np.linalg.norm(w_bio) * np.linalg.norm(w_branch))
    assert abs(cos) < 1e-10


def test_fork_is_separable_in_the_ground_truth(forked):
    """Sanity floor, not a ceiling for branch_asw.

    branch_asw is measured in decipher_v (2-D); this is latent_z (3-D). Silhouette is not
    comparable across spaces of different dimension, so a trained model can and does score above
    this number. All this asserts is that the fork is present in the data at all.
    """
    assert _gt_branch_asw(forked) > 0.25


def test_bifurcation_needs_a_second_dimension():
    with pytest.raises(ValueError, match="second direction"):
        bif.simulate_multivariate_bifurcation(n_z_dims=1, branching_t=BRANCHING_T)


# --------------------------------------------------------------------------- 3. rigid translation


@pytest.fixture(scope="module")
def shift_pair():
    """Same cells, with and without a batch offset.

    The shift is passed into the generator rather than drawn from the cell stream, so these two
    runs contain identical cells and their difference IS the offset.
    """
    kw = dict(
        n_batches=5,
        n_samples=500,
        n_genes=200,
        n_z_dims=3,
        biological_sigma=0.1,
        seed=3,
        branching_t=BRANCHING_T,
        branch_prob=0.5,
    )
    return (
        bif.shift_magnitudes_multivariate_bifurcation(shift_sigma=0.0, **kw),
        bif.shift_magnitudes_multivariate_bifurcation(shift_sigma=2.0, **kw),
    )


def test_shift_does_not_touch_the_cell_stream(shift_pair):
    d0, d2 = shift_pair
    assert np.array_equal(d0.obs["latent_t"].values, d2.obs["latent_t"].values)
    assert np.array_equal(d0.obs["branch_id"].values, d2.obs["branch_id"].values)


def test_offset_is_a_rigid_translation_identical_on_both_arms(shift_pair):
    d0, d2 = shift_pair
    offset = _latent_z(d2) - _latent_z(d0)
    for i, b in enumerate(sorted(d2.obs["batch"].unique())):
        m = (d2.obs["batch"] == b).values
        # constant within the batch
        assert offset[m].std(axis=0).max() < 1e-12
        # equal to the documented, sqrt(n_z_dims)-normalized shift
        assert np.allclose(offset[m].mean(axis=0), d2.uns["shift_matrix"][i] / np.sqrt(3))
        # and the same on both arms -- no arm-dependent deformation
        plus = offset[m & (d2.obs["branch_id"].values == 1)].mean(axis=0)
        minus = offset[m & (d2.obs["branch_id"].values == -1)].mean(axis=0)
        assert np.abs(plus - minus).max() < 1e-12


def test_removing_the_offset_restores_the_unshifted_geometry(shift_pair):
    d0, d2 = shift_pair
    assert abs(_gt_branch_asw(d2, center_per_batch=True) - _gt_branch_asw(d0)) < 0.02


def test_the_uncorrected_batch_effect_really_does_obscure_the_fork(shift_pair):
    """If this fails the sweep is pointless -- there would be nothing for a variant to correct."""
    d0, d2 = shift_pair
    assert _gt_branch_asw(d2) < _gt_branch_asw(d0) - 0.05


# --------------------------------------------------------------------------- 4. trajectory split


def _fake_clustered(cluster_arms, n_per=50):
    """Minimal adata carrying the three obs columns branch_cluster_orders reads."""
    rng = np.random.default_rng(0)
    rows = []
    for ci, (cluster, arm) in enumerate(cluster_arms.items()):
        for _ in range(n_per):
            rows.append(
                {
                    "decipher_clusters": cluster,
                    "branch_id": arm,
                    "latent_t": ci / len(cluster_arms) + rng.random() * 0.05,
                }
            )
    obs = pd.DataFrame(rows)
    obs["decipher_clusters"] = obs["decipher_clusters"].astype("category")
    return ad.AnnData(X=np.zeros((len(obs), 2)), obs=obs)


def test_two_arms_give_two_trajectories_sharing_the_trunk():
    orders = bif.branch_cluster_orders(
        _fake_clustered({"c0": 0, "c1": 0, "c2": 1, "c3": -1, "c4": 1, "c5": -1})
    )
    assert set(orders) == {"traj_plus", "traj_minus"}
    assert set(orders["traj_plus"]) & set(orders["traj_minus"]) == {"c0", "c1"}
    assert "c3" not in orders["traj_plus"] and "c2" not in orders["traj_minus"]
    assert orders["traj_plus"] == sorted(orders["traj_plus"])  # ordered by mean latent_t


@pytest.mark.parametrize(
    "cluster_arms, why",
    [
        ({"c0": 0, "c1": 0, "c2": 0, "c3": 0}, "no fork at all (the straight-line control)"),
        ({"c0": 0, "c1": 0, "c2": 1, "c3": 1}, "only one arm survived"),
    ],
)
def test_falls_back_to_a_single_trajectory(cluster_arms, why):
    """At high shift_sigma the fork can wash out. The sweep must degrade, not crash."""
    orders = bif.branch_cluster_orders(_fake_clustered(cluster_arms))
    assert set(orders) == {"trajectory"}, why
    assert orders["trajectory"] == sorted(orders["trajectory"])


# --------------------------------------------------------------- 5. the no-fork (branching off)
#
# The straight-line half of a sweep runs with branching_t=None, which writes NO branch_id column --
# that absence is what keeps the control path byte-for-byte identical to aug1. Everything
# downstream has to treat "no branch_id" as "no fork" rather than raising. Before these were
# guarded, branch_cluster_orders and branch_silhouette both raised KeyError, i.e. half of every
# bifurcation-vs-straight-line sweep died on the first run.


@pytest.fixture(scope="module")
def unforked():
    d = bif.shift_magnitudes_multivariate_bifurcation(
        n_batches=2, shift_sigma=0.5, n_samples=100, n_genes=20, branching_t=None
    )
    d.obs["decipher_clusters"] = pd.Categorical(["c0"] * 100 + ["c1"] * 100)
    return d


def test_no_branch_id_column_when_branching_is_off(unforked):
    assert "branch_id" not in unforked.obs.columns
    assert "branch_coord" not in unforked.obs.columns


def test_cluster_orders_degrades_to_one_trajectory_without_branch_id(unforked):
    orders = bif.branch_cluster_orders(unforked)
    assert set(orders) == {"trajectory"}
    # every valid cluster is kept, ordered by mean latent_t (not by cluster name -- these two
    # clusters are batch 0 vs batch 1, whose mean pseudotimes are in arbitrary order)
    got = orders["trajectory"]
    assert set(got) == {"c0", "c1"}
    means = unforked.obs.groupby("decipher_clusters", observed=True)["latent_t"].mean()
    assert got == sorted(got, key=lambda c: means[c])


def test_branch_silhouette_is_nan_without_a_fork(unforked):
    assert np.isnan(bif.branch_silhouette(unforked, branching_t=0.3))
    # and also when branching_t itself says there is no fork
    assert np.isnan(bif.branch_silhouette(unforked, branching_t=None))


@pytest.mark.parametrize(
    "branching_t, batch_mode, expected",
    [
        (0.3, "z", "bif_z"),
        (0.3, "genes", "bif_genes"),
        (None, "z", "nobif_z"),
        (None, "genes", "nobif_genes"),
    ],
)
def test_data_tag_separates_every_data_arm(branching_t, batch_mode, expected):
    """Output names must distinguish the data, or runs silently overwrite each other."""
    assert bif._data_tag(branching_t, batch_mode) == expected


def test_data_tags_are_all_distinct():
    tags = {bif._data_tag(bt, bm) for bt in (0.3, None) for bm in ("z", "genes")}
    assert len(tags) == 4


def test_trained_h5ad_name_separates_every_swept_axis():
    """Every axis the sweep varies must appear in the output name.

    This has now bitten twice. First it was bifurcation/batch_mode, fixed by _data_tag. Then
    decipher_seed: the grid went from 1 to 3 decipher seeds, and because the h5ad name did not
    encode it, all three wrote to ONE path -- 2 of every 3 results destroyed, last-writer-wins,
    no error raised. The figure name had decipher_seed all along; only the h5ad lacked it.

    Guards the composition rather than an exact string, so reordering the name stays fine but
    dropping an axis does not.
    """
    import itertools

    axes = dict(
        shift_sigma=[0.5, 2.0],
        seed=[3, 4],
        model_tag=["model2", "model5"],
        branching_t=[0.3, None],
        batch_mode=["z", "genes"],
        decipher_seed=[1, 2, 3],
    )

    def name(shift_sigma, seed, model_tag, branching_t, batch_mode, decipher_seed):
        return (
            f"sigma{shift_sigma}_seed{seed}_{model_tag}"
            f"_{bif._data_tag(branching_t, batch_mode)}"
            f"_decipherseed_{decipher_seed}.h5ad"
        )

    combos = list(itertools.product(*axes.values()))
    assert len({name(*c) for c in combos}) == len(
        combos
    ), "two distinct sweep cells share an output filename"

    # specifically: decipher_seed alone must separate two otherwise-identical runs
    base = dict(shift_sigma=2.0, seed=3, model_tag="model2", branching_t=0.3, batch_mode="z")
    assert name(**base, decipher_seed=1) != name(**base, decipher_seed=2)


def test_min_cells_filter_still_applies():
    adata = _fake_clustered({"c0": 0, "c1": 1, "c2": -1})
    keep = ~((adata.obs.decipher_clusters == "c2") & (np.arange(adata.n_obs) % 50 > 4))
    orders = bif.branch_cluster_orders(adata[keep].copy(), min_cells=10)
    assert "c2" not in next(iter(orders.values()))
