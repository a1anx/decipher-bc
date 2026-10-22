"""Tests for the shared BMMC h5ad builder, on a tiny synthetic AnnData (no real data)."""

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

_PATH = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Healthy Human Bone Marrow Mononuclear Cells"
    / "1008_bmmc_build_shared.py"
)
_spec = importlib.util.spec_from_file_location("bmmc_build_shared", _PATH)
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)

LINEAGE_MAP = pd.DataFrame(
    {
        "cell_type": ["HSC", "Ery", "Mono", "TCell"],
        "lineage": ["HSC", "erythroid", "myeloid", "exclude"],
        "rank": [0, 1, 1, np.nan],
    }
)
N_GENES = 40
SAMPLES = {"s1": 100, "s2": 60}


@pytest.fixture(scope="module")
def built():
    rng = np.random.default_rng(1)
    obs = pd.concat(
        [
            pd.DataFrame(
                {
                    "DonorNumber": f"d{i}",
                    "Site": "site1",
                    "batch": s,
                    "cell_type": rng.choice(LINEAGE_MAP["cell_type"], n),
                    "GEX_pseudotime_order": rng.random(n),
                }
            )
            for i, (s, n) in enumerate(SAMPLES.items())
        ],
        ignore_index=True,
    )
    obs.index = [f"c{i}" for i in range(len(obs))]
    counts = sp.csr_matrix(rng.poisson(rng.gamma(1, 2, N_GENES), (len(obs), N_GENES)))
    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs)
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.layers["counts"] = counts.astype(np.float32)
    return adata, shared.build_shared(adata, LINEAGE_MAP, n_hvg=30, n_hvg_small=10)


def test_excluded_cell_types_are_dropped(built):
    source, out = built
    assert "TCell" not in set(out.obs["cell_type"])
    assert out.n_obs == (source.obs["cell_type"] != "TCell").sum()
    assert out.obs["lineage"].isin(["HSC", "erythroid", "myeloid"]).all()


def test_hvg_flags_are_nested_with_exact_counts(built):
    var = built[1].var
    assert var["hvg_30"].sum() == 30
    assert var["hvg_10"].sum() == 10
    assert not (var["hvg_10"] & ~var["hvg_30"]).any()


def test_counts_layer_is_integer_csr_and_x_equals_it(built):
    out = built[1]
    assert sp.issparse(out.layers["counts"]) and out.layers["counts"].format == "csr"
    assert np.issubdtype(out.layers["counts"].dtype, np.integer)
    assert (out.X != out.layers["counts"]).nnz == 0


def test_split_is_stratified_ninety_ten_per_sample(built):
    obs = built[1].obs
    frac = obs.groupby("sample", observed=True)["split"].apply(lambda s: (s == "val").mean())
    assert (frac.sub(0.1).abs() < 0.02).all()


def test_split_is_deterministic():
    strata = pd.Series(["a"] * 50 + ["b"] * 30)
    first = shared.stratified_split(strata, 0.1, 0)
    assert first.equals(shared.stratified_split(strata, 0.1, 0))
    assert first.value_counts().to_dict() == {"train": 72, "val": 8}


def test_obs_columns_renamed_and_pseudotime_kept(built):
    source, out = built
    for col in (
        "donor",
        "site",
        "sample",
        "cell_type",
        "lineage",
        "lineage_rank",
        "ref_pseudotime",
    ):
        assert col in out.obs.columns
    expected = source.obs.loc[out.obs_names, "GEX_pseudotime_order"].to_numpy()
    np.testing.assert_allclose(out.obs["ref_pseudotime"].to_numpy(), expected)


def test_unknown_cell_type_raises(built):
    bad = built[0].copy()
    bad.obs["cell_type"] = "Mystery"
    with pytest.raises(ValueError, match="missing from the lineage map"):
        shared.filter_lineage_cells(bad, LINEAGE_MAP)


def test_exclude_lineages_drops_only_those_cells(built):
    source = built[0]
    out = shared.filter_lineage_cells(source, LINEAGE_MAP, exclude_lineages=("myeloid",))
    expected = source.obs_names[source.obs["cell_type"].isin(["HSC", "Ery"])]
    assert list(out.obs_names) == list(expected)
    assert set(out.obs["lineage"]) == {"HSC", "erythroid"}
