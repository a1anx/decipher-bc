"""Tests for the Oetjen 6-sample shared-file builder on a tiny synthetic dataset."""

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
    / "Oetjen Bone Marrow"
    / "1015_build_oetjen6_shared.py"
)
_spec = importlib.util.spec_from_file_location("oetjen6_shared", _PATH)
oetjen6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oetjen6)

LINEAGE_MAP = pd.DataFrame(
    {
        "cell_type": ["HSPCs", "Erythrocytes", "CD14+ Monocytes", "CD20+ B cells", "NK cells"],
        "lineage": ["HSC", "erythroid", "myeloid", "lymphoid", "exclude"],
        "rank": [0, 1, 1, 1, np.nan],
    }
)
TYPES = ["HSPCs", "Erythrocytes", "CD14+ Monocytes", "CD20+ B cells", "NK cells"]


def _toy(samples=("A", "P", "K"), per_type: int = 10, n_genes: int = 40):
    """Cells: every sample x type x per_type; one cell per sample fails QC (absent from labels)."""
    rng = np.random.default_rng(0)
    rows = [(s, t) for s in samples for t in TYPES for _ in range(per_type)]
    names = [f"c{i}-{s}" for i, (s, _) in enumerate(rows)]
    counts = sp.csr_matrix(rng.poisson(rng.gamma(1, 2, n_genes), (len(rows), n_genes)))
    obs = pd.DataFrame(
        {
            "sample": [s for s, _ in rows],
            "donor": [s for s, _ in rows],
            "age": 50,
            "scib_dpt_pseudotime": np.nan,
            "n_counts": 1.0,
            "n_genes": 1.0,
        },
        index=names,
    )
    adata = ad.AnnData(counts.astype(np.float32), obs=obs)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.layers["counts"] = counts.astype(np.float32)
    labels = pd.DataFrame(
        {"celltype": [t for _, t in rows], "celltype_conf": 1.0}, index=names
    ).drop(index=[names[0], names[per_type * len(TYPES)]])
    return adata, labels


def _build(**kw):
    adata, labels = _toy()
    return oetjen6.build_shared(
        adata, labels, LINEAGE_MAP, samples=("A", "P"), n_hvg=20, n_hvg_small=5, **kw
    )


def test_keeps_only_chosen_samples_qc_cells_and_branch_lineages():
    out = _build()
    got = pd.crosstab(out.obs["cell_type"].astype(str), out.obs["sample"].astype(str))
    assert got.to_dict() == {
        "A": {"CD14+ Monocytes": 10, "Erythrocytes": 10, "HSPCs": 9},
        "P": {"CD14+ Monocytes": 10, "Erythrocytes": 10, "HSPCs": 9},
    }


def test_keeps_lymphoid_when_not_excluded():
    out = _build(exclude_lineages=())
    assert set(out.obs["lineage"]) == {"HSC", "erythroid", "myeloid", "lymphoid"}


def test_counts_are_integer_csr_and_equal_x():
    out = _build()
    assert sp.issparse(out.layers["counts"]) and out.layers["counts"].dtype == np.int32
    np.testing.assert_array_equal(out.X.toarray(), out.layers["counts"].toarray())


def test_split_is_stratified_per_sample():
    out = _build()
    got = pd.crosstab(out.obs["sample"], out.obs["split"])
    assert got.loc["A", "val"] == 3 and got.loc["P", "val"] == 3


def test_hvg_flags_have_requested_sizes():
    out = _build()
    assert out.var["hvg_20"].sum() == 20 and out.var["hvg_5"].sum() == 5
    assert out.var["hvg_5_nbatches"].between(0, 2).all()


def test_unknown_sample_raises():
    adata, labels = _toy()
    with pytest.raises(ValueError, match="Z"):
        oetjen6.build_shared(adata, labels, LINEAGE_MAP, samples=("A", "Z"))


def test_unmapped_cell_type_raises():
    adata, labels = _toy()
    labels.iloc[0, 0] = "Mystery cells"
    with pytest.raises(ValueError, match="Mystery"):
        oetjen6.build_shared(adata, labels, LINEAGE_MAP, samples=("A",))


def test_five_samples_keep_only_those_with_categories_in_given_order():
    adata, labels = _toy(samples=("A", "P", "G", "T", "W", "R"))
    chosen = ("P", "G", "T", "W", "R")
    out = oetjen6.build_shared(adata, labels, LINEAGE_MAP, samples=chosen, n_hvg=20, n_hvg_small=5)
    assert list(out.obs["sample"].cat.categories) == list(chosen)
    assert out.obs["sample"].value_counts(sort=False).to_dict() == {
        "P": 29,
        "G": 30,
        "T": 30,
        "W": 30,
        "R": 30,
    }


def test_cli_defaults_reproduce_oetjen6_paths():
    args = oetjen6.parse_args([])
    assert (args.samples, args.out_dir, args.name) == (
        ["A", "U", "J", "M", "P", "G"],
        _PATH.parent / "1015_oetjen6",
        "oetjen6",
    )


def test_cli_parses_oetjen5_subset():
    args = oetjen6.parse_args(["--samples", "P", "G", "T", "W", "R", "--name", "oetjen5"])
    assert (args.samples, args.name) == (["P", "G", "T", "W", "R"], "oetjen5")
