"""Pure helpers of `Real Data/Oetjen Bone Marrow/1015_oetjen6_m5_decipher.py`: the batch column
model5 is conditioned on and the file-split override of `decipher_m5`'s train/val split. No
training (the script imports decipher_m5, wandb and scanpy only inside `main`)."""

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1015_oetjen6_m5_decipher.py"
)
_spec = importlib.util.spec_from_file_location("oetjen6_m5", SCRIPT)
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)


def _adata(split: list[str]) -> ad.AnnData:
    obs = pd.DataFrame({"split": split}, index=[f"c{i}" for i in range(len(split))])
    return ad.AnnData(np.zeros((len(split), 1), dtype=np.float32), obs=obs)


def test_batch_from_sample_keeps_only_present_samples_as_categories():
    sample = pd.Series(
        pd.Categorical(["A", "U", "A", "J"], categories=["A", "G", "J", "U"]), name="sample"
    )
    batch = m5.batch_from_sample(sample)
    assert list(batch.cat.categories) == ["A", "J", "U"]
    assert list(batch) == ["A", "U", "A", "J"]


def test_file_split_copies_the_files_train_val_cells():
    adata = _adata(["train", "val", "train", "val", "train"])
    m5.file_split(adata, val_frac=0.5, seed=7)
    expected = pd.Categorical(
        ["train", "validation", "train", "validation", "train"],
        categories=["train", "validation"],
    )
    pd.testing.assert_series_equal(
        adata.obs["decipher_split"],
        pd.Series(expected, index=adata.obs_names, name="decipher_split"),
    )


def test_file_split_raises_on_an_unknown_split_value():
    adata = _adata(["train", "test"])
    with pytest.raises(ValueError, match="test"):
        m5.file_split(adata, val_frac=0.1, seed=0)
