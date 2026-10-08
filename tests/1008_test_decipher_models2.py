"""No-training checks for the 1008 split-fix sweep (Claude Files/plans/1007_splitfix-relay/).

Figure layout: the 1008 pipeline writes figures into the same relative tree the user hand-sorted
`0918/figs/` into, so a 1008 run and its 0918 twin pair up path for path.
"""

import importlib
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "Simulated Data", "Simulated Data Sweep Pipeline")
FIGS_0918 = os.path.join(
    REPO, "Simulated Data", "Sweeps and Results", "shift_sigma_sweep_bifurcation", "0918", "figs"
)
sys.path.insert(0, PIPELINE)

splitfix = importlib.import_module("1008_simulated_data_pipeline_splitfix")


@pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("kind", ["reconstruction", "vspace"])
@pytest.mark.parametrize("model", ["native", "model5"])
def test_fig_path_exists_in_the_sorted_0918_tree(tmp_path, model, kind, sigma):
    path = splitfix._fig_path(str(tmp_path), kind, model, "genes", 0.3, sigma, 3, 1)
    relative = os.path.relpath(path, tmp_path)
    assert os.path.isfile(os.path.join(FIGS_0918, relative)), relative


def test_fig_path_mixes_sigma_formats_like_0918(tmp_path):
    recon = splitfix._fig_path(str(tmp_path), "reconstruction", "model5", "genes", 0.3, 10.0, 3, 2)
    vspace = splitfix._fig_path(str(tmp_path), "vspace", "native", "genes", 0.3, 10.0, 4, 1)
    assert os.path.relpath(recon, tmp_path) == os.path.join(
        "model5", "bif", "sigma10.0", "reconstruction",
        "reconstruction_sigma10.0_seed3_model5_bif_genes_decipherseed_2.png",
    )  # fmt: skip
    assert os.path.relpath(vspace, tmp_path) == os.path.join(
        "native_genes", "bif", "sigma10.0", "vspace",
        "vspace_sigma10_seed4_native_bif_genes_decipherseed_1.png",
    )  # fmt: skip
