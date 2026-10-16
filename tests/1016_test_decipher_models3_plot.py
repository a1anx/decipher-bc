"""decipher_models3 v-space plot: x-axis label for every form of `color`.

Regression for `len(color)` on the raw argument: `color=None` raised TypeError (so
`decipher_train` failed without `plot_kwargs`), and a string `color` dropped the x-axis label.
"""

import matplotlib

matplotlib.use("Agg")

import anndata as ad  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from decipher_models3.plot.decipher import decipher as plot_decipher_v  # noqa: E402


def _toy_adata(n_cells: int = 20) -> ad.AnnData:
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["a", "b"] * (n_cells // 2)),
            "score": rng.normal(size=n_cells),
        },
        index=[str(i) for i in range(n_cells)],
    )
    adata = ad.AnnData(np.zeros((n_cells, 2)), obs=obs)
    adata.obsm["decipher_v"] = rng.normal(size=(n_cells, 2))
    return adata


def _panel_x_labels(fig) -> list[str]:
    return [ax.get_xlabel() for ax in fig.axes if ax._label != "<colorbar>"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("color", [None, "batch"])
def test_single_panel_has_x_label(color):
    fig = plot_decipher_v(_toy_adata(), color=color)

    assert _panel_x_labels(fig) == ["Decipher 1"]


def test_two_colors_in_one_row_both_have_x_label():
    fig = plot_decipher_v(_toy_adata(), color=["batch", "score"], ncols=2)

    assert _panel_x_labels(fig) == ["Decipher 1", "Decipher 1"]


def test_three_colors_in_two_rows_label_only_the_bottom_row():
    fig = plot_decipher_v(_toy_adata(), color=["batch", "score", "batch"], ncols=2)

    assert _panel_x_labels(fig) == ["", "", "Decipher 1"]
