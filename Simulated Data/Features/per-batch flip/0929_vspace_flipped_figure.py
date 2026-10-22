"""Redraw the sweep's v-space figure with `batch03` reflected across its mirror line.

Same `dc.pl.decipher` call, title and dpi as the figure the sweep driver writes
(`0918_simulated_data_pipeline_bifurcation.py`), but `obsm["decipher_v"]` is replaced by the
flipped v from `0929_tip_line_reflect.flip_batch`, and the legend reads "Flipped batch03".
Writes next to this script; the sweep's own figure is not touched.
"""

import importlib
import sys
from pathlib import Path

import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))  # repo root, for decipher_models
flip_mod = importlib.import_module("0929_tip_line_reflect")
dc = importlib.import_module("decipher_models")

FLIP = "batch03"
a = ad.read_h5ad(flip_mod.DEFAULT_H5AD)
a.obs_names_make_unique()
a.obsm["decipher_v"], _, _ = flip_mod.flip_batch(a, FLIP)
a.obs["batch"] = a.obs["batch"].astype("category").cat.rename_categories({FLIP: f"Flipped {FLIP}"})

fig = dc.pl.decipher(
    a, color=["batch", "latent_t", "decipher_time"], basis="decipher_v", ncols=3, show=False
)
# the longer label is clipped by the neighbouring panel; move the legend under the first panel
leg = fig.axes[0].get_legend()
if leg is not None:
    leg.set_loc("upper center")
    leg.set_bbox_to_anchor((0.5, -0.06))
    leg._ncols = 3
fig.suptitle(
    "Set3 zx2 | bif | sigma=2 | seed=3 | decipher seed=1 | flipped batch03 (branch-point reflect)",
    y=1.05,
    fontsize=11,
)
out = HERE / "0929_vspace_sigma2_seed3_model5_bif_genes_decipherseed_1_flipped.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(out)
