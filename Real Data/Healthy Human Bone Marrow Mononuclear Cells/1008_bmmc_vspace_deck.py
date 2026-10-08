"""V-space deck for the BMMC T1 runs: native Decipher (left) vs Decipher-BC model5 (right).

Same template and layout as `Claude Files/writeups/1008_vspace_native_vs_model5_splitfix_bif.pptx`
(built by `1008_build_splitfix_decks.py`, whose helpers this reuses): one slide per batch variable
(donor, site, sample), decipherseeds 1-3 as rows. Each picture is one run's v-space in three panels:
cell type | the slide's batch variable | decipher_time. There is no ground-truth latent time, so
cell type takes that panel.

Trajectories (no ground-truth cluster order, unlike the simulations): erythroid HSC -> Reticulocyte
and myeloid HSC -> CD16+ Mono (CD14+ Mono when no cluster is mostly CD16+ Mono); each endpoint is
the Leiden cluster holding the most cells of that type. `decipher_time` is the package's single
column, so on the shared HSC clusters the second trajectory overwrites the first, as in the
simulation decks. Reads the run h5ads read-only; no retraining.
"""

import importlib
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt

import decipher_m5 as dc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "1008_bmmc"
RUNS = OUT / "runs" / "full"
FIGS = OUT / "figs" / "vspace"
DECK = REPO / "Claude Files" / "writeups" / "1008_vspace_native_vs_bc_bmmc.pptx"
sys.path.insert(0, str(REPO / "Simulated Data" / "Simulated Data Sweep Pipeline"))
decks = importlib.import_module("1008_build_splitfix_decks")

BATCH_VARS = ["donor", "site", "sample"]
SEEDS = [1, 2, 3]
TEMPLATE_TITLE = "Sigma = 0.1, Seed = 3"


def majority_cluster(cell_type: pd.Series, clusters: pd.Series, label: str) -> str:
    """The cluster id holding the most cells of `label`."""
    counts = pd.crosstab(clusters.astype(str), cell_type.astype(str))
    if label not in counts.columns:
        raise ValueError(f"no cells of type {label!r}")
    return counts[label].idxmax()


def myeloid_end_label(cell_type: pd.Series, clusters: pd.Series) -> str:
    """CD16+ Mono if some cluster is mostly CD16+ Mono, else CD14+ Mono."""
    top = pd.crosstab(clusters.astype(str), cell_type.astype(str)).idxmax(axis=1)
    return "CD16+ Mono" if (top == "CD16+ Mono").any() else "CD14+ Mono"


def add_trajectories(adata: ad.AnnData) -> dict[str, list[str]]:
    """Cluster, build the two trajectories and set `decipher_time`; return each path's clusters."""
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    ct, cl = adata.obs["cell_type"], adata.obs["decipher_clusters"]
    hsc = majority_cluster(ct, cl, "HSC")
    ends = {
        "erythroid": majority_cluster(ct, cl, "Reticulocyte"),
        "myeloid": majority_cluster(ct, cl, myeloid_end_label(ct, cl)),
    }
    dc.tl.trajectories(
        adata,
        *[
            dc.tl.TConfig(name, start_cluster_or_marker=hsc, end_cluster_or_marker=end)
            for name, end in ends.items()
        ],
    )
    dc.tl.decipher_time(adata)
    return {k: list(v["cluster_ids"]) for k, v in adata.uns["decipher"]["trajectories"].items()}


def vspace_png(config: str, seed: int, batch_var: str) -> Path:
    return FIGS / f"vspace_{config}_ds{seed}_by_{batch_var}.png"


def write_figure(adata: ad.AnnData, config: str, seed: int, batch_var: str) -> Path:
    panels = ["cell_type", batch_var, "decipher_time"]
    fig = dc.pl.decipher(adata, color=panels, basis="decipher_v", ncols=3, wspace=0.45)
    # Run h5ads carry no training config, so the package titles every panel "No Batch Correction".
    for ax, name in zip(fig.axes, panels):
        ax.set_title(name)
    fig.suptitle(f"BMMC | {config} | decipher seed={seed}", y=1.05, fontsize=11)
    out = vspace_png(config, seed, batch_var)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def write_figures() -> pd.DataFrame:
    """All v-space PNGs: native by each batch variable, BC-<var> by its own variable."""
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in ["native"] + [f"BC-{v}" for v in BATCH_VARS]:
        for seed in SEEDS:
            run_id = f"{config}_hvg2000_ds{seed}"
            adata = ad.read_h5ad(RUNS / f"{run_id}.h5ad")
            paths = add_trajectories(adata)
            for batch_var in BATCH_VARS if config == "native" else [config[3:]]:
                write_figure(adata, config, seed, batch_var)
            rows += [{"run_id": run_id, "trajectory": k, "clusters": " ".join(v)}
                     for k, v in paths.items()]  # fmt: skip
            print(run_id, paths, flush=True)
    return pd.DataFrame(rows)


def build_deck() -> None:
    parts, order, titles = decks.read_template()
    skeleton = parts[titles[TEMPLATE_TITLE]].decode()
    slides = []
    for batch_var in BATCH_VARS:
        left = [vspace_png("native", s, batch_var) for s in SEEDS]
        right = [vspace_png(f"BC-{batch_var}", s, batch_var) for s in SEEDS]
        pics = decks.recon_pics(left, right)  # keeps each PNG's aspect ratio
        xml = decks.slide_xml(
            skeleton,
            f"BMMC, batch = {batch_var}",
            "decipherseed = 1,2,3 · HSC → erythroid + myeloid",
            "Base Decipher",
            f"Decipher-BC (model5), on {batch_var}",
            pics,
            small_sub=True,
        )
        slides.append((xml, [p[0] for p in pics]))
    decks.write_deck(DECK, slides, parts, order)


if __name__ == "__main__":
    write_figures().to_csv(FIGS / "trajectory_paths_preview.csv", index=False)
    build_deck()
    print("wrote", DECK)
