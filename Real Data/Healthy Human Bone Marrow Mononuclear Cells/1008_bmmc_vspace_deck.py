"""V-space deck for the BMMC T1 runs: native Decipher (left) vs Decipher-BC model5 (right).

Same template and layout as `Claude Files/writeups/1008_vspace_native_vs_model5_splitfix_bif.pptx`
(built by `1008_build_splitfix_decks.py`, whose helpers this reuses): one slide per batch variable
(donor, site, sample), decipherseeds 1-3 as rows. Each picture is one run's v-space in three panels:
cell type | the slide's batch variable | projected decipher time. There is no ground-truth latent time, so
cell type takes that panel.

Trajectories (no ground-truth cluster order, unlike the simulations): erythroid HSC -> Reticulocyte
and myeloid HSC -> CD16+ Mono (CD14+ Mono when no cluster is mostly CD16+ Mono); each endpoint is
the Leiden cluster holding the most cells of that type. The package's `decipher_time` only times
cells whose cluster lies on a path, which leaves about half of the wide CD14+ Mono cloud untimed.
So every cell of a lineage (HSC + its branch) is projected onto that trajectory's curve with the
package's own KNN regression on the curve points; erythroid and myeloid cells take their own
trajectory's time and HSC the mean of the two. Before any of this, v is rotated the same way in
every run (`rotate_v`: v1 along HSC -> G/M prog -> CD14+ -> CD16+ Mono, erythroid arm up); the
rotation is rigid, so distances and every metric are unchanged. Reads the run h5ads read-only.
"""

import importlib
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

import decipher_m5 as dc
from decipher_m5.tools.decipher import rot

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
MYELOID_ORDER = ["HSC", "G/M prog", "CD14+ Mono", "CD16+ Mono"]
ERYTHROID_TYPES = ["MK/E prog", "Proerythroblast", "Erythroblast", "Normoblast", "Reticulocyte"]


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


def rotate_v(v: np.ndarray, cell_type: np.ndarray) -> np.ndarray:
    """The 2x2 orthogonal matrix that orients v the same way in every run.

    v1 follows MYELOID_ORDER: `dc.tl.decipher_rotate_space(v1_col="cell_type",
    v1_order=MYELOID_ORDER)`'s search and score, applied to the stored v (that function reloads the
    model and needs counts, which run h5ads lack). That score is mirror-symmetric across v1, so v2 is
    then flipped to put the erythroid cells' mean on the positive side.
    """
    rank = pd.Series(cell_type).map({c: i for i, c in enumerate(MYELOID_ORDER)}).to_numpy()
    on = ~np.isnan(rank)
    if len(np.unique(rank[on])) < 2:
        raise ValueError(f"need at least two of {MYELOID_ORDER} to orient v1")

    def score(r: np.ndarray) -> float:
        w = v[on] @ r
        return np.corrcoef(w[:, 0], rank[on])[1, 0] - abs(np.corrcoef(w[:, 1], rank[on])[1, 0])

    candidates = [rot(t, u) for t in np.linspace(0, 2 * np.pi, 100) for u in (1, -1)]
    best = max(candidates, key=score)
    if (v[cell_type_is_erythroid(cell_type)] @ best)[:, 1].mean() < 0:
        best = best @ np.diag([1.0, -1.0])
    return best


def cell_type_is_erythroid(cell_type: np.ndarray) -> np.ndarray:
    return np.isin(cell_type, ERYTHROID_TYPES)


def add_trajectories(adata: ad.AnnData) -> dict[str, list[str]]:
    """Orient v, cluster, build the two trajectories and the times; return each path's clusters."""
    rotation = rotate_v(adata.obsm["decipher_v"], adata.obs["cell_type"].astype(str).to_numpy())
    adata.obsm["decipher_v"] = adata.obsm["decipher_v"] @ rotation
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
    trajs = adata.uns["decipher"]["trajectories"]
    adata.obs["projected_time"] = projected_time(
        adata.obsm["decipher_v"], adata.obs["lineage"].to_numpy(), trajs
    )
    return {k: list(v["cluster_ids"]) for k, v in trajs.items()}


def project_onto(v: np.ndarray, trajectory: dict, n_neighbors: int = 10) -> np.ndarray:
    """Time of each point in `v` from its nearest trajectory points (as `dc.tl.decipher_time`)."""
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(trajectory["points"], trajectory["times"])
    return knn.predict(v)


def projected_time(v: np.ndarray, lineage: np.ndarray, trajectories: dict) -> np.ndarray:
    """Each lineage's cells on its own trajectory; HSC the mean of both; other cells NaN."""
    t = np.full(len(v), np.nan)
    for name in ("erythroid", "myeloid"):
        sel = lineage == name
        t[sel] = project_onto(v[sel], trajectories[name])
    hsc = lineage == "HSC"
    t[hsc] = np.mean([project_onto(v[hsc], trajectories[n]) for n in ("erythroid", "myeloid")], 0)
    return t


def vspace_png(config: str, seed: int, batch_var: str) -> Path:
    return FIGS / f"vspace_{config}_ds{seed}_by_{batch_var}.png"


def write_figure(adata: ad.AnnData, config: str, seed: int, batch_var: str) -> Path:
    panels = ["cell_type", batch_var, "projected_time"]
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
