"""Ordering and divergence scores — copied from the Decipher paper's own benchmark.

PROVENANCE. `gene_marker` and `compute_location_score` are copied verbatim from
`decipher_reproducibility/benchmark/3-evaluate-metric.ipynb` (the notebook that produces
Supplementary Figure 4 of Nazaret et al., Genome Biology 2025,
doi 10.1186/s13059-025-03682-8). A verbatim copy of that notebook also sits beside this
file as `3-evaluate-metric.ipynb`.

Only compatibility edits were made, for pandas 2.x / anndata 0.12. Every original line is
left in place, commented, with a dated reason — nothing has been deleted.

WHAT THE TWO SCORES MEAN
------------------------
Eight groups are formed. On the healthy side, `immature` cells are split by marker into
`normal_stage1` (CD34+), `normal_stage3` (MPO+) and `normal_stage2` (neither). On the
disease side the groups are the paper's manual staging: `perturbed_immature`, then
`perturbed_blast0..blast3`.

Seven pairs are declared adjacent, forming the expected maturation chain:

    normal_stage1 — normal_stage2 — normal_stage3                   healthy maturation
    normal_stage1 — perturbed_immature                              the shared root
    perturbed_immature — blast0 — blast1 — blast2 — blast3          leukemic maturation

Mean pairwise distance is computed in `adata.obsm[obsm_key]` between all 28 group pairs
and divided by `intrinsic_distance` (mean distance between two 2000-cell subsamples), which
makes the scores comparable across embeddings of different scale.

    score_ordering   = mean(dist | same origin, non-adjacent) / mean(same origin, adjacent)
    score_divergence = mean(dist | cross origin, non-adjacent) - 2 * mean(cross origin, adjacent)

Higher is better for both. `score_divergence` is the over-correction guard: the only
adjacent cross-origin pair is normal_stage1—perturbed_immature, so a model that collapses
AML onto healthy inflates the subtracted term and the score falls.

REQUIREMENTS on the AnnData passed in
-------------------------------------
- `obs["origin"]`            values exactly "normal" and "perturbed"
- `obs["cell_type_merged"]`  values "immature", "blast0", "blast1", "blast2", "blast3"
- `var_names`                must contain CD34 and MPO
- `X`                        must be sparse; `gene_marker` calls `.X.toarray()`
- at least 2000 cells        `sc.pp.subsample(adata, n_obs=2000)` is called

NOTE ON FITS WITHOUT A HEALTHY SAMPLE. With no cells at `origin == "normal"` the three
normal_* groups are empty, the guard at the top of the distance loop skips them, and
`score_divergence` becomes NaN because its mask (`origin1 != origin2`) never fires.
`score_ordering` is still defined over the leukemic chain alone.
"""

import itertools

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial


def gene_marker(adata, gene, threshold=0.9):
    return (
        adata[:, gene].X.toarray()[:, 0]
        > np.quantile(adata[:, gene].X.toarray()[:, 0], threshold)
    ).astype(float)


def compute_location_score(
    adata, obsm_key, mode=0, get_distances=False, scale=5, power=1, agg="mean"
):
    # form 3 stages for normal data
    # - Stage 1: cd34+
    # - Stage 3: mpo+
    # - Stage 2: the in-between cells

    normal = adata[adata.obs["origin"] == "normal"]
    # WAS (2026-09-29): normal_immature = normal[normal.obs["cell_type_merged"] == "immature"]
    # anndata 0.12 raises ImplicitModificationWarning when .obs is written on a view, and
    # the two assignments below do exactly that. .copy() makes it an owned object. No
    # change to any value.
    normal_immature = normal[normal.obs["cell_type_merged"] == "immature"].copy()
    normal_immature.obs["CD34+"] = gene_marker(normal_immature, "CD34", 0.9)
    normal_immature.obs["MPO+"] = gene_marker(normal_immature, "MPO", 0.7)
    normal_stage1 = normal_immature[normal_immature.obs["CD34+"] > 0]
    normal_stage2 = normal_immature[
        (normal_immature.obs["CD34+"] < 1) & (normal_immature.obs["MPO+"] < 1)
    ]
    normal_stage3 = normal_immature[normal_immature.obs["MPO+"] > 0]

    perturbed = adata[adata.obs["origin"] == "perturbed"]
    perturbed_immature = perturbed[perturbed.obs["cell_type_merged"] == "immature"]
    perturbed_blast0 = perturbed[perturbed.obs["cell_type_merged"] == "blast0"]
    perturbed_blast1 = perturbed[perturbed.obs["cell_type_merged"] == "blast1"]
    perturbed_blast2 = perturbed[perturbed.obs["cell_type_merged"] == "blast2"]
    perturbed_blast3 = perturbed[perturbed.obs["cell_type_merged"] == "blast3"]

    # save benchmark group
    key = "benchmark_group"
    adata.obs[key] = None
    adata.obs.loc[perturbed_immature.obs.index, key] = "perturbed_immature"
    adata.obs.loc[perturbed_blast0.obs.index, key] = "perturbed_blast0"
    adata.obs.loc[perturbed_blast1.obs.index, key] = "perturbed_blast1"
    adata.obs.loc[perturbed_blast2.obs.index, key] = "perturbed_blast2"
    adata.obs.loc[perturbed_blast3.obs.index, key] = "perturbed_blast3"
    adata.obs.loc[normal_stage1.obs.index, key] = "normal_stage1"
    adata.obs.loc[normal_stage2.obs.index, key] = "normal_stage2"
    adata.obs.loc[normal_stage3.obs.index, key] = "normal_stage3"

    groups = {
        "normal_stage1": normal_stage1,
        "normal_stage2": normal_stage2,
        "normal_stage3": normal_stage3,
        "perturbed_immature": perturbed_immature,
        "perturbed_blast0": perturbed_blast0,
        "perturbed_blast1": perturbed_blast1,
        "perturbed_blast2": perturbed_blast2,
        "perturbed_blast3": perturbed_blast3,
    }
    neighbor_groups = [
        ("normal_stage1", "normal_stage2"),
        ("normal_stage2", "normal_stage3"),
        ("normal_stage1", "perturbed_immature"),
        ("perturbed_immature", "perturbed_blast0"),
        ("perturbed_blast0", "perturbed_blast1"),
        ("perturbed_blast1", "perturbed_blast2"),
        ("perturbed_blast2", "perturbed_blast3"),
    ]
    neighbor_groups = set(map(tuple, map(sorted, neighbor_groups)))
    score = 0

    sampled_distances = scipy.spatial.distance.cdist(
        sc.pp.subsample(adata, n_obs=2000, copy=True).obsm[obsm_key],
        sc.pp.subsample(adata, n_obs=2000, copy=True).obsm[obsm_key],
    )
    intrinsic_distance = sampled_distances.mean()

    distances = []

    for n1, n2 in itertools.combinations(groups.keys(), 2):
        if not len(groups[n1].obsm[obsm_key]) or not len(groups[n2].obsm[obsm_key]):
            continue
        distance = np.mean(
            (
                scipy.spatial.distance.cdist(
                    groups[n1].obsm[obsm_key],
                    groups[n2].obsm[obsm_key],
                )
            ).flatten()
        )
        distances.append(
            [n1, n2, tuple(sorted([n1, n2])) not in neighbor_groups, distance]
        )

    distances = pd.DataFrame(
        distances,
        columns=["n1", "n2", "d>1", "distance"],
    )
    distances["distance"] /= intrinsic_distance

    distances["origin1"] = distances["n1"].str.split("_").str[0]
    distances["origin2"] = distances["n2"].str.split("_").str[0]

    df = distances

    df["score_ordering"] = (df["distance"] ** power) * (df["origin1"] == df["origin2"])
    # WAS (2026-09-29): df_ordering = df.groupby(["d>1"]).mean()
    # pandas 2.x raises TypeError on .mean() over the string columns n1/n2/origin1/origin2.
    # pandas 1.x silently dropped them, which is what numeric_only=True does explicitly.
    # Same numbers, no behaviour change.
    df_ordering = df.groupby(["d>1"]).mean(numeric_only=True)
    score_ordering = (
        df_ordering.loc[True, "score_ordering"]
        / df_ordering.loc[False, "score_ordering"]
    )

    df["score_divergence"] = (df["distance"] ** power) * (
        df["origin1"] != df["origin2"]
    )
    # WAS (2026-09-29): df["score_divergence"].replace(0, np.nan, inplace=True)
    # Chained assignment with inplace= is a no-op under pandas 3 copy-on-write and already
    # warns in 2.3. Assigning the result back is the documented replacement.
    df["score_divergence"] = df["score_divergence"].replace(0, np.nan)
    # WAS (2026-09-29): df_divergence = df.groupby(["d>1"]).mean()   # same reason as above
    df_divergence = df.groupby(["d>1"]).mean(numeric_only=True)
    score_divergence = (
        df_divergence.loc[True, "score_divergence"]
        - 2 * df_divergence.loc[False, "score_divergence"]
    )
    return score_ordering, score_divergence
