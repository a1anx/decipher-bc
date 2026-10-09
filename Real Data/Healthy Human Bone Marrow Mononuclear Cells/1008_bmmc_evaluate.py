"""Stage 5 evaluation of the BMMC T1 runs: cross-evaluation matrix, criteria, figures.

Reads `1008_bmmc/runs/full/*.h5ad` (decipher_v / decipher_z only) and `1008_bmmc/bmmc_shared.h5ad`
read-only, and writes `1008_bmmc/eval/`:

- per_run_metrics.csv: long table (run x embedding x scope x metric x variable). Scopes are `global`,
  `set_erythroid` (HSC + erythroid), `set_myeloid` (HSC + myeloid) and `donor1` (donor1's cells,
  site only). Embeddings are `v` (primary), `z`, and `pca` (the 50-PC baseline, config `PCA`).
- cross_eval_matrix.csv (+ .png): config x (embedding|scope|metric|variable), "mean ± sd" over seeds.
  cross_eval_agg.csv is the same as numbers (mean, sd, n).
- vs_pca_baseline.csv, within_donor.csv, vs_simulation.csv, criteria.csv, trajectory_paths.csv.
- figs/: v-space figures, all in the rotated v (`rotate_v` of 1008_bmmc_vspace_deck.py).

Per run: v is rotated rigidly (metrics unchanged), Leiden clusters on z, a trajectory
HSC -> Reticulocyte (erythroid) and HSC -> CD16+ Mono / CD14+ Mono (myeloid) is built, and every
cell of the trajectory's set (HSC + branch) gets its time by KNN projection on the curve (the
package's `decipher_time` regression, but not restricted to on-path clusters). The package's own
on-path coverage is recorded beside each correlation.

The PCA baseline within a set uses the global 50-PC PCA (fit on all cells, as in
`eda/mixing/baseline_pca.csv`) restricted to the set. Global PCA rows join to baseline_pca.csv.

Run: .venv/bin/python "Real Data/Healthy Human Bone Marrow Mononuclear Cells/1008_bmmc_evaluate.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_THREADS = os.environ.get("BMMC_EVAL_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "1008_bmmc"
RUNS = OUT / "runs" / "full"
SHARED = OUT / "bmmc_shared.h5ad"
BASELINE = OUT / "eda" / "mixing" / "baseline_pca.csv"
SIM_LOG = (
    REPO
    / "Simulated Data"
    / "Sweeps and Results"
    / "shift_sigma_sweep_bifurcation"
    / "1008"
    / "sweep1008_splitfix_sweep_log.csv"
)

BATCH_VARS = ["donor", "site", "sample"]
CONFIGS = ["native", "BC-donor", "BC-site", "BC-sample"]
SEEDS = [1, 2, 3]
TRAJECTORIES = ("erythroid", "myeloid")
SET_SCOPES = {"erythroid": "set_erythroid", "myeloid": "set_myeloid"}
PROGENITOR = {"erythroid": "MK/E prog", "myeloid": "G/M prog"}
END_LABEL = {"erythroid": "Reticulocyte"}
WITHIN_DONOR = "donor1"
KBET_K = 50
ASW_CELLS_PER_TYPE = 3000
LABEL_ASW_SAMPLE = 10000
KEY = ["config", "embedding", "scope", "metric", "variable"]
CELL_TYPE_ORDER = [
    "HSC",
    "MK/E prog",
    "Proerythroblast",
    "Erythroblast",
    "Normoblast",
    "Reticulocyte",
    "G/M prog",
    "CD14+ Mono",
    "CD16+ Mono",
]

# ----------------------------------------------------------------------------- module loading


def load_by_path(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _metrics() -> ModuleType:
    return load_by_path(HERE / "1008_bmmc_metrics.py", "bmmc_metrics")


@lru_cache(maxsize=None)
def _deck() -> ModuleType:
    return load_by_path(HERE / "1008_bmmc_vspace_deck.py", "bmmc_vspace_deck")


@lru_cache(maxsize=None)
def _eda() -> ModuleType:
    return load_by_path(HERE / "1008_bmmc_eda_mixing.py", "bmmc_eda_mixing")


# ----------------------------------------------------------------------------- pure scoring


def set_masks(lineage: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks of the scoring sets: HSC + erythroid and HSC + myeloid."""
    lineage = np.asarray(lineage).astype(str)
    return {
        SET_SCOPES[name]: np.isin(lineage, ["HSC", name]) for name in TRAJECTORIES
    }  # fmt: skip


def asw_subsample(celltype: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Sorted indices keeping at most ``cap`` random cells of each cell type."""
    rng = np.random.default_rng(seed)
    keep = []
    for ct in np.unique(celltype):
        ix = np.flatnonzero(celltype == ct)
        keep.append(ix if len(ix) <= cap else rng.choice(ix, cap, replace=False))
    return np.sort(np.concatenate(keep))


def score_cells(
    emb: np.ndarray,
    obs: pd.DataFrame,
    embedding: str,
    scope: str,
    variables: list[str],
    with_branch: bool = False,
) -> list[dict]:
    """Mixing and cell-type metrics of one embedding on one set of cells (long-format rows).

    Per batch variable with at least 2 levels: iLISI median (raw and normalised), kBET acceptance
    and batch ASW. Per set: cell-type ASW and cLISI score (1 = pure). ``with_branch`` adds the
    erythroid-vs-myeloid ASW (the analogue of the simulation's branch_asw).
    """
    m = _metrics()
    celltype = obs["cell_type"].astype(str).to_numpy()
    asw_ix = asw_subsample(celltype, ASW_CELLS_PER_TYPE)
    rows = []

    def add(metric: str, variable: str, value: float) -> None:
        rows.append(
            dict(
                embedding=embedding,
                scope=scope,
                metric=metric,
                variable=variable,
                value=float(value),
                n_cells=len(obs),
            )
        )

    for var in variables:
        batch = obs[var].astype(str).to_numpy()
        n_levels = len(np.unique(batch))
        if n_levels < 2:
            continue
        lisi = m.lisi_median(m.ilisi(emb, batch), n_levels)
        add("ilisi_median", var, lisi["median"])
        add("ilisi_median_norm", var, lisi["median_normalized"])
        add("kbet_acceptance", var, m.kbet_acceptance(emb, batch, KBET_K))
        add("batch_asw", var, m.silhouette_batch(emb[asw_ix], batch[asw_ix], celltype[asw_ix]))
    n_types = len(np.unique(celltype))
    if n_types >= 2:
        med = m.lisi_median(m.clisi(emb, celltype), n_types)
        add("clisi_median", "cell_type", med["median"])
        add("clisi_score", "cell_type", 1 - med["median_normalized"])
        asw = m.silhouette_label(
            emb, celltype, sample_size=min(LABEL_ASW_SAMPLE, len(celltype)), seed=0
        )
        add("celltype_asw", "cell_type", asw)
    if with_branch:
        lineage = obs["lineage"].astype(str).to_numpy()
        on = np.isin(lineage, TRAJECTORIES)
        asw = m.silhouette_label(
            emb[on], lineage[on], sample_size=min(LABEL_ASW_SAMPLE, int(on.sum())), seed=0
        )
        add("branch_asw", "lineage", asw)
    return rows


def trajectory_rows(
    name: str, time: np.ndarray, ref: np.ndarray, on_path: np.ndarray, scope: str
) -> list[dict]:
    """Correlation of one trajectory's time with its reference over the whole scoring set.

    Rows: traj_pearson (|r|), traj_r2, traj_spearman (sign-chosen), traj_sign, traj_coverage
    (fraction of the set on the path's clusters).
    """
    time, ref = np.asarray(time, float), np.asarray(ref, float)
    ok = ~np.isnan(time) & ~np.isnan(ref)  # the reference is NaN for a few progenitor cells
    corr = _metrics().trajectory_corr(time[ok], ref[ok])
    values = {
        "traj_pearson": corr.pearson,
        "traj_r2": corr.r2,
        "traj_spearman": corr.spearman,
        "traj_sign": corr.sign,
        "traj_coverage": float(np.mean(on_path)),
    }
    return [
        dict(
            embedding="v",
            scope=scope,
            metric=k,
            variable=name,
            value=float(v),
            n_cells=int(ok.sum()),
        )
        for k, v in values.items()
    ]


def cluster_majority_types(cell_type: pd.Series, clusters: pd.Series) -> dict[str, str]:
    """Majority cell type of each cluster id."""
    top = pd.crosstab(clusters.astype(str), cell_type.astype(str)).idxmax(axis=1)
    return top.to_dict()


def path_record(
    name: str,
    path_ids: list[str],
    start: str,
    end: str,
    cell_type: pd.Series,
    clusters: pd.Series,
    in_set: np.ndarray,
) -> dict:
    """One trajectory_paths.csv row: path clusters, their majority types, progenitor check."""
    majority = cluster_majority_types(cell_type, clusters)
    counts = pd.crosstab(clusters.astype(str), cell_type.astype(str))
    prog_label = PROGENITOR[name]
    prog_cluster = counts[prog_label].idxmax() if prog_label in counts.columns else ""
    on_path = clusters.astype(str).isin(path_ids).to_numpy()
    return dict(
        trajectory=name,
        start_cluster=start,
        end_cluster=end,
        cluster_ids=" ".join(path_ids),
        majority_types=" > ".join(majority[c] for c in path_ids),
        progenitor_label=prog_label,
        progenitor_cluster=prog_cluster,
        passes_progenitor=bool(prog_cluster in path_ids),
        coverage=float(on_path[in_set].mean()),
    )


# ----------------------------------------------------------------------------- aggregation


def aggregate_seeds(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Mean, sample SD (ddof=1; NaN for one seed) and n of ``value`` over seeds per key."""
    keys = keys or KEY
    g = df.groupby(keys, sort=False, dropna=False)["value"]
    return g.agg(mean="mean", sd=lambda x: x.std(ddof=1), n="count").reset_index()


def matrix_wide(agg: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    """Rows = config, columns = embedding|scope|metric|variable, cells "mean ± sd", plus n_seeds."""
    a = agg[agg["config"].isin(configs)].copy()
    a["col"] = a["embedding"] + "|" + a["scope"] + "|" + a["metric"] + "|" + a["variable"]
    a["text"] = a.apply(
        lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}" if pd.notna(r["sd"]) else f"{r['mean']:.3f}",
        axis=1,
    )
    wide = a.pivot(index="config", columns="col", values="text").reindex(configs)
    wide.insert(0, "n_seeds", a.groupby("config")["n"].min().reindex(configs))
    return wide


def _cell(agg: pd.DataFrame, config: str, embedding: str, scope: str, metric: str, var: str):
    sel = agg[
        (agg.config == config)
        & (agg.embedding == embedding)
        & (agg.scope == scope)
        & (agg.metric == metric)
        & (agg.variable == var)
    ]
    if len(sel) != 1:
        raise ValueError(
            f"expected 1 row for {config, embedding, scope, metric, var}, got {len(sel)}"
        )
    return sel.iloc[0]


def ilisi_verdict(native_mean: float, native_sd: float, bc_mean: float) -> str:
    """'improves' if BC beats native by more than native's seed SD, 'within noise' if it is
    within one SD either way, 'worse' if below by more than one SD, 'undetermined' without SD."""
    if np.isnan(native_sd):
        return "undetermined"
    diff = bc_mean - native_mean
    if diff > native_sd:
        return "improves"
    if diff < -native_sd:
        return "worse"
    return "within noise"


def trajectory_not_worse(native_mean: float, native_sd: float, bc_mean: float) -> bool:
    """BC is not lower than native by more than native's seed SD."""
    return bool(native_mean - bc_mean <= native_sd + 1e-9)


def path_flags(paths: pd.DataFrame, config: str) -> str:
    """Runs of ``config`` whose trajectory misses its progenitor cluster, 'run:trajectory ...'."""
    bad = paths[(paths["config"] == config) & (~paths["passes_progenitor"].astype(bool))]
    return " ".join(f"{r.run_id}:{r.trajectory}" for r in bad.itertuples())


def build_criteria(agg: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """One row per conditioned variable: the pre-specified criteria for BC-<var> vs native.

    1. iLISI (median, v, global) on that variable: BC-<var> minus native, 'improves' when it
       exceeds native's seed SD.
    2. Trajectory Spearman (v, over the trajectory's set) of each trajectory: BC-<var> not lower
       than native by more than native's seed SD.
    Path-missing-progenitor runs are listed, not used to filter.
    """
    rows = []
    for var in BATCH_VARS:
        bc = f"BC-{var}"
        nat = _cell(agg, "native", "v", "global", "ilisi_median", var)
        cur = _cell(agg, bc, "v", "global", "ilisi_median", var)
        row = dict(
            variable=var,
            config=bc,
            ilisi_native_mean=nat["mean"],
            ilisi_native_sd=nat["sd"],
            ilisi_bc_mean=cur["mean"],
            ilisi_bc_sd=cur["sd"],
            ilisi_diff=cur["mean"] - nat["mean"],
            ilisi_exceeds_native_sd=bool(cur["mean"] - nat["mean"] > nat["sd"]),
            ilisi_verdict=ilisi_verdict(nat["mean"], nat["sd"], cur["mean"]),
        )
        norm_nat = _cell(agg, "native", "v", "global", "ilisi_median_norm", var)["mean"]
        norm_bc = _cell(agg, bc, "v", "global", "ilisi_median_norm", var)["mean"]
        row["ilisi_norm_diff"] = norm_bc - norm_nat
        ok = []
        for traj in TRAJECTORIES:
            scope = SET_SCOPES[traj]
            tn = _cell(agg, "native", "v", scope, "traj_spearman", traj)
            tb = _cell(agg, bc, "v", scope, "traj_spearman", traj)
            good = trajectory_not_worse(tn["mean"], tn["sd"], tb["mean"])
            ok.append(good)
            row.update(
                {
                    f"{traj}_spearman_native_mean": tn["mean"],
                    f"{traj}_spearman_native_sd": tn["sd"],
                    f"{traj}_spearman_bc_mean": tb["mean"],
                    f"{traj}_spearman_bc_sd": tb["sd"],
                    f"{traj}_spearman_diff": tb["mean"] - tn["mean"],
                    f"{traj}_not_worse": good,
                }
            )
        row["trajectories_not_worse"] = all(ok)
        row["passes_all"] = bool(row["ilisi_verdict"] == "improves" and all(ok))
        row["path_flags_bc"] = path_flags(paths, bc)
        row["path_flags_native"] = path_flags(paths, "native")
        rows.append(row)
    return pd.DataFrame(rows)


def pick_best_config(criteria: pd.DataFrame) -> str:
    """Best BC config: passing all criteria first, then largest normalised iLISI gain."""
    ranked = criteria.sort_values(["passes_all", "ilisi_norm_diff"], ascending=False)
    return str(ranked.iloc[0]["config"])


def baseline_to_rows(base: pd.DataFrame) -> pd.DataFrame:
    """Global rows of baseline_pca.csv in the per-run layout (value, perm null as own column)."""
    g = base[(base.scope == "global") & (base.group == "all")]
    perm = g[g.metric.str.endswith("_perm")].copy()
    perm["metric"] = perm["metric"].str.removesuffix("_perm")
    real = g[~g.metric.str.endswith("_perm")]
    out = real.merge(
        perm[["metric", "variable", "value"]],
        on=["metric", "variable"],
        how="left",
        suffixes=("", "_perm"),
    )
    out = out.rename(columns={"value_perm": "pca_perm"})
    out["scope"] = "global"
    return out[["scope", "metric", "variable", "value", "pca_perm"]].rename(
        columns={"value": "pca_value"}
    )


def join_pca(agg: pd.DataFrame, pca: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Model mean ± sd next to the PCA value (global: baseline_pca.csv as is; other scopes: the
    recomputed PCA rows) and the baseline's permutation null where it exists."""
    recomputed = pca.rename(columns={"value": "pca_value"})[
        ["scope", "metric", "variable", "pca_value"]
    ].copy()
    recomputed["pca_perm"] = np.nan
    recomputed["pca_source"] = "recomputed"
    recomputed["pca_recomputed_value"] = recomputed["pca_value"]
    base = baseline.copy()
    base["pca_source"] = "baseline_pca.csv"
    base = base.merge(
        recomputed[recomputed.scope == "global"][
            ["scope", "metric", "variable", "pca_recomputed_value"]
        ],
        on=["scope", "metric", "variable"],
        how="left",
    )
    keys = ["scope", "metric", "variable"]
    extra = recomputed.merge(base[keys], on=keys, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    pca_all = pd.concat([base, extra], ignore_index=True)
    models = agg[agg.embedding.isin(["v", "z"])]
    out = models.merge(pca_all, on=["scope", "metric", "variable"], how="left")
    out["diff_vs_pca"] = out["mean"] - out["pca_value"]
    return out


def sim_ranges(sim: pd.DataFrame) -> pd.DataFrame:
    """Per model and shift_sigma min / median / max of branch_asw and rho over the ledger."""
    ok = sim[sim["error"].isna()]
    long = ok.melt(
        id_vars=["model", "shift_sigma"],
        value_vars=["branch_asw", "rho"],
        var_name="sim_metric",
        value_name="value",
    )
    g = long.groupby(["model", "shift_sigma", "sim_metric"])["value"]
    return g.agg(sim_min="min", sim_median="median", sim_max="max", sim_n="count").reset_index()


def compare_simulation(agg: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Real values (v, global) beside the simulation range per sigma, where comparable.

    branch_asw <-> real erythroid-vs-myeloid ASW (and cell-type ASW as a looser analogue);
    rho (|Spearman| of decipher time vs latent time) <-> real trajectory Spearman. native <->
    native, BC-* <-> model5. There is no simulation iLISI, so iLISI is not placed.
    """
    pairs = [
        ("branch_asw", "global", "branch_asw", "lineage"),
        ("branch_asw", "global", "celltype_asw", "cell_type"),
        ("rho", SET_SCOPES["erythroid"], "traj_spearman", "erythroid"),
        ("rho", SET_SCOPES["myeloid"], "traj_spearman", "myeloid"),
    ]
    rows = []
    for sim_metric, scope, metric, variable in pairs:
        real = agg[
            (agg.embedding == "v")
            & (agg.scope == scope)
            & (agg.metric == metric)
            & (agg.variable == variable)
            & agg.config.isin(CONFIGS)
        ]
        for r in real.itertuples():
            model = "native" if r.config == "native" else "model5"
            for s in ranges[
                (ranges.model == model) & (ranges.sim_metric == sim_metric)
            ].itertuples():
                rows.append(
                    dict(
                        config=r.config,
                        real_metric=metric,
                        real_variable=variable,
                        real_scope=scope,
                        real_mean=r.mean,
                        real_sd=r.sd,
                        sim_metric=sim_metric,
                        sim_model=model,
                        shift_sigma=s.shift_sigma,
                        sim_min=s.sim_min,
                        sim_median=s.sim_median,
                        sim_max=s.sim_max,
                        real_in_sim_range=bool(s.sim_min <= r.mean <= s.sim_max),
                    )
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- per-run work


def run_name(config: str, seed: int) -> str:
    return f"{config}_hvg2000_ds{seed}"


def parse_run(run_id: str) -> tuple[str, int]:
    config, rest = run_id.split("_hvg2000_ds")
    return config, int(rest)


def process_run(
    run_id: str, cache_dir: Path, fig_dir: Path | None
) -> tuple[list[dict], list[dict]]:
    """Orient v, build trajectories and times, score everything for one run.

    Returns (metric rows, trajectory path rows); pickles what the figures need to cache_dir
    and, when fig_dir is given, draws that run's figures.
    """
    import anndata as ad

    import decipher_m5 as dc

    deck = _deck()
    config, seed = parse_run(run_id)
    adata = ad.read_h5ad(RUNS / f"{run_id}.h5ad")
    obs = adata.obs
    cell_type = obs["cell_type"].astype(str)
    v = adata.obsm["decipher_v"]
    adata.obsm["decipher_v_not_rotated"] = v.copy()
    adata.obsm["decipher_v"] = v @ deck.rotate_v(v, cell_type.to_numpy())
    v = adata.obsm["decipher_v"]
    if "decipher" not in adata.uns:
        adata.uns["decipher"] = {}
    dc.tl.cell_clusters(adata)
    clusters = adata.obs["decipher_clusters"].astype(str)
    hsc = deck.majority_cluster(cell_type, clusters, "HSC")
    ends = {
        "erythroid": deck.majority_cluster(cell_type, clusters, "Reticulocyte"),
        "myeloid": deck.majority_cluster(
            cell_type, clusters, deck.myeloid_end_label(cell_type, clusters)
        ),
    }
    dc.tl.trajectories(
        adata,
        *[
            dc.tl.TConfig(n, start_cluster_or_marker=hsc, end_cluster_or_marker=e)
            for n, e in ends.items()
        ],
    )
    trajs = adata.uns["decipher"]["trajectories"]
    masks = set_masks(obs["lineage"].to_numpy())
    rows, paths = [], []
    ref = {
        "erythroid": obs["ref_pseudotime"].to_numpy(float),
        "myeloid": obs["lineage_rank"].to_numpy(float),
    }
    for name in TRAJECTORIES:
        sel = masks[SET_SCOPES[name]]
        t = np.full(len(obs), np.nan)
        t[sel] = deck.project_onto(v[sel], trajs[name])
        adata.obs[f"decipher_time_{name}"] = t
        path_ids = [str(c) for c in trajs[name]["cluster_ids"]]
        rec = path_record(name, path_ids, hsc, ends[name], cell_type, clusters, sel)
        paths.append(dict(run_id=run_id, config=config, seed=seed, **rec))
        on_path = clusters.isin(path_ids).to_numpy()[sel]
        rows += trajectory_rows(name, t[sel], ref[name][sel], on_path, SET_SCOPES[name])
    dc.tl.decipher_time(adata)  # the package's own time: record its coverage per set
    for name in TRAJECTORIES:
        sel = masks[SET_SCOPES[name]]
        covered = adata.obs["decipher_time"].notna().to_numpy()[sel].mean()
        rows.append(
            dict(
                embedding="v",
                scope=SET_SCOPES[name],
                metric="pkg_time_coverage",
                variable=name,
                value=float(covered),
                n_cells=int(sel.sum()),
            )
        )
    scopes = {"global": np.ones(len(obs), bool), **masks}
    donor1 = (obs["donor"].astype(str) == WITHIN_DONOR).to_numpy()
    for embedding, emb in (("v", v), ("z", adata.obsm["decipher_z"])):
        for scope, sel in scopes.items():
            rows += score_cells(
                emb[sel], obs[sel], embedding, scope, BATCH_VARS, with_branch=scope == "global"
            )
        rows += score_cells(emb[donor1], obs[donor1], embedding, WITHIN_DONOR, ["site"])
    cache = dict(
        v=np.asarray(v, np.float32),
        obs=adata.obs[
            [
                "cell_type",
                "donor",
                "site",
                "sample",
                "lineage",
                "lineage_rank",
                "ref_pseudotime",
                "decipher_time_erythroid",
                "decipher_time_myeloid",
                "decipher_clusters",
            ]
        ].copy(),
        trajs={
            n: dict(
                cluster_locations=np.asarray(t["cluster_locations"]),
                cluster_ids=list(t["cluster_ids"]),
            )
            for n, t in trajs.items()
        },
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{run_id}.pkl", "wb") as f:
        pickle.dump(cache, f)
    for r in rows:
        r.update(run_id=run_id, config=config, seed=seed)
    if fig_dir is not None:
        draw_run_figures(cache, config, seed, fig_dir)
    return rows, paths


def _process_run_star(args: tuple) -> tuple[list[dict], list[dict]]:
    return process_run(*args)


def score_pca(counts_obs: pd.DataFrame, emb: np.ndarray) -> list[dict]:
    """PCA baseline rows (config PCA) on the global, set and donor1 scopes."""
    rows = []
    scopes = {
        "global": np.ones(len(counts_obs), bool),
        **set_masks(counts_obs["lineage"].to_numpy()),
    }
    for scope, sel in scopes.items():
        rows += score_cells(
            emb[sel], counts_obs[sel], "pca", scope, BATCH_VARS, with_branch=scope == "global"
        )
    d1 = (counts_obs["donor"].astype(str) == WITHIN_DONOR).to_numpy()
    rows += score_cells(emb[d1], counts_obs[d1], "pca", WITHIN_DONOR, ["site"])
    for r in rows:
        r.update(run_id="PCA", config="PCA", seed=0)
    return rows


def run_pca() -> list[dict]:
    eda = _eda()
    counts, obs, var = eda.load_shared(SHARED)
    emb, _ = eda.normalized_pca(counts, var["hvg_2000"].to_numpy(bool))
    return score_pca(obs, emb)


# ----------------------------------------------------------------------------- figures


def _palette(categories: list[str]) -> dict[str, tuple]:
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    return {c: cmap(i % 20) for i, c in enumerate(categories)}


def scatter_panel(
    fig,
    ax,
    v: np.ndarray,
    values: pd.Series,
    title: str,
    mask: np.ndarray | None = None,
    traj: dict | None = None,
    cmap: str = "viridis",
) -> None:
    """One v-space scatter: cells outside ``mask`` grey, the rest coloured, trajectory on top."""
    mask = np.ones(len(v), bool) if mask is None else mask
    ax.scatter(v[~mask, 0], v[~mask, 1], s=0.6, c="#d0d0d0", rasterized=True, linewidths=0)
    categorical = values.dtype.kind in "OUS" or str(values.dtype) == "category"
    if categorical:
        vals = values.astype(str).to_numpy()
        if title.startswith("cell_type"):
            cats = [c for c in CELL_TYPE_ORDER if c in set(vals[mask])]
        else:
            cats = sorted(set(vals[mask]))
        pal = _palette(cats)
        for c in cats:
            sel = mask & (vals == c)
            ax.scatter(
                v[sel, 0], v[sel, 1], s=0.6, color=pal[c], rasterized=True, linewidths=0, label=c
            )
        ax.legend(
            fontsize=5, markerscale=6, frameon=False, loc="best", ncol=2 if len(cats) > 6 else 1
        )
    else:
        x = values.to_numpy(float)
        ok = mask & ~np.isnan(x)
        ax.scatter(
            v[mask & ~ok, 0], v[mask & ~ok, 1], s=0.6, c="#d0d0d0", rasterized=True, linewidths=0
        )
        sc = ax.scatter(
            v[ok, 0], v[ok, 1], s=0.6, c=x[ok], cmap=cmap, rasterized=True, linewidths=0
        )
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01).ax.tick_params(labelsize=6)
    if traj:
        for name, t in traj.items():
            loc = t["cluster_locations"]
            ax.plot(loc[:, 0], loc[:, 1], "-o", color="black", lw=1.2, ms=3, mec="white", mew=0.5)
            for cid, (x0, y0) in zip(t["cluster_ids"], loc):
                ax.annotate(
                    str(cid), (x0, y0), fontsize=5, xytext=(2, 2), textcoords="offset points"
                )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_run_figures(cache: dict, config: str, seed: int, fig_dir: Path) -> None:
    """Overview and per-lineage figures of one run in rotated v."""
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    v, obs, trajs = cache["v"], cache["obs"], cache["trajs"]
    masks = set_masks(obs["lineage"].to_numpy())
    # Overview: identity panels, then each trajectory's time and reference.
    panels = [
        ("cell_type", obs["cell_type"], None, None),
        ("donor", obs["donor"], None, None),
        ("site", obs["site"], None, None),
        ("sample", obs["sample"], None, None),
        (
            "erythroid time",
            obs["decipher_time_erythroid"],
            masks["set_erythroid"],
            {"erythroid": trajs["erythroid"]},
        ),
        ("erythroid ref_pseudotime", obs["ref_pseudotime"], masks["set_erythroid"], None),
        (
            "myeloid time",
            obs["decipher_time_myeloid"],
            masks["set_myeloid"],
            {"myeloid": trajs["myeloid"]},
        ),
        ("myeloid lineage_rank", obs["lineage_rank"], masks["set_myeloid"], None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, (title, values, mask, traj) in zip(axes.ravel(), panels):
        scatter_panel(fig, ax, v, values, title, mask, traj)
    fig.suptitle(f"BMMC | {config} | decipherseed {seed} | rotated v", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"vspace_{config}_ds{seed}.png", dpi=110)
    plt.close(fig)
    # Per lineage: that set coloured, everything else grey, the path on top.
    for name in TRAJECTORIES:
        mask = masks[SET_SCOPES[name]]
        ref_name = "ref_pseudotime" if name == "erythroid" else "lineage_rank"
        lp = [
            (f"{name} time", obs[f"decipher_time_{name}"]),
            (f"reference {ref_name}", obs[ref_name]),
            ("cell_type", obs["cell_type"]),
            ("donor", obs["donor"]),
            ("site", obs["site"]),
            ("sample", obs["sample"]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        for ax, (title, values) in zip(axes.ravel(), lp):
            scatter_panel(fig, ax, v, values, title, mask, {name: trajs[name]})
        fig.suptitle(
            f"BMMC | {config} | ds{seed} | HSC + {name} (others grey) | path = cluster centres",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"lineage_{name}_{config}_ds{seed}.png", dpi=110)
        plt.close(fig)


def draw_seed_grid(config: str, cache_dir: Path, fig_dir: Path, color_var: str) -> None:
    """Seeds 1-3 of one config: cell type, a batch variable, and both trajectory times."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(SEEDS), 4, figsize=(18, 4.3 * len(SEEDS)))
    for row, seed in enumerate(SEEDS):
        with open(cache_dir / f"{run_name(config, seed)}.pkl", "rb") as f:
            c = pickle.load(f)
        obs, v, tr = c["obs"], c["v"], c["trajs"]
        m = set_masks(obs["lineage"].to_numpy())
        scatter_panel(fig, axes[row, 0], v, obs["cell_type"], f"cell_type (ds{seed})")
        scatter_panel(fig, axes[row, 1], v, obs[color_var], f"{color_var} (ds{seed})")
        scatter_panel(
            fig, axes[row, 2], v, obs["decipher_time_erythroid"], f"erythroid time (ds{seed})",
            m["set_erythroid"], {"erythroid": tr["erythroid"]},
        )  # fmt: skip
        scatter_panel(
            fig, axes[row, 3], v, obs["decipher_time_myeloid"], f"myeloid time (ds{seed})",
            m["set_myeloid"], {"myeloid": tr["myeloid"]},
        )  # fmt: skip
    fig.suptitle(f"BMMC | {config} | decipherseeds 1-3 | rotated v", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"seeds_{config}.png", dpi=100)
    plt.close(fig)


def draw_heatmap(agg: pd.DataFrame, path: Path) -> None:
    """Config x metric heatmaps of the v embedding per scope; colour = z-score across configs."""
    import matplotlib.pyplot as plt

    scopes = ["global", "set_erythroid", "set_myeloid"]
    metrics = [
        "ilisi_median",
        "kbet_acceptance",
        "batch_asw",
        "celltype_asw",
        "clisi_score",
        "traj_spearman",
    ]
    fig, axes = plt.subplots(len(scopes), 1, figsize=(16, 3.2 * len(scopes)))
    for ax, scope in zip(axes, scopes):
        a = agg[
            (agg.embedding == "v")
            & (agg.scope == scope)
            & agg.metric.isin(metrics)
            & agg.config.isin(CONFIGS)
        ].copy()
        a["col"] = a["metric"] + "\n" + a["variable"]
        cols = [c for m in metrics for c in dict.fromkeys(a[a.metric == m]["col"])]
        mean = a.pivot(index="config", columns="col", values="mean").reindex(CONFIGS)[cols]
        sd = a.pivot(index="config", columns="col", values="sd").reindex(CONFIGS)[cols]
        z = (mean - mean.mean()) / mean.std(ddof=0).replace(0, np.nan)
        ax.imshow(z.fillna(0).to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mean.iat[i, j]:.2f}\n±{sd.iat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
        ax.set_xticks(range(len(cols)), cols, fontsize=6)
        ax.set_yticks(range(len(CONFIGS)), CONFIGS, fontsize=8)
        ax.set_title(
            f"{scope} (v; colour = z-score across configs, text = mean ± sd over seeds)", fontsize=9
        )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def main() -> None:
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT / "eval")
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (smoke test)")
    ap.add_argument("--skip-pca", action="store_true")
    args = ap.parse_args()
    out, cache_dir, fig_dir = args.out, args.out / "_cache", args.out / "figs"
    out.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_csv(RUNS / "ledger.csv")
    bad = ledger[ledger["status"] != "ok"]
    if len(bad):
        raise RuntimeError(f"runs not ok: {bad['run_id'].tolist()}")
    run_ids = [run_name(c, s) for c in CONFIGS for s in SEEDS]
    missing = set(run_ids) - set(ledger["run_id"])
    if missing:
        raise RuntimeError(f"expected runs missing from the ledger: {sorted(missing)}")
    if args.limit:
        run_ids = run_ids[: args.limit]
    jobs = [(r, cache_dir, fig_dir if parse_run(r)[1] == 1 else None) for r in run_ids]

    rows, paths = [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        pca_job = None if args.skip_pca else pool.apply_async(run_pca)
        for r, p in pool.imap_unordered(_process_run_star, jobs):
            rows += r
            paths += p
            print("done", r[0]["run_id"], flush=True)
        if pca_job:
            rows += pca_job.get()
    per_run = pd.DataFrame(rows)
    cols = [
        "run_id",
        "config",
        "seed",
        "embedding",
        "scope",
        "metric",
        "variable",
        "value",
        "n_cells",
    ]
    per_run = per_run[cols].sort_values(
        ["config", "seed", "embedding", "scope", "metric", "variable"]
    )
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    paths_df = pd.DataFrame(paths).sort_values(["config", "seed", "trajectory"])
    paths_df.to_csv(out / "trajectory_paths.csv", index=False)

    models = per_run[per_run.config.isin(CONFIGS)]
    agg = aggregate_seeds(models)
    agg.to_csv(out / "cross_eval_agg.csv", index=False)
    matrix_wide(agg, CONFIGS).to_csv(out / "cross_eval_matrix.csv")
    draw_heatmap(agg, out / "cross_eval_matrix.png")

    pca = per_run[per_run.config == "PCA"]
    if len(pca):
        baseline = baseline_to_rows(pd.read_csv(BASELINE))
        join_pca(agg, pca, baseline).to_csv(out / "vs_pca_baseline.csv", index=False)
        d1 = pca[pca.scope == WITHIN_DONOR][["metric", "variable", "value"]]
        d1 = d1.rename(columns={"value": "pca_value"})
        wd = agg[agg.scope == WITHIN_DONOR].merge(d1, on=["metric", "variable"], how="left")
        wd.to_csv(out / "within_donor.csv", index=False)
    ranges = sim_ranges(pd.read_csv(SIM_LOG))
    compare_simulation(agg, ranges).to_csv(out / "vs_simulation.csv", index=False)

    criteria = build_criteria(agg, paths_df)
    criteria.to_csv(out / "criteria.csv", index=False)
    best = pick_best_config(criteria)
    print("best BC config by the criteria:", best)
    if not args.limit:
        color = {"native": "sample", "BC-donor": "donor", "BC-site": "site", "BC-sample": "sample"}
        for config in ("native", best):
            draw_seed_grid(config, cache_dir, fig_dir, color[config])
    print("wrote", out)


if __name__ == "__main__":
    main()
