"""Build the oetjen5 comparison tables from each run folder's meta, metrics and v-space files.

Writes <out>/suite_per_seed.csv (one row per model x seed), <out>/suite_mean.csv (per model:
mean, min, max of every numeric column, plus the `beats_native` verdict against `native_d3` when
that folder exists), <out>/suite_strat_skipped.csv (cell types left out of the stratified
metrics) and, when λ-scan folders exist, <out>/suite_scan.csv (one row per scan folder, seed 0).

Stratified mixing (`*_strat_*` columns): graph iLISI and batch ASW computed within each cell type
on v and on z, with the same functions as the global scIB metrics, then the cell-weighted mean
over types with cells from >= 2 samples and more than KNN_K cells. Caveat: `cell_type` is the
kNN-transferred label; only sample P carries true scIB labels.

Folders (under --root): final runs `1016_oetjen5_d3_<preset>` (plus the older `native`, `model5`
folders in SOURCES), seeds 0-2; tagged finals `1016_oetjen5_d3_<preset>_<tag>_final` (driver
`--tagged-final`), found by scanning --root and added as final arms named `<preset>_<tag>` after
the fixed arms, seeds 0-2, never read as scans; λ scans `1016_oetjen5_d3_<preset>_<li|lr|lc><λ>` with `.` -> `p`
(e.g. `_li0p1`, `_lc10`), seed 0. A missing final folder fails unless --allow-missing.

`--choose <preset>` prints the λ selection table for that preset's scan folders against
`native_d3` seeds 0-2 and a last line `CHOSEN <tag> <flag>`; it writes no tables.
`--min-lambda X` keeps only the scan folders with λ >= X (default: all).
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "oetjen6_native", HERE / "1015_oetjen6_native_decipher.py"
)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)

SEEDS = (0, 1, 2)
SCAN_SEED = 0
D3_PRESETS = ("model1", "model2", "model3", "model4", "model6")
NEW_PRESETS = (
    "native_mmd",
    "model2_mmd",
    "model5_mmd",
    "model4r",
    "model5r",
    "model7",
    "model7i",
    "model8",
    "model8v",
    "model5rz",
    "model5r0",
    "model7iw",
    "model7im",
    "model5r_mmd",
    "model8r",
)
NATIVE_REF = "native_d3"
D3_MODEL5 = "model5_d3"  # d3 model5 final (decipher_models3), the same-package model5 reference
# model label -> run folder name
SOURCES = {
    "native": "1016_oetjen5_native",
    **{p: f"1016_oetjen5_d3_{p}" for p in D3_PRESETS},
    "model5": "1016_oetjen5_m5e",
    NATIVE_REF: "1016_oetjen5_d3_native",
    D3_MODEL5: "1016_oetjen5_d3_model5",
    **{p: f"1016_oetjen5_d3_{p}" for p in NEW_PRESETS},
}
MODEL_ORDER = [
    "native",
    "model1",
    "model2",
    "model3",
    "model4",
    "model5",
    "model6",
    NATIVE_REF,
    *NEW_PRESETS,
    D3_MODEL5,
]
SCAN_DIR = re.compile(r"^1016_oetjen5_d3_(?P<preset>.+)_(?P<tag>(?:li|lr|lc)(?P<lam>[0-9p]+))$")
FINAL_TAGGED_DIR = re.compile(
    r"^1016_oetjen5_d3_(?P<preset>.+)_(?P<tag>(?:li|lr|lc)[0-9p]+)_final$"
)
# output column -> (space, metric, branch) row of seed<k>_metrics.csv
METRIC_ROWS = {
    "v_graph_ilisi": ("v", "graph_ilisi", "all"),
    "z_graph_ilisi": ("z", "graph_ilisi", "all"),
    "v_batch_asw": ("v", "batch_asw", "all"),
    "z_batch_asw": ("z", "batch_asw", "all"),
    "v_graph_connectivity": ("v", "graph_connectivity", "all"),
    "z_graph_connectivity": ("z", "graph_connectivity", "all"),
    "v_traj_cons_erythroid": ("v", "trajectory_conservation", "erythroid"),
    "v_traj_cons_myeloid": ("v", "trajectory_conservation", "myeloid"),
    "z_traj_cons_erythroid": ("z", "trajectory_conservation", "erythroid"),
    "z_traj_cons_myeloid": ("z", "trajectory_conservation", "myeloid"),
    "decipher_time_cons_erythroid": ("v", "decipher_time_conservation", "erythroid"),
    "decipher_time_cons_myeloid": ("v", "decipher_time_conservation", "myeloid"),
}
# output column -> key in seed<k>_meta.json
META_KEYS = {
    "epochs_run": "n_epochs_run",
    "best_epoch": "best_epoch",
    "r2_overall": "r2_overall",
    "r2_median_per_gene": "r2_median_per_gene",
    "sanity_pass": "sanity_pass",
}
# space -> v-space CSV columns of that embedding
SPACE_COLS = {"v": ["v1", "v2"], "z": [f"z{i}" for i in range(1, 11)]}
STRAT_METRICS = {"graph_ilisi": "strat_ilisi", "batch_asw": "strat_batch_asw"}
TRAJ_CRITERIA = [
    "v_traj_cons_erythroid",
    "v_traj_cons_myeloid",
    "decipher_time_cons_erythroid",
    "decipher_time_cons_myeloid",
]
# success criterion: (column, comparison, native bound)
CRITERIA = [
    ("v_strat_ilisi", ">", "max"),
    *[(col, ">=", "min") for col in TRAJ_CRITERIA],
    ("best_val_nll", "<=", "max"),
]
COMPARE = {">": np.greater, ">=": np.greater_equal, "<=": np.less_equal}


def best_val_nll(meta: dict, loss: pd.DataFrame) -> float:
    """Meta's best_val_nll when present, else the minimum val_nll of the loss CSV."""
    if "best_val_nll" in meta:
        return float(meta["best_val_nll"])
    return float(loss["val_nll"].min())


def split_types(batch: np.ndarray, labels: np.ndarray) -> tuple[list[str], dict[str, str]]:
    """Cell types the stratified metrics can use (cells from >= 2 batches and more than KNN_K
    cells), and the skipped ones as {type: reason}."""
    kept, skipped = [], {}
    for label in np.unique(labels):
        ix = labels == label
        n_batches = len(np.unique(batch[ix]))
        if n_batches < 2:
            skipped[label] = f"{n_batches} batch"
        elif ix.sum() <= native.KNN_K:
            skipped[label] = f"{ix.sum()} cells <= k={native.KNN_K}"
        else:
            kept.append(label)
    return kept, skipped


def per_type_mixing(
    emb: np.ndarray, batch: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Graph iLISI and batch ASW of each usable cell type, on a kNN graph of that type's cells
    only. Returns (table with cell_type, n_cells, graph_ilisi, batch_asw; skipped types)."""
    kept, skipped = split_types(batch, labels)
    rows = []
    for label in kept:
        ix = labels == label
        distances, _ = native.knn_graph(emb[ix])
        rows.append(
            {
                "cell_type": label,
                "n_cells": int(ix.sum()),
                "graph_ilisi": native.graph_ilisi(distances, batch[ix]),
                "batch_asw": native.batch_asw(emb[ix], batch[ix], labels[ix]),
            }
        )
    return pd.DataFrame(rows, columns=["cell_type", "n_cells", "graph_ilisi", "batch_asw"]), skipped


def cell_weighted_mean(per_type: pd.DataFrame, metric: str) -> float:
    """Mean of `metric` over cell types, each weighted by its number of cells."""
    if per_type.empty:
        raise ValueError("no cell type usable for the stratified metrics")
    return float(np.average(per_type[metric], weights=per_type["n_cells"]))


def stratified_metrics(vspace: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    """`<space>_strat_ilisi` / `<space>_strat_batch_asw` for v and z of one run's v-space table,
    and the skipped cell types."""
    batch = vspace[native.BATCH_KEY].astype(str).to_numpy()
    labels = vspace[native.LABEL_KEY].astype(str).to_numpy()
    row, skipped = {}, {}
    for space, cols in SPACE_COLS.items():
        per_type, skipped = per_type_mixing(vspace[cols].to_numpy(), batch, labels)
        for metric, suffix in STRAT_METRICS.items():
            row[f"{space}_{suffix}"] = cell_weighted_mean(per_type, metric)
    return row, skipped


def seed_row(run_dir: Path, seed: int) -> tuple[dict, dict[str, str]]:
    """One flat row of table values for a run folder's seed, and its skipped cell types;
    raises KeyError/FileNotFoundError."""
    meta = json.loads((run_dir / f"seed{seed}_meta.json").read_text())
    loss = pd.read_csv(run_dir / "csv" / f"seed{seed}_loss.csv")
    metrics = pd.read_csv(run_dir / "csv" / f"seed{seed}_metrics.csv")
    vspace = pd.read_csv(run_dir / "csv" / f"seed{seed}_vspace.csv")
    row = {"seed": seed}
    for col, key in META_KEYS.items():
        row[col] = meta[key]
    row["best_val_nll"] = best_val_nll(meta, loss)
    for col, (space, metric, branch) in METRIC_ROWS.items():
        hit = metrics[
            (metrics["space"] == space)
            & (metrics["metric"] == metric)
            & (metrics["branch"] == branch)
        ]["value"]
        if len(hit) != 1 or pd.isna(hit.iloc[0]):
            raise KeyError(f"metric row {space}/{metric}/{branch} missing or empty")
        row[col] = float(hit.iloc[0])
    strat, skipped = stratified_metrics(vspace)
    row.update(strat)
    return row, skipped


def run_rows(
    run_dir: Path, seeds: tuple[int, ...], label: str, skipped_rows: list[dict]
) -> list[dict]:
    """Rows of `seeds` for one run folder; appends its skipped types to `skipped_rows`."""
    rows = []
    for seed in seeds:
        try:
            row, skipped = seed_row(run_dir, seed)
        except (OSError, KeyError, ValueError) as err:
            raise RuntimeError(f"{label} ({run_dir.name}) seed {seed}: {err}") from err
        rows.append(row)
        for cell_type, reason in skipped.items():
            print(f"{label} seed {seed}: skipped {cell_type} ({reason})", file=sys.stderr)
            skipped_rows.append(
                {"run": label, "seed": seed, "cell_type": cell_type, "reason": reason}
            )
    return rows


def tagged_final_dirs(root: Path) -> dict[str, str]:
    """{`<preset>_<tag>`: folder name} of every tagged-final folder under `root`, sorted by name."""
    found = {}
    for path in sorted(root.iterdir()):
        m = FINAL_TAGGED_DIR.match(path.name)
        if path.is_dir() and m:
            found[f"{m['preset']}_{m['tag']}"] = path.name
    return found


def build_per_seed(
    root: Path,
    allow_missing: bool,
    models: list[str],
    skipped_rows: list[dict],
    sources: dict[str, str] = SOURCES,
) -> pd.DataFrame:
    """Per-seed table for `models` (folder names from `sources`); a missing folder raises (before
    any metric is computed) unless `allow_missing`, and a bad seed raises with its name."""
    missing = [m for m in models if not (root / sources[m]).is_dir()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"run folders not found under {root}: {missing}")
    rows = []
    for model in models:
        run_dir = root / sources[model]
        if model in missing:
            print(f"skipping {model}: {run_dir} not found", file=sys.stderr)
            continue
        rows += [{"model": model, **r} for r in run_rows(run_dir, SEEDS, model, skipped_rows)]
    return pd.DataFrame(rows)


def scan_dirs(
    root: Path, preset: str | None = None, min_lambda: float | None = None
) -> list[tuple[str, str, float, Path]]:
    """(preset, tag, λ, folder) of every scan folder under `root` (of `preset` only, if given, and
    with λ >= `min_lambda`, if given), sorted by preset, λ, tag."""
    found = []
    for path in root.iterdir():
        m = SCAN_DIR.match(path.name)
        if not (path.is_dir() and m and (preset is None or m["preset"] == preset)):
            continue
        lam = float(m["lam"].replace("p", "."))
        if min_lambda is None or lam >= min_lambda:
            found.append((m["preset"], m["tag"], lam, path))
    return sorted(found, key=lambda f: (f[0], f[2], f[1]))


def build_scan(
    root: Path,
    skipped_rows: list[dict],
    preset: str | None = None,
    min_lambda: float | None = None,
) -> pd.DataFrame:
    """Seed-0 rows of every scan folder (of `preset`, if given, with λ >= `min_lambda`, if given),
    with preset, tag, lambda."""
    rows = []
    for name, tag, lam, path in scan_dirs(root, preset, min_lambda):
        for r in run_rows(path, (SCAN_SEED,), f"{name}_{tag}", skipped_rows):
            rows.append({"preset": name, "tag": tag, "lambda": lam, **r})
    return pd.DataFrame(rows)


def criteria_table(values: dict[str, float], native_rows: pd.DataFrame) -> pd.DataFrame:
    """Each success criterion: value, native bound (max or min over `native_rows`), pass."""
    rows = []
    for col, op, stat in CRITERIA:
        bound = float(native_rows[col].agg(stat))
        value = float(values[col])
        rows.append(
            {
                "criterion": col,
                "value": value,
                "rule": f"{op} native {stat}",
                "bound": bound,
                "pass": bool(COMPARE[op](value, bound)),
            }
        )
    return pd.DataFrame(rows)


def choose_lambda(
    scan_rows: pd.DataFrame, native_rows: pd.DataFrame
) -> tuple[pd.DataFrame, str, str]:
    """λ selection over one preset's seed-0 scan rows (one row per tag, columns tag, lambda and
    the CRITERIA columns) against native_d3's seed rows.

    A λ passes iff every criterion passes; the passing λ with the highest stratified v iLISI is
    chosen (`pass`). If none passes: the largest λ keeping all four trajectory criteria
    (`none-pass-traj-kept`), else the smallest λ (`none-pass-smallest`). Ties go to the smaller
    λ. Returns (per-λ criteria table, chosen tag, flag)."""
    if scan_rows.empty or scan_rows["tag"].duplicated().any():
        raise ValueError("scan rows must be non-empty with one row per tag")
    if native_rows.empty:
        raise ValueError("no native rows")
    scan = scan_rows.sort_values(["lambda", "tag"]).reset_index(drop=True)
    tables, passes, traj_kept = [], [], []
    for _, row in scan.iterrows():
        table = criteria_table(row.to_dict(), native_rows)
        tables.append(table.assign(tag=row["tag"], **{"lambda": row["lambda"]}))
        passes.append(bool(table["pass"].all()))
        traj_kept.append(bool(table.loc[table["criterion"].isin(TRAJ_CRITERIA), "pass"].all()))
    table = pd.concat(tables, ignore_index=True)[
        ["tag", "lambda", "criterion", "value", "rule", "bound", "pass"]
    ]
    passes, traj_kept = np.array(passes), np.array(traj_kept)
    if passes.any():
        ilisi = scan["v_strat_ilisi"].where(passes, -np.inf)
        return table, scan.loc[int(ilisi.idxmax()), "tag"], "pass"
    if traj_kept.any():
        return table, scan.loc[int(np.flatnonzero(traj_kept)[-1]), "tag"], "none-pass-traj-kept"
    return table, scan.loc[0, "tag"], "none-pass-smallest"


def beats_native(arm_rows: pd.DataFrame, native_rows: pd.DataFrame) -> dict[str, bool]:
    """Final 3-seed verdict: the arm's lowest-seed stratified v iLISI > native max; its mean
    trajectory and decipher-time conservations >= native min; its mean best val NLL <= native
    max. Returns `beats_<criterion>` per criterion and `beats_native` (all of them)."""
    values = {col: arm_rows[col].mean() for col, _, _ in CRITERIA}
    values["v_strat_ilisi"] = arm_rows["v_strat_ilisi"].min()
    table = criteria_table(values, native_rows)
    out = {f"beats_{c}": bool(p) for c, p in zip(table["criterion"], table["pass"])}
    out["beats_native"] = bool(table["pass"].all())
    return out


def build_mean(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Per model: mean, min, max of each numeric column (seed and sanity_pass excluded); plus the
    beats_native columns when native_d3 is in the table."""
    cols = [c for c in per_seed.columns if c not in ("model", "seed", "sanity_pass")]
    agg = per_seed.groupby("model", sort=False)[cols].agg(["mean", "min", "max"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg["sanity_pass_all"] = per_seed.groupby("model", sort=False)["sanity_pass"].all()
    out = agg.reset_index()
    native_rows = per_seed[per_seed["model"] == NATIVE_REF]
    if native_rows.empty:
        return out
    verdicts = [beats_native(per_seed[per_seed["model"] == m], native_rows) for m in out["model"]]
    return pd.concat([out, pd.DataFrame(verdicts)], axis=1)


def print_choice(table: pd.DataFrame, tag: str, flag: str) -> None:
    """The per-λ criteria table, then `CHOSEN <tag> <flag>`."""
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"CHOSEN {tag} {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=HERE, help="folder holding the run folders")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default <root>/1016_oetjen5_suite"
    )
    parser.add_argument("--allow-missing", action="store_true", help="skip missing run folders")
    parser.add_argument(
        "--choose", metavar="PRESET", help="print the λ selection for PRESET's scan folders"
    )
    parser.add_argument(
        "--min-lambda", type=float, default=None, help="with --choose: only scan folders, λ >= this"
    )
    args = parser.parse_args()
    if args.min_lambda is not None and not args.choose:
        parser.error("--min-lambda needs --choose")
    skipped_rows: list[dict] = []
    if args.choose:
        native_rows = build_per_seed(args.root, False, [NATIVE_REF], skipped_rows)
        scan = build_scan(args.root, skipped_rows, args.choose, args.min_lambda)
        if scan.empty:
            raise FileNotFoundError(
                f"no scan folders for preset {args.choose} with lambda >= {args.min_lambda or 0} "
                f"in {args.root}"
            )
        print_choice(*choose_lambda(scan, native_rows))
        return
    out_dir = args.out_dir or args.root / "1016_oetjen5_suite"
    tagged = tagged_final_dirs(args.root)
    per_seed = build_per_seed(
        args.root,
        args.allow_missing,
        [*MODEL_ORDER, *tagged],
        skipped_rows,
        {**SOURCES, **tagged},
    )
    if per_seed.empty:
        sys.exit("no run folders found")
    scan = build_scan(args.root, skipped_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "suite_per_seed.csv", index=False)
    build_mean(per_seed).to_csv(out_dir / "suite_mean.csv", index=False)
    pd.DataFrame(skipped_rows, columns=["run", "seed", "cell_type", "reason"]).to_csv(
        out_dir / "suite_strat_skipped.csv", index=False
    )
    if not scan.empty:
        scan.to_csv(out_dir / "suite_scan.csv", index=False)
    print(f"wrote {len(per_seed)} rows ({len(scan)} scan rows) to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as err:
        sys.exit(f"ERROR: {err}")
