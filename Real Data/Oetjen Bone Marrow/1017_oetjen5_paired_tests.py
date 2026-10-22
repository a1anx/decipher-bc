"""Paired per-seed tests between oetjen5 arms (batch-invariance follow-up, plan §C2).

For each pair (arm, ref) and each metric, d_k = arm_k - ref_k over the seeds present in both run
folders (seed<k>_meta.json), then: n, mean d, min/max d, an exact two-sided sign test (zeros
dropped) and an exact two-sided sign-flip permutation test over all 2^n flips. A pair needs at
least MIN_PAIRED paired seeds; seeds present in only one folder are listed, never dropped
silently.

Pairs: every arm with >= MIN_PAIRED seeds against native_d3, plus EXTRA_PAIRS. Arms are the
suite table's SOURCES, the NEW_FINALS folders and the tagged finals
`1016_oetjen5_d3_<preset>_<tag>_final` (arm `<preset>_<tag>`); λ-scan folders (no `_final`) are
never arms. Absent folders are listed and skipped. Note: the suite's `model5` is
`1016_oetjen5_m5e` (decipher_m5 model5e), not a d3 run, so model5rz pairs with `model5_d3`
(`1016_oetjen5_d3_model5`, EXTRA_FOLDERS) instead.

Gates per arm against native_d3, side by side (the suite table's verdicts are unchanged):
- floor: the suite's `beats_native` on seeds 0-2 only;
- mean: the arm's mean of each of the four conservations over its paired seeds >= native's mean
  over the same seeds - DELTA;
- ilisi: paired sign-flip p < ALPHA and mean d > 0 for the stratified v iLISI.

Writes <out>/paired_tests.csv (one row per pair x metric) and <out>/paired_tests.md.
"""

import argparse
import importlib.util
import itertools
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "oetjen5_suite", HERE / "1016_oetjen5_suite_table.py"
)
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)

NATIVE_REF = suite.NATIVE_REF
MIN_PAIRED = 3
DELTA = 0.05
ALPHA = 0.05
FLOOR_SEEDS = (0, 1, 2)
TIE_TOL = 1e-9
METRICS = ["v_strat_ilisi", "best_val_nll", *suite.TRAJ_CRITERIA]
NEW_FINALS = ("model5rz", "model5r0", "model7iw", "model7im", "model5r_mmd", "model8r")
# the suite's model5 is decipher_m5 model5e; A's capacity control pairs with a d3 model5 run
EXTRA_FOLDERS = {"model5_d3": "1016_oetjen5_d3_model5"}
TAGGED_FINAL = re.compile(r"^1016_oetjen5_d3_(?P<preset>.+)_(?P<tag>(?:li|lr|lc)[0-9p]+)_final$")
# (arm, ref) pairs beyond each arm vs native_d3
EXTRA_PAIRS = [
    ("model7", "model5r"),
    ("model8v", "model8"),
    ("model5r", "model4r"),
    ("model5rz", "model5_d3"),
    ("model5r", "model5rz"),
    ("model5r0", "model5r"),
    ("model7iw", "model7i"),
    ("model7im", "model7i"),
]
SEED_FILE = re.compile(r"^seed(?P<seed>\d+)_meta\.json$")


def seeds_present(run_dir: Path) -> tuple[int, ...]:
    """Sorted seeds with a seed<k>_meta.json in `run_dir`."""
    found = (SEED_FILE.match(p.name) for p in run_dir.iterdir())
    return tuple(sorted(int(m["seed"]) for m in found if m))


def arm_folders(root: Path) -> tuple[dict[str, Path], list[str]]:
    """({arm: folder} of the arms present under `root`, names of the known arms whose folder is
    absent). Known: suite SOURCES, NEW_FINALS, EXTRA_FOLDERS; plus every tagged-final folder found.
    """
    known = {
        **suite.SOURCES,
        **{p: f"1016_oetjen5_d3_{p}" for p in NEW_FINALS},
        **EXTRA_FOLDERS,
    }
    present = {arm: root / name for arm, name in known.items() if (root / name).is_dir()}
    absent = [arm for arm in known if arm not in present]
    for path in sorted(root.iterdir()):
        m = TAGGED_FINAL.match(path.name)
        if path.is_dir() and m:
            present[f"{m['preset']}_{m['tag']}"] = path
    return present, absent


def sign_test_p(d: np.ndarray) -> float:
    """Exact two-sided sign test of median 0, zeros dropped; 1.0 when every d is 0."""
    nonzero = d[d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = int(min((nonzero > 0).sum(), (nonzero < 0).sum()))
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip p: the fraction of all 2^n sign flips of `d` whose |mean| is
    >= the observed |mean| (observed included, TIE_TOL for float ties)."""
    if len(d) == 0:
        raise ValueError("sign_flip_p needs at least one difference")
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    flipped = np.abs(signs @ d) / len(d)
    return float(np.mean(flipped >= abs(d.mean()) - TIE_TOL))


def paired_seeds(arm_rows: pd.DataFrame, ref_rows: pd.DataFrame) -> dict[str, list[int]]:
    """{"paired", "arm_only", "ref_only"} sorted seed lists of two per-seed tables."""
    arm, ref = set(arm_rows["seed"]), set(ref_rows["seed"])
    return {
        "paired": sorted(arm & ref),
        "arm_only": sorted(arm - ref),
        "ref_only": sorted(ref - arm),
    }


def compare_pair(
    arm_rows: pd.DataFrame, ref_rows: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    """One row per metric: n, mean_d, min_d, max_d, n_pos, n_neg, sign_p, signflip_p over the
    seeds both tables hold (d = arm - ref), with the paired and unpaired seeds. Raises ValueError
    with fewer than MIN_PAIRED paired seeds."""
    seeds = paired_seeds(arm_rows, ref_rows)
    if len(seeds["paired"]) < MIN_PAIRED:
        raise ValueError(
            f"{len(seeds['paired'])} paired seeds < {MIN_PAIRED} (paired {seeds['paired']}, "
            f"arm only {seeds['arm_only']}, ref only {seeds['ref_only']})"
        )
    arm = arm_rows.set_index("seed").loc[seeds["paired"]]
    ref = ref_rows.set_index("seed").loc[seeds["paired"]]
    rows = []
    for metric in metrics:
        d = (arm[metric] - ref[metric]).to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "n": len(d),
                "mean_d": float(d.mean()),
                "min_d": float(d.min()),
                "max_d": float(d.max()),
                "n_pos": int((d > 0).sum()),
                "n_neg": int((d < 0).sum()),
                "sign_p": sign_test_p(d),
                "signflip_p": sign_flip_p(d),
                "paired_seeds": " ".join(map(str, seeds["paired"])),
                "arm_only_seeds": " ".join(map(str, seeds["arm_only"])),
                "ref_only_seeds": " ".join(map(str, seeds["ref_only"])),
            }
        )
    return pd.DataFrame(rows)


def mean_rule(arm_rows: pd.DataFrame, native_rows: pd.DataFrame) -> dict[str, bool]:
    """Per conservation criterion: arm mean >= native mean - DELTA over the paired seeds, and
    `mean_rule` (all of them). Raises ValueError with fewer than MIN_PAIRED paired seeds."""
    paired = paired_seeds(arm_rows, native_rows)["paired"]
    if len(paired) < MIN_PAIRED:
        raise ValueError(f"{len(paired)} paired seeds < {MIN_PAIRED}")
    arm = arm_rows.set_index("seed").loc[paired]
    nat = native_rows.set_index("seed").loc[paired]
    out = {f"mean_{c}": bool(arm[c].mean() >= nat[c].mean() - DELTA) for c in suite.TRAJ_CRITERIA}
    out["mean_rule"] = all(out.values())
    return out


def floor_rule(arm_rows: pd.DataFrame, native_rows: pd.DataFrame) -> bool | None:
    """The suite's `beats_native` on seeds FLOOR_SEEDS only; None when the arm lacks one."""
    arm = arm_rows[arm_rows["seed"].isin(FLOOR_SEEDS)]
    nat = native_rows[native_rows["seed"].isin(FLOOR_SEEDS)]
    if set(arm["seed"]) != set(FLOOR_SEEDS) or set(nat["seed"]) != set(FLOOR_SEEDS):
        return None
    return suite.beats_native(arm, nat)["beats_native"]


def gate_row(arm_rows: pd.DataFrame, native_rows: pd.DataFrame, ilisi: pd.Series) -> dict:
    """Floor, mean and iLISI rules of one arm against native_d3; `ilisi` is that pair's
    v_strat_ilisi row of compare_pair."""
    return {
        "n": int(ilisi["n"]),
        "floor_rule": floor_rule(arm_rows, native_rows),
        **mean_rule(arm_rows, native_rows),
        "ilisi_mean_d": float(ilisi["mean_d"]),
        "ilisi_signflip_p": float(ilisi["signflip_p"]),
        "ilisi_rule": bool(ilisi["signflip_p"] < ALPHA and ilisi["mean_d"] > 0),
    }


def all_pairs(arms: list[str]) -> list[tuple[str, str]]:
    """Each arm (except native_d3) vs native_d3, then EXTRA_PAIRS."""
    return [(a, NATIVE_REF) for a in arms if a != NATIVE_REF] + EXTRA_PAIRS


def run_tests(
    rows: dict[str, pd.DataFrame], pairs: list[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(pair x metric table, gate table per arm vs native_d3, skipped pairs with reasons) for
    per-seed tables `rows` keyed by arm."""
    tables, gates, skipped = [], [], []
    for arm, ref in pairs:
        missing = [a for a in (arm, ref) if a not in rows]
        if missing:
            skipped.append(f"{arm} vs {ref}: no run folder or seeds for {', '.join(missing)}")
            continue
        try:
            table = compare_pair(rows[arm], rows[ref], METRICS)
        except ValueError as err:
            skipped.append(f"{arm} vs {ref}: {err}")
            continue
        tables.append(table.assign(arm=arm, ref=ref))
        if ref == NATIVE_REF:
            ilisi = table.set_index("metric").loc["v_strat_ilisi"]
            gates.append({"arm": arm, **gate_row(rows[arm], rows[ref], ilisi)})
    cols = ["arm", "ref", *compare_pair_columns()]
    results = pd.concat(tables, ignore_index=True)[cols] if tables else pd.DataFrame(columns=cols)
    return results, pd.DataFrame(gates), skipped


def compare_pair_columns() -> list[str]:
    """Column order of compare_pair's table."""
    return [
        "metric",
        "n",
        "mean_d",
        "min_d",
        "max_d",
        "n_pos",
        "n_neg",
        "sign_p",
        "signflip_p",
        "paired_seeds",
        "arm_only_seeds",
        "ref_only_seeds",
    ]


def md_table(df: pd.DataFrame) -> str:
    """A pipe table of `df`, floats to 4 decimals."""

    def fmt(x) -> str:
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_markdown(
    path: Path,
    results: pd.DataFrame,
    gates: pd.DataFrame,
    skipped: list[str],
    seeds: dict[str, tuple[int, ...]],
    absent: list[str],
) -> None:
    """paired_tests.md: seeds per arm, absent arms, skipped pairs, gate table, then per pair the
    metric table with its unpaired seeds."""
    out = ["# oetjen5 paired tests", ""]
    out += [f"d = arm − ref per seed; sign-flip p exact over 2ⁿ flips; δ = {DELTA}, α = {ALPHA}."]
    out += ["", "## Seeds per arm", ""]
    out += [f"- {arm}: {' '.join(map(str, s))}" for arm, s in seeds.items()]
    out += ["", "## Absent arms", "", *(f"- {a}" for a in absent or ["none"])]
    out += ["", "## Skipped pairs", "", *(f"- {s}" for s in skipped or ["none"])]
    out += ["", "## Gates vs native_d3", ""]
    out += ["floor = suite `beats_native` on seeds 0–2 (None: arm lacks one); mean = each mean"]
    out += [f"conservation ≥ native mean − {DELTA} on paired seeds; ilisi = p < {ALPHA} and d > 0."]
    out += ["", md_table(gates) if not gates.empty else "none"]
    for (arm, ref), table in results.groupby(["arm", "ref"], sort=False):
        first = table.iloc[0]
        out += ["", f"## {arm} vs {ref}", ""]
        out += [
            f"paired seeds: {first['paired_seeds']}; {arm} only: "
            f"{first['arm_only_seeds'] or 'none'}; {ref} only: {first['ref_only_seeds'] or 'none'}",
            "",
        ]
        out += [md_table(table[compare_pair_columns()[:9]])]
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=HERE, help="folder holding the run folders")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default <root>/1016_oetjen5_suite"
    )
    args = parser.parse_args()
    folders, absent = arm_folders(args.root)
    if NATIVE_REF not in folders:
        raise FileNotFoundError(f"{NATIVE_REF} run folder not found under {args.root}")
    seeds = {arm: seeds_present(path) for arm, path in folders.items()}
    strat_skipped: list[dict] = []
    rows = {
        arm: pd.DataFrame(suite.run_rows(path, seeds[arm], arm, strat_skipped))
        for arm, path in folders.items()
        if seeds[arm]
    }
    results, gates, skipped = run_tests(rows, all_pairs(list(folders)))
    out_dir = args.out_dir or args.root / "1016_oetjen5_suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "paired_tests.csv", index=False)
    write_markdown(out_dir / "paired_tests.md", results, gates, skipped, seeds, absent)
    print(f"absent arms: {absent or 'none'}")
    for line in skipped:
        print(f"skipped pair: {line}")
    print(
        f"wrote {len(results)} rows ({results.groupby(['arm', 'ref']).ngroups} pairs) to {out_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError) as err:
        sys.exit(f"ERROR: {err}")
