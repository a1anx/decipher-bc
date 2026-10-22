"""Split-fix test sweep: does fixing the train/val split change training?

ADDED 2026-10-08. Copy of 0918_sweep_3models_bifurcation.py (left untouched) that trains the
same cells of the 0918 grid -- arms native:genes and model5:genes, bifurcation on, the 0918
sigmas, sim seeds 3-5, decipher seeds 1-3 = 126 runs -- through `decipher_models2`, the copy
of `decipher_models` in which the train/val split is fixed in code (relay Step 4; the
simulator's `ad.concat` repeats names across batches, and the original split picks validation
cells by name, giving ~58/42 instead of 90/10). The data are unchanged, so each run here differs from its control
twin by the split (plus numerics-neutral speed-ups) alone.

COMPARE TO the control ledger (`sweep1008_control`, the same 126 jobs through the unmodified 0918
path on the same machine type): the KEY columns are identical, so the two ledgers join 1:1.
The CSV also records n_train / n_val and the package, so the split of each run is auditable.

Usage
-----
    python "Simulated Data/Simulated Data Sweep Pipeline/1008_sweep_splitfix_bifurcation.py" --dry-run
    nohup python ... --workers 1 > sweep.log 2>&1 &

`--package decipher_m5 --tag <tag>` trains the same cells through `decipher_m5` (model5 with
embedding tables) instead; its figures go to `<MMDD>/figs_decipher_m5/`.

Re-running resumes: rows already completed in the CSV are skipped. The output dir (%m%d) is fixed
once at launch and passed to every job, so a run crossing midnight does not split across dirs.
"""

import argparse
import importlib
import os
import sys
import time
from datetime import datetime
from multiprocessing import get_context

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ----------------------------------------------------------------------------------- the grid

# (display name, preset, batch_mode)
ARMS = [
    ("Base Decipher (genes)", "native", "genes"),
    ("Set3 zx2", "model5", "genes"),
]
BIFURCATIONS = {"bif": 0.3}  # label -> branching_t
SHIFT_SIGMAS = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
SEEDS = [3, 4, 5]  # simulation seed: which DATASET (projections, cells, shifts)
# training seed: SVI init only, same dataset. Three of them separate training variance from
# simulation variance -- with one, an rho gap between arms could just be a lucky initialization.
DECIPHER_SEEDS = [1, 2, 3]

# data size unchanged from the existing aug1 sweeps, for comparability
SIM_KWARGS = dict(
    n_batches=5, n_samples=500, n_genes=200, n_z_dims=3, biological_sigma=0.1, beta=0.1
)

NOTEBOOK_TAG = "sweep1008_splitfix"
PACKAGE = "decipher_models2"  # default model package (--package); recorded per row
DEFAULT_WORKERS = 1  # this VM has 1 vCPU; raise with --workers on a bigger one

DEVICE = "cpu"  # the T4's driver is not installed; see the plan. Tiny model, 168 independent
# jobs, 32 cores -- CPU throughput wins. Kept a parameter so a benchmark can flip it.

# Columns that identify a run. Resume matches on these.
KEY = ["model", "batch_mode", "bifurcation", "shift_sigma", "seed", "decipher_seed"]


def jobs(notebook_tag=NOTEBOOK_TAG, run_date=None, package=PACKAGE):
    """The full grid, as a list of dicts. Order is deterministic.

    `notebook_tag` is carried IN each job rather than read from module scope by the worker,
    because "spawn" re-imports this module in every child: a value assigned in the parent's
    main() would silently revert to the module default inside run_one, and the outputs would be
    written under the wrong tag. Same for `run_date`.
    """
    out = []
    for name, preset, batch_mode in ARMS:
        for bif_label, branching_t in BIFURCATIONS.items():
            for shift_sigma in SHIFT_SIGMAS:
                for seed in SEEDS:
                    for decipher_seed in DECIPHER_SEEDS:
                        out.append(
                            dict(
                                arm=name,
                                model=preset,
                                batch_mode=batch_mode,
                                bifurcation=bif_label,
                                branching_t=branching_t,
                                shift_sigma=shift_sigma,
                                seed=seed,
                                decipher_seed=decipher_seed,
                                notebook_tag=notebook_tag,
                                run_date=run_date,
                                package=package,
                            )
                        )
    return out


# ------------------------------------------------------------------------------ the worker


def _init_worker():
    """Per-process setup. Every line here is load-bearing.

    a) Thread count -- THREE SEPARATE POOLS, not one.
       MEASURED after the first launch went to load average 509 with 16 workers: setting
       OMP_NUM_THREADS/MKL_NUM_THREADS + torch.set_num_threads(1) was not enough.
         - numba (pulled in by scanpy.pp.neighbors -> pynndescent, used every single run)
           defaults to one thread PER CORE -- 32 here -- and ignores OMP_NUM_THREADS entirely;
           it only reads NUMBA_NUM_THREADS, and only at numba's own import time.
         - torch's INTEROP thread pool (separate from the intraop pool set.num_threads
           controls) defaulted to 16 and was untouched by any of the above.
       16 workers x up to 32 numba threads is up to 512 -- almost exactly the observed load.
       All of this must be set before the respective library is first imported/used, which is
       why the env vars are set before ANY of matplotlib/torch/numba is imported.
    b) Agg backend, because each run saves a reconstruction figure and workers have no display.
    c) A private model directory per process. decipher_save_model names runs
       "<timestamp>-<random words>" and decipher_rotate_space READS THAT DIRECTORY BACK via
       decipher_load_model -- so two workers colliding on a run_id would have one load the
       other's weights and report metrics for a model it never trained, silently. A private
       absolute folder makes collision impossible, and also fixes that the default
       "./_decipher_models/" is relative to cwd.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[var] = "1"

    import matplotlib

    matplotlib.use("Agg")

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)  # separate pool from set_num_threads; not covered by it

    # Numerics-neutral speed-up (~20% per epoch): skip Pyro's per-step argument validation.
    # Sweep workers only; tests and notebooks keep validation on.
    import pyro

    pyro.enable_validation(False)

    save_folder = os.path.join(_REPO_ROOT, "_decipher_models", f"sweep_{os.getpid()}")
    for package in ("decipher_models2", "decipher_m5"):
        importlib.import_module(f"{package}.utils").DECIPHER_GLOBALS["save_folder"] = save_folder


def run_one(job):
    """Train one model on one dataset. Returns a CSV row; never raises."""
    import numpy as np

    bif = importlib.import_module("1008_simulated_data_pipeline_splitfix")

    record = {k: job[k] for k in KEY}
    record["arm"] = job["arm"]
    started = time.time()
    try:
        (
            rho,
            rho_plus,
            rho_minus,
            branch_asw,
            trained_path,
            r2_overall,
            r2_per_gene_median,
            _adata,
        ) = bif.train_and_compute_rho_r2_bifurcation(
            job["model"],
            job["decipher_seed"],
            shift_sigma=job["shift_sigma"],
            seed=job["seed"],
            batch_mode=job["batch_mode"],
            branching_t=job["branching_t"],
            notebook_tag=job["notebook_tag"],
            run_date=job["run_date"],
            arm=job["arm"],
            package=job["package"],
            **SIM_KWARGS,
        )
        record.update(
            rho=rho,
            rho_plus=rho_plus,
            rho_minus=rho_minus,
            branch_asw=branch_asw,
            r2_overall=r2_overall,
            r2_per_gene_median=r2_per_gene_median,
            trained_h5ad=trained_path,
            n_train=_adata.uns["n_train"],
            n_val=_adata.uns["n_val"],
            n_epochs=_adata.uns["n_epochs"],
            git_sha=_adata.uns["git_sha"],
            git_dirty=_adata.uns["git_dirty"],
            **{f"seconds_{k}": round(v, 1) for k, v in _adata.uns["stage_seconds"].items()},
            package=job["package"],
            error=None,
        )
    except Exception as e:  # one bad cell must not take down 167 others
        record.update(
            rho=np.nan,
            rho_plus=np.nan,
            rho_minus=np.nan,
            branch_asw=np.nan,
            r2_overall=np.nan,
            r2_per_gene_median=np.nan,
            trained_h5ad=None,
            n_train=np.nan,
            n_val=np.nan,
            n_epochs=np.nan,
            git_sha=None,
            git_dirty=None,
            package=job["package"],
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        # decipher_train ends by calling plot_decipher_v without closing the figure, so every
        # run leaks one otherwise.
        import matplotlib.pyplot as plt

        plt.close("all")
    record["seconds"] = round(time.time() - started, 1)
    return record


# ------------------------------------------------------------------------------ the driver


def completed_keys(csv_path):
    """Keys of runs already finished successfully, for resume."""
    if not os.path.exists(csv_path):
        return set(), None
    df = pd.read_csv(csv_path)
    if df.empty:
        return set(), df
    ok = df[df["error"].isna()] if "error" in df.columns else df
    # branching_t is float-or-None; the CSV carries the label, so KEY compares cleanly
    return set(map(tuple, ok[KEY].astype(str).values.tolist())), df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="list the grid and exit")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N jobs")
    ap.add_argument("--tag", default=NOTEBOOK_TAG)
    ap.add_argument("--package", default=PACKAGE, choices=["decipher_models2", "decipher_m5"])
    ap.add_argument("--models", nargs="+", default=None, help="keep only these presets")
    # Subsetting flags exist so a smoke run is a normal invocation of THIS FILE. Do not try to
    # shrink the grid by importing this module and reassigning SHIFT_SIGMAS: the pool uses the
    # "spawn" start method, which re-executes __main__ from its path in every child, so a module
    # loaded from a heredoc or via importlib.spec_from_file_location makes every worker die with
    # FileNotFoundError: '<stdin>' -- and Pool respawns them forever, burning CPU with zero
    # progress and no visible error. Measured: 41 minutes, 0 runs completed.
    ap.add_argument("--sigmas", type=float, nargs="+", default=None, help="override SHIFT_SIGMAS")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override SEEDS")
    ap.add_argument(
        "--decipher-seeds",
        type=int,
        nargs="+",
        default=None,
        help="override DECIPHER_SEEDS",
    )
    args = ap.parse_args()

    global SHIFT_SIGMAS, SEEDS, DECIPHER_SEEDS
    if args.sigmas:
        SHIFT_SIGMAS = args.sigmas
    if args.seeds:
        SEEDS = args.seeds
    if args.decipher_seeds:
        DECIPHER_SEEDS = args.decipher_seeds

    today = datetime.now().strftime("%m%d")  # once: every job and the CSV share this dir
    out_dir = os.path.join(
        _HERE, "..", "Sweeps and Results", "shift_sigma_sweep_bifurcation", today
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.tag}_sweep_log.csv")

    all_jobs = jobs(
        notebook_tag=args.tag,
        run_date=today,
        package=args.package,
    )
    done, prior = completed_keys(csv_path)
    if args.models:
        all_jobs = [j for j in all_jobs if j["model"] in args.models]
    todo = [j for j in all_jobs if tuple(str(j[k]) for k in KEY) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(
        f"grid      : {len(all_jobs)} runs  ({len(ARMS)} arms x {len(BIFURCATIONS)} bifurcation "
        f"x {len(SHIFT_SIGMAS)} sigmas x {len(SEEDS)} sim seeds "
        f"x {len(DECIPHER_SEEDS)} decipher seeds)"
    )
    print(f"already ok: {len(done)}")
    print(f"to run    : {len(todo)}   workers: {args.workers}   device: {DEVICE}")
    print(f"package   : {args.package}")
    print(f"csv       : {csv_path}")
    if args.dry_run:
        for j in todo[:12]:
            print("   ", {k: j[k] for k in KEY})
        if len(todo) > 12:
            print(f"    ... and {len(todo) - 12} more")
        return 0
    if not todo:
        print("nothing to do.")
        return 0

    rows = [] if prior is None else prior.to_dict("records")
    t0 = time.time()
    # spawn, not fork: workers must not inherit the parent's RNG state or torch thread config
    ctx = get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            rows.append(rec)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            status = rec["error"] or f"rho={rec['rho']:.3f} r2={rec['r2_overall']:.3f}"
            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i)
            print(
                f"[{i:3d}/{len(todo)}] {rec['arm']:22s} {rec['bifurcation']:5s} "
                f"{rec['batch_mode']:5s} sigma={rec['shift_sigma']:<4} seed={rec['seed']} "
                f"{rec['seconds']:6.1f}s  {status}   ETA {eta/60:.0f}m",
                flush=True,
            )

    # r.get("error") is wrong here: rows loaded from a resumed CSV carry error as pandas'
    # float('nan') (truthy in Python), while rows written live this run use None (falsy) --
    # so a resumed run overcounts failures by exactly its prior row count. pd.notna() treats
    # both consistently. Confirmed against this run's own CSV: this line printed "16 failed"
    # for a sweep whose CSV had zero non-null error values.
    failed = sum(1 for r in rows if pd.notna(r.get("error")))
    print(f"\ndone in {(time.time() - t0)/60:.1f} min. {len(rows)} rows, {failed} failed.")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
