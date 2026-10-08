"""Stage 3/4 training driver for the BMMC study: native vs batch-conditioned Decipher.

Trains {native, BC-donor, BC-site, BC-sample} x decipherseeds x HVG set on the shared h5ad
(`1008_bmmc/bmmc_shared.h5ad`, built by 1008_bmmc_build_shared.py) through `decipher_models2`.
BC = the `model5` preset (batch concatenated into the decoder x-side); native = `native` preset.
One dataset, so the only seed is the decipherseed (repeat training seed).

Triage tiers (Stage 4 budget ~12 h of training):
    T1  4 configs x decipherseeds 1-3, 2000 HVG   (12 runs)
    T2  4 configs x decipherseeds 4-5, 2000 HVG   (+8)
    T3  native + BC-donor x decipherseeds 1-3, 5000 HVG   (+6)

Usage
-----
    python "Real Data/.../1008_bmmc_run_decipher.py" --tier T1 --dry-run
    nohup python ... --tier T1 --tag full --workers 5 > full_T1.log 2>&1 &

Outputs go to `1008_bmmc/runs/<tag>/`: one `<run_id>.h5ad` per run (obs + obsm decipher_v/z, no X)
and `ledger.csv`. Re-running resumes: ledger rows with status ok are skipped.

Train/val split: `decipher_train` always rebuilds `obs["decipher_split"]` itself, so a worker
replaces its module-level `_make_train_val_split` (looked up at call time) with one that copies
our `obs["split"]`. No package file is edited.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_HERE, "1008_bmmc")
SHARED_H5AD = os.path.join(OUT, "bmmc_shared.h5ad")
SHARED_SHA = os.path.join(OUT, "bmmc_shared.sha256")

PACKAGE = "decipher_models2"
WANDB_ENTITY = "jpark-columbia"
WANDB_PROJECT = "decipher-bc-bmmc"
WANDB_JOB_TYPE = "bmmc_run"
DEVICE = "cpu"
LEARNING_RATE = 1e-3  # simulation default
BETA = 0.1  # simulation default
DEFAULT_MAX_EPOCHS = 1000  # DecipherConfig default; early stopping (patience 10) ends runs sooner
DEFAULT_WORKERS = 2

# config name -> (preset, obs column the batch label is read from or None)
CONFIGS = {
    "native": ("native", None),
    "BC-donor": ("model5", "donor"),
    "BC-site": ("model5", "site"),
    "BC-sample": ("model5", "sample"),
}
TIERS = {
    "T1": dict(configs=list(CONFIGS), seeds=[1, 2, 3], hvg=2000),
    "T2": dict(configs=list(CONFIGS), seeds=[4, 5], hvg=2000),
    "T3": dict(configs=["native", "BC-donor"], seeds=[1, 2, 3], hvg=5000),
}
OBS_KEEP = ["split", "donor", "site", "sample", "cell_type", "lineage", "lineage_rank"]
OBS_KEEP += ["ref_pseudotime"]
LEDGER_COLUMNS = [
    "run_id", "config", "conditioned_on", "n_batches", "seed", "hvg", "n_cells", "n_train",
    "n_val", "n_epochs", "seconds_total", "seconds_per_epoch", "peak_rss_mb",
    "shared_h5ad_sha256", "package", "git_sha", "git_dirty", "wandb_run_id", "wandb_url",
    "status", "error",
]  # fmt: skip


# ----------------------------------------------------------------------- pure grid / subsample


def run_id_for(config: str, hvg: int, seed: int) -> str:
    return f"{config}_hvg{hvg}_ds{seed}"


def make_jobs(
    tier: str | None = None,
    configs: list[str] | None = None,
    seeds: list[int] | None = None,
    hvgs: list[int] | None = None,
    **shared,
) -> list[dict]:
    """Job dicts for a tier (comma-separated allowed, e.g. "T1,T2") or an explicit grid.

    `shared` (tag, use_wandb, max_cells, max_epochs, out_dir) is copied into every job: spawn
    re-imports this module in each worker, so per-run settings must travel in the job dict.
    """
    if tier:
        cells = []
        for name in tier.split(","):
            t = TIERS[name]
            cells += [(c, s, t["hvg"]) for c in t["configs"] for s in t["seeds"]]
    else:
        cells = [
            (c, s, h)
            for c in configs or list(CONFIGS)
            for s in seeds or [1, 2, 3]
            for h in hvgs or [2000]
        ]
    out = []
    for config, seed, hvg in cells:
        preset, batch_col = CONFIGS[config]
        out.append(
            dict(
                run_id=run_id_for(config, hvg, seed),
                config=config,
                preset=preset,
                conditioned_on=batch_col or "none",
                seed=seed,
                hvg=hvg,
                **shared,
            )
        )
    return out


def stratified_subsample(labels, max_cells: int, seed: int = 0) -> np.ndarray:
    """Sorted positional indices of ~max_cells cells, proportional per label, >= 1 per label."""
    labels = np.asarray(labels)
    n = len(labels)
    if max_cells >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    keep = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        k = max(1, round(len(idx) * max_cells / n))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


# ----------------------------------------------------------------------------------- helpers


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_shared_sha() -> str:
    """The sha256 recorded next to the shared h5ad, checked against the file; raises on mismatch."""
    with open(SHARED_SHA) as f:
        recorded = f.read().split()[0]
    actual = _sha256_of(SHARED_H5AD)
    if actual != recorded:
        raise RuntimeError(f"{SHARED_H5AD} sha256 {actual} != recorded {recorded}")
    return recorded


def _git_state() -> tuple:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_HERE,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


# ------------------------------------------------------------------------------------ worker


def _init_worker():
    """Pin every thread pool to 1 before numeric imports (numba and torch interop separately)."""
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
    torch.set_num_interop_threads(1)

    import pyro

    pyro.enable_validation(False)  # numerics-neutral speed-up, as in the split-fix sweep


def _use_shared_split(adata) -> None:
    """Make `decipher_train` use `adata.obs["split"]` (train/val) as its train/validation split."""
    import importlib

    import pandas as pd
    import scipy.sparse as sp

    mod = importlib.import_module("decipher_models2.tools.decipher")

    def _from_obs_split(adata_, val_frac, seed):
        split = np.where(adata_.obs["split"].astype(str) == "val", "validation", "train")
        adata_.obs["decipher_split"] = pd.Categorical(split, categories=["train", "validation"])

    mod._make_train_val_split = _from_obs_split

    def _sparse_integer_check(adata_):
        # Same test as the package's, on the stored values only: the package densifies X and
        # its allclose temporaries peak at ~5 GB on the full matrix.
        values = adata_.X.data if sp.issparse(adata_.X) else np.asarray(adata_.X)
        if not np.array_equal(values, np.round(values)):
            raise ValueError("adata.X must hold integer counts")

    mod.check_adata_has_integer_counts = _sparse_integer_check


def _start_wandb(job: dict):
    """One wandb run per job; None when disabled or init fails (never fatal)."""
    if not job["use_wandb"]:
        return None
    try:
        import wandb

        return wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group=job["config"],
            job_type=WANDB_JOB_TYPE,
            name=f"{job['run_id']}_{job['tag']}",
            tags=[job["config"], f"hvg{job['hvg']}", job["tag"]],
            config={
                "config": job["config"],
                "preset": job["preset"],
                "conditioned_on": job["conditioned_on"],
                "decipher_seed": job["seed"],
                "hvg": job["hvg"],
                "max_cells": job["max_cells"],
                "max_epochs": job["max_epochs"],
                "package": PACKAGE,
            },
            reinit="finish_previous",
        )
    except Exception as e:  # degrade to CSV-only
        print(f"  [wandb] init failed, CSV-only: {type(e).__name__}: {e}", flush=True)
        return None


def _train(job: dict, run) -> dict:
    """Load, subset, train, save the h5ad; return ledger fields."""
    import importlib
    import resource

    import anndata as ad

    sha = verified_shared_sha()
    # backed read + gene subset first: loading the whole 13,953-gene matrix peaks near 4.6 GB
    backed = ad.read_h5ad(SHARED_H5AD, backed="r")
    adata = backed[:, backed.var[f"hvg_{job['hvg']}"].to_numpy()].to_memory()
    backed.file.close()
    del adata.layers["counts"]  # identical to X
    if job["max_cells"]:
        adata = adata[stratified_subsample(adata.obs["sample"], job["max_cells"])].copy()
    adata.obs_names_make_unique()

    batch_col = None if job["conditioned_on"] == "none" else job["conditioned_on"]
    if batch_col:
        adata.obs["batch"] = adata.obs[batch_col].astype(str).astype("category")
    n_batches = adata.obs["batch"].nunique() if batch_col else 0

    dc = importlib.import_module(PACKAGE)
    DecipherConfig = importlib.import_module(f"{PACKAGE}.tools._decipher").DecipherConfig
    from decipher_models2.presets import PRESETS
    from decipher_models2.utils import DECIPHER_GLOBALS

    DECIPHER_GLOBALS["save_folder"] = os.path.join(
        _REPO_ROOT, "_decipher_models", "bmmc_1008", job["tag"], job["run_id"]
    )
    _use_shared_split(adata)
    config = DecipherConfig(
        **PRESETS[job["preset"]],
        learning_rate=LEARNING_RATE,
        beta=BETA,
        n_batches=n_batches,
        seed=job["seed"],
        n_epochs=job["max_epochs"],
    )
    t0 = time.perf_counter()
    decipher, _ = dc.tl.decipher_train(
        adata, config, batch_key="batch", device=DEVICE, plot_kwargs={"color": "lineage"}
    )
    seconds_total = time.perf_counter() - t0

    train_losses, val_losses = decipher.train_losses_, decipher.val_losses_
    n_epochs = len(val_losses)
    split = adata.obs["decipher_split"]
    git_sha, git_dirty = _git_state()
    rec = dict(
        n_batches=int(config.n_batches),
        n_cells=adata.n_obs,
        n_train=int((split == "train").sum()),
        n_val=int((split == "validation").sum()),
        n_epochs=n_epochs,
        seconds_total=round(seconds_total, 1),
        seconds_per_epoch=round(seconds_total / max(n_epochs, 1), 2),
        shared_h5ad_sha256=sha,
        package=PACKAGE,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )

    out = ad.AnnData(obs=adata.obs[OBS_KEEP].copy())
    out.obsm["decipher_v"] = np.asarray(adata.obsm["decipher_v"])
    out.obsm["decipher_z"] = np.asarray(adata.obsm["decipher_z"])
    os.makedirs(job["out_dir"], exist_ok=True)
    out.write_h5ad(os.path.join(job["out_dir"], f"{job['run_id']}.h5ad"))

    if run is not None:
        run.config.update(config.to_dict(), allow_val_change=True)
        for epoch, (tr, va) in enumerate(zip(train_losses, val_losses)):
            run.log({"train_elbo": tr, "val_nll": va}, step=epoch)
        final = dict(
            final_train_elbo=train_losses[-1],
            final_val_nll=val_losses[-1],
            n_epochs_trained=n_epochs,
            seconds_total=rec["seconds_total"],
            seconds_per_epoch=rec["seconds_per_epoch"],
            n_train=rec["n_train"],
            n_val=rec["n_val"],
        )
        run.log(final)
        run.summary.update(final)
    rec["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)
    return rec


def run_one(job: dict) -> dict:
    """Train one job. Returns a ledger row; never raises."""
    record = {k: job[k] for k in ("run_id", "config", "conditioned_on", "seed", "hvg")}
    run = _start_wandb(job)
    record["wandb_run_id"] = getattr(run, "id", None)
    record["wandb_url"] = getattr(run, "url", None)
    try:
        record.update(_train(job, run), status="ok", error=None)
    except Exception as e:  # one bad run must not kill the sweep
        record.update(status="failed", error=f"{type(e).__name__}: {e}")
        if run is not None:
            run.summary["status"] = "failed"
            run.summary["error"] = record["error"]
    finally:
        if run is not None:
            import matplotlib.pyplot as plt

            plt.close("all")
            run.finish()
    return record


# ------------------------------------------------------------------------------------ driver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tier", help="T1, T2, T3 or a comma list, e.g. T1,T2")
    ap.add_argument("--configs", help=f"comma list from {list(CONFIGS)} (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", help="decipherseeds (default 1 2 3)")
    ap.add_argument("--hvg", type=int, nargs="+", choices=[2000, 5000], help="default 2000")
    ap.add_argument("--max-cells", type=int, default=None, help="stratified subsample (smoke)")
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wandb", action="store_true", help="debug only; CSV ledger only")
    args = ap.parse_args(argv)

    out_dir = os.path.join(OUT, "runs", args.tag)
    ledger_path = os.path.join(out_dir, "ledger.csv")
    all_jobs = make_jobs(
        tier=args.tier,
        configs=args.configs.split(",") if args.configs else None,
        seeds=args.seeds,
        hvgs=args.hvg,
        tag=args.tag,
        use_wandb=not args.no_wandb,
        max_cells=args.max_cells,
        max_epochs=args.max_epochs,
        out_dir=out_dir,
    )
    prior = pd.read_csv(ledger_path) if os.path.exists(ledger_path) else None
    done = set() if prior is None else set(prior.loc[prior["status"] == "ok", "run_id"])
    todo = [j for j in all_jobs if j["run_id"] not in done]
    print(
        f"grid: {len(all_jobs)} runs, already ok: {len(all_jobs) - len(todo)}, to run: {len(todo)}"
    )
    print(f"workers: {args.workers}  device: {DEVICE}  ledger: {ledger_path}")
    print("wandb: " + ("disabled" if args.no_wandb else f"{WANDB_ENTITY}/{WANDB_PROJECT}"))
    if args.dry_run:
        for j in todo:
            print("   ", j["run_id"], j["preset"], j["conditioned_on"])
        return 0
    if not todo:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    rows = [] if prior is None else prior[prior["run_id"].isin(done)].to_dict("records")
    t0 = time.time()
    # spawn, not fork: workers must not inherit RNG state or torch thread config.
    # maxtasksperchild=1: a fresh process per job, so peak_rss_mb is that job's own.
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, maxtasksperchild=1) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            rows.append(rec)
            pd.DataFrame(rows).reindex(columns=LEDGER_COLUMNS).to_csv(ledger_path, index=False)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i) / 60
            print(
                f"[{i}/{len(todo)}] {rec['run_id']} {rec['status']} "
                f"{rec.get('seconds_total', '')}s {rec.get('n_epochs', '')}ep "
                f"{rec['error'] or ''}  ETA {eta:.0f}m",
                flush=True,
            )
    failed = sum(r["status"] != "ok" for r in rows)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(rows)} rows, {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
