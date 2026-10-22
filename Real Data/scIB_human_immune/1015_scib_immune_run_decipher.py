"""Native vs model5 Decipher (decipher_m5) on the scIB human-immune atlas, Villani dropped.

Run from the repo root with `.venv/bin/python`:
    python "Real Data/scIB_human_immune/1015_scib_immune_run_decipher.py" --dry-run
    nohup python ".../1015_scib_immune_run_decipher.py" --tag seurat_v3 > run.log 2>&1 &

Input: 2000 batch-aware HVGs (flavor seurat_v3 on the raw-counts layer, batch_key=batch), raw counts as X, 90/10 train/val split
stratified by batch. model5 conditions on `batch` (9 batches); native is batch-blind.
Per run: reconstruction diagnostic and v-space PNGs,
a small h5ad (obs + decipher_v/z) and a ledger row. Re-running resumes ok rows.

Reuses the worker setup, shared-split patch and git helper from the BMMC driver.
"""

import argparse
import importlib
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_HERE, "out")
SRC_H5AD = os.path.join(_HERE, "data", "Immune_ALL_human.h5ad")
SHARED_H5AD = os.path.join(OUT, "immune_no_villani_hvg2000_seurat_v3.h5ad")

PACKAGE = "decipher_m5"
DEVICE = "cpu"
LEARNING_RATE = 1e-3  # as in the simulation and BMMC runs
BETA = 0.1
DEFAULT_MAX_EPOCHS = 1000  # early stopping (patience 10) ends runs sooner
DEFAULT_WORKERS = 2
N_HVG = 2000
HVG_FLAVOR = "seurat_v3"  # raw counts, as in the scvi-tools tutorials
DROP_BATCH = "Villani"
SPLIT_SEED = 0
CONFIGS = {"native": "native", "model5": "model5"}  # config name -> preset
OBS_KEEP = ["split", "batch_label", "cell_type", "tissue", "dpt_pseudotime"]
SHARED_OBS = ["split", "batch", "cell_type", "tissue", "dpt_pseudotime"]
LEDGER_COLUMNS = [
    "run_id", "config", "n_batches", "seed", "n_cells", "n_train", "n_val", "n_epochs", "best_epoch",
    "final_train_elbo", "final_val_nll", "r2_overall", "seconds_total", "peak_rss_mb",
    "git_sha", "git_dirty", "status", "error",
]  # fmt: skip


def _bmmc():
    """The BMMC driver module (name starts with a digit, so importlib)."""
    here = os.path.join(_REPO_ROOT, "Real Data", "Healthy Human Bone Marrow Mononuclear Cells")
    if here not in sys.path:
        sys.path.insert(0, here)
    return importlib.import_module("1008_bmmc_run_decipher")


def build_shared() -> str:
    """Drop Villani, pick 2000 batch-aware HVGs, put raw counts in X, add a stratified split."""
    if os.path.exists(SHARED_H5AD):
        return SHARED_H5AD
    import anndata as ad
    import scanpy as sc

    adata = ad.read_h5ad(SRC_H5AD)
    adata = adata[adata.obs["batch"] != DROP_BATCH].copy()
    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")
    adata.obs["tissue"] = adata.obs["tissue"].astype(str)
    adata.obs["cell_type"] = adata.obs["final_annotation"].astype(str)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=N_HVG, batch_key="batch", flavor=HVG_FLAVOR, layer="counts", subset=False
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    adata.X = adata.layers["counts"].copy()
    del adata.layers["counts"]
    rng = np.random.default_rng(SPLIT_SEED)
    split = np.full(adata.n_obs, "train", dtype=object)
    for b in adata.obs["batch"].cat.categories:
        idx = np.where(adata.obs["batch"] == b)[0]
        split[rng.choice(idx, int(round(0.1 * len(idx))), replace=False)] = "val"
    adata.obs["split"] = split
    adata.obs = adata.obs[SHARED_OBS]
    os.makedirs(OUT, exist_ok=True)
    adata.write_h5ad(SHARED_H5AD)
    print(f"built {SHARED_H5AD}: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    return SHARED_H5AD


def make_jobs(configs=None, seeds=None, **shared) -> list[dict]:
    return [
        dict(run_id=f"{c}_hvg{N_HVG}_ds{s}", config=c, preset=CONFIGS[c], seed=s, **shared)
        for c in configs or list(CONFIGS)
        for s in seeds or [1, 2, 3]
    ]


def write_vspace(adata, job, fig_dir):
    import matplotlib.pyplot as plt

    dc = importlib.import_module(PACKAGE)
    panels = ["cell_type", "batch_label", "tissue"]
    shuffled = adata[np.random.default_rng(0).permutation(adata.n_obs)].copy()  # no draw-order bias
    fig = dc.pl.decipher(shuffled, color=panels, basis="decipher_v", ncols=3, wspace=0.45)
    for ax, name in zip(fig.axes, panels):  # the package otherwise titles every panel
        ax.set_title(name)
    fig.suptitle(
        f"scIB immune | {job['config']} | decipher seed={job['seed']}", y=1.05, fontsize=11
    )
    path = os.path.join(fig_dir, f"vspace_{job['config']}_ds{job['seed']}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def write_reconstruction(decipher, adata, job, fig_dir):
    import matplotlib.pyplot as plt

    dc = importlib.import_module(PACKAGE)
    fig, axes = dc.pl.reconstruction_r2_log1p(decipher, adata, figsize=(21, 5))
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(labelsize=8)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(8)
    fig.suptitle(f"{job['config']} | decipher seed={job['seed']}", y=1.02, fontsize=11)
    path = os.path.join(fig_dir, f"reconstruction_{job['config']}_ds{job['seed']}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _train(job) -> dict:
    import anndata as ad
    import resource

    bmmc = _bmmc()
    adata = ad.read_h5ad(SHARED_H5AD)
    if job["max_cells"]:
        keep = adata.obs.groupby(["batch", "split"], observed=True).sample(
            frac=job["max_cells"] / adata.n_obs, random_state=0
        )  # per batch x split, so every batch stays in train and val
        adata = adata[adata.obs_names.isin(keep.index)].copy()
    adata.obs_names_make_unique()
    adata.obs["batch"] = adata.obs["batch"].astype(str).astype("category")  # drop unused levels
    adata.obs["batch_label"] = adata.obs["batch"]  # plotting copy; native must not see `batch`
    if job["config"] == "native":
        del adata.obs["batch"]
    n_batches = 0 if job["config"] == "native" else adata.obs["batch"].nunique()

    dc = importlib.import_module(PACKAGE)
    DecipherConfig = importlib.import_module(f"{PACKAGE}.tools._decipher").DecipherConfig
    from decipher_m5.presets import PRESETS
    from decipher_m5.utils import DECIPHER_GLOBALS

    DECIPHER_GLOBALS["save_folder"] = os.path.join(
        _REPO_ROOT, "_decipher_models", "scib_immune_1015", job["tag"], job["run_id"]
    )
    bmmc._use_shared_split(adata)
    config = DecipherConfig(
        **PRESETS[job["preset"]], learning_rate=job["lr"], beta=BETA, n_batches=n_batches,
        seed=job["seed"], n_epochs=job["max_epochs"],
    )  # fmt: skip
    t0 = time.perf_counter()
    decipher, _ = dc.tl.decipher_train(
        adata, config, batch_key="batch", device=DEVICE, plot_kwargs={"color": "cell_type"}
    )
    seconds = time.perf_counter() - t0
    train_losses, val_losses = decipher.train_losses_, decipher.val_losses_
    split = adata.obs["decipher_split"]
    sha, dirty = bmmc._git_state()
    rec = dict(
        n_batches=int(config.n_batches), n_cells=adata.n_obs,
        n_train=int((split == "train").sum()), n_val=int((split == "validation").sum()),
        n_epochs=len(val_losses), best_epoch=decipher.best_epoch_, final_train_elbo=train_losses[-1], final_val_nll=val_losses[-1],
        seconds_total=round(seconds, 1), git_sha=sha, git_dirty=dirty,
    )  # fmt: skip

    fig_dir = os.path.join(job["out_dir"], "figs")
    os.makedirs(fig_dir, exist_ok=True)
    r2 = dc.tl.reconstruction_r2_log1p(decipher, adata)
    rec["r2_overall"] = float(r2["r2_overall"])
    write_reconstruction(decipher, adata, job, fig_dir)
    write_vspace(adata, job, fig_dir)

    out = ad.AnnData(obs=adata.obs[OBS_KEEP].copy())
    out.obsm["decipher_v"] = np.asarray(adata.obsm["decipher_v"])
    out.obsm["decipher_z"] = np.asarray(adata.obsm["decipher_z"])
    out.write_h5ad(os.path.join(job["out_dir"], f"{job['run_id']}.h5ad"))
    rec["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)
    return rec


def run_one(job) -> dict:
    """Train one job. Returns a ledger row; never raises."""
    record = {k: job[k] for k in ("run_id", "config", "seed")}
    try:
        record.update(_train(job), status="ok", error=None)
    except Exception as e:  # one bad run must not kill the sweep
        record.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        import matplotlib.pyplot as plt

        plt.close("all")
    return record


def _worker_init():
    _bmmc()._init_worker()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--configs", help=f"comma list from {list(CONFIGS)} (default: both)")
    ap.add_argument("--seeds", type=int, nargs="+", help="decipherseeds (default 1 2 3)")
    ap.add_argument("--max-cells", type=int, default=None, help="stratified subsample (smoke)")
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate (default 1e-3)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--tag", default="seurat_v3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_dir = os.path.join(OUT, "runs", args.tag)
    ledger_path = os.path.join(out_dir, "ledger.csv")
    all_jobs = make_jobs(
        configs=args.configs.split(",") if args.configs else None, seeds=args.seeds,
        tag=args.tag, max_cells=args.max_cells,
        max_epochs=args.max_epochs, out_dir=out_dir, lr=args.lr,
    )  # fmt: skip
    prior = pd.read_csv(ledger_path) if os.path.exists(ledger_path) else None
    done = set() if prior is None else set(prior.loc[prior["status"] == "ok", "run_id"])
    todo = [j for j in all_jobs if j["run_id"] not in done]
    print(f"grid: {len(all_jobs)} runs, to run: {len(todo)}  workers: {args.workers}")
    if args.dry_run or not todo:
        for j in todo:
            print("   ", j["run_id"], j["preset"])
        return 0

    build_shared()
    os.makedirs(out_dir, exist_ok=True)
    rows = [] if prior is None else prior[prior["run_id"].isin(done)].to_dict("records")
    t0 = time.time()
    # spawn + one task per child: no inherited RNG/thread state, per-job peak RSS
    with get_context("spawn").Pool(
        args.workers, initializer=_worker_init, maxtasksperchild=1
    ) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            rows.append(rec)
            pd.DataFrame(rows).reindex(columns=LEDGER_COLUMNS).to_csv(ledger_path, index=False)
            eta = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {rec['run_id']} {rec['status']} {rec.get('n_epochs', '')}ep "
                  f"{rec['error'] or ''}  ETA {eta:.0f}m", flush=True)  # fmt: skip
    failed = sum(r["status"] != "ok" for r in rows)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(rows)} rows, {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
