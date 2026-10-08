"""Track when a batch lands in a mirrored v frame during Decipher training (simulated data only).

Reruns ONE sweep job through the sweep driver's `train_and_compute_rho_r2_bifurcation` (default:
the 0918 batch03 case -- sigma=2, seed=3, model5, bif, gene-mode, decipher seed 1) and, after
every epoch, records for each batch the sign of det of the Procrustes map decipher_v -> latent_v
(+1 / -1; a batch whose sign differs from the others is mirrored), how ambiguous that sign is
(`ambig`, 0 = clear, 1 = undecidable), and how far the batch sits from the other batches in v
(`overlap`, median nearest-neighbour distance).

Neither the driver nor `decipher_train` is edited. `decipher_train` has no callback, but with
`plot_every_k_epochs=1` it calls `_decipher_to_adata` (fills `adata.obsm["decipher_v"]`) and then
`plot_decipher_v(adata, ...)` once per epoch, plus once more after training. This script wraps
both for the duration of the call: `decipher_train` gets `plot_every_k_epochs=1`, and
`plot_decipher_v` becomes the hook (it draws nothing). Training itself is unchanged -- the hook
uses no RNG and the model is already in eval mode when it runs. A side effect of the plot switch:
`decipher_train` also writes a blank `decipher_training.gif` into `_decipher_models/<run_id>/`.

Outputs, next to this script:
  <name>.csv   one row per epoch: epoch, phase, det_<batch>, ambig_<batch>, overlap_<batch>, n_minority
  wandb        same run as the driver's loss curves; metrics `flip/*` against x-axis `flip/epoch`
               (logged after training, so they don't collide with the driver's step-indexed
               `train_elbo` / `val_nll`). --no-wandb for CSV only.

Usage:  python 0929_track_flip_during_training.py [--sigma 2.0] [--seed 3] [--model model5]
            [--batch-mode genes] [--decipher-seed 1] [--no-wandb]
The driver also writes its usual figures / h5ad under `Sweeps and Results/.../<today>/`.
"""

import argparse
import csv
import importlib
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Simulated Data" / "Simulated Data Sweep Pipeline"))

WANDB_ENTITY = "jpark-columbia"
WANDB_PROJECT = "decipher-bc-sigma-sweep"
# same as SIM_KWARGS in 0918_sweep_3models_bifurcation.py
SIM_KWARGS = dict(
    n_batches=5, n_samples=500, n_genes=200, n_z_dims=3, biological_sigma=0.1, beta=0.1
)


def flip_metrics(adata):
    """Per batch: Procrustes det sign of decipher_v -> latent_v, and median NN distance to the rest."""
    V, L = adata.obsm["decipher_v"], adata.obsm["latent_v"]
    batch = adata.obs["batch"].astype(str).values
    out = {}
    for b in sorted(set(batch)):
        m = batch == b
        Vc, Lc = V[m] - V[m].mean(0), L[m] - L[m].mean(0)
        R, _ = orthogonal_procrustes(Vc, Lc)
        out[f"det_{b}"] = int(np.sign(np.linalg.det(R)))
        # How firmly the sign is decided: the best rotation and the best reflection fit with
        # trace s1+s2 and s1-s2 (singular values of Vc.T @ Lc). ambig -> 1 means the batch is
        # nearly a line in v, so both fit about equally well and the sign is not to be trusted.
        s1, s2 = np.linalg.svd(Vc.T @ Lc, compute_uv=False)
        out[f"ambig_{b}"] = float((s1 - s2) / (s1 + s2))
        out[f"overlap_{b}"] = float(np.median(cKDTree(V[~m]).query(V[m])[0]))
    dets = np.array([v for k, v in out.items() if k.startswith("det_")])
    majority = 1 if dets.sum() >= 0 else -1
    out["n_minority"] = int((dets != majority).sum())
    return out


class FlipRecorder:
    """Stands in for `plot_decipher_v`; called once per epoch and once after training."""

    def __init__(self, csv_path):
        self.csv_path, self.rows, self.epoch, self._writer = csv_path, [], 0, None

    def __call__(self, adata, *args, **kwargs):
        # decipher_save_model sets run_id right before decipher_train's final call
        phase = "final" if "run_id" in adata.uns.get("decipher", {}) else "train"
        row = {"epoch": self.epoch, "phase": phase, **flip_metrics(adata)}
        self.rows.append(row)
        if self._writer is None:
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        print(
            f"  [flip] epoch {row['epoch']:3d} {phase:5s} "
            + " ".join(f"{k[4:]}={v:+d}" for k, v in row.items() if k.startswith("det_")),
            flush=True,
        )
        if phase == "train":
            self.epoch += 1

    def close(self):
        if self._writer is not None:
            self._file.close()


def init_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb

        return wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            group="flip_tracking",
            job_type="flip_tracking",
            name=f"flip_{args.model}_bif_{args.batch_mode}_sigma{args.sigma}_seed{args.seed}"
            f"_decipherseed{args.decipher_seed}",
            tags=[args.model, "flip_tracking", f"sigma_{args.sigma}"],
            config=dict(
                model=args.model,
                batch_mode=args.batch_mode,
                shift_sigma=args.sigma,
                seed=args.seed,
                decipher_seed=args.decipher_seed,
                **SIM_KWARGS,
            ),
        )
    except Exception as e:  # wandb must never kill the run (sweep convention)
        print(f"  [wandb] init failed, continuing CSV-only: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model", default="model5")
    ap.add_argument("--batch-mode", default="genes")
    ap.add_argument("--decipher-seed", type=int, default=1)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    bif = importlib.import_module("0918_simulated_data_pipeline_bifurcation")
    dc = importlib.import_module("decipher_models")
    dc_tools = importlib.import_module("decipher_models.tools.decipher")

    csv_path = HERE / (
        f"0929_flip_tracking_{args.model}_sigma{args.sigma}_seed{args.seed}_bif_{args.batch_mode}"
        f"_decipherseed_{args.decipher_seed}.csv"
    )
    recorder = FlipRecorder(csv_path)
    run = init_wandb(args)

    orig_train, orig_plot = dc.tl.decipher_train, dc_tools.plot_decipher_v
    dc.tl.decipher_train = partial(orig_train, plot_every_k_epochs=1)
    dc_tools.plot_decipher_v = recorder
    try:
        bif.train_and_compute_rho_r2_bifurcation(
            args.model,
            args.decipher_seed,
            shift_sigma=args.sigma,
            seed=args.seed,
            batch_mode=args.batch_mode,
            branching_t=0.3,
            notebook_tag="flip_tracking",
            wandb_run=run,
            **SIM_KWARGS,
        )
    finally:
        dc.tl.decipher_train, dc_tools.plot_decipher_v = orig_train, orig_plot
        recorder.close()

    if run is not None:
        run.define_metric("flip/epoch")
        run.define_metric("flip/*", step_metric="flip/epoch")
        for row in recorder.rows:
            run.log({f"flip/{k}": v for k, v in row.items() if k != "phase"})
        run.finish()
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
