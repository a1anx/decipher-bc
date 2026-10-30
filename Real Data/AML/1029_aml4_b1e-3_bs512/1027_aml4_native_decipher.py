"""Native Decipher (published `scdecipher` from PyPI, batch-blind) on the aml4 h5ad.

`1027_aml4_common.py` (loaded by path from this folder, with `scib_metrics.py` through it) holds
the evaluation shared with the d3 driver. The parser, the hyperparameters (`--lr`, `--beta`,
`--batch-size`, `--dim-z`) and the wrapped training step (`train_package`: the package's SVI and
validation wrapped, not edited, to record per-epoch train ELBO / val NLL on the file's split;
last-epoch weights, as the package keeps them) live here. v-space panels as in the d3 driver,
titled "(No Batch Correction)". Run from this folder in an environment with `scdecipher` installed:
    python 1027_aml4_native_decipher.py --seed 0 --beta 1e-3 --batch-size 512 --out-dir native
Defaults: `aml4/aml4_shared.h5ad` (all genes), dataset `aml4`, scdecipher package defaults
(`--lr 5e-3 --beta 0.1 --batch-size 64 --dim-z 10`), package early stopping (patience 10 on val
NLL), `--max-epochs 1000`; out-dir `native/` next to this script. Each epoch is printed as it
finishes and the per-epoch losses go to `csv/seed<k>_loss.csv` after training. The package cannot
resume, so a rerun starts over.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_cspec = importlib.util.spec_from_file_location("aml4_common", HERE / "1027_aml4_common.py")
common = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(common)
native = common.native

SRC = common.SRC


DIM_Z = 10  # the package default


def build_parser() -> argparse.ArgumentParser:
    """`--seed`, `--max-epochs`, `--out-dir`, `--src`, `--dataset` with the aml4 defaults, plus
    the hyperparameters; `--out-dir` defaults to `native/` next to this script."""
    hp = common.HYPERPARAMS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="decipher seed (init, batch order)")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=None, help="default native/ here")
    parser.add_argument(
        "--src", type=Path, default=SRC, help="<src stem>.sha256 must sit beside it"
    )
    parser.add_argument("--dataset", default=common.DATASET)
    parser.add_argument("--lr", type=float, default=hp["lr"])
    parser.add_argument("--beta", type=float, default=hp["beta"])
    parser.add_argument("--batch-size", type=int, default=hp["batch_size"])
    parser.add_argument("--dim-z", type=int, default=DIM_Z)
    return parser


def train_hyperparams(args: argparse.Namespace) -> dict:
    """The DecipherConfig kwargs of the hyperparameter flags."""
    return {
        "learning_rate": args.lr,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "dim_z": args.dim_z,
    }


def train_package(adata, config, save_folder: Path, on_epoch=None):
    """`decipher.tools.decipher.decipher_train` on the file's `split` with the package wrapped (not
    edited) to record per epoch train_elbo (mean negative ELBO per training cell) and val_nll
    (package validation NLL per cell); `on_epoch(row)` is called with each row. Validation runs
    under `torch.no_grad()` (the package keeps the autograd graph of every val batch x 5 samples,
    ~10 GB on aml4; the values are unchanged). Returns (model, loss DataFrame
    epoch/train_elbo/val_nll). Raises if the package stopped on a training-step error (it prints it
    and returns without saving) or the rows differ from its val losses."""
    import pandas as pd
    import torch
    from decipher.tools import decipher as dtool
    from decipher.utils import DECIPHER_GLOBALS

    rows, step = [], {"loss": 0.0, "n": 0}

    class RecordingSVI(dtool.SVI):
        def step(self, *a, **kw):
            loss = super().step(*a, **kw)
            step["loss"] += loss
            step["n"] += a[0].shape[0]
            return loss

    orig_pll = dtool.predictive_log_likelihood

    def recording_pll(*a, **kw):
        with torch.no_grad():
            out_ll = orig_pll(*a, **kw)
        row = {
            "epoch": len(rows),
            "train_elbo": step["loss"] / step["n"],
            "val_nll": -out_ll / len(a[1].dataset),
        }
        rows.append(row)
        step.update(loss=0.0, n=0)
        if on_epoch is not None:
            on_epoch(row)
        return out_ll

    def file_split(adata, val_frac, seed):
        adata.obs["decipher_split"] = (
            adata.obs["split"].map({"train": "train", "val": "validation"}).astype("category")
        )

    dtool.SVI, dtool.predictive_log_likelihood = RecordingSVI, recording_pll
    dtool._make_train_val_split = file_split
    DECIPHER_GLOBALS["save_folder"] = str(save_folder)

    run_id_before = adata.uns.get("decipher", {}).get("run_id")
    decipher, val_losses = dtool.decipher_train(adata, config, plot_every_k_epochs=-2)
    if adata.uns.get("decipher", {}).get("run_id") in (None, run_id_before):
        raise RuntimeError(
            f"scdecipher stopped on a training-step error after {len(rows)} complete epochs "
            "(see its ERROR line above); the model was not saved"
        )
    loss = pd.DataFrame(rows)
    assert np.allclose(loss["val_nll"], val_losses), "recorded val_nll != package val_losses"
    return decipher, loss


def vspace_figure(adata):
    """`dc.pl.decipher` of the aml4 panels in three columns, titled "(No Batch Correction)"."""
    import decipher as dc

    fig = dc.pl.decipher(adata, color=common.PANELS, ncols=3, wspace=0.6, sort_order=False)
    common.add_title_suffix(fig, " (No Batch Correction)")
    return fig


def main() -> None:
    args = build_parser().parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from importlib.metadata import version

    import anndata as ad
    import decipher as dc
    import matplotlib.pyplot as plt
    import torch

    out = args.out_dir or common.out_dir(args.dataset, "native")
    out = out if out.is_absolute() else Path.cwd() / out
    for sub in ["csv", "reconstruction", "vspace", "_models", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    stages, t_stage = {}, time.time()

    def lap(name: str) -> None:
        nonlocal t_stage
        stages[name] = round(time.time() - t_stage, 1)
        t_stage = time.time()
        print(f"[{tag}] {name}: {stages[name]} s", flush=True)

    # 1. data (X is integer counts, all genes; the counts layer is the same matrix)
    h5ad_sha = native.sha256(args.src)
    expected_sha = args.src.with_suffix(".sha256").read_text().split()[0]
    adata = ad.read_h5ad(args.src)
    del adata.layers["counts"]
    hyper = train_hyperparams(args)
    config = dc.tl.DecipherConfig(seed=args.seed, **hyper, n_epochs=args.max_epochs)
    lap("load")

    # 2. train (package wrapped; each epoch printed, the loss CSV written after training)
    def on_epoch(row: dict) -> None:
        print(
            f"[{tag}] epoch {row['epoch']}: train_elbo {row['train_elbo']:.4f}, "
            f"val_nll {row['val_nll']:.4f}",
            flush=True,
        )

    decipher, loss = train_package(adata, config, out / "_models", on_epoch=on_epoch)
    loss.to_csv(out / "csv" / f"{tag}_loss.csv", index=False)
    plt.close("all")  # the package draws a final v-space figure
    lap("train")
    seconds_per_epoch = round(stages["train"] / len(loss), 2)

    # 3-7. rotate, trajectory, metrics, reconstruction, v-space (shared with d3)
    def recon(sub):
        decipher.eval()
        x = sub.X.toarray().astype(np.float32)
        with torch.no_grad():
            return native.reconstruction_r2(x, decipher.impute_gene_expression_numpy(x))

    ev = common.evaluate(
        adata,
        dc,
        out,
        tag,
        lap,
        loss,
        recon,
        f"{args.dataset} native Decipher (PyPI), seed {args.seed}",
        vspace_figure,
    )

    # 8. meta
    summary = common.run_summary(loss, ev, int(loss["val_nll"].idxmin()))
    lap("save")
    meta = {
        "dataset": args.dataset,
        "src": common.rel_path(args.src, HERE),
        "package": "scdecipher",
        "scdecipher": version("scdecipher"),
        "preset": "native",
        "seed": args.seed,
        "hyperparameters": hyper,
        "config": config.to_dict(),
        "batch_source": None,
        "h5ad_sha256": h5ad_sha,
        "h5ad_sha256_matches": h5ad_sha == expected_sha,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "run_id": adata.uns["decipher"]["run_id"],
        "weights_from": "last epoch (the package does not restore the best)",
        **summary,
        "rotation": ev["rotation"],
        "trajectory_cluster_ids": {common.TRAJECTORY: ev["trajectory"]["path_cluster_ids"]},
        "trajectory": ev["trajectory"],
        "metric_subsample_n": ev["metric_subsample_n"],
        "batch_asw_skipped_labels": ev["batch_asw_skipped_labels"],
        "seconds": stages,
        "seconds_per_epoch": seconds_per_epoch,
    }
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
