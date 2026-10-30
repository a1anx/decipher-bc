"""A decipher_models3 batch arm (model5, model4r, model5r; batch = `patient`) on the aml4 h5ad.

`d3_helpers.py` holds the parser, run names, file split and batch norms; `1027_aml4_common.py`
holds the leukemic trajectory and the evaluation shared with the PyPI-native driver. Both are
loaded by path from this folder. One process per preset x seed: train -> rotate v1 onto the stage
rank -> clusters and the leukemic trajectory -> decipher time -> metrics -> reconstruction ->
v-space PNG (patient, cell_type_merged, cell_type, decipher_time, library, NPM1_mut; titles
"(Batch = patient)"). Run from this folder in an environment with `decipher_models3` installed:
    python 1027_aml4_d3_decipher.py --preset model5 --seed 0 --beta 1e-3 --batch-size 512 \
        --out-dir model5
Defaults: `aml4/aml4_shared.h5ad` (all 6,341 genes, X = integer counts), dataset `aml4`, the
scdecipher package defaults (`--lr 5e-3 --beta 0.1 --batch-size 64`, dim_z 10), early
stopping on val NLL with best-epoch restore, `--max-epochs 1000`; out-dir `<preset>/` next to this
script. Each epoch's train ELBO and val NLL go to `csv/seed<k>_loss.csv` after training.

Memory: the package's `check_adata_has_integer_counts` densifies X and builds int64/float64 copies
(~16 GB peak on aml4); it is replaced (as `_make_train_val_split` is) by
`common.check_integer_counts`, the same test on the sparse values.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_cspec = importlib.util.spec_from_file_location("aml4_common", HERE / "1027_aml4_common.py")
common = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(common)
d3 = common.load_by_path(HERE / "d3_helpers.py", "aml4_d3_helpers")
native = common.native
lambda_overrides = d3.lambda_overrides

SRC = common.SRC
AML_PRESETS = ("model5", "model4r", "model5r")


def build_parser():
    """The d3 command line with the aml4 src, dataset and package-default hyperparameters as
    defaults."""
    parser = d3.build_parser()
    parser.description = __doc__.splitlines()[0]
    parser.set_defaults(src=SRC, dataset=common.DATASET, **common.HYPERPARAMS)
    return parser


def run_names(
    dataset: str,
    preset: str,
    seed: int,
    lambdas: dict[str, float],
    scan: bool,
    final: bool,
    root: Path = HERE,
    tagged_final: bool = False,
) -> dict:
    """`d3.run_names` (same signature). Raises for a preset outside `AML_PRESETS`."""
    if preset not in AML_PRESETS:
        raise ValueError(f"preset {preset!r} is not an aml4 arm; choose from {AML_PRESETS}")
    return d3.run_names(
        dataset, preset, seed, lambdas, scan, final, root=root, tagged_final=tagged_final
    )


def vspace_figure(adata):
    """`dc.pl.decipher` of the aml4 panels in three columns, titled "(Batch = patient)"."""
    import decipher_models3 as dc

    return dc.pl.decipher(
        adata,
        color=common.PANELS,
        ncols=3,
        wspace=0.6,
        sort_order=False,
        batch_label=common.BATCH_SOURCE,
    )


def main() -> None:
    args = build_parser().parse_args()
    lambdas = lambda_overrides(args)
    names = run_names(
        args.dataset,
        args.preset,
        args.seed,
        lambdas,
        args.scan,
        args.final,
        tagged_final=args.tagged_final,
    )

    import matplotlib

    matplotlib.use("Agg")
    import anndata as ad
    import matplotlib.pyplot as plt
    import torch

    import decipher_models3 as dc
    import decipher_models3.tools.decipher as dtool
    from decipher_models3.presets import PRESETS
    from decipher_models3.utils import DECIPHER_GLOBALS

    print(f"decipher_models3: {dc.__file__}", flush=True)
    out = args.out_dir or names["out_dir"]
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
    adata.obs[common.BATCH_COL] = d3.batch_from_sample(adata.obs[common.BATCH_SOURCE])
    batch_levels = [str(c) for c in adata.obs[common.BATCH_COL].cat.categories]
    hyper = d3.train_hyperparams(args)
    config = dc.tl.DecipherConfig(
        **{**PRESETS[args.preset], **lambdas}, seed=args.seed, **hyper, n_epochs=args.max_epochs
    )
    lap("load")

    # 2. train (file split, save folder; the package restores the best-val-NLL weights)
    dtool._make_train_val_split = d3.file_split
    dtool.check_adata_has_integer_counts = common.check_integer_counts  # no dense copy
    DECIPHER_GLOBALS["save_folder"] = str(out / "_models")
    decipher, val_losses = dc.tl.decipher_train(
        adata,
        config,
        batch_key=common.BATCH_COL,
        plot_every_k_epochs=-2,
        plot_kwargs={"color": common.LABEL_KEY},
    )
    plt.close("all")
    loss = pd.DataFrame(
        {
            "epoch": range(len(decipher.train_losses_)),
            "train_elbo": decipher.train_losses_,
            "val_nll": decipher.val_losses_,
        }
    )
    assert np.allclose(loss["val_nll"], val_losses), "recorded val_nll != package val_losses"
    loss.to_csv(out / "csv" / f"{tag}_loss.csv", index=False)
    lap("train")
    seconds_per_epoch = round(stages["train"] / len(loss), 2)

    # 3-7. rotate, trajectory, metrics, reconstruction, v-space (shared with native)
    def recon(sub):
        with torch.no_grad():
            return dc.tl.reconstruction_r2_log1p(decipher, sub)

    ev = common.evaluate(
        adata,
        dc,
        out,
        tag,
        lap,
        loss,
        recon,
        f"{args.dataset} decipher_models3 {args.preset}, seed {args.seed}",
        vspace_figure,
    )

    # 8. batch norms, meta
    z_for_shift = d3.model_z(decipher, adata) if config.recon_batch != "concat" else None
    batch_norms = {
        "batch_emb_norm": d3.batch_emb_norms(decipher, batch_levels),
        "batch_shift_norm": d3.batch_shift_norms(decipher, batch_levels, z_for_shift),
        "batch_shift_norm_centered": d3.batch_shift_norms_centered(
            decipher, batch_levels, z_for_shift
        ),
    }
    summary = common.run_summary(loss, ev, decipher.best_epoch_)
    lap("save")
    meta = {
        "dataset": args.dataset,
        "src": common.rel_path(args.src, HERE),
        "package": "decipher_models3",
        "decipher_models3_file": common.rel_path(dc.__file__, HERE.parents[2]),
        "preset": args.preset,
        "seed": args.seed,
        "run_mode": names["mode"],
        "lambda_overrides": lambdas,
        "lambda_tag": names["lambda_tag"],
        "hyperparameters": hyper,
        "config": config.to_dict(),
        "batch_key": common.BATCH_COL,
        "batch_source": common.BATCH_SOURCE,
        "batch_levels": batch_levels,
        "h5ad_sha256": h5ad_sha,
        "h5ad_sha256_matches": h5ad_sha == expected_sha,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "run_id": adata.uns["decipher"]["run_id"],
        "weights_from": "best val-NLL epoch (decipher_models3 restore)",
        **d3.penalty_meta(decipher),
        **summary,
        **batch_norms,
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
