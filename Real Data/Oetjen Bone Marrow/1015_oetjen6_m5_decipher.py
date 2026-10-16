"""decipher_m5 model5 (batch = `sample`) on an Oetjen shared h5ad.

Same framework as `1015_oetjen6_native_decipher.py`, whose helpers and constants are imported from
that file by path so the two runs share one definition of the data, rotation, trajectories, scIB
metrics and figures. One process per seed: train -> rotate -> decipher time -> scIB metrics on v
and z -> reconstruction diagnostic -> v-space PNG.
Run from the repo root with the repo venv. oetjen6 (defaults):
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        "Real Data/Oetjen Bone Marrow/1015_oetjen6_m5_decipher.py" --seed 0
oetjen5:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        "Real Data/Oetjen Bone Marrow/1015_oetjen6_m5_decipher.py" --seed 0 \
        --src "Real Data/Oetjen Bone Marrow/1016_oetjen5/oetjen5_shared.h5ad" --dataset oetjen5 \
        --out-dir "Real Data/Oetjen Bone Marrow/1016_oetjen5_m5"
`--src` / `--dataset` behave as in the native script (sha256 from `<src stem>.sha256`; dataset in
the wandb config and tags, the run name `model5_<dataset>_seed<k>` and the plot title).

What differs from the native run:
    model           `decipher_m5` preset `model5` (batch enters encoder_x_to_z and decoder_z_to_x)
    batch           `obs["batch"]` = `obs["sample"]` (one level per sample), `batch_key="batch"`
    weights kept    the lowest-val-NLL epoch (`decipher_m5` restores it; native keeps the last)
    losses          read from `decipher.train_losses_` / `val_losses_` (same per-cell units as the
                    native wrappers) and replayed to wandb after training, step = epoch
    reconstruction  `decipher_m5.tl.reconstruction_r2_log1p`: each cell decoded in its own batch
    z               `decipher_z` is the batch-averaged frame; `decipher_z_raw` (the guide's own z,
                    batch included) is written to the v-space CSV but not scored
The split is the file's own 90/10 `split` column, as in the native run (same train/val cells).

Writes under --out-dir (default `1015_oetjen6_m5/` next to this script) the native file layout,
plus in `seed<k>_meta.json`: batch_key, batch levels, best epoch and the L2 norm of each sample's
decoder shift row (`batch_ctx_dec.weight`), a cheap "how much batch did model5 absorb" number.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "oetjen6_native", HERE / "1015_oetjen6_native_decipher.py"
)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)

PRESET = "model5"
BATCH_COL = "batch"
WANDB = {"entity": "jpark-columbia", "project": "decipher-bc-oetjen", "group": "m5-model5"}


def batch_from_sample(sample: pd.Series) -> pd.Series:
    """The batch column model5 is conditioned on: `sample` as a category with only the levels
    present (an unused level would add an untrained batch row)."""
    return sample.astype("category").cat.remove_unused_categories()


def file_split(adata, val_frac: float, seed: int) -> None:
    """Replacement for `decipher_m5`'s `_make_train_val_split`: copy the file's `split` column
    (train/val) into `decipher_split` (train/validation). `val_frac` and `seed` are ignored."""
    split = adata.obs["split"].astype(str)
    unknown = set(split) - {"train", "val"}
    if unknown:
        raise ValueError(f"obs['split'] has values other than train/val: {sorted(unknown)}")
    adata.obs["decipher_split"] = pd.Categorical(
        np.where(split == "val", "validation", "train"), categories=["train", "validation"]
    )


def scib_metrics(spaces: dict, obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """scIB metrics of the native run on each space (must include the `pca` reference): batch ASW,
    graph iLISI and graph connectivity on all cells, and per branch the trajectory conservation of
    DPT on v and z and of `decipher_time_<branch>` against DPT on the PCA.

    Returns (metrics with columns space/metric/branch/value, per-cell DPT columns
    `dpt_<space>_<branch>` indexed like `obs`)."""
    batch = obs[native.BATCH_KEY].astype(str).to_numpy()
    labels = obs[native.LABEL_KEY].astype(str).to_numpy()
    rows = []
    for space, emb in spaces.items():
        distances, connectivities = native.knn_graph(emb)
        for metric, value in [
            ("batch_asw", native.batch_asw(emb, batch, labels)),
            ("graph_ilisi", native.graph_ilisi(distances, batch)),
            ("graph_connectivity", native.graph_connectivity(connectivities, labels)),
        ]:
            rows.append({"space": space, "metric": metric, "branch": "all", "value": value})
    dpts = pd.DataFrame(index=obs.index)
    for name, (_, _, col) in native.TRAJECTORIES.items():
        in_branch = obs[col].to_numpy()
        is_root = labels[in_branch] == native.ROOT_CELL_TYPE
        for space, emb in spaces.items():
            dpt = np.full(len(obs), np.nan)
            dpt[in_branch] = native.branch_dpt(emb[in_branch], is_root)
            dpts[f"dpt_{space}_{name}"] = dpt
        reference = dpts[f"dpt_pca_{name}"].to_numpy()
        assert np.isfinite(reference).sum() >= 3, f"reference DPT for {name} could not be computed"
        for space, metric, pseudotime in [
            ("v", "trajectory_conservation", dpts[f"dpt_v_{name}"]),
            ("z", "trajectory_conservation", dpts[f"dpt_z_{name}"]),
            ("v", "decipher_time_conservation", obs[f"decipher_time_{name}"]),
        ]:
            value = native.trajectory_conservation(pseudotime.to_numpy(), reference)
            rows.append({"space": space, "metric": metric, "branch": name, "value": value})
    return pd.DataFrame(rows), dpts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="decipher seed (init, batch order)")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=HERE / "1015_oetjen6_m5")
    parser.add_argument(
        "--src", type=Path, default=native.SRC, help="shared h5ad; <src stem>.sha256 beside it"
    )
    parser.add_argument(
        "--dataset", default=native.DATASET, help="name for wandb config/tags, run name, title"
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import anndata as ad
    import matplotlib.pyplot as plt
    import torch

    import decipher_m5 as dc
    import decipher_m5.tools.decipher as dtool
    from decipher_m5.presets import PRESETS
    from decipher_m5.utils import DECIPHER_GLOBALS

    torch.set_num_threads(1)
    out = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    for sub in ["csv", "reconstruction", "vspace", "_models", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    stages, t_stage = {}, time.time()

    def lap(name: str) -> None:
        nonlocal t_stage
        stages[name] = round(time.time() - t_stage, 1)
        t_stage = time.time()
        print(f"[{tag}] {name}: {stages[name]} s", flush=True)

    # 1. data
    h5ad_sha = native.sha256(args.src)
    expected_sha = args.src.with_suffix(".sha256").read_text().split()[0]
    adata = ad.read_h5ad(args.src)
    adata = adata[:, adata.var["hvg_2000"]].copy()
    adata.X = adata.layers["counts"].copy()
    del adata.layers["counts"]
    adata.obs[BATCH_COL] = batch_from_sample(adata.obs["sample"])
    batch_levels = [str(c) for c in adata.obs[BATCH_COL].cat.categories]
    config = dc.tl.DecipherConfig(
        **PRESETS[PRESET],
        seed=args.seed,
        learning_rate=native.LEARNING_RATE,
        beta=native.BETA,
        n_epochs=args.max_epochs,
    )

    run = None
    if not args.no_wandb:
        import wandb

        assert hasattr(wandb, "init"), f"`wandb` resolved to a non-package: {wandb.__path__}"
        run = wandb.init(
            **WANDB,
            name=f"{PRESET}_{args.dataset}_{tag}",
            dir=str(out),
            tags=[args.dataset],
            config={
                "dataset": args.dataset,
                "package": "decipher_m5",
                "preset": PRESET,
                "seed": args.seed,
                **config.to_dict(),
                "batch_key": BATCH_COL,
                "batch_source": "sample",
                "h5ad_sha256": h5ad_sha,
                "h5ad_sha256_matches": h5ad_sha == expected_sha,
                "n_cells_total": adata.n_obs,
                "n_genes": adata.n_vars,
            },
        )
    lap("load")

    # 2. train (file split, save folder; the package restores the best-val-NLL weights)
    dtool._make_train_val_split = file_split
    DECIPHER_GLOBALS["save_folder"] = str(out / "_models")
    decipher, val_losses = dc.tl.decipher_train(
        adata,
        config,
        batch_key=BATCH_COL,
        plot_every_k_epochs=-2,
        plot_kwargs={"color": "lineage"},  # its final v-space figure fails with color=None
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
    if run is not None:
        for row in loss.to_dict("records"):
            run.log({"train_elbo": row["train_elbo"], "val_nll": row["val_nll"]}, step=row["epoch"])
    lap("train")
    seconds_per_epoch = round(stages["train"] / len(loss), 2)

    # 3. rotate (reloads the saved model by run_id)
    dc.tl.decipher_rotate_space(adata, **native.ROTATION)
    lap("rotate")

    # 4. decipher time (package column, then one column per branch over all its cells)
    adata.obs[list(native.BRANCHES)] = native.branch_columns(adata.obs["lineage"])
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    dc.tl.trajectories(
        adata,
        *[dc.tl.TConfig(n, s, e, col, True) for n, (s, e, col) in native.TRAJECTORIES.items()],
    )
    dc.tl.decipher_time(adata)
    trajs = adata.uns["decipher"]["trajectories"]
    for name, (_, _, col) in native.TRAJECTORIES.items():
        adata.obs[f"decipher_time_{name}"] = native.branch_time(
            adata.obsm["decipher_v"],
            trajs[name]["points"],
            trajs[name]["times"],
            adata.obs[col].to_numpy(),
        )
    paths = {n: list(t["cluster_ids"]) for n, t in trajs.items()}
    counts = pd.crosstab(adata.obs["decipher_clusters"], adata.obs["cell_type"])
    native.traj_cluster_table(paths, counts).to_csv(
        out / "csv" / f"{tag}_traj_clusters.csv", index=False
    )
    sanity = native.trajectory_sanity(paths, counts)
    print(f"[{tag}] trajectory sanity: {json.dumps(sanity)}", flush=True)
    lap("decipher_time")

    # 5. scIB metrics on v, z and the uncorrected PCA baseline
    spaces = {
        "pca": native.uncorrected_pca(adata.X),
        "v": adata.obsm["decipher_v"],
        "z": adata.obsm["decipher_z"],
    }
    metrics, dpts = scib_metrics(spaces, adata.obs)
    adata.obs[list(dpts.columns)] = dpts
    metrics.to_csv(out / "csv" / f"{tag}_metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    lap("metrics")

    # 6. reconstruction diagnostic on all cells, each cell decoded in its own batch
    with torch.no_grad():
        r2 = dc.tl.reconstruction_r2_log1p(decipher, adata)
    r2_median = float(np.nanmedian(r2["r2_per_gene"]))
    fig = native.plot_reconstruction(
        loss, r2, f"{args.dataset} decipher_m5 model5, seed {args.seed}"
    )
    recon_png = out / "reconstruction" / f"{tag}_recon.png"
    fig.savefig(recon_png, dpi=120)
    plt.close(fig)
    lap("reconstruction")

    # 7. v-space PNG, cells in shuffled order
    order = np.random.default_rng(0).permutation(adata.n_obs)
    fig = dc.pl.decipher(
        adata[order], color=native.VSPACE_PANELS, ncols=2, wspace=0.6, sort_order=False
    )
    vspace_png = out / "vspace" / f"{tag}_vspace.png"
    fig.savefig(vspace_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    lap("vspace_png")

    # 8. tables, meta, wandb summary
    time_cols = [f"decipher_time_{n}" for n in native.TRAJECTORIES]
    table = adata.obs[
        native.OBS_KEEP
        + [
            "decipher_split",
            *native.BRANCHES,
            "decipher_clusters",
            "decipher_time",
            *time_cols,
            *dpts.columns,
        ]
    ]
    table = table.copy()
    table[["v1", "v2"]] = adata.obsm["decipher_v"]
    table[["v1_unrotated", "v2_unrotated"]] = adata.obsm["decipher_v_not_rotated"]
    for key, prefix in [("decipher_z", "z"), ("decipher_z_raw", "z_raw")]:
        z = adata.obsm[key]
        table[[f"{prefix}{i + 1}" for i in range(z.shape[1])]] = z
    table.index.name = "cell_id"
    table.to_csv(out / "csv" / f"{tag}_vspace.csv")

    shift = decipher.batch_ctx_dec.weight.detach().cpu().numpy()  # B x G
    shift_norms = dict(zip(batch_levels, np.linalg.norm(shift, axis=1).round(4).tolist()))
    metric_summary = {
        f"{r.space}/{r.metric}" + ("" if r.branch == "all" else f"/{r.branch}"): r.value
        for r in metrics.itertuples()
    }
    summary = {
        "r2_overall": r2["r2_overall"],
        "r2_median_per_gene": r2_median,
        "sanity_pass": sanity["pass"],
        "n_epochs_run": len(loss),
        "best_epoch": decipher.best_epoch_,
        "best_val_nll": float(loss["val_nll"].min()),
        "final_val_nll": float(loss["val_nll"].iloc[-1]),
        **metric_summary,
    }
    if run is not None:
        run.log(
            {"reconstruction": wandb.Image(str(recon_png)), "vspace": wandb.Image(str(vspace_png))}
        )
        run.summary.update({**summary, "batch_shift_norm": shift_norms})
    lap("save")
    meta = {
        "dataset": args.dataset,
        "src": str(args.src),
        "package": "decipher_m5",
        "preset": PRESET,
        "seed": args.seed,
        "config": config.to_dict(),
        "batch_key": BATCH_COL,
        "batch_source": "sample",
        "batch_levels": batch_levels,
        "h5ad_sha256": h5ad_sha,
        "h5ad_sha256_matches": h5ad_sha == expected_sha,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "run_id": adata.uns["decipher"]["run_id"],
        "weights_from": "best val-NLL epoch (decipher_m5 restore)",
        **summary,
        "batch_shift_norm": shift_norms,
        "rotation": adata.uns["decipher"]["rotation"].tolist(),
        "trajectory_cluster_ids": paths,
        "sanity": sanity,
        "wandb_url": run.url if run is not None else None,
        "seconds": stages,
        "seconds_per_epoch": seconds_per_epoch,
    }
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
