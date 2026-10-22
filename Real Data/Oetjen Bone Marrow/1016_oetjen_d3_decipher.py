"""A decipher_models3 preset (any key of PRESETS; batch = `sample`) on an Oetjen shared h5ad.

Same framework as `1015_oetjen6_m5_decipher.py`: its helpers (file split, scIB metrics, batch
column, embedding norms) and, through it, the native script's constants, rotation,
branch time, sanity check and plots are imported by path, so data, split, metrics and figures are
one definition. One process per seed: train -> rotate -> decipher time -> scIB metrics on v and z
-> reconstruction diagnostic -> v-space PNG. Run from the repo root with the repo venv:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        "Real Data/Oetjen Bone Marrow/1016_oetjen_d3_decipher.py" --preset model4 --seed 0
Defaults: oetjen5 (`1016_oetjen5/oetjen5_shared.h5ad`), out-dir `1016_<dataset>_d3_<preset>/`
next to this script.

λ overrides and out-dir naming (`run_names`):
    --lambda-inv/--lambda-r/--lambda-c  a given value goes into the DecipherConfig kwargs as
                    lambda_inv/lambda_r/lambda_c; its tag is li/lr/lc + `{λ:g}` with `.` -> `p`
    --scan          exactly one λ; out-dir `1016_<dataset>_d3_<preset>_<tag>`
    --final         out-dir `1016_<dataset>_d3_<preset>`
    --tagged-final  with --final and exactly one λ (else it raises): a final run that keeps the
                    preset's untagged folder free; out-dir `1016_<dataset>_d3_<preset>_<tag>_final`;
                    the meta key `tagged_final` holds the tag (null otherwise)
    neither         as --final

What differs from the m5 run:
    model           `decipher_models3` preset (one shared learned `batch_emb`; model1-3 concat it
                    onto the prior input, model4/6 onto the decoder input; see that package's guide)
    package         decipher_models3
    meta            `batch_emb_norm` (L2 norm of each e_b) for every batch arm, null for native;
                    `batch_shift_norm` is the L2 norm of the per-gene logit shift W_e e_b for
                    model4/5/6 (concat_x) and null for native and model1-3 (concat_z: the batch
                    shifts the prior on z, not the genes); for model4r/5r (recon_batch="residual")
                    it is the mean over all cells of ||r([z; e_b])||, each cell decoded under
                    every batch b, z = z_loc of q(z | x, own batch) (the unrotated decipher_z_raw,
                    the z the decoder is trained on). That includes r's batch-independent part
                    (output bias, z-only terms), so it is NOT comparable with concat_x ||W_e e_b||;
                    `batch_shift_norm_centered` (model4r/5r only, null otherwise) is the mean over
                    cells of ||r([z; e_b]) - mean_b' r([z; e_b'])||, which drops that part;
                    model7 (recon_batch="residual") uses the same two formulas; for model7i
                    (recon_batch="residual_in_span") the correction is the logit shift W ρ([z; e_b])
                    (W = decoder_z_to_x's weight, no bias c), so the two norms are the mean over cells
                    of ||W ρ([z; e_b])|| and of ||W (ρ([z; e_b]) - mean_b' ρ([z; e_b']))||;
                    for model5rz (recon_batch="concat_plus_residual_z") `batch_shift_norm` is
                    ||W_e e_b|| as for model5 (the e_b columns of decoder_z_to_x) and
                    `batch_shift_norm_centered` is the same centered norm of r_z(z) =
                    `decoder_residual_z(z)`, which has no e_b input, so it is asserted to be 0;
                    also `run_mode`, `lambda_overrides`,
                    `lambda_tag`, `decipher_models3_file`, and
                    `penalty_history` (penalty name -> per-epoch mean of the unweighted training
                    penalty, {} for arms without one)
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from decipher_models3.presets import PRESETS

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("oetjen_m5", HERE / "1015_oetjen6_m5_decipher.py")
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)
native = m5.native

PRESET_CHOICES = tuple(PRESETS)
DATASET = "oetjen5"
SRC = HERE / "1016_oetjen5" / "oetjen5_shared.h5ad"
LAMBDA_PREFIX = {"lambda_inv": "li", "lambda_r": "lr", "lambda_c": "lc"}


def default_out_dir(dataset: str, preset: str, root: Path = HERE) -> Path:
    """Output folder under `root` (this script's folder) when `--out-dir` is not given."""
    return root / f"1016_{dataset}_d3_{preset}"


def lambda_tag(lambdas: dict[str, float]) -> str | None:
    """`li`/`lr`/`lc` + `f"{λ:g}"` with `.` -> `p` per given λ, joined by `_` (`lc0p1`, `lc10`);
    None when no λ is given."""
    if not lambdas:
        return None
    return "_".join(f"{LAMBDA_PREFIX[k]}{v:g}".replace(".", "p") for k, v in lambdas.items())


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
    """Default out-dir, run mode and λ tag of one run.

    scan: exactly one λ, which tags the out-dir.
    tagged_final: needs final and exactly one λ; out-dir `<preset>_<tag>_final`, key
    `tagged_final` = tag.
    final or neither: the untagged out-dir."""
    if scan and final:
        raise ValueError("--scan and --final are exclusive")
    if tagged_final and (not final or len(lambdas) != 1):
        raise ValueError(
            f"--tagged-final needs --final and exactly one λ override, got final={final}, "
            f"λ={sorted(lambdas)}"
        )
    if scan and len(lambdas) != 1:
        raise ValueError(f"--scan needs exactly one λ override, got {sorted(lambdas)}")
    tag = lambda_tag(lambdas)
    if scan:
        return {
            "mode": "scan",
            "out_dir": root / f"1016_{dataset}_d3_{preset}_{tag}",
            "lambda_tag": tag,
        }
    if tagged_final:
        return {
            "mode": "final",
            "out_dir": root / f"1016_{dataset}_d3_{preset}_{tag}_final",
            "lambda_tag": tag,
            "tagged_final": tag,
        }
    return {
        "mode": "final" if final else None,
        "out_dir": default_out_dir(dataset, preset, root),
        "lambda_tag": tag,
    }


def penalty_meta(decipher) -> dict[str, dict[str, list[float]]]:
    """The meta entry `penalty_history`: penalty name -> per-epoch mean, {} without a penalty."""
    return {"penalty_history": decipher.penalty_history_}


def batch_emb_norms(decipher, batch_levels: list[str]) -> dict[str, float] | None:
    """L2 norm of each batch's `batch_emb` row e_b; None when the model has no table (native)."""
    if decipher.batch_emb is None:
        return None
    return m5.batch_emb_norms(decipher, batch_levels)


def concat_x_shift_norms(decipher, batch_levels: list[str]) -> dict[str, float]:
    """L2 norm per batch level of the per-gene logit shift `W_e @ e_b`, with
    `W_e = decoder_z_to_x.layers[0].weight[:, dim_z:]` and `e_b` row b of `batch_emb`. This is the
    shift only when `layers_z_to_x` is empty, otherwise it raises."""
    if len(decipher.config.layers_z_to_x) != 0:
        raise ValueError(
            "concat_x batch shift is W_e e_b only for layers_z_to_x=(); got "
            f"{tuple(decipher.config.layers_z_to_x)}"
        )
    w = decipher.decoder_z_to_x.layers[0].weight.detach().cpu().numpy()  # G x (dim_z + D)
    emb = decipher.batch_emb.weight.detach().cpu().numpy()  # B x D
    if len(batch_levels) != emb.shape[0]:
        raise ValueError(f"{len(batch_levels)} batch levels but {emb.shape[0]} batch rows")
    shift = emb @ w[:, decipher.config.dim_z :].T  # B x G
    return dict(zip(batch_levels, np.linalg.norm(shift, axis=1).round(4).tolist()))


def _residual_per_batch(decipher, batch_levels: list[str], z):
    """(function e_b -> the batch's logit correction for every cell of z, table rows e_b); raises
    when the batch levels and the table rows differ.

    The correction is r([z; e_b]) under recon_batch="residual" (model4r/5r/7) and W ρ([z; e_b])
    under "residual_in_span" (model7i), W = `decoder_z_to_x.layers[0].weight` (G x dim_z, no bias),
    which is the logit shift only when `layers_z_to_x` is empty (it raises otherwise). Under
    "concat_plus_residual_z" (model5rz) it is r_z(z) = `decoder_residual_z(z)`, the same for every
    e_b (h has no e_b input)."""
    import torch

    emb = decipher.batch_emb.weight.detach()  # B x D
    if len(batch_levels) != emb.shape[0]:
        raise ValueError(f"{len(batch_levels)} batch levels but {emb.shape[0]} batch rows")
    z = torch.as_tensor(z, dtype=emb.dtype)

    if decipher.config.recon_batch == "concat_plus_residual_z":
        # h sees z only: the correction is the same for every e_b
        def h(e):
            return decipher.decoder_residual_z(z)

        return h, emb

    if decipher.config.recon_batch == "residual_in_span":
        if len(decipher.config.layers_z_to_x) != 0:
            raise ValueError(
                "residual_in_span batch shift is W ρ only for layers_z_to_x=(); got "
                f"{tuple(decipher.config.layers_z_to_x)}"
            )
        a = decipher.decoder_z_to_x.layers[0].weight.detach()  # G x dim_z

        def h(e):
            zh = torch.cat([z, e.expand(len(z), -1)], dim=-1)
            return decipher.decoder_residual_in_span(zh) @ a.T

        return h, emb

    def h(e):
        return decipher.decoder_residual(torch.cat([z, e.expand(len(z), -1)], dim=-1))

    return h, emb


def residual_shift_norms(decipher, batch_levels: list[str], z) -> dict[str, float]:
    """Mean over cells of ||correction_b(z)|| per batch level: r([z; e_b]), r = `decoder_residual`
    (recon_batch="residual"), or W ρ([z; e_b]) (recon_batch="residual_in_span"); every cell of `z`
    (n_cells x dim_z) is decoded under every batch b."""
    import torch

    h, emb = _residual_per_batch(decipher, batch_levels, z)
    with torch.no_grad():
        means = [h(e).norm(dim=-1).mean().item() for e in emb]
    return dict(zip(batch_levels, np.round(means, 4).tolist()))


def residual_shift_norms_centered(decipher, batch_levels: list[str], z) -> dict[str, float]:
    """Mean over cells of ||r([z; e_b]) - mean_b' r([z; e_b'])|| per batch level: r minus its
    average over batches for the same cell, which drops r's batch-independent part (output bias
    and z-only terms). Under recon_batch="residual_in_span" the correction is W ρ([z; e_b]).
    Two passes, so only two n_cells x G arrays are held at once."""
    import torch

    h, emb = _residual_per_batch(decipher, batch_levels, z)
    with torch.no_grad():
        h_mean = sum(h(e) for e in emb) / len(emb)
        means = [(h(e) - h_mean).norm(dim=-1).mean().item() for e in emb]
    return dict(zip(batch_levels, np.round(means, 4).tolist()))


def batch_shift_norms(decipher, batch_levels: list[str], z=None) -> dict[str, float] | None:
    """`concat_x_shift_norms` for concat_x arms (model4/5/6), `residual_shift_norms` on `z` for
    recon_batch="residual" (model4r/5r/7) or "residual_in_span" (model7i); None for concat_z arms
    (model1-3) and for a model without a table (native)."""
    if decipher.config.batch_embedding_mode == "concat_z" or decipher.batch_emb is None:
        return None
    if decipher.config.recon_batch == "concat_plus_residual_z":
        return concat_x_shift_norms(decipher, batch_levels)
    if decipher.config.recon_batch != "concat":
        if z is None:
            raise ValueError(
                f"recon_batch={decipher.config.recon_batch!r} needs z (n_cells x dim_z) for the "
                "shift norm"
            )
        return residual_shift_norms(decipher, batch_levels, z)
    return concat_x_shift_norms(decipher, batch_levels)


def batch_shift_norms_centered(
    decipher, batch_levels: list[str], z=None
) -> dict[str, float] | None:
    """`residual_shift_norms_centered` on `z` for recon_batch="residual" (model4r/5r/7) or
    "residual_in_span" (model7i); None for every other arm (concat_x model4/5/6, concat_z
    model1-3, native)."""
    if decipher.batch_emb is None or decipher.config.recon_batch == "concat":
        return None
    if z is None:
        raise ValueError(
            f"recon_batch={decipher.config.recon_batch!r} needs z (n_cells x dim_z) for the "
            "shift norm"
        )
    if decipher.config.recon_batch == "concat_plus_residual_z":
        # h sees z only, so its output must be bit-identical under every batch and the centred
        # norm is exactly 0; comparing outputs avoids the float residue of sum_b h / B for B not
        # a power of 2 (oetjen5 has B = 5)
        import torch

        h, emb = _residual_per_batch(decipher, batch_levels, z)
        with torch.no_grad():
            first = h(emb[0])
            assert all(torch.equal(first, h(e)) for e in emb[1:]), "r_z(z) carries a batch part"
        return dict.fromkeys(batch_levels, 0.0)
    return residual_shift_norms_centered(decipher, batch_levels, z)


def model_z(decipher, adata):
    """z_loc of q(z | x, b) with each cell's own batch: the unrotated `decipher_z_raw`, the z
    the reconstruction decoder sees in training."""
    from decipher_models3.tools._decipher.data import get_batch_idx, get_dense_X

    decipher.eval()
    _, _, z_raw = decipher.compute_v_z_numpy(
        get_dense_X(adata), get_batch_idx(adata, decipher.config)
    )
    return z_raw


def build_parser() -> argparse.ArgumentParser:
    """The driver's command line (also parsed by the queue helper to name logs and out-dirs)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="decipher seed (init, batch order)")
    parser.add_argument("--preset", choices=PRESET_CHOICES, required=True)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default 1016_<dataset>_d3_<preset>/ here"
    )
    parser.add_argument(
        "--src", type=Path, default=SRC, help="shared h5ad; <src stem>.sha256 beside it"
    )
    parser.add_argument("--dataset", default=DATASET, help="name for the out-dir and title")
    for flag in LAMBDA_PREFIX:
        parser.add_argument(f"--{flag.replace('_', '-')}", type=float, default=None)
    parser.add_argument("--scan", action="store_true", help="λ-scan run (exactly one λ)")
    parser.add_argument("--final", action="store_true", help="chosen-λ run, tag final")
    parser.add_argument(
        "--tagged-final",
        action="store_true",
        help="with --final and exactly one λ: out-dir/group/run name carry the λ tag",
    )
    return parser


def lambda_overrides(args: argparse.Namespace) -> dict[str, float]:
    """The λ flags that were given, keyed by their DecipherConfig field name."""
    return {k: getattr(args, k) for k in LAMBDA_PREFIX if getattr(args, k) is not None}


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
    from decipher_models3.utils import DECIPHER_GLOBALS

    print(f"decipher_models3: {dc.__file__}", flush=True)
    torch.set_num_threads(1)
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

    # 1. data
    h5ad_sha = native.sha256(args.src)
    expected_sha = args.src.with_suffix(".sha256").read_text().split()[0]
    adata = ad.read_h5ad(args.src)
    adata = adata[:, adata.var["hvg_2000"]].copy()
    adata.X = adata.layers["counts"].copy()
    del adata.layers["counts"]
    adata.obs[m5.BATCH_COL] = m5.batch_from_sample(adata.obs["sample"])
    batch_levels = [str(c) for c in adata.obs[m5.BATCH_COL].cat.categories]
    config = dc.tl.DecipherConfig(
        **{**PRESETS[args.preset], **lambdas},
        seed=args.seed,
        learning_rate=native.LEARNING_RATE,
        beta=native.BETA,
        n_epochs=args.max_epochs,
    )

    lap("load")

    # 2. train (file split, save folder; the package restores the best-val-NLL weights)
    dtool._make_train_val_split = m5.file_split
    DECIPHER_GLOBALS["save_folder"] = str(out / "_models")
    decipher, val_losses = dc.tl.decipher_train(
        adata,
        config,
        batch_key=m5.BATCH_COL,
        plot_every_k_epochs=-2,
        plot_kwargs={"color": "lineage"},
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
    metrics, dpts = m5.scib_metrics(spaces, adata.obs)
    adata.obs[list(dpts.columns)] = dpts
    metrics.to_csv(out / "csv" / f"{tag}_metrics.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    lap("metrics")

    # 6. reconstruction diagnostic on all cells, each cell decoded in its own batch
    with torch.no_grad():
        r2 = dc.tl.reconstruction_r2_log1p(decipher, adata)
    r2_median = float(np.nanmedian(r2["r2_per_gene"]))
    fig = native.plot_reconstruction(
        loss, r2, f"{args.dataset} decipher_models3 {args.preset}, seed {args.seed}"
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

    # 8. tables, meta
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

    z_for_shift = model_z(decipher, adata) if config.recon_batch != "concat" else None
    batch_norms = {
        "batch_emb_norm": batch_emb_norms(decipher, batch_levels),
        "batch_shift_norm": batch_shift_norms(decipher, batch_levels, z_for_shift),
        "batch_shift_norm_centered": batch_shift_norms_centered(
            decipher, batch_levels, z_for_shift
        ),
    }
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
    lap("save")
    meta = {
        "dataset": args.dataset,
        "src": str(args.src),
        "package": "decipher_models3",
        "decipher_models3_file": dc.__file__,
        "preset": args.preset,
        "seed": args.seed,
        "run_mode": names["mode"],
        "lambda_overrides": lambdas,
        "lambda_tag": names["lambda_tag"],
        "tagged_final": names.get("tagged_final"),
        "config": config.to_dict(),
        "batch_key": m5.BATCH_COL,
        "batch_source": "sample",
        "batch_levels": batch_levels,
        "h5ad_sha256": h5ad_sha,
        "h5ad_sha256_matches": h5ad_sha == expected_sha,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "run_id": adata.uns["decipher"]["run_id"],
        "weights_from": "best val-NLL epoch (decipher_models3 restore)",
        **penalty_meta(decipher),
        **summary,
        **batch_norms,
        "rotation": adata.uns["decipher"]["rotation"].tolist(),
        "trajectory_cluster_ids": paths,
        "sanity": sanity,
        "seconds": stages,
        "seconds_per_epoch": seconds_per_epoch,
    }
    (out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
