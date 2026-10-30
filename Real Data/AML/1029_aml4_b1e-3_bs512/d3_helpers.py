"""Run naming, command line, file split and batch-norm helpers of the decipher_models3 driver.

λ overrides and run naming (`run_names`):
    --lambda-inv/--lambda-r/--lambda-c  a given value goes into the DecipherConfig kwargs as
                    lambda_inv/lambda_r/lambda_c; its tag is li/lr/lc + `{λ:g}` with `.` -> `p`
    --scan          exactly one λ; out-dir `<preset>_<tag>`, run mode "scan"
    --final         out-dir `<preset>`, run mode "final"
    --tagged-final  with --final and exactly one λ (else it raises); out-dir
                    `<preset>_<tag>_final`, run mode "final"
    neither         out-dir `<preset>`, run mode None

Batch norms written to the meta:
    batch_emb_norm             L2 norm of each batch's embedding row e_b; None without a table
    batch_shift_norm           concat_x arms (model5): L2 norm of the per-gene logit shift
                               W_e e_b; residual arms (model4r/5r): mean over all cells of
                               ||r([z; e_b])||, each cell decoded under every batch b, z = z_loc of
                               q(z | x, own batch); this includes r's batch-independent part
                               (output bias, z-only terms), so it is not comparable with
                               ||W_e e_b||
    batch_shift_norm_centered  residual arms only: mean over cells of
                               ||r([z; e_b]) - mean_b' r([z; e_b'])||, which drops that part
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from decipher_models3.presets import PRESETS

HERE = Path(__file__).resolve().parent
PRESET_CHOICES = tuple(PRESETS)
LAMBDA_PREFIX = {"lambda_inv": "li", "lambda_r": "lr", "lambda_c": "lc"}


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
    """Default out-dir under `root`, run mode, group, run name, tags and λ tag of one run.

    scan: exactly one λ, which tags the out-dir and run name, group `d3-<preset>-scan`.
    tagged_final: needs final and exactly one λ; out-dir `<preset>_<tag>_final`, group
    `d3-<preset>-<tag>`, run name with `_<tag>`, key `tagged_final` = tag.
    final or neither: out-dir `<preset>`, untagged group and run name; tags add "final" for
    final."""
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
    name = f"{preset}_{dataset}_seed{seed}"
    if scan:
        return {
            "mode": "scan",
            "out_dir": root / f"{preset}_{tag}",
            "group": f"d3-{preset}-scan",
            "name": f"{name}_{tag}",
            "tags": [dataset, "scan"],
            "lambda_tag": tag,
        }
    if tagged_final:
        return {
            "mode": "final",
            "out_dir": root / f"{preset}_{tag}_final",
            "group": f"d3-{preset}-{tag}",
            "name": f"{name}_{tag}",
            "tags": [dataset, "final"],
            "lambda_tag": tag,
            "tagged_final": tag,
        }
    return {
        "mode": "final" if final else None,
        "out_dir": root / preset,
        "group": f"d3-{preset}",
        "name": name,
        "tags": [dataset, "final"] if final else [dataset],
        "lambda_tag": tag,
    }


def penalty_meta(decipher) -> dict[str, dict[str, list[float]]]:
    """The meta entry `penalty_history`: penalty name -> per-epoch mean, {} without a penalty."""
    return {"penalty_history": decipher.penalty_history_}


def batch_emb_norms(decipher, batch_levels: list[str]) -> dict[str, float] | None:
    """L2 norm of each batch's `batch_emb` row e_b; None when the model has no table (native)."""
    if decipher.batch_emb is None:
        return None
    emb = decipher.batch_emb.weight.detach().cpu().numpy()
    return dict(zip(batch_levels, np.linalg.norm(emb, axis=1).round(4).tolist()))


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
        # a power of 2
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


def batch_from_sample(sample: pd.Series) -> pd.Series:
    """The batch column the model is conditioned on: `sample` as a category with only the levels
    present (an unused level would add an untrained batch row)."""
    return sample.astype("category").cat.remove_unused_categories()


def file_split(adata, val_frac: float, seed: int) -> None:
    """Replacement for the package's `_make_train_val_split`: copy the file's `split` column
    (train/val) into `decipher_split` (train/validation). `val_frac` and `seed` are ignored."""
    split = adata.obs["split"].astype(str)
    unknown = set(split) - {"train", "val"}
    if unknown:
        raise ValueError(f"obs['split'] has values other than train/val: {sorted(unknown)}")
    adata.obs["decipher_split"] = pd.Categorical(
        np.where(split == "val", "validation", "train"), categories=["train", "validation"]
    )


def build_parser() -> argparse.ArgumentParser:
    """The driver's command line; the caller sets the `--src`, `--dataset`, `--lr`, `--beta` and
    `--batch-size` defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="decipher seed (init, batch order)")
    parser.add_argument("--preset", choices=PRESET_CHOICES, required=True)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=None, help="default <preset>/ here")
    parser.add_argument(
        "--src", type=Path, default=None, help="shared h5ad; <src stem>.sha256 beside it"
    )
    parser.add_argument("--dataset", default=None, help="name for the run name, title and meta")
    for flag in LAMBDA_PREFIX:
        parser.add_argument(f"--{flag.replace('_', '-')}", type=float, default=None)
    parser.add_argument("--scan", action="store_true", help="λ-scan run (exactly one λ)")
    parser.add_argument("--final", action="store_true", help="chosen-λ run, tag final")
    parser.add_argument(
        "--tagged-final",
        action="store_true",
        help="with --final and exactly one λ: out-dir/group/run name carry the λ tag",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="default: the package's (64)")
    return parser


def train_hyperparams(args: argparse.Namespace) -> dict:
    """DecipherConfig kwargs from `--lr`, `--beta`, `--batch-size`: learning_rate, beta and
    batch_size only when given (else the package default)."""
    out = {"learning_rate": args.lr, "beta": args.beta}
    if args.batch_size is not None:
        out["batch_size"] = args.batch_size
    return out


def lambda_overrides(args: argparse.Namespace) -> dict[str, float]:
    """The λ flags that were given, keyed by their DecipherConfig field name."""
    return {k: getattr(args, k) for k in LAMBDA_PREFIX if getattr(args, k) is not None}
