r"""Bifurcating variant of `simulated_data_pipeline_aug1.py`.

ADDED 2026-09-18. Same architecture as aug1 -- random linear v -> z projection, multi-batch
concat, z-mode / gene-mode batch shift, linear z -> x decoder -- with ONE generative change:
the ground-truth manifold is a Y instead of a straight line.

    aug1:  latent_t ~ U(0,1)                        (n, 1)
           z_mean   = latent_t @ w_bio              w_bio: (1, n_z_dims)

    here:  latent_t   ~ U(0,1)                                   (n, 1)
           branch_id  in {-1,+1} w.p. branch_prob, 0 when latent_t <= branching_t
           branch_c   = branch_scale * branch_id * clip(latent_t - branching_t, 0, None)
           latent_v   = [latent_t, branch_c]                     (n, 2)
           z_mean     = latent_v @ W_v     W_v = vstack([w_bio, w_branch])  (2, n_z_dims)

                        /---- arm +1
           ------------O  branching_t
                        \---- arm -1

`latent_v` is 2-D, which matches DecipherConfig.dim_v = 2, so ground-truth v and
obsm["decipher_v"] live in spaces of the same dimension.

NO HOLES. The chunk / hole machinery in `simulations_original.py::simulation_correlated_1` is
deliberately not carried over -- this file adds bifurcation and nothing else.

The batch mechanism is untouched: the shift is still added to z_mean for every cell regardless of
arm, i.e. a rigid translation of the whole Y. `branching_t=None` disables branching entirely and
reproduces aug1 byte-for-byte; that is the A/B control and the test that the batch machinery was
copied rather than perturbed.

One deliberate exception to "byte-for-byte": this file repairs an aug1 bug by writing
uns["w_bio"] and uns["latent_z_names"] onto the concatenated object, which ad.concat's
merge="same" drops whenever n_batches > 1. See the note at the end of
`shift_magnitudes_multivariate_bifurcation`. X, obs, layers and the batch uns keys are identical;
the control's uns is a strict superset of aug1's.
"""

import importlib
import os
import sys
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

# `decipher_models` and `decipher` are directories at the repo root, not installed packages -- they
# import only when the repo root is on sys.path. Deriving it from __file__ means this script (and
# the notebooks that import it) work from any working directory.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from decipher_models.presets import PRESETS as MODEL_PRESETS  # noqa: E402


def simulate_multivariate_bifurcation(
    n_samples: int = 500,
    n_genes: int = 50,
    n_z_dims: int = 3,
    shift_cov: np.ndarray = None,  # (n_z_dims, n_z_dims). Ignored if `shift` is provided.
    shift: np.ndarray = None,  # length n_z_dims, applied to all dims
    biological_sigma: float = 0.1,  # biological noise at z level
    seed: int = 0,  # shared ACROSS batches: w_bio, w_branch, and W (z->x)
    cell_seed: int = 0,  # per-cell / per-batch: latent_t, branch_id, z noise, shift draw
    batch_label: str = None,  # what goes in adata.obs["batch"]
    gene_shift: np.ndarray = None,  # length n_genes; gene-mode offset applied after `@ W`
    # ADDED 2026-09-18 -- the bifurcation knobs. branching_t=None => no branch at all.
    branching_t: float = 0.3,  # pseudotime at which the trajectory forks
    branch_prob: float = 0.5,  # P(arm = +1); 0.5 gives balanced arms
    branch_scale: float = 1.0,  # multiplier on the branch coordinate (how far the arms open)
    orthogonalize_branch: bool = True,  # Gram-Schmidt w_branch against w_bio
):
    """Simulate one batch's worth of cells on a BIFURCATING trajectory.

    Mirrors Decipher-BC's generative hierarchy v -> z -> x, with a batch shift applied to z, and
    with the v -> z stage driven by a 2-D latent v = (pseudotime, branch coordinate).

    Inputs
    ------
    Everything from `simulated_data_pipeline_aug1.simulate_multivariate`, unchanged, plus:

    branching_t : float or None
        Pseudotime of the fork. Cells with latent_t <= branching_t are on the shared trunk and get
        branch_id 0. None disables branching entirely, reproducing aug1 byte-for-byte (no extra
        draws are taken from either RNG stream, so every downstream value is identical).
    branch_prob : float
        Probability a post-fork cell lands on arm +1. 0.5 gives balanced arms.
    branch_scale : float
        Multiplier on the branch coordinate. Larger = the arms open wider relative to the trunk.
    orthogonalize_branch : bool
        Gram-Schmidt `w_branch` against `w_bio`, renormalized to `w_bio`'s norm. Without it a bad
        `seed` can draw a `w_branch` nearly parallel to `w_bio`, which collapses the Y into a line
        and silently produces a dataset with no fork to find.

    Returns
    -------
    adata : AnnData
        X : (n_samples, n_genes) float array of pre-count expression.
        obs :
            latent_t     -- ground-truth pseudotime
            branch_id    -- -1 / 0 / +1; 0 on the shared trunk          [ADDED 2026-09-18]
            branch_coord -- the signed branch coordinate                [ADDED 2026-09-18]
            shift, batch, latent_z0..latent_z{n_z_dims - 1}  -- as in aug1
        obsm :
            latent_v     -- (n_samples, 2) ground-truth v               [ADDED 2026-09-18]
        uns :
            latent_z_names, w_bio  -- as in aug1
            w_branch, branching_t, branch_prob, branch_scale            [ADDED 2026-09-18]
            (the branch keys are written only when branching is on, so the control path's uns is
            structurally identical to aug1's too)
    """
    bifurcating = branching_t is not None
    if bifurcating and n_z_dims < 2:
        raise ValueError(
            f"a bifurcation needs a second direction to open into: n_z_dims={n_z_dims} < 2. "
            f"Pass branching_t=None for the straight-line (aug1) control."
        )

    rng = np.random.default_rng(cell_seed)

    # --- v: batch-free pseudotime (1D) ---
    latent_t = rng.uniform(0, 1, size=(n_samples, 1))  # shape (n_samples, 1)

    # `seed` stream: what every batch must agree on. w_bio is drawn FIRST, exactly as in aug1, so
    # the trunk direction is bit-identical at the same seed and only the branch direction is new.
    proj_rng = np.random.default_rng(seed)
    w_bio = proj_rng.standard_normal((1, n_z_dims))  # shape (1, n_z_dims)

    if bifurcating:
        w_branch = proj_rng.standard_normal((1, n_z_dims))
        if orthogonalize_branch:
            # project out the component along w_bio, then restore the original scale, so the arms
            # open into a direction the trunk does not already span
            w_branch = w_branch - (w_branch @ w_bio.T) / (w_bio @ w_bio.T) * w_bio
            w_branch = w_branch / np.linalg.norm(w_branch) * np.linalg.norm(w_bio)

        # CELL stream. NOTE: with branching on, this draw is consumed BEFORE the
        # rng.normal(z_mean, biological_sigma) draw below, so the biological-noise realization
        # differs from aug1 at the same cell_seed. Unavoidable, and harmless -- the datasets are
        # different by construction. branching_t=None takes no draw here, which is what keeps the
        # control path byte-for-byte identical.
        branch_id = (rng.random((n_samples, 1)) < branch_prob).astype(int) * 2 - 1
        branch_id = branch_id * (latent_t > branching_t)  # 0 on the shared trunk
        branch_coord = branch_scale * branch_id * np.clip(latent_t - branching_t, 0, None)

        latent_v = np.concatenate([latent_t, branch_coord], axis=1)  # (n_samples, 2)
        w_v = np.concatenate([w_bio, w_branch], axis=0)  # (2, n_z_dims)
    else:
        w_branch = None
        branch_id = np.zeros((n_samples, 1), dtype=int)
        branch_coord = np.zeros((n_samples, 1))
        latent_v = latent_t
        w_v = w_bio

    z_mean = latent_v @ w_v  # shape (n_samples, n_z_dims)

    # ---- everything below this line is copied from aug1 verbatim ----
    #
    # Two mutually exclusive places the batch effect can live.
    #
    #   z-mode    (gene_shift is None) -- offset added to z_mean, BEFORE `@ W`.
    #   gene-mode (gene_shift given)   -- offset added to pre_x, AFTER `@ W`, and NOTHING is added
    #             to z_mean, so latent_z carries no batch offset.
    if gene_shift is None:
        # ---- z-mode ----
        if shift is None:
            if shift_cov is None:
                shift_cov = np.eye(n_z_dims)  # identity default
            shift = rng.multivariate_normal(np.zeros(n_z_dims), shift_cov)

        # normalize by sqrt(n_z_dims) so sigma has the same meaning across dim choices
        shift = shift / np.sqrt(n_z_dims)

        # applied to every cell regardless of arm: a rigid translation of the whole Y
        z_mean = z_mean + shift
    else:
        # ---- gene-mode ----
        gene_shift = np.asarray(gene_shift, dtype=float).ravel()
        if gene_shift.shape[0] != n_genes:
            raise ValueError(
                f"gene_shift has length {gene_shift.shape[0]}, expected n_genes={n_genes}"
            )
        shift = np.zeros(n_z_dims)

    # sample z ~ MVN(z_mean, biological_sigma^2 * I)
    latent_z = rng.normal(z_mean, biological_sigma)  # shape (n, n_z_dims)

    # --- x: z -> x ---
    W = np.random.default_rng(seed + 1).standard_normal((n_z_dims, n_genes))
    pre_x = latent_z @ W
    pre_x_nobatch = None
    if gene_shift is not None:
        # the exact gene-mode ceiling: keeping the pre-offset matrix makes it EXACT rather than
        # approximate, because round / min-subtract / clip happen after the offset and are not
        # invertible.
        pre_x_nobatch = pre_x.copy()
        pre_x = pre_x + gene_shift

    adata = sc.AnnData(X=pre_x)
    if pre_x_nobatch is not None:
        adata.layers["pre_x_nobatch"] = pre_x_nobatch
    adata.obs["latent_t"] = latent_t[:, 0]
    adata.obs["shift"] = (
        "gene_mode -- see uns['gene_shift_matrix']"
        if gene_shift is not None
        else str(np.round(shift, 2).tolist())
    )
    # "batch00" < "batch01" < ... sorts into generation order, so pandas category codes match the
    # row order of uns["shift_matrix"]. See the long note in aug1.
    adata.obs["batch"] = (
        batch_label if batch_label is not None else "_".join(f"{s:.2f}" for s in shift)
    )
    for i in range(latent_z.shape[1]):
        adata.obs[f"latent_z{i}"] = latent_z[:, i]
    adata.uns["latent_z_names"] = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata.uns["w_bio"] = w_bio  # save for reproducibility

    # ADDED 2026-09-18 -- the bifurcation ground truth. Written only when branching is on so the
    # control path produces an AnnData structurally identical to aug1's.
    if bifurcating:
        adata.obs["branch_id"] = branch_id[:, 0]
        adata.obs["branch_coord"] = branch_coord[:, 0]
        adata.obsm["latent_v"] = latent_v
        adata.uns["w_branch"] = w_branch
        adata.uns["branching_t"] = branching_t
        adata.uns["branch_prob"] = branch_prob
        adata.uns["branch_scale"] = branch_scale

    return adata


def shift_magnitudes_multivariate_bifurcation(
    n_batches: int = 5,
    shift_sigma: float = 0.0,
    n_samples: int = 500,
    n_genes: int = 200,
    n_z_dims: int = 3,
    biological_sigma: float = 0.1,
    seed: int = 0,
    batch_mode: str = "z",  # "z" or "genes", as in aug1
    # ADDED 2026-09-18
    branching_t: float = 0.3,
    branch_prob: float = 0.5,
    branch_scale: float = 1.0,
    orthogonalize_branch: bool = True,
):
    """Draw one shift vector per batch and concatenate the simulated batches into one AnnData.

    Identical to `simulated_data_pipeline_aug1.shift_magnitudes_multivariate_from_normal` except
    that it wraps `simulate_multivariate_bifurcation` and threads the four branch knobs through.
    The batch machinery -- the mag_rng stream, z-mode / gene-mode, the batch labels, the count
    conversion, the counts_nobatch ceiling -- is unchanged.
    """
    # dedicated RNG stream so shift draws never collide with the projection stream (`seed`),
    # decoder stream (`seed+1`), or cell streams (`seed+i+1`)
    mag_rng = np.random.default_rng(seed + 1000)

    if batch_mode not in ("z", "genes"):
        raise ValueError(f"batch_mode must be 'z' or 'genes', got {batch_mode!r}")

    if batch_mode == "z":
        # (n_batches, n_z_dims): shift distribution is MVN(0, shift_sigma^2 * I) in n_z_dims
        shift_matrix = mag_rng.multivariate_normal(
            mean=np.zeros(n_z_dims),
            cov=(shift_sigma**2) * np.eye(n_z_dims),
            size=n_batches,
        )
        gene_shift_matrix = None
    else:
        # (n_batches, n_genes): a free offset per gene per batch, added after `@ W`.
        shift_matrix = None
        gene_shift_matrix = mag_rng.normal(0.0, shift_sigma, size=(n_batches, n_genes))

    adata_concat = None
    for i in range(n_batches):
        shift = None if shift_matrix is None else shift_matrix[i]
        gene_shift = None if gene_shift_matrix is None else gene_shift_matrix[i]
        adata_sim = simulate_multivariate_bifurcation(
            n_samples=n_samples,
            n_genes=n_genes,
            n_z_dims=n_z_dims,
            shift=shift,
            gene_shift=gene_shift,
            biological_sigma=biological_sigma,
            seed=seed,
            cell_seed=seed + i + 1,
            batch_label=f"batch{i:02d}",
            branching_t=branching_t,
            branch_prob=branch_prob,
            branch_scale=branch_scale,
            orthogonalize_branch=orthogonalize_branch,
        )
        if adata_concat is None:
            adata_concat = adata_sim
        else:
            adata_concat = ad.concat(
                [adata_concat, adata_sim],
                axis=0,
                join="outer",
                label=None,
                merge="same",
            )

    # After all batches concatenated: convert to non-negative integer counts
    X_all = adata_concat.X
    X_all = np.clip(np.round(X_all - X_all.min(axis=0)), 0, None).astype(int)
    adata_concat.X = X_all
    adata_concat.layers["counts"] = X_all.copy()

    if batch_mode == "genes":
        # the exact gene-mode ceiling: what this dataset would have been with gene_shift = 0 and
        # everything else held fixed
        C = np.asarray(adata_concat.layers["pre_x_nobatch"])
        adata_concat.layers["counts_nobatch"] = np.clip(
            np.round(C - C.min(axis=0)), 0, None
        ).astype(int)
        del adata_concat.layers["pre_x_nobatch"]

    # record swept parameters so downstream plots can read them back
    adata_concat.uns["shift_sigma"] = shift_sigma
    adata_concat.uns["n_z_dims"] = n_z_dims
    adata_concat.uns["batch_mode"] = batch_mode
    if batch_mode == "z":
        # (n_batches, n_z_dims), raw / pre-normalization -- needs /sqrt(n_z_dims) before comparing
        # it to anything measured off latent_z
        adata_concat.uns["shift_matrix"] = shift_matrix
    else:
        adata_concat.uns["gene_shift_matrix"] = gene_shift_matrix

    # ADDED 2026-09-18 -- restore the per-batch uns keys that ad.concat drops, and add the branch
    # ones.
    #
    # MEASURED, and it is a pre-existing aug1 bug: ad.concat(..., merge="same") silently DROPS
    # uns["w_bio"] and uns["latent_z_names"] whenever n_batches > 1. The "same" strategy keeps a
    # key only if it compares equal across all inputs, and that comparison does not hold for a
    # numpy array or a list, even when every batch holds an identical one. So:
    #
    #     n_batches=1 -> uns has w_bio, latent_z_names
    #     n_batches>1 -> uns has neither
    #
    # Every multi-batch h5ad aug1 has ever written is therefore missing the projection that
    # defines its own ground truth. Nothing downstream read it, so nothing broke -- but the
    # bifurcation analysis DOES need it: the (w_bio, w_branch) plane is the basis the Y is legible
    # in. Assigning after the concat is unconditional and cheap, so it is done for both.
    adata_concat.uns["w_bio"] = adata_sim.uns["w_bio"]
    adata_concat.uns["latent_z_names"] = adata_sim.uns["latent_z_names"]
    if branching_t is not None:
        adata_concat.uns["w_branch"] = adata_sim.uns["w_branch"]
        adata_concat.uns["branching_t"] = branching_t
        adata_concat.uns["branch_prob"] = branch_prob
        adata_concat.uns["branch_scale"] = branch_scale

    return adata_concat


def branch_cluster_orders(adata, min_cells=10, cluster_key="decipher_clusters"):
    """Split the decipher clusters into one ordered cluster list per arm of the Y.

    Each arm's list is (shared trunk clusters + that arm's clusters), ordered by mean latent_t --
    the same ordering rule aug1 already uses, just applied per arm. A cluster is assigned to an arm
    by the MODE of branch_id among its cells, so clusters straddling the fork go wherever they
    mostly sit.

    Returns
    -------
    dict : {trajectory_name: [cluster_id, ...]}
        Two entries ("traj_plus", "traj_minus") normally. Falls back to a single
        {"trajectory": ...} over all valid clusters -- exactly aug1's behavior -- in either of the
        two no-fork cases: the data was generated with branching_t=None (no `branch_id` column at
        all), or the fork washed out at high shift_sigma so one arm has no clusters of its own.
        The sweep must degrade, not crash.
    """
    counts = adata.obs[cluster_key].value_counts()
    valid = set(counts[counts > min_cells].index)

    by_time = (
        adata.obs.groupby(cluster_key, observed=True)["latent_t"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    by_time = [c for c in by_time if c in valid]

    # ADDED 2026-09-18: branching_t=None writes no `branch_id` column (that is what keeps the
    # control path byte-for-byte identical to aug1), so the straight-line half of a sweep reaches
    # here with nothing to split on. Same answer as a washed-out fork: one trajectory.
    if "branch_id" not in adata.obs.columns:
        return {"trajectory": by_time}

    arm = adata.obs.groupby(cluster_key, observed=True)["branch_id"].agg(lambda s: s.mode().iat[0])

    has_plus = any(arm[c] == 1 for c in by_time)
    has_minus = any(arm[c] == -1 for c in by_time)
    if not (has_plus and has_minus):
        return {"trajectory": by_time}

    return {
        "traj_plus": [c for c in by_time if arm[c] in (0, 1)],
        "traj_minus": [c for c in by_time if arm[c] in (0, -1)],
    }


def _data_tag(branching_t, batch_mode):
    """Filename fragment identifying the DATA a run was trained on, e.g. "bif_z", "nobif_genes".

    ADDED 2026-09-18. Output names previously encoded only (sigma, seed, model), so every
    combination of the two data axes at the same coordinates wrote the SAME path and silently
    overwrote each other -- the results of the losing run vanished with no error.

    batch_mode stays in the name even though it is implied by the model for model2/model5,
    because it is NOT implied for `native`, which is run against both regimes as the baseline.

    THE RULE, because this bug has now appeared twice: every axis the sweep varies must appear
    in the output filename. First it was bifurcation/batch_mode (this function). Then the grid
    widened from 1 to 3 decipher_seeds and the h5ad name did not carry decipher_seed, so all
    three wrote to one path and 2 of every 3 results were destroyed with no error. The caller
    appends `_decipherseed_{decipher_seed}`; `tests/0918_test_bifurcation_pipeline.py::
    test_trained_h5ad_name_separates_every_swept_axis` guards the whole composition. Add a new
    sweep axis -> extend that test first.
    """
    return f"{'bif' if branching_t is not None else 'nobif'}_{batch_mode}"


def branch_silhouette(adata, branching_t, rep_key="decipher_v"):
    """Silhouette of branch_id in `rep_key`, restricted to post-fork cells.

    Returns NaN when there is no fork to measure -- either the data has no `branch_id` column
    (branching_t=None) or only one arm survives among the post-fork cells -- so a straight-line or
    collapsed run reports rather than raises.

    NOT comparable to the same quantity computed on latent_z: silhouette depends on the
    dimension of the space, and this is measured in decipher_v (2-D) while latent_z is n_z_dims.
    Compare branch_asw within decipher_v, across sigma, within a model.
    """
    if branching_t is None or "branch_id" not in adata.obs.columns:
        return np.nan
    post = (adata.obs["latent_t"] > branching_t).values
    labels = adata.obs["branch_id"].values[post]
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(silhouette_score(np.asarray(adata.obsm[rep_key])[post], labels))


# --------------------------------------------------------------------------------------------
# Model selection
#
# CHANGED 2026-09-18. This used to be aug1's if/elif chain over ten separate packages
# (decipher_vz, decipher_zx, ...). `decipher_models` replaced all of them with ONE package whose
# architecture is chosen by config flags, so the chain is now a preset table.
#
#     batch_conditioning  : "none" | "decoder_only" | "decoder_encoder"   (is the guide conditioned)
#     batch_embedding_mode: "concat_z" | "additive" | "concat_x"          (which family)
#     mean_field_v        : False | True                                  (encoder architecture)
#
# The 1 -> 2 -> 3 progression is the same in every family, which is why this is a product of two
# flags rather than ten names: "decoder_only" = guide blind, "decoder_encoder" = guide
# conditioned, and mean_field_v=True is the third step. See batch-conditioning-comparison.md.
#
# The table itself lives in decipher_models.presets -- shared with
# tests/test_variant_equivalence.py, so the sweep and the equivalence test can never disagree
# about what "model4" means. Bad flag VALUES are caught by DecipherConfig.__post_init__ and bad
# field names by the dataclass constructor, so this file does no validation of its own.


def _build_model(model):
    """Return (dc, DecipherConfig, preset_kwargs) for one model name."""
    if model not in MODEL_PRESETS:
        raise ValueError(f"unknown model {model!r}. Expected one of: {', '.join(MODEL_PRESETS)}")
    dc = importlib.import_module("decipher_models")
    DecipherConfig = importlib.import_module("decipher_models.tools._decipher").DecipherConfig
    return dc, DecipherConfig, MODEL_PRESETS[model]


def train_and_compute_rho_r2_bifurcation(
    model,
    decipher_seed,
    n_batches: int = 5,
    shift_sigma: float = 0.0,
    n_samples: int = 500,
    n_genes: int = 200,
    n_z_dims: int = 3,
    biological_sigma: float = 0.1,
    seed: int = 0,
    dim_z: int = None,
    beta: float = 0.1,
    notebook_tag: str = None,
    batch_mode: str = "z",
    branching_t: float = 0.3,
    branch_prob: float = 0.5,
    branch_scale: float = 1.0,
    wandb_run=None,
):
    """Generate bifurcating data, train one variant, and score it.

    Same shape as `simulated_data_pipeline_aug1.train_and_compute_rho_r2`, with two differences:
    the data has a fork in it, and pseudotime is scored along ONE TRAJECTORY PER ARM rather than a
    single spline that would have to zigzag across the fork.

    wandb_run : wandb Run or None
        ADDED 2026-09-18. When given, the per-epoch loss curves, the diagnostic figure and the
        final metrics are logged to it. Taken as a parameter rather than returning the decipher
        object because the losses and the figure only exist inside this function, and adding a
        return value would break every existing caller's unpacking.
        None keeps this function wandb-free, so the tests and the smoke script are unaffected.

    Returns
    -------
    (rho, rho_plus, rho_minus, branch_asw, trained_h5ad_path, r2_overall, r2_per_gene_median, adata)
    """
    adata = shift_magnitudes_multivariate_bifurcation(
        n_batches=n_batches,
        shift_sigma=shift_sigma,
        n_samples=n_samples,
        n_genes=n_genes,
        n_z_dims=n_z_dims,
        biological_sigma=biological_sigma,
        seed=seed,
        batch_mode=batch_mode,
        branching_t=branching_t,
        branch_prob=branch_prob,
        branch_scale=branch_scale,
    )

    if dim_z is None:
        dim_z = n_z_dims

    dc, DecipherConfig, preset = _build_model(model)
    model_tag = model

    config = DecipherConfig(
        learning_rate=1e-3, seed=decipher_seed, dim_z=dim_z, beta=beta, **preset
    )
    decipher, _ = dc.tl.decipher_train(
        adata,
        config,
        plot_kwargs={"color": "batch", "title": f"shift_sigma={shift_sigma}"},
    )

    if wandb_run is not None:
        # decipher_train fills in dim_genes/n_cells/n_batches/batch_key via
        # config.initialize_from_adata, so the config is only complete after training.
        wandb_run.config.update(config.to_dict(), allow_val_change=True)
        for epoch, (train_elbo, val_nll) in enumerate(
            zip(decipher.train_losses_, decipher.val_losses_)
        ):
            wandb_run.log({"train_elbo": train_elbo, "val_nll": val_nll}, step=epoch)

    # Reconstruction diagnostic -- computed right after training so we still have the in-memory
    # decipher object with train_losses_ / val_losses_.
    r2_result = dc.tl.reconstruction_r2_log1p(decipher, adata)
    r2_overall = r2_result["r2_overall"]
    r2_per_gene_median = float(np.nanmedian(r2_result["r2_per_gene"]))
    adata.uns["r2_overall"] = r2_overall
    adata.uns["r2_per_gene_median"] = r2_per_gene_median

    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(
        script_dir, "..", "Simulated Adata", "shift_sigma_sweep_bifurcation", today
    )

    # Save reconstruction diagnostic figure
    fig, axes = dc.pl.reconstruction_r2_log1p(decipher, adata, figsize=(21, 5))
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(labelsize=8)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(8)
    fig.suptitle(
        f"{model_tag} | sigma={shift_sigma} | decipher seed={decipher_seed}",
        y=1.02,
        fontsize=11,
    )
    figs_dir = os.path.join(out_root, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            figs_dir,
            f"reconstruction_sigma{shift_sigma}_seed{seed}_{model_tag}"
            f"_{_data_tag(branching_t, batch_mode)}_decipherseed_{decipher_seed}.png",
        ),
        dpi=120,
        bbox_inches="tight",
    )
    if wandb_run is not None:
        import wandb

        wandb_run.log({"reconstruction_diagnostic": wandb.Image(fig)})
    plt.close(fig)

    # Trajectories -- one per arm, sharing the trunk clusters.
    #
    # decipher_time (native-decipher/decipher/tools/trajectory_inference.py:402) loops over the
    # trajectories and writes each one's KNN-regressed time onto the cells in its clusters. The
    # trunk clusters belong to BOTH arms, so the second loop overwrites the first there -- that is
    # fine and not a bug: both arms share the same trunk geometry and both measure arc length from
    # the same root, so the two estimates agree and pseudotime stays globally consistent.
    dc.tl.cell_clusters(adata, leiden_resolution=1.0, n_neighbors=10, seed=0)
    orders = branch_cluster_orders(adata)
    dc.tl.trajectories(
        adata, *[dc.tl.TConfig(name, cluster_ids_list=ids) for name, ids in orders.items()]
    )
    dc.tl.decipher_rotate_space(adata)
    dc.tl.decipher_time(adata)

    # Metrics
    m = adata.obs["decipher_time"].notna()
    rho, _ = spearmanr(adata.obs["decipher_time"][m], adata.obs["latent_t"][m])
    adata.uns["rho"] = abs(rho)

    # per-arm rho. Trunk cells (branch_id == 0) are in neither, but are already counted in pooled.
    # With branching_t=None there are no arms, so both are NaN -- meaningful, not missing.
    arm_rhos = {"rho_plus": np.nan, "rho_minus": np.nan}
    if "branch_id" in adata.obs.columns:
        for arm_name, arm_value in (("rho_plus", 1), ("rho_minus", -1)):
            sel = m & (adata.obs["branch_id"] == arm_value)
            if sel.sum() < 3:
                continue
            r, _ = spearmanr(adata.obs["decipher_time"][sel], adata.obs["latent_t"][sel])
            arm_rhos[arm_name] = abs(r)
    adata.uns.update(arm_rhos)

    branch_asw = branch_silhouette(adata, branching_t)
    adata.uns["branch_asw"] = branch_asw
    adata.uns["trajectory_names"] = list(orders.keys())

    if wandb_run is not None:
        # Both log() and summary: log() makes it plottable against the sweep axes, summary makes
        # it sortable in the runs table. NaN arm metrics on the no-bifurcation half are recorded
        # as-is -- they mean "no fork existed", not "measurement missing".
        final = {
            "rho": abs(rho),
            "rho_plus": arm_rhos["rho_plus"],
            "rho_minus": arm_rhos["rho_minus"],
            "branch_asw": branch_asw,
            "r2_overall": r2_overall,
            "r2_per_gene_median": r2_per_gene_median,
            "n_trajectories": len(orders),
        }
        wandb_run.log(final)
        for k, v in final.items():
            wandb_run.summary[k] = v
        wandb_run.summary["status"] = "success"

    # Save trained adata. As in aug1, two sweeps on the same calendar day that both leave
    # notebook_tag unset will overwrite each other's h5ads -- pass a distinct tag per notebook.
    trained_dir = os.path.join(out_root, "trained")
    os.makedirs(trained_dir, exist_ok=True)
    prefix = f"{notebook_tag}_" if notebook_tag else ""
    trained_h5ad_path = os.path.join(
        trained_dir,
        f"{prefix}sigma{shift_sigma}_seed{seed}_{model_tag}"
        f"_{_data_tag(branching_t, batch_mode)}_decipherseed_{decipher_seed}.h5ad",
    )
    adata.write(trained_h5ad_path)

    return (
        abs(rho),
        arm_rhos["rho_plus"],
        arm_rhos["rho_minus"],
        branch_asw,
        trained_h5ad_path,
        r2_overall,
        r2_per_gene_median,
        adata,
    )


if __name__ == "__main__":

    # ---- sweep ----
    NOTEBOOK_TAG = "bifurcation_sep18"

    shift_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]
    seeds = [3, 4]
    decipher_seeds = [1]
    n_z_dims = 3
    n_samples = 500
    n_genes = 200
    biological_sigma = 0.1
    branching_t = 0.3
    branch_prob = 0.5

    # Display name -> MODEL_PRESETS key. All ten arms now come from the one decipher_models
    # package; add or drop a line to change the sweep's scope.
    models = {
        "Base Decipher": "native",
        "Set1 concat->z  model1": "model1",
        "Set1 concat->z  model2": "model2",
        "Set1 concat->z  model3": "model3",
        "Set2 add->z     model1": "model1-add",
        "Set2 add->z     model2": "model2-add",
        "Set2 add->z     model3": "model3-add",
        "Set3 concat->x  model4": "model4",
        "Set3 concat->x  model5": "model5",
        "Set3 concat->x  model6": "model6",
    }

    today = datetime.now().strftime("%m%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(
        script_dir, "..", "Simulated Adata", "shift_sigma_sweep_bifurcation", today
    )
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{NOTEBOOK_TAG}_sweep_log_{shift_sigmas}_{seeds}.csv")

    run_log = []
    for name, model in models.items():
        for shift_sigma in shift_sigmas:
            for sd in seeds:
                for decipher_seed in decipher_seeds:
                    record = {
                        "model": name,
                        "n_z_dims": n_z_dims,
                        "shift_sigma": shift_sigma,
                        "seed": sd,
                        "decipher_seed": decipher_seed,
                        "branching_t": branching_t,
                        "branch_prob": branch_prob,
                    }
                    try:
                        (
                            rho,
                            rho_plus,
                            rho_minus,
                            branch_asw,
                            trained_path,
                            r2_overall,
                            r2_per_gene_median,
                            _,
                        ) = train_and_compute_rho_r2_bifurcation(
                            model,
                            decipher_seed,
                            shift_sigma=shift_sigma,
                            n_samples=n_samples,
                            n_genes=n_genes,
                            biological_sigma=biological_sigma,
                            seed=sd,
                            n_z_dims=n_z_dims,
                            branching_t=branching_t,
                            branch_prob=branch_prob,
                            notebook_tag=NOTEBOOK_TAG,
                        )
                        record.update(
                            {
                                "rho": rho,
                                "rho_plus": rho_plus,
                                "rho_minus": rho_minus,
                                "branch_asw": branch_asw,
                                "trained_h5ad": trained_path,
                                "r2_overall": r2_overall,
                                "r2_per_gene_median": r2_per_gene_median,
                                "error": None,
                            }
                        )
                    except Exception as e:
                        print(
                            f"[{name}] n_z_dims={n_z_dims} shift_sigma={shift_sigma} "
                            f"seed={sd} decipher_seed={decipher_seed} "
                            f"failed: {type(e).__name__}: {e}"
                        )
                        record.update(
                            {
                                "rho": np.nan,
                                "rho_plus": np.nan,
                                "rho_minus": np.nan,
                                "branch_asw": np.nan,
                                "trained_h5ad": None,
                                "r2_overall": np.nan,
                                "r2_per_gene_median": np.nan,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                    run_log.append(record)
                    pd.DataFrame(run_log).to_csv(csv_path, index=False)
