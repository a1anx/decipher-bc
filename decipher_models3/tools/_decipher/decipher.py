"""DecipherConfig and Decipher for the seven arms of `decipher_models3`.

Rewritten 2026-10-16 from `decipher_models/tools/_decipher/decipher.py`. Every batch arm has ONE
learned table `batch_emb = nn.Embedding(n_batches, dim_batch_embedding)`, shared by every network
that sees the batch, and its row e_b is concatenated onto that network's input (the guides take
log1p x):

    arm       prior                    reconstruction h        guide d_z            v encoder
    native    f(v)                     h(z)                    d_z(log1p x)         full
    model1    f([v; e_b])              h(z)                    d_z(log1p x)         full
    model2    f([v; e_b])              h(z)                    d_z([log1p x; e_b])  full
    model3    f([v; e_b])              h(z)                    d_z([log1p x; e_b])  mean-field
    model4    f(v)                     h([z; e_b])             d_z(log1p x)         full
    model5    f(v)                     h([z; e_b])             d_z([log1p x; e_b])  full
    model6    f(v)                     h([z; e_b])             d_z([log1p x; e_b])  mean-field
    model4r   f(v)                     W z + c + r([z; e_b])   d_z(log1p x)         full
    model5r   f(v)                     W z + c + r([z; e_b])   d_z([log1p x; e_b])  full
    model7    f_mean(v) + t([v; e_b])  W z + c + r([z; e_b])   d_z([log1p x; e_b])  full
    model7i   f_mean(v) + t([v; e_b])  W(z + ρ([z; e_b])) + c  d_z([log1p x; e_b])  full
    model5rz  f(v)                     h([z; e_b]) + r_z(z)    d_z([log1p x; e_b])  full

f = (f_mean, f_var) is the prior network `decoder_v_to_z`, h the reconstruction decoder
`decoder_z_to_x` (h([z; e_b]) = W z + W_e e_b + c), d_z and d_v the guide networks (the tex's
d^{->z}, d^{->v}: `encoder_x_to_z`, `encoder_zx_to_v`), L = dim_z, D = dim_batch_embedding.
mean-field: the v encoder sees log1p x only.

concat_z (model1-3) is `decipher_models`' math and init: `batch_emb` is built BEFORE the networks
with PyTorch's default N(0, 1) init, so the same seed gives `decipher_models`' state_dict.
concat_x (model4-6) replaces `decipher_models`' fixed one-hot context with the shared embedding of
`decipher_m5` model5e: `batch_emb` is built AFTER the networks and zero-initialized, so the
untrained model is batch-blind and the same seed gives model5e's state_dict for model5.
The `additive` family (Set 2) does not exist here.

`recon_batch="residual"` (model4r, model5r; concat_x only) moves e_b out of h: the logits are
W z + c + r([z; e_b]), W z + c = `decoder_z_to_x(z)` (batch-free gene programs) and
r = `decoder_residual` (L + D -> 64 -> G, ReLU, output layer zero at init), so the batch
correction depends on the cell state and need not lie in span(W). The table is N(0, 1) and is built
after every network (r included), so the encoders and the prior keep model4/5's seed draws.

`prior_batch="residual"` (model7, model7i; concat_x only) anchors the prior MrVI-style: z_loc =
f_mean(v) + t([v; e_b]) with f = (f_mean, f_var) = `decoder_v_to_z` batch-blind (native width)
and t = `prior_residual` (2 + D -> 32 -> L, ReLU, output layer zero at init); the z-scale comes
from f_var alone. Training adds the penalty "prior_residual" = mean over cells of ||t||^2, weight
lambda_r. `recon_batch="residual_in_span"` (model7i) is MrVI's shipped decoder: logits =
W(z + ρ([z; e_b])) + c, ρ = `decoder_residual_in_span` (L + D -> 32 -> L, output layer
zero), so the batch correction depends on z but stays inside span(W). Both use the N(0, 1) table
built after every network, as 1b.

`recon_batch="concat_plus_residual_z"` (model5rz; concat_x only, 2026-10-17) keeps model5's
h([z; e_b]) and adds r_z(z) = `decoder_residual_z` (L -> 64 -> G, output layer zero): model5r's
capacity without e_b inside r_z. `batch_emb_init` ("auto" = the rules above, or "zero" / "normal")
and `residual_hidden` (r's or r_z's / ρ's hidden width, default 64 / 32) are config overrides; the
zeroing is applied after every network is built, so neither moves an RNG draw.

`prior="vamp"` (model8v = model8 + VampPrior) replaces p(v) = N(0, I) with
(1/K) sum_k q(v | u_k, b_k): the guide's v distribution at K learned pseudo-inputs u_k (count space,
softplus), with b_k round-robin over the batches. The pseudo-inputs are built last, from a private
generator, so every other weight keeps its seed draw.
"""

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union, get_args, get_origin, get_type_hints

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
import torch.utils.data
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus

from decipher_models3.tools._decipher.module import ConditionalDenseNN


@dataclass(unsafe_hash=True)
class DecipherConfig:
    dim_z: int = 10
    dim_v: int = 2
    layers_v_to_z: Sequence = (64,)
    layers_z_to_x: Sequence = tuple()

    beta: float = 1e-1
    seed: int = 0

    learning_rate: float = 5e-3
    val_frac: float = 0.1
    batch_size: int = 64
    n_epochs: int = 1000
    early_stopping_patience: Optional[int] = 10

    dim_genes: int = None
    n_cells: int = None

    # ---- BATCH ----
    # `batch_conditioning` says WHICH HALF of the model sees the batch label, and reads the
    # same way in every mode:
    #   "none"            -- native. No batch anywhere.
    #   "decoder_only"    -- batch reaches the generative model only. That is decoder_v_to_z
    #                        under "concat_z", and decoder_z_to_x under "concat_x".
    #   "decoder_encoder" -- batch also reaches the guide, via encoder_x_to_z.
    #
    # `batch_embedding_mode` says WHERE it enters. Both modes use one learned table
    # `batch_emb` (n_batches x dim_batch_embedding) whose row e_b is concatenated onto inputs:
    #   "concat_z" -- Set 1. e_b joins decoder_v_to_z's input, so the batch shifts the prior on
    #                 z. Default N(0, 1) init, as `decipher_models`.
    #   "concat_x" -- Set 3. The b -> z edge is deleted; the prior is identical to base Decipher.
    #                 e_b joins decoder_z_to_x's input. Zero init, as `decipher_m5` model5e.
    # "additive" (Set 2) was removed in decipher_models3 and raises.
    batch_conditioning: Literal["none", "decoder_only", "decoder_encoder"] = "none"
    mean_field_v: bool = False
    batch_embedding_mode: Literal["concat_z", "concat_x"] = "concat_z"
    n_batches: int = 0
    dim_batch_embedding: int = 8
    batch_key: Optional[str] = None
    # The batch labels, in code order, of the FULL training AnnData (set by `decipher_train`).
    # Every batch code is built against these (`data.batch_codes`), never against a subset's own
    # categories: AnnData subsetting drops unused categories, which shifts a subset's codes.
    # None without a batch column (n_batches == 0).
    batch_levels: Optional[list[str]] = None
    # How e_b enters the reconstruction under "concat_x" (raises with any other mode):
    #   "concat"   -- logits = h([z; e_b]): a per-batch shift W_e e_b, the same in every cell.
    #   "residual" -- logits = W z + c + r([z; e_b]): W has no e_b columns, r is a zero-output
    #                 MLP (`decoder_residual`), so the shift depends on z. Table init N(0, 1).
    #   "residual_in_span" -- logits = W(z + ρ([z; e_b])) + c: ρ = `decoder_residual_in_span`
    #                 (L + D -> 32 -> L, zero output layer), so the shift depends on z
    #                 but lies in span(W). Table init N(0, 1).
    #   "concat_plus_residual_z" -- logits = h([z; e_b]) + r_z(z): model5's per-batch shift W_e e_b
    #                 plus a batch-blind r_z = `decoder_residual_z` (L -> 64 -> G, zero output
    #                 layer). r_z never sees e_b (model5rz, the capacity control of model5r).
    recon_batch: Literal["concat", "residual", "residual_in_span", "concat_plus_residual_z"] = (
        "concat"
    )
    # Init of the `batch_emb` table:
    #   "auto"   -- the per-arm rule: zero under concat_x with recon_batch "concat" or
    #               "concat_plus_residual_z" and no residual prior (model4-6, the untrained model is
    #               batch-blind), N(0, 1) otherwise (concat_z, residual, residual_in_span, residual
    #               prior).
    #   "zero" / "normal" -- force that init. The table always draws PyTorch's N(0, 1) init at its
    #               usual place in the construction order; the field only decides whether that
    #               table is zeroed in place at the END of construction, so no other weight's RNG
    #               draw moves. Needs a batch arm (batch_conditioning != "none").
    batch_emb_init: Literal["auto", "zero", "normal"] = "auto"
    # Hidden width of the decoder's batch network: r or r_z (recon_batch "residual" or
    # "concat_plus_residual_z", default RESIDUAL_HIDDEN = 64) or ρ ("residual_in_span", default
    # ANCHOR_HIDDEN = 32). None keeps those defaults. t (prior_batch="residual") is never affected.
    # Raises on an arm with neither r nor ρ.
    residual_hidden: Optional[int] = None
    # How e_b enters the prior p(z | v, b):
    #   "none"     -- f(v), batch-blind (native and the concat_x arms model4-6, 4r, 5r).
    #   "concat"   -- f([v; e_b]): the concat_z wiring (model1-3).
    #   "residual" -- z_loc = f_mean(v) + t([v; e_b]), z-scale f_var(v) only; t =
    #                 `prior_residual` (2 + D -> 32 -> L, zero output layer). concat_x
    #                 only. Adds the training penalty "prior_residual" (weight lambda_r).
    # None (the default) is filled in by __post_init__ from batch_embedding_mode: "concat" for a
    # concat_z batch arm, "none" otherwise, so the existing presets and saved configs keep their
    # math without naming the field. "none" / "concat" must match that derived value.
    prior_batch: Optional[Literal["none", "concat", "residual"]] = None
    # -----------------

    # ---- TRAINING PENALTIES ----
    # Added to the training loss only, never to `predictive_log_likelihood` (val NLL). Any active
    # penalty switches `decipher_train` from `svi.step` to `penalized_step`; see
    # `Decipher.penalty_weights` / `Decipher.penalties`. Off by default.
    #   v_invariance="mmd" -- penalty "mmd_v" = `mmd_v_penalty` on the guide's v samples,
    #                         weight lambda_inv.
    #   prior_batch="residual" -- penalty "prior_residual" = mean over cells of ||t([v; e_b])||^2
    #                         on the guide's v samples, weight lambda_r.
    #   cycle_v=True       -- penalty "cycle_v" = `Decipher.cycle_v_penalty`, latent
    #                         cycle-consistency of v under a swapped batch (sysVI), weight
    #                         lambda_c. Needs concat_x + decoder_encoder (model5's wiring).
    v_invariance: Literal["none", "mmd"] = "none"
    lambda_inv: float = 1.0
    lambda_r: float = 1.0
    cycle_v: bool = False
    lambda_c: float = 1.0
    # ----------------------------

    # ---- PRIOR ON v ----
    #   "normal" -- N(0, I) (default).  "gamma" -- Gamma(0.3, 0.8), with a Gamma guide on v.
    #   "vamp"   -- VampPrior (Tomczak & Welling 2018), p(v) = (1/K) sum_k q(v | u_k, b_k): the
    #               guide's own v distribution at K = n_pseudo_inputs learned pseudo-inputs u_k
    #               (count space, softplus), pseudo-input k under batch b_k = k mod n_batches.
    #               See `Decipher.vamp_components`. The v KL becomes a Monte Carlo term.
    prior: str = "normal"
    n_pseudo_inputs: int = 100

    _initialized_from_adata: bool = False

    def __post_init__(self):
        """Enforce the Literal annotations above, which Python does not enforce on its own.

        ADDED 2026-09-18. `@dataclass` ignores `Literal` at runtime, so before this
        `DecipherConfig(batch_embedding_mode="concat")` -- the pre-rename spelling, still present
        in older notebooks and scripts -- was accepted silently. The model then fell through to
        whichever branch the unrecognized value missed and trained a DIFFERENT ARCHITECTURE than
        the caller asked for, with nothing in the losses, the h5ad or the sweep log to reveal it.
        A whole sigma sweep could be published under the wrong model names.

        Failing at construction makes that class of error impossible. Every caller benefits --
        notebooks, sweeps, tests -- not just the ones that remember to validate.
        """
        if self.prior_batch is None:
            self.prior_batch = self._derived_prior_batch()
        if self.batch_levels is not None:
            # A config read back from an h5ad holds a numpy array here.
            self.batch_levels = [str(level) for level in self.batch_levels]
        for name, allowed in _LITERAL_CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"DecipherConfig.{name}={value!r} is not one of {list(allowed)}. "
                    f"If this used to work, the flag was renamed -- check the field's comment "
                    f"above rather than assuming the old value still means what it did."
                )
        if self.lambda_inv < 0:
            # A negative weight would reward batch separation in v.
            raise ValueError(f"DecipherConfig.lambda_inv={self.lambda_inv!r} must be >= 0")
        if self.lambda_c < 0:
            # A negative weight would reward breaking the cycle, i.e. batch-dependent v.
            raise ValueError(f"DecipherConfig.lambda_c={self.lambda_c!r} must be >= 0")
        if self.prior == "vamp" and self.n_pseudo_inputs <= 0:
            raise ValueError(
                f"prior='vamp' needs n_pseudo_inputs >= 1; got n_pseudo_inputs="
                f"{self.n_pseudo_inputs!r}"
            )
        cycle_wiring = (self.batch_embedding_mode, self.batch_conditioning)
        if self.cycle_v and cycle_wiring != ("concat_x", "decoder_encoder"):
            # The cycle decodes under b' and re-encodes under b', so both the decoder and the
            # guide on z must take e_b.
            raise ValueError(
                f"cycle_v=True needs batch_embedding_mode='concat_x' and "
                f"batch_conditioning='decoder_encoder' (model5's wiring); got "
                f"batch_embedding_mode={self.batch_embedding_mode!r}, "
                f"batch_conditioning={self.batch_conditioning!r}"
            )
        if self.lambda_r < 0:
            # A negative weight would reward a large prior residual t.
            raise ValueError(f"DecipherConfig.lambda_r={self.lambda_r!r} must be >= 0")
        concat_x_batch = (
            self.batch_conditioning != "none" and self.batch_embedding_mode == "concat_x"
        )
        if self.recon_batch != "concat" and not concat_x_batch:
            raise ValueError(
                f"DecipherConfig.recon_batch={self.recon_batch!r} needs a batch decoder: "
                f"batch_embedding_mode='concat_x' and batch_conditioning != 'none', got "
                f"{self.batch_embedding_mode!r} / {self.batch_conditioning!r}."
            )
        if self.residual_hidden is not None:
            if self.recon_batch not in _RECON_WITH_HIDDEN:
                raise ValueError(
                    f"DecipherConfig.residual_hidden={self.residual_hidden!r} needs a decoder batch "
                    f"network r or ρ (recon_batch in {list(_RECON_WITH_HIDDEN)}), got "
                    f"recon_batch={self.recon_batch!r}."
                )
            if self.residual_hidden < 1:
                raise ValueError(
                    f"DecipherConfig.residual_hidden={self.residual_hidden!r} must be >= 1"
                )
        if self.batch_emb_init != "auto" and self.batch_conditioning == "none":
            raise ValueError(
                f"DecipherConfig.batch_emb_init={self.batch_emb_init!r} needs a batch arm, got "
                f"batch_conditioning='none' (no batch_emb table)."
            )
        if self.prior_batch == "residual":
            if not concat_x_batch:
                raise ValueError(
                    f"DecipherConfig.prior_batch='residual' needs batch_embedding_mode="
                    f"'concat_x' and batch_conditioning != 'none', got "
                    f"{self.batch_embedding_mode!r} / {self.batch_conditioning!r}."
                )
        elif self.prior_batch != self._derived_prior_batch():
            raise ValueError(
                f"DecipherConfig.prior_batch={self.prior_batch!r} contradicts "
                f"batch_embedding_mode={self.batch_embedding_mode!r} / batch_conditioning="
                f"{self.batch_conditioning!r}, which give {self._derived_prior_batch()!r}."
            )

    def _derived_prior_batch(self) -> str:
        """The prior wiring `batch_embedding_mode` implies: "concat" for a concat_z batch arm."""
        concat_z_batch = (
            self.batch_conditioning != "none" and self.batch_embedding_mode == "concat_z"
        )
        return "concat" if concat_z_batch else "none"

    def initialize_from_adata(
        self, adata, batch_key: str = "batch", batch_levels: Optional[Sequence[str]] = None
    ) -> None:
        """Set dim_genes and n_cells from `adata`, and the batch levels.

        `batch_levels` are the batch labels of the full training AnnData. Pass them whenever
        `adata` is a subset (the training split): a subset's categories miss any level with no
        cells in it. When None they are read from `adata.obs[batch_key]` (None and n_batches 0
        without that column).
        """
        self.dim_genes = adata.shape[1]
        self.n_cells = adata.shape[0]

        # Save the key so the model instance knows what it was trained on
        self.batch_key = batch_key

        if batch_levels is None and batch_key in adata.obs:
            batch_levels = adata.obs[batch_key].astype("category").cat.categories
        self.batch_levels = None if batch_levels is None else [str(c) for c in batch_levels]
        self.n_batches = 0 if self.batch_levels is None else len(self.batch_levels)

        self._initialized_from_adata = True

    def to_dict(self):
        res = dataclasses.asdict(self)
        res["layers_v_to_z"] = list(res["layers_v_to_z"])
        res["layers_z_to_x"] = list(res["layers_z_to_x"])
        return res


# Allowed values for every Literal-annotated field, read off the annotations themselves so this can
# never drift from them. Computed once at import; __post_init__ runs per instance.
def _literal_args(hint) -> Optional[tuple]:
    """The allowed values of a Literal[...] or Optional[Literal[...]] hint, else None."""
    if get_origin(hint) is Union:
        hint = next((a for a in get_args(hint) if get_origin(a) is Literal), None)
    return get_args(hint) if get_origin(hint) is Literal else None


# recon_batch values whose decoder has a hidden-layer batch network (r or ρ), the arms
# `residual_hidden` applies to.
_RECON_WITH_HIDDEN = ("residual", "concat_plus_residual_z", "residual_in_span")

_LITERAL_CHOICES = {
    name: _literal_args(hint)
    for name, hint in get_type_hints(DecipherConfig).items()
    if _literal_args(hint) is not None
}

# Gaussian-kernel bandwidths of the v-invariance MMD, as multiples of the median pairwise distance.
MMD_BANDWIDTH_SCALES = (0.1, 0.5, 1.0, 2.0, 5.0)
# Floor on the median pairwise distance, so v samples that all coincide give a finite kernel.
_MMD_MIN_MEDIAN = 1e-6


def multi_bandwidth_gaussian_kernel(v: torch.Tensor) -> torch.Tensor:
    """K_ij = sum_s exp(-|v_i - v_j|^2 / (2 (s m)^2)) for s in MMD_BANDWIDTH_SCALES, (n, n).

    m is the median Euclidean distance over pairs i < j (the lower median for an even count),
    computed without gradient, so the bandwidths follow the scale of v but are not optimized.
    Needs n >= 2.
    """
    sq_dist = (v[:, None, :] - v[None, :, :]).pow(2).sum(-1)
    with torch.no_grad():
        i, j = torch.triu_indices(v.shape[0], v.shape[0], offset=1)
        median = sq_dist[i, j].median().sqrt().clamp_min(_MMD_MIN_MEDIAN)
    return sum(torch.exp(-sq_dist / (2 * (s * median) ** 2)) for s in MMD_BANDWIDTH_SCALES)


def mmd_v_penalty(v: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
    """sum over batches b of MMD^2(v | batch b, all v), on one minibatch of v samples (n, d_v).

    U-statistic form with the kernel K of `multi_bandwidth_gaussian_kernel`: every mean runs over
    pairs i != j only (n_b cells in batch b, n in the minibatch),

        MMD^2_b = sum_{i != j in b} K_ij / (n_b (n_b - 1))
                + sum_{i != j} K_ij / (n (n - 1))
                - 2 sum_{i in b, j != i} K_ij / (n_b (n - 1)),

    so its expectation is exactly 0 when every batch has the same v distribution, it is exactly 0
    when the minibatch holds one batch, and it can be slightly negative. Batches with fewer than 2
    cells in the minibatch are skipped (the first mean needs 2); with none left the penalty is 0.
    """
    if batch_idx is None:
        raise ValueError("mmd_v_penalty needs per-cell batch codes; got batch_idx=None")
    labels, counts = torch.unique(batch_idx, return_counts=True)
    kept = labels[counts >= 2]
    if len(kept) == 0:
        return v.new_zeros(())
    n = v.shape[0]
    k_off = multi_bandwidth_gaussian_kernel(v).masked_fill(
        torch.eye(n, dtype=torch.bool, device=v.device), 0.0
    )
    within_all = k_off.sum() / (n * (n - 1))
    total = v.new_zeros(())
    for b in kept:
        in_b = batch_idx == b
        n_b = int(in_b.sum())
        within_b = k_off[in_b][:, in_b].sum() / (n_b * (n_b - 1))
        cross = k_off[in_b].sum() / (n_b * (n - 1))
        total = total + within_b + within_all - 2 * cross
    return total


def draw_other_batch(batch_idx: torch.Tensor, n_batches: int) -> torch.Tensor:
    """For each cell of batch a, a code b' != a drawn uniformly from the other n_batches - 1.

    b' = (a + s) mod n_batches with s uniform on {1, ..., n_batches - 1}, from the global torch
    RNG, so a seeded run draws the same codes. Needs n_batches >= 2.
    """
    if n_batches < 2:
        raise ValueError(f"draw_other_batch needs n_batches >= 2; got {n_batches}")
    shift = torch.randint(1, n_batches, batch_idx.shape, device=batch_idx.device)
    return (batch_idx + shift) % n_batches


@contextmanager
def batch_norm_stats_frozen(module: nn.Module):
    """Inside the block, every BatchNorm1d of `module` normalizes as usual (the minibatch's own
    statistics in train mode, the running ones in eval mode) but leaves its running statistics
    and `num_batches_tracked` untouched. Used for forward passes on inputs that are not data."""
    norms = [m for m in module.modules() if isinstance(m, nn.BatchNorm1d)]
    tracked = [m.track_running_stats for m in norms]
    for m in norms:
        m.track_running_stats = False
    try:
        yield
    finally:
        for m, flag in zip(norms, tracked):
            m.track_running_stats = flag


def round_robin_batches(n: int, n_batches: int) -> torch.Tensor:
    """Batch codes k mod n_batches for k = 0..n-1, (n,) long: every batch gets floor(n / B) or
    ceil(n / B) of the n slots, the first n mod B batches one more."""
    return torch.arange(n) % n_batches


def vamp_mixture(loc: torch.Tensor, scale: torch.Tensor) -> dist.MixtureSameFamily:
    """The equal-weight mixture (1/K) sum_k N(v; loc_k, diag scale_k^2) of K diagonal Gaussians,
    loc and scale (K, d): event shape (d,), batch shape (). Its log density is
    logsumexp_k log N(v; loc_k, scale_k) - log K (uniform mixing logits, log-softmaxed)."""
    return dist.MixtureSameFamily(
        dist.Categorical(logits=loc.new_zeros(loc.shape[0])),
        dist.Normal(loc, scale).to_event(1),
    )


RESIDUAL_HIDDEN = 64  # width of r's hidden layer under recon_batch="residual"
# Hidden width of t (prior_batch="residual") and of ρ (recon_batch="residual_in_span").
ANCHOR_HIDDEN = 32


def zero_output_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Linear -> ReLU -> Linear whose output layer (weight and bias) starts at zero.

    The net outputs exactly 0 at init, so a residual built from it leaves its base path unchanged
    until training moves the output layer. The first layer keeps PyTorch's default init, so the
    output layer gets a nonzero gradient from the first step. Used for r under
    recon_batch="residual" (L + D -> 64 -> G), ρ under recon_batch="residual_in_span"
    (L + D -> 32 -> L) and t under prior_batch="residual" (2 + D -> 32 -> L).
    """
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Decipher(nn.Module):
    """Decipher _decipher for single-cell data.

    Parameters
    ----------
    config : DecipherConfig or dict
        Configuration for the decipher _decipher.
    """

    def __init__(
        self,
        config: Union[DecipherConfig, dict] = DecipherConfig(),
    ):
        super().__init__()
        if isinstance(config, dict):
            config = DecipherConfig(**config)

        if not config._initialized_from_adata:
            raise ValueError(
                "DecipherConfig must be initialized from an AnnData object, "
                "use `DecipherConfig.initialize_from_adata(adata)` to do so."
            )

        if config.v_invariance != "none" and config.n_batches == 0:
            # Otherwise every training step raises inside decipher_train, which only prints it.
            raise ValueError(
                f"v_invariance={config.v_invariance!r} needs batches, but n_batches=0 "
                f"(batch_key={config.batch_key!r} is not a column of the training AnnData)"
            )
        if config.cycle_v and config.n_batches < 2:
            # No b' != b exists. Skipping the penalty would train model5 under the model8 name.
            raise ValueError(
                f"cycle_v=True needs at least 2 batches to swap between, but n_batches="
                f"{config.n_batches} (batch_key={config.batch_key!r})"
            )

        self.config = config
        self.dummy_param = nn.Parameter(torch.empty(0))

        # Which networks get [input; e_b]. No table (native, or n_batches == 0) -> native widths.
        has_table = config.batch_conditioning != "none" and config.n_batches > 0
        concat_z = config.batch_embedding_mode == "concat_z"
        self._emb_in_encoder = has_table and config.batch_conditioning == "decoder_encoder"
        self._emb_in_prior = has_table and config.prior_batch == "concat"
        self._emb_in_recon = has_table and not concat_z
        # residual / in_span: e_b reaches the reconstruction through r / ρ only, never through W.
        residual = self._emb_in_recon and config.recon_batch == "residual"
        in_span = self._emb_in_recon and config.recon_batch == "residual_in_span"
        # concat_plus_residual_z: e_b enters through h's e_b columns, r_z sees z only.
        residual_z = self._emb_in_recon and config.recon_batch == "concat_plus_residual_z"
        prior_residual = has_table and config.prior_batch == "residual"
        D = config.dim_batch_embedding

        # The construction order fixes the RNG draws, so the same seed gives the same init as the
        # package each family reproduces. concat_z (decipher_models): table first, default init.
        self.batch_emb = None
        if has_table and concat_z:
            self.batch_emb = nn.Embedding(config.n_batches, D)

        # 2. Encoder
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes + (D if self._emb_in_encoder else 0),
            [128],
            [self.config.dim_z] * 2,
        )
        self.encoder_zx_to_v = ConditionalDenseNN(
            (
                self.config.dim_genes
                if self.config.mean_field_v
                else self.config.dim_genes + self.config.dim_z
            ),
            [128],
            [self.config.dim_v, self.config.dim_v],
        )

        # 3. Decoder
        ## v -> z. Batch-blind under "concat_x": that mode deletes the b -> z edge, so the prior
        ## is identical to base Decipher's and z is only PERMITTED to carry batch, not required to.
        self.decoder_v_to_z = ConditionalDenseNN(
            input_dim=self.config.dim_v + (D if self._emb_in_prior else 0),
            hidden_dims=self.config.layers_v_to_z,
            output_dims=[self.config.dim_z] * 2,
        )
        ## z -> x (reconstruction). Batch-conditioned only under "concat_x".
        ## With layers_z_to_x = () the per-batch logit shift is W_e e_b, with
        ## W_e = decoder_z_to_x.layers[0].weight[:, dim_z:] (rank <= D in gene space).
        ## Under recon_batch="residual" / "residual_in_span" this is W z + c with no e_b columns.
        self.decoder_z_to_x = ConditionalDenseNN(
            input_dim=self.config.dim_z
            + (D if self._emb_in_recon and not (residual or in_span) else 0),
            hidden_dims=config.layers_z_to_x,
            output_dims=[self.config.dim_genes],
        )
        ## r([z; e_b]) under recon_batch="residual", zero at init; None otherwise.
        self.decoder_residual = None
        if residual:
            self.decoder_residual = zero_output_mlp(
                self.config.dim_z + D,
                config.residual_hidden or RESIDUAL_HIDDEN,
                self.config.dim_genes,
            )
        ## r_z(z) under recon_batch="concat_plus_residual_z": batch-blind, zero at init; None
        ## otherwise. Built where r is built under "residual", so the table still comes after it.
        self.decoder_residual_z = None
        if residual_z:
            self.decoder_residual_z = zero_output_mlp(
                self.config.dim_z, config.residual_hidden or RESIDUAL_HIDDEN, self.config.dim_genes
            )
        ## ρ([z; e_b]) under recon_batch="residual_in_span", zero at init; None otherwise.
        self.decoder_residual_in_span = None
        if in_span:
            self.decoder_residual_in_span = zero_output_mlp(
                self.config.dim_z + D,
                config.residual_hidden or ANCHOR_HIDDEN,
                self.config.dim_z,
            )
        ## t([v; e_b]) under prior_batch="residual", zero at init; None otherwise. Built after
        ## the decoders, so every other network keeps model5r's seed draws.
        self.prior_residual = None
        if prior_residual:
            self.prior_residual = zero_output_mlp(
                self.config.dim_v + D, ANCHOR_HIDDEN, self.config.dim_z
            )

        # concat_x (decipher_m5 model5e): table last, zero init, so the untrained model is
        # batch-blind; gradients reach e_b through the nonzero first-layer weights.
        # residual: table last with the default N(0, 1) init (batch-invariance plan, 1b). r's
        # output layer is zero, so e_b gets no gradient through r on the first step either way;
        # with N(0, 1) each batch already has its own r input once that layer moves. The same
        # rule holds for ρ and t (model7, model7i).
        # The zeroing itself happens at the end of __init__ (see batch_emb_init).
        if has_table and not concat_z:
            self.batch_emb = nn.Embedding(config.n_batches, D)

        # prior="vamp": K pseudo-inputs u_k = softplus(pseudo_inputs_raw_k) in count space, built
        # last. The init U ~ N(0, 1) (u_k = softplus(U_k) in (0, ~3) counts) is drawn from a private
        # generator seeded with config.seed, so the global RNG and every draw above are untouched.
        self.pseudo_inputs_raw = None
        if config.prior == "vamp":
            generator = torch.Generator().manual_seed(config.seed)
            self.pseudo_inputs_raw = nn.Parameter(
                torch.randn(config.n_pseudo_inputs, config.dim_genes, generator=generator)
            )

        # batch_emb_init: every table above drew PyTorch's N(0, 1) init at its fixed place in the
        # order; "zero" (or "auto" on a zero-init arm) zeroes it in place now, after every other
        # weight, so the field never moves an RNG draw. zeros_ consumes no randomness.
        if self.batch_emb is not None and self._batch_emb_init(concat_z, prior_residual) == "zero":
            nn.init.zeros_(self.batch_emb.weight)

        self._epsilon = 1e-5

        self.theta = None

    def _batch_emb_init(self, concat_z: bool, prior_residual: bool) -> str:
        """The table init `config.batch_emb_init` resolves to: "zero" or "normal".

        "auto" is the per-arm rule: zero for a concat_x arm whose reconstruction takes e_b only
        through h's columns ("concat", "concat_plus_residual_z") and has no residual prior
        (model4-6, decipher_m5 model5e); N(0, 1) for concat_z (decipher_models) and every arm
        with a zero-output batch network on e_b (r, ρ, t), which would otherwise see one input
        for every batch.
        """
        if self.config.batch_emb_init != "auto":
            return self.config.batch_emb_init
        concat_recon = self.config.recon_batch in ("concat", "concat_plus_residual_z")
        return "zero" if not concat_z and concat_recon and not prior_residual else "normal"

    @property
    def device(self):
        return self.dummy_param.device

    # ---- batch helpers: every forward path goes through these ----

    def _batch_vec(self, batch_idx: Optional[torch.Tensor], like: torch.Tensor):
        """Per-cell rows e_b of `batch_emb`, shape (n_cells, D), or None without a table.

        None -- not zeros -- when the model has no table (native, or n_batches == 0), so every
        network keeps its native width. `batch_idx=None` falls back to batch 0 for every cell.
        """
        if self.batch_emb is None:
            return None
        if batch_idx is None:
            batch_idx = like.new_zeros(like.shape[0]).long()
        return self.batch_emb(batch_idx)

    def _with_batch_emb(self, h: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """[h; e_b], or `h` unchanged when there is no table."""
        e = self._batch_vec(batch_idx, h)
        return h if e is None else torch.cat([h, e], dim=-1)

    def _encode_z(self, x: torch.Tensor, batch_idx: Optional[torch.Tensor]):
        """encoder_x_to_z on log1p'd `x`: (z_loc, z_scale before softplus).

        e_b is concatenated only under "decoder_encoder"; otherwise the guide is batch-blind.
        """
        h = self._with_batch_emb(x, batch_idx) if self._emb_in_encoder else x
        return self.encoder_x_to_z(h)

    def _prior_z(self, v: torch.Tensor, batch_idx: Optional[torch.Tensor]):
        """decoder_v_to_z: (z_loc, z_scale before softplus). e_b is concatenated onto its input
        only under prior_batch="concat"; under "residual" z_loc gains t([v; e_b]) and the scale
        stays f_var(v)'s."""
        h = self._with_batch_emb(v, batch_idx) if self._emb_in_prior else v
        z_loc, z_scale = self.decoder_v_to_z(h)
        if self.prior_residual is not None:
            z_loc = z_loc + self.prior_residual(self._with_batch_emb(v, batch_idx))
        return z_loc, z_scale

    def _decode_x(self, z: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """Reconstruction logits (before softmax). e_b is concatenated only under "concat_x":
        onto decoder_z_to_x's input ("concat"), onto r's input, added to W z + c ("residual"), or
        onto the input of ρ, added to z before W ("residual_in_span"), or onto
        decoder_z_to_x's input with the batch-blind r_z(z) added ("concat_plus_residual_z").
        """
        if self.decoder_residual is not None:
            return self.decoder_z_to_x(z) + self.decoder_residual(
                self._with_batch_emb(z, batch_idx)
            )
        if self.decoder_residual_z is not None:
            return self.decoder_z_to_x(
                self._with_batch_emb(z, batch_idx)
            ) + self.decoder_residual_z(z)
        if self.decoder_residual_in_span is not None:
            return self.decoder_z_to_x(
                z + self.decoder_residual_in_span(self._with_batch_emb(z, batch_idx))
            )
        h = self._with_batch_emb(z, batch_idx) if self._emb_in_recon else z
        return self.decoder_z_to_x(h)

    def _encode_v(self, z: torch.Tensor, x: torch.Tensor):
        """encoder_zx_to_v on [z; x], or on x alone under mean_field_v: (v_loc, v_scale before
        softplus). `x` is already log1p'd."""
        h = x if self.config.mean_field_v else torch.cat([z, x], dim=-1)
        return self.encoder_zx_to_v(h)

    # ---- VampPrior on v (prior="vamp") ----

    def pseudo_inputs(self) -> torch.Tensor:
        """u_k = softplus(pseudo_inputs_raw_k): the K pseudo-inputs in count space, (K, G), >= 0."""
        return softplus(self.pseudo_inputs_raw)

    def pseudo_input_batches(self) -> Optional[torch.Tensor]:
        """b_k = k mod n_batches, (K,) long (balanced round-robin); None without batches."""
        if self.config.n_batches == 0:
            return None
        batches = round_robin_batches(self.config.n_pseudo_inputs, self.config.n_batches)
        return batches.to(self.pseudo_inputs_raw.device)

    def vamp_components(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(loc, scale), each (K, dim_v), of the VampPrior components q(v | u_k, b_k).

        The guide's own path on each pseudo-input: z_k = z_loc of encode_z(log1p u_k, b_k),
        (v_loc_k, s_k) = encode_v(z_k, log1p u_k), scale_k = softplus(s_k) + eps,
        exactly as the guide's v scale. z_k is the z-encoder's loc, not a sample, so each
        component is a Gaussian and the prior density is deterministic. The passes run with
        BatchNorm running statistics frozen: pseudo-inputs are not data.
        """
        with batch_norm_stats_frozen(self):
            x = torch.log1p(self.pseudo_inputs())
            z_loc, _ = self._encode_z(x, self.pseudo_input_batches())
            v_loc, v_scale = self._encode_v(z_loc, x)
        return v_loc, softplus(v_scale) + self._epsilon

    # ---- training penalties: added to the training loss only, never to val NLL ----
    # To add a penalty: give it a config flag and weight, add `name: weight` here when its flag is
    # on, and compute `name: value` in `penalties`. `penalized_step` checks the two key sets match.

    def penalty_weights(self) -> dict[str, float]:
        """Weight of each active training penalty, by name; {} when no penalty flag is on."""
        weights = {}
        if self.config.v_invariance == "mmd":
            weights["mmd_v"] = float(self.config.lambda_inv)
        if self.prior_residual is not None:
            weights["prior_residual"] = float(self.config.lambda_r)
        if self.config.cycle_v:
            weights["cycle_v"] = float(self.config.lambda_c)
        return weights

    def penalties(
        self,
        guide_trace: poutine.Trace,
        x: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Unweighted scalar training penalties of one minibatch, by name (the keys of
        `penalty_weights`).

        `guide_trace` is the guide trace the ELBO of this step was computed on, so a penalty on a
        latent uses the same sample as the ELBO. `x` (raw counts) and `batch_idx` are the
        minibatch the model and guide were called with.
        """
        weights = self.penalty_weights()
        values = {}
        if "mmd_v" in weights:
            values["mmd_v"] = mmd_v_penalty(guide_trace.nodes["v"]["value"], batch_idx)
        if "prior_residual" in weights:
            v = guide_trace.nodes["v"]["value"]
            t = self.prior_residual(self._with_batch_emb(v, batch_idx))
            values["prior_residual"] = t.pow(2).sum(-1).mean()
        if "cycle_v" in weights:
            values["cycle_v"] = self.cycle_v_penalty(guide_trace, x, batch_idx)
        return values

    def cycle_v_penalty(
        self, guide_trace: poutine.Trace, x: torch.Tensor, batch_idx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Latent cycle-consistency of v under a swapped batch (sysVI), mean_i |v_i - v'_i|^2.

        For cell i of batch a, with z_i the guide trace's z sample and v_i the guide's v_loc
        computed from that z_i (the guide's return value):
            b'_i  = `draw_other_batch` (uniform over the other batches),
            x^_i  = l_i softmax(decode(z_i, b'_i)), l_i = sum_g x_ig: expected counts, no sample,
            z'_i  = z_loc of encode_z(log1p x^_i, b'_i),
            v'_i  = v_loc of encode_v(z'_i, log1p x^_i), built as the guide builds it.
        Gradients flow through both v_i and v'_i. The decode and re-encode run with BatchNorm
        running statistics frozen, so x^ never enters the statistics used at evaluation.
        """
        if batch_idx is None:
            raise ValueError("cycle_v_penalty needs per-cell batch codes; got batch_idx=None")
        z = guide_trace.nodes["z"]["value"]
        v_loc = guide_trace.nodes["_RETURN"]["value"][1]
        other = draw_other_batch(batch_idx, self.config.n_batches)
        with batch_norm_stats_frozen(self):
            x_hat = x.sum(-1, keepdim=True) * softmax(self._decode_x(z, other), dim=-1)
            x_hat = torch.log1p(x_hat)
            z_cycle, _ = self._encode_z(x_hat, other)
            v_cycle, _ = self._encode_v(z_cycle, x_hat)
        return (v_loc - v_cycle).pow(2).sum(-1).mean()

    def model(self, x, batch_idx=None):
        pyro.module("decipher", self)

        self.theta = pyro.param(
            "theta",
            x.new_ones(self.config.dim_genes),
            constraint=constraints.positive,
        )

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                elif self.config.prior == "vamp":
                    prior = vamp_mixture(*self.vamp_components())
                else:
                    raise ValueError("Invalid prior, must be normal, gamma or vamp")
                v = pyro.sample("v", prior)

            # v -> z prior. p(z|v,b) under "concat_z", p(z|v) otherwise.
            z_loc, z_scale = self._prior_z(v, batch_idx)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # z -> x reconstruction. p(x|z,b) under "concat_x", p(x|z) otherwise.
            mu = self._decode_x(z, batch_idx)

            mu = softmax(mu, dim=-1)
            library_size = x.sum(axis=-1, keepdim=True)
            # Parametrization of Negative Binomial by the mean and inverse dispersion
            # See https://github.com/pytorch/pytorch/issues/42449
            # noinspection PyTypeChecker
            logit = torch.log(library_size * mu + self._epsilon) - torch.log(
                self.theta + self._epsilon
            )
            # noinspection PyUnresolvedReferences
            x_dist = dist.NegativeBinomial(total_count=self.theta + self._epsilon, logits=logit)
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, batch_idx=None):
        pyro.module("decipher", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)

            # Batch-blind unless the guide is batch-conditioned ("decoder_encoder").
            z_loc, z_scale = self._encode_z(x, batch_idx)
            z_scale = softplus(z_scale) + self._epsilon
            posterior_z = dist.Normal(z_loc, z_scale).to_event(1)
            z = pyro.sample("z", posterior_z)

            v_loc, v_scale = self._encode_v(z, x)
            v_scale = softplus(v_scale) + self._epsilon
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "gamma":
                    posterior_v = dist.Gamma(softplus(v_loc), v_scale).to_event(1)
                elif self.config.prior in ("normal", "student-normal", "vamp"):
                    posterior_v = dist.Normal(v_loc, v_scale).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal, gamma or vamp")
                pyro.sample("v", posterior_v)
        return z_loc, v_loc, z_scale, v_scale

    def _encoder_common_frame(self, x):
        """encoder_x_to_z's output with the batch context held identical for every cell.

        Averaged over all n_batches contexts rather than pinned to a reference batch. Under
        concatenation a different reference is a different *function*: cells move by different
        amounts and rho moves with them. Averaging removes the choice instead of requiring it
        to be recorded and held fixed across variants, and keeps every forward pass on a
        context the encoder actually saw during training. A batch-blind guide has no context to
        hold, so its output is returned as is.

        `x` is already log1p'd.
        """
        if not self._emb_in_encoder:
            z, _ = self._encode_z(x, None)
            return z

        acc = None
        for b in range(self.config.n_batches):
            batch_idx = torch.full((x.shape[0],), b, dtype=torch.long)
            zb, _ = self._encode_z(x, batch_idx)
            acc = zb if acc is None else acc + zb
        return acc / self.config.n_batches

    @torch.no_grad()
    def compute_v_z_numpy(self, x: np.array, batch_idx=None):
        """Compute decipher_v, decipher_z and decipher_z_raw for the given counts.

        One definition across every variant:

            decipher_z_raw = the coordinate the guide actually produces for the cell, batch
                             effect included.
            decipher_z     = the same computation with the batch context held identical for
                             every cell, so all cells sit in one shared frame.

        "Held identical" means averaging the encoder over all n_batches contexts. When the guide
        is batch-blind -- `batch_conditioning` of "none" or "decoder_only" -- there is no context
        to hold, and the two coordinates are equal. That is correct rather than a regression:
        nothing was added to z, so there is nothing to subtract.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).
        batch_idx : torch.Tensor, optional
            Integer batch codes of shape (n_cells,). Defaults to batch 0 for all cells; pass
            the real codes via `get_batch_idx(adata, config)`.

        Returns
        -------
        v : (n_cells, dim_v)
            The decipher v space. Computed from `z_raw`, since that is the coordinate the
            guide produced.
        z : (n_cells, dim_z)
            Batch-corrected. Use for Leiden, trajectories, integration metrics and plots.
        z_raw : (n_cells, dim_z)
            Batch effect included. Use for anything feeding decoder_z_to_x -- the counts it
            reproduces still contain the batch effect.
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)

        x = torch.log1p(x)

        z_raw, _ = self._encode_z(x, batch_idx)
        z = self._encoder_common_frame(x)

        v_loc, _ = self._encode_v(z_raw, x)
        return v_loc.detach().numpy(), z.detach().numpy(), z_raw.detach().numpy()

    @torch.no_grad()
    def impute_gene_expression_numpy(self, x, batch_idx=None):
        """Reconstruct counts from the guide's z.

        `batch_idx` must be the real per-cell codes. Without them every cell is reconstructed
        as if it belonged to batch 0, which is scored against counts that carry each cell's own
        batch effect -- wrong for every batch but the first. Callers pass them via
        `get_batch_idx()`.
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        # guide() returns the raw (batch-containing) z, which is what decoder_z_to_x expects:
        # the observed counts still contain the batch effect.
        z_loc, _, _, _ = self.guide(x, batch_idx)
        mu = self._decode_x(z_loc, batch_idx)
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
