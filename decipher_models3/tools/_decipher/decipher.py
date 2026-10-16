"""DecipherConfig and Decipher for the seven arms of `decipher_models3`.

Rewritten 2026-10-16 from `decipher_models/tools/_decipher/decipher.py`. Every batch arm has ONE
learned table `batch_emb = nn.Embedding(n_batches, dim_batch_embedding)`, shared by every network
that sees the batch, and its row e_b is concatenated onto that network's input (x~ = log1p x):

    arm      prior            reconstruction   guide on z          v encoder
    native   f_vz(v)          f_zx(z)          g_xz(x~)            full
    model1   f_vz([v; e_b])   f_zx(z)          g_xz(x~)            full
    model2   f_vz([v; e_b])   f_zx(z)          g_xz([x~; e_b])     full
    model3   f_vz([v; e_b])   f_zx(z)          g_xz([x~; e_b])     mean-field (x~ only)
    model4   f_vz(v)          f_zx([z; e_b])   g_xz(x~)            full
    model5   f_vz(v)          f_zx([z; e_b])   g_xz([x~; e_b])     full
    model6   f_vz(v)          f_zx([z; e_b])   g_xz([x~; e_b])     mean-field (x~ only)

concat_z (model1-3) is `decipher_models`' math and init: `batch_emb` is built BEFORE the networks
with PyTorch's default N(0, 1) init, so the same seed gives `decipher_models`' state_dict.
concat_x (model4-6) replaces `decipher_models`' fixed one-hot context with the shared embedding of
`decipher_m5` model5e: `batch_emb` is built AFTER the networks and zero-initialized, so the
untrained model is batch-blind and the same seed gives model5e's state_dict for model5.
The `additive` family (Set 2) does not exist here.
"""

import dataclasses
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
    # -----------------

    prior: str = "normal"

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
        for name, allowed in _LITERAL_CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"DecipherConfig.{name}={value!r} is not one of {list(allowed)}. "
                    f"If this used to work, the flag was renamed -- check the field's comment "
                    f"above rather than assuming the old value still means what it did."
                )

    def initialize_from_adata(self, adata, batch_key="batch"):
        self.dim_genes = adata.shape[1]
        self.n_cells = adata.shape[0]

        # Save the key so the model instance knows what it was trained on
        self.batch_key = batch_key

        # --- AUTOMATIC BATCH COUNTING ---
        if batch_key in adata.obs:
            self.n_batches = adata.obs[batch_key].astype("category").cat.categories.size
        else:
            self.n_batches = 0
        # --------------------------------

        self._initialized_from_adata = True

    def to_dict(self):
        res = dataclasses.asdict(self)
        res["layers_v_to_z"] = list(res["layers_v_to_z"])
        res["layers_z_to_x"] = list(res["layers_z_to_x"])
        return res


# Allowed values for every Literal-annotated field, read off the annotations themselves so this can
# never drift from them. Computed once at import; __post_init__ runs per instance.
_LITERAL_CHOICES = {
    name: get_args(hint)
    for name, hint in get_type_hints(DecipherConfig).items()
    if get_origin(hint) is Literal
}


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

        self.config = config
        self.dummy_param = nn.Parameter(torch.empty(0))

        # Which networks get [input; e_b]. No table (native, or n_batches == 0) -> native widths.
        has_table = config.batch_conditioning != "none" and config.n_batches > 0
        concat_z = config.batch_embedding_mode == "concat_z"
        self._emb_in_encoder = has_table and config.batch_conditioning == "decoder_encoder"
        self._emb_in_prior = has_table and concat_z
        self._emb_in_recon = has_table and not concat_z
        d_e = config.dim_batch_embedding

        # The construction order fixes the RNG draws, so the same seed gives the same init as the
        # package each family reproduces. concat_z (decipher_models): table first, default init.
        self.batch_emb = None
        if has_table and concat_z:
            self.batch_emb = nn.Embedding(config.n_batches, d_e)

        # 2. Encoder
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes + (d_e if self._emb_in_encoder else 0),
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
            input_dim=self.config.dim_v + (d_e if self._emb_in_prior else 0),
            hidden_dims=self.config.layers_v_to_z,
            output_dims=[self.config.dim_z] * 2,
        )
        ## z -> x (reconstruction). Batch-conditioned only under "concat_x".
        ## With layers_z_to_x = () the per-batch logit shift is W_e e_b, with
        ## W_e = decoder_z_to_x.layers[0].weight[:, dim_z:] (rank <= d_e in gene space).
        self.decoder_z_to_x = ConditionalDenseNN(
            input_dim=self.config.dim_z + (d_e if self._emb_in_recon else 0),
            hidden_dims=config.layers_z_to_x,
            output_dims=[self.config.dim_genes],
        )

        # concat_x (decipher_m5 model5e): table last, zero init, so the untrained model is
        # batch-blind; gradients reach e_b through the nonzero first-layer weights.
        if has_table and not concat_z:
            self.batch_emb = nn.Embedding(config.n_batches, d_e)
            nn.init.zeros_(self.batch_emb.weight)

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    # ---- batch helpers: every forward path goes through these ----

    def _batch_vec(self, batch_idx: Optional[torch.Tensor], like: torch.Tensor):
        """Per-cell rows e_b of `batch_emb`, shape (n_cells, d_e), or None without a table.

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
        """decoder_v_to_z: (z_loc, z_scale before softplus). e_b is concatenated only under
        "concat_z"."""
        h = self._with_batch_emb(v, batch_idx) if self._emb_in_prior else v
        return self.decoder_v_to_z(h)

    def _decode_x(self, z: torch.Tensor, batch_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """decoder_z_to_x logits (before softmax). e_b is concatenated only under "concat_x"."""
        h = self._with_batch_emb(z, batch_idx) if self._emb_in_recon else z
        return self.decoder_z_to_x(h)

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
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
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

            if self.config.mean_field_v:
                v_loc, v_scale = self.encoder_zx_to_v(x)
            else:
                zx = torch.cat([z, x], dim=-1)
                v_loc, v_scale = self.encoder_zx_to_v(zx)
            v_scale = softplus(v_scale) + self._epsilon
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "gamma":
                    posterior_v = dist.Gamma(softplus(v_loc), v_scale).to_event(1)
                elif self.config.prior == "normal" or self.config.prior == "student-normal":
                    posterior_v = dist.Normal(v_loc, v_scale).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
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

        if self.config.mean_field_v:
            v_loc, _ = self.encoder_zx_to_v(x)
        else:
            zx = torch.cat([z_raw, x], dim=-1)
            v_loc, _ = self.encoder_zx_to_v(zx)
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
