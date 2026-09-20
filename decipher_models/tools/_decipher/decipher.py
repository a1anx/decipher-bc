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

from decipher_models.tools._decipher.module import ConditionalDenseNN


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
    #   "decoder_only"    -- batch reaches the generative model only. That is decoder_v_to_z
    #                        under "concat_z"/"additive", and decoder_z_to_x under "concat_x".
    #   "decoder_encoder" -- batch also reaches the guide, via encoder_x_to_z.
    #
    # `batch_embedding_mode` says WHERE and HOW it enters:
    #   "concat_z" -- Set 1. A learned nn.Embedding concatenated into decoder_v_to_z's input
    #                 (and encoder_x_to_z's), so the batch shifts the biological prior on z.
    #   "additive" -- Set 2. A zero-init nn.Embedding(n_batches, dim_z) added onto z_loc after
    #                 a batch-blind network runs. Same graphical model as Set 1.
    #   "concat_x" -- Set 3. The b -> z edge is deleted outright; the prior is identical to base
    #                 Decipher. Batch reaches only the reconstruction, as a fixed one-hot
    #                 `context` of width n_batches. `dim_batch_embedding` is unread here.
    batch_conditioning: Literal["none", "decoder_only", "decoder_encoder"] = "none"
    mean_field_v: bool = False
    batch_embedding_mode: Literal["concat_z", "additive", "concat_x"] = "concat_z"
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

        bc = config.batch_conditioning
        mode = config.batch_embedding_mode
        batch_on = bc != "none"
        encoder_on = bc == "decoder_encoder"

        # ---- batch conditioning ----
        # Attribute names deliberately match the standalone variant packages, so a state_dict
        # from any of them loads straight into this model (tests/test_variant_equivalence.py).
        if batch_on and mode == "concat_z":
            self.batch_emb = torch.nn.Embedding(
                num_embeddings=config.n_batches,
                embedding_dim=config.dim_batch_embedding,
            )
        elif batch_on and mode == "additive":
            if encoder_on:
                # Two tables, not one. A single shared table applied additively to z_loc on both
                # the generative and the variational b -> z edge cancels exactly out of the z
                # log-density ratio in the ELBO, so the gradient that would identify it is
                # algebraically deleted. Matches model2-add / model3-add.
                self.batch_shift_prior = self._make_batch_shift(config)  # generative, model()
                self.batch_shift_post = self._make_batch_shift(config)  # variational, guide()
            else:
                # Only one b -> z edge exists (the guide is blind), so there is no second edge
                # to give a second table to, and nothing to cancel. Matches model1-add.
                self.batch_shift = self._make_batch_shift(config)
        # "concat_x" learns no batch table at all: the context is a fixed one-hot. At n_batches=5
        # that gives the first layer 5 dedicated input slots, no less expressive than an
        # nn.Embedding(5, d) feeding the same layer, and cheaper.
        # -----------------------------

        encoder_extra = (
            self.config.dim_batch_embedding if (encoder_on and mode == "concat_z") else 0
        )
        decoder_extra = self.config.dim_batch_embedding if (batch_on and mode == "concat_z") else 0
        encoder_context = self.config.n_batches if (encoder_on and mode == "concat_x") else 0
        recon_context = self.config.n_batches if (batch_on and mode == "concat_x") else 0

        # 2. Encoder
        # ConditionalDenseNN joins the context onto the input at the first layer, so under
        # "concat_x" the batch reaches BOTH output heads -- z_loc and z_scale come out of one
        # final Linear and are split afterwards. The additive table reaches z_loc only.
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes + encoder_extra,
            [128],
            [self.config.dim_z] * 2,
            context_dim=encoder_context,
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
            input_dim=self.config.dim_v + decoder_extra,
            hidden_dims=self.config.layers_v_to_z,
            output_dims=[self.config.dim_z] * 2,
        )
        ## z -> x (reconstruction). Batch-conditioned only under "concat_x".
        ## With layers_z_to_x = () this is a single Linear(dim_z + n_batches, dim_genes), and
        ## since the context is concatenated first, weight[:, :n_batches] is n_batches free
        ## gene-space vectors -- one per batch, directly comparable to uns["gene_shift_matrix"].
        self.decoder_z_to_x = ConditionalDenseNN(
            input_dim=self.config.dim_z,
            hidden_dims=config.layers_z_to_x,
            output_dims=[self.config.dim_genes],
            context_dim=recon_context,
        )

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    # ---- "additive" mode helpers (Set 2) ----

    @staticmethod
    def _make_batch_shift(config):
        """Build one zero-initialized (n_batches, dim_z) shift table.

        Returns None when the model has no batches, so every lookup site degrades to the
        batch-blind model instead of raising IndexError.
        """
        if config.n_batches == 0:
            return None
        table = torch.nn.Embedding(
            num_embeddings=config.n_batches,
            embedding_dim=config.dim_z,
        )
        torch.nn.init.zeros_(table.weight)  # intercepts start at zero
        return table

    def _shift(self, table, batch_idx, like):
        """Per-cell shift rows, shape (n_cells, dim_z). Zeros when there is no table."""
        if table is None:
            return like.new_zeros(like.shape[0], self.config.dim_z)
        return table(batch_idx)

    def _to_common_frame(self, z_raw, table, batch_idx, like):
        """Move every cell into one shared frame.

        Subtracts the cell's own batch row and adds the mean row, so all batches sit in the
        same frame and `z_raw - z` is the centred per-batch shift. Centring on the mean rather
        than batch 0's row avoids privileging whichever batch sorts first.
        """
        if table is None:
            return z_raw
        return z_raw - self._shift(table, batch_idx, like) + table.weight.mean(dim=0, keepdim=True)

    @property
    def _prior_shift(self):
        """The additive table on the generative b -> z edge, under either naming."""
        if hasattr(self, "batch_shift_prior"):
            return self.batch_shift_prior
        return getattr(self, "batch_shift", None)

    # ---- "concat_x" mode helpers (Set 3) ----

    def _batch_context(self, batch_idx, like):
        """One-hot batch context, shape (n_cells, n_batches), or None when there are none.

        Returned as None -- not zeros -- when n_batches == 0, because a ConditionalDenseNN
        built with context_dim=0 never reads its `context` argument. That makes the mode
        degrade to the batch-blind model instead of raising.
        """
        if self.config.n_batches == 0:
            return None
        if batch_idx is None:
            batch_idx = like.new_zeros(like.shape[0]).long()
        return torch.nn.functional.one_hot(batch_idx, num_classes=self.config.n_batches).to(
            like.dtype
        )

    def _encoder_context(self, batch_idx, like):
        """The context for encoder_x_to_z: None unless the guide is conditioned in concat_x."""
        if self.config.batch_embedding_mode != "concat_x":
            return None
        if self.config.batch_conditioning != "decoder_encoder":
            return None
        return self._batch_context(batch_idx, like)

    def _recon_context(self, batch_idx, like):
        """The context for decoder_z_to_x: None unless batch is wired into concat_x."""
        if self.config.batch_embedding_mode != "concat_x":
            return None
        if self.config.batch_conditioning == "none":
            return None
        return self._batch_context(batch_idx, like)

    def model(self, x, batch_idx=None):
        pyro.module("decipher", self)

        self.theta = pyro.param(
            "theta",
            x.new_ones(self.config.dim_genes),
            constraint=constraints.positive,
        )

        bc = self.config.batch_conditioning
        mode = self.config.batch_embedding_mode
        if bc != "none":
            if batch_idx is None:
                batch_idx = x.new_zeros(x.shape[0]).long()
            if mode == "concat_z":
                batch_vec = self.batch_emb(batch_idx)

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                v = pyro.sample("v", prior)

            # v -> z prior. Under "concat_x" this edge is batch-blind by design.
            if bc != "none" and mode == "concat_z":
                v_combined = torch.cat([v, batch_vec], dim=-1)
                z_loc, z_scale = self.decoder_v_to_z(v_combined)
            else:
                z_loc, z_scale = self.decoder_v_to_z(v)
                if bc != "none" and mode == "additive":
                    z_loc = z_loc + self._shift(self._prior_shift, batch_idx, x)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # z -> x reconstruction. p(x|z,b) under "concat_x", p(x|z) otherwise.
            mu = self.decoder_z_to_x(z, context=self._recon_context(batch_idx, x))

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

        bc = self.config.batch_conditioning
        mode = self.config.batch_embedding_mode
        if bc != "none" and batch_idx is None:
            batch_idx = x.new_zeros(x.shape[0]).long()
        if bc == "decoder_encoder" and mode == "concat_z":
            batch_vec = self.batch_emb(batch_idx)

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)

            if bc == "decoder_encoder" and mode == "concat_z":
                z_loc, z_scale = self.encoder_x_to_z(torch.cat([x, batch_vec], dim=-1))
            else:
                # _encoder_context is None unless this is concat_x with a conditioned guide.
                z_loc, z_scale = self.encoder_x_to_z(x, context=self._encoder_context(batch_idx, x))
                if bc == "decoder_encoder" and mode == "additive":
                    z_loc = z_loc + self._shift(self.batch_shift_post, batch_idx, x)
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
        an additive table the reference is free -- it adds one constant vector to every cell,
        so every distance and every rank correlation is unchanged by the choice. Under
        concatenation a different reference is a different *function*: cells move by different
        amounts and rho moves with them. Averaging removes the choice instead of requiring it
        to be recorded and held fixed across variants, and keeps every forward pass on a
        context the encoder actually saw during training.

        `x` is already log1p'd.
        """
        n_batches = self.config.n_batches
        if n_batches == 0:
            z, _ = self.encoder_x_to_z(x)
            return z

        acc = None
        for b in range(n_batches):
            batch_idx = torch.full((x.shape[0],), b, dtype=torch.long)
            if self.config.batch_embedding_mode == "concat_z":
                zb, _ = self.encoder_x_to_z(torch.cat([x, self.batch_emb(batch_idx)], dim=-1))
            else:  # concat_x
                zb, _ = self.encoder_x_to_z(x, context=self._batch_context(batch_idx, x))
            acc = zb if acc is None else acc + zb
        return acc / n_batches

    def compute_v_z_numpy(self, x: np.array, batch_idx=None):
        """Compute decipher_v, decipher_z and decipher_z_raw for the given counts.

        One definition across every variant:

            decipher_z_raw = the coordinate the guide actually produces for the cell, batch
                             effect included.
            decipher_z     = the same computation with the batch context held identical for
                             every cell, so all cells sit in one shared frame.

        How "held identical" is realised depends on the mode: averaging the encoder over all
        n_batches contexts under "concat_z"/"concat_x", and `_to_common_frame` (subtract the
        cell's own row, add the mean row) under "additive". When the guide is batch-blind --
        `batch_conditioning` of "none" or "decoder_only" -- there is no context to hold, and
        the two coordinates are equal. That is correct rather than a regression: nothing was
        added to z, so there is nothing to subtract.

        Note this differs from standalone model1-add, which corrected its blind guide's output
        using the *generative* table. Holding the context fixed is the rule here, and a blind
        guide has no context.

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

        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long)

        x = torch.log1p(x)

        bc = self.config.batch_conditioning
        mode = self.config.batch_embedding_mode

        if bc != "decoder_encoder":
            # the guide is batch-blind, so the two coordinates coincide
            z_raw, _ = self.encoder_x_to_z(x)
            z = z_raw
        elif mode == "concat_z":
            z_raw, _ = self.encoder_x_to_z(torch.cat([x, self.batch_emb(batch_idx)], dim=-1))
            z = self._encoder_common_frame(x)
        elif mode == "additive":
            z_loc, _ = self.encoder_x_to_z(x)
            z_raw = z_loc + self._shift(self.batch_shift_post, batch_idx, x)
            z = self._to_common_frame(z_raw, self.batch_shift_post, batch_idx, x)
        else:  # concat_x
            z_raw, _ = self.encoder_x_to_z(x, context=self._batch_context(batch_idx, x))
            z = self._encoder_common_frame(x)

        if self.config.mean_field_v:
            v_loc, _ = self.encoder_zx_to_v(x)
        else:
            zx = torch.cat([z_raw, x], dim=-1)
            v_loc, _ = self.encoder_zx_to_v(zx)
        return v_loc.detach().numpy(), z.detach().numpy(), z_raw.detach().numpy()

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
        mu = self.decoder_z_to_x(z_loc, context=self._recon_context(batch_idx, z_loc))
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
