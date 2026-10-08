"""DecipherConfig and Decipher for the native and model5 presets.

Copied from `decipher_models2` on 2026-10-08 and trimmed to the one-hot `concat_x` batch path:
the `concat_z` and `additive` branches, their tables and helpers are gone.
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

from decipher_m5.tools._decipher.module import ConditionalDenseNN


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
    # `batch_conditioning`:
    #   "none"            -- native. No batch anywhere.
    #   "decoder_encoder" -- model5. The b -> z edge is deleted; the prior is identical to base
    #                        Decipher. Batch enters decoder_z_to_x and encoder_x_to_z as a fixed
    #                        one-hot `context` of width n_batches, at the first layer only.
    batch_conditioning: Literal["none", "decoder_encoder"] = "none"
    mean_field_v: bool = False
    n_batches: int = 0
    batch_key: Optional[str] = None
    # -----------------

    prior: str = "normal"

    _initialized_from_adata: bool = False

    def __post_init__(self):
        """Enforce the Literal annotations above, which Python does not enforce on its own.

        ADDED 2026-09-18. `@dataclass` ignores `Literal` at runtime, so before this an unknown
        flag value was accepted silently. The model then fell through to whichever branch the
        value missed and trained a DIFFERENT ARCHITECTURE than the caller asked for, with nothing
        in the losses, the h5ad or the sweep log to reveal it. Here `batch_conditioning` accepts
        only "none" | "decoder_encoder"; the dropped "decoder_only" raises.

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

        # No batch table is learned: the context is a fixed one-hot of width n_batches, the same
        # on the encoder and the decoder side (0 for native, which never reads it).
        context_dim = config.n_batches if config.batch_conditioning == "decoder_encoder" else 0

        # 2. Encoder
        # ConditionalDenseNN joins the context onto the input at the first layer, so the batch
        # reaches BOTH output heads -- z_loc and z_scale come out of one final Linear and are
        # split afterwards.
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes,
            [128],
            [self.config.dim_z] * 2,
            context_dim=context_dim,
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
        ## v -> z. Batch-blind: the b -> z edge is deleted, so the prior is identical to base
        ## Decipher's and z is only PERMITTED to carry batch, not required to.
        self.decoder_v_to_z = ConditionalDenseNN(
            input_dim=self.config.dim_v,
            hidden_dims=self.config.layers_v_to_z,
            output_dims=[self.config.dim_z] * 2,
        )
        ## z -> x (reconstruction). Batch-conditioned under model5.
        ## With layers_z_to_x = () this is a single Linear(dim_z + n_batches, dim_genes), and
        ## since the context is concatenated first, weight[:, :n_batches] is n_batches free
        ## gene-space vectors -- one per batch, directly comparable to uns["gene_shift_matrix"].
        self.decoder_z_to_x = ConditionalDenseNN(
            input_dim=self.config.dim_z,
            hidden_dims=config.layers_z_to_x,
            output_dims=[self.config.dim_genes],
            context_dim=context_dim,
        )

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    # ---- batch context helpers ----

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
        """The context for encoder_x_to_z: None unless the guide is batch-conditioned."""
        if self.config.batch_conditioning == "none":
            return None
        return self._batch_context(batch_idx, like)

    def _recon_context(self, batch_idx, like):
        """The context for decoder_z_to_x: None unless the decoder is batch-conditioned."""
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

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                v = pyro.sample("v", prior)

            # v -> z prior. Batch-blind by design.
            z_loc, z_scale = self.decoder_v_to_z(v)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # z -> x reconstruction. p(x|z,b) under model5, p(x|z) under native.
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

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)

            # _encoder_context is None unless the guide is batch-conditioned.
            z_loc, z_scale = self.encoder_x_to_z(x, context=self._encoder_context(batch_idx, x))
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
            zb, _ = self.encoder_x_to_z(x, context=self._batch_context(batch_idx, x))
            acc = zb if acc is None else acc + zb
        return acc / n_batches

    @torch.no_grad()
    def compute_v_z_numpy(self, x: np.array, batch_idx=None):
        """Compute decipher_v, decipher_z and decipher_z_raw for the given counts.

        One definition for both presets:

            decipher_z_raw = the coordinate the guide actually produces for the cell, batch
                             effect included.
            decipher_z     = the same computation with the batch context held identical for
                             every cell, so all cells sit in one shared frame.

        "Held identical" means averaging the encoder over all n_batches contexts. When the
        guide is batch-blind (native) there is no context to hold, and the two coordinates are
        equal: nothing was added to z, so there is nothing to subtract.

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

        if self.config.batch_conditioning == "none":
            # the guide is batch-blind, so the two coordinates coincide
            z_raw, _ = self.encoder_x_to_z(x)
            z = z_raw
        else:
            z_raw, _ = self.encoder_x_to_z(x, context=self._batch_context(batch_idx, x))
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
        mu = self.decoder_z_to_x(z_loc, context=self._recon_context(batch_idx, z_loc))
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
