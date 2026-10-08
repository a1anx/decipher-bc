import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
import torch.utils.data
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus

from decipher_vz_add.tools._decipher.module import ConditionalDenseNN


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
    n_batches: int = 0
    batch_key: Optional[str] = None
    # -----------------
    
    prior: str = "normal"

    _initialized_from_adata: bool = False
    
    # updated for batch
    def initialize_from_adata(self, adata, batch_key = "batch"):
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
        
        # ---- per-batch additive shift in z space ----
        # CHANGED 2026-08-24 -- fix 3 + fix 7 (batch_shift_review_handoff.md).
        #
        # WAS: self.batch_shift = torch.nn.Embedding(
        #          num_embeddings=config.n_batches,
        #          embedding_dim=config.dim_z
        #      )
        #
        # Deliberately still ONE table, unlike models 2 and 3. This arm's guide is
        # q(z|x) -- batch-blind -- so the plate diagram has exactly one b -> z edge, in
        # model() only. Fix 2 (splitting into _prior/_post) does NOT apply here; there
        # is no second edge to give a second table to. This is also why batch_shift is
        # identified in this arm at all: it appears in exactly one place in the loss, so
        # nothing cancels it (§2.1).
        #
        #  fix 3 -- nn.Embedding defaults to Normal(0,1), measured per-dim std ~1.02,
        #    against a true per-batch shift std of 0.058-0.173 and a decoder_v_to_z
        #    output layer whose weights start at std ~0.073. Zero is the standard start
        #    for an intercept, and at zero the z prior is exactly base Decipher's.
        #  fix 7 -- nn.Embedding(0, dim_z) with n_batches=0 raised IndexError against
        #    the zeros fallback instead of degrading to the batch-blind model.
        self.batch_shift = self._make_batch_shift(config)  # generative edge, model()
        # -------------------------
        
        # 2. Encoder
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes, [128], [self.config.dim_z] * 2
        )
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.config.dim_genes + self.config.dim_z,
            [128],
            [self.config.dim_v, self.config.dim_v],
        )
        
        # 3. Decoder
        ## v -> z (batch-blind; batch enters only as an additive shift on z_loc)
        self.decoder_v_to_z = ConditionalDenseNN(
            input_dim=self.config.dim_v,
            hidden_dims=self.config.layers_v_to_z,
            output_dims=[self.config.dim_z] * 2,
        )
        ## z -> x (reconstruction)
        # Input is now the biological latent z, without the batch embedding
        self.decoder_z_to_x = ConditionalDenseNN(
            input_dim=self.config.dim_z,
            hidden_dims=config.layers_z_to_x, 
            output_dims=[self.config.dim_genes]
        )
        

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    # ---- batch-shift helpers (added 2026-08-24, fixes 3/5/7) ----

    @staticmethod
    def _make_batch_shift(config):
        """Build one zero-initialized (n_batches, dim_z) shift table.

        Returns None when the model has no batches, so every lookup site degrades to
        the batch-blind model instead of raising IndexError (fix 7).
        """
        if config.n_batches == 0:
            return None
        table = torch.nn.Embedding(
            num_embeddings=config.n_batches,
            embedding_dim=config.dim_z,
        )
        torch.nn.init.zeros_(table.weight)  # fix 3: intercepts start at zero
        return table

    def _shift(self, table, batch_idx, like):
        """Per-cell shift rows, shape (n_cells, dim_z). Zeros when there is no table."""
        if table is None:
            return like.new_zeros(like.shape[0], self.config.dim_z)
        return table(batch_idx)

    def _to_common_frame(self, z_raw, table, batch_idx, like):
        """Move every cell into one shared frame (fix 5).

        Subtracts the cell's own batch row and adds the mean row, so all batches sit in
        the same frame and `z_raw - z` is the centred per-batch shift. Centring on the
        mean rather than batch 0's row avoids privileging whichever batch sorts first
        and makes finding 8 (no zero-sum constraint) explicit.
        """
        if table is None:
            return z_raw
        return z_raw - self._shift(table, batch_idx, like) + table.weight.mean(dim=0, keepdim=True)

    def model(self, x, batch_idx=None):
        pyro.module("decipher", self)

        self.theta = pyro.param(
            "theta",
            x.new_ones(self.config.dim_genes),
            constraint=constraints.positive,
        )
        
        # Just in case batch_idx is not provided during a naked model() call
        if batch_idx is None:
            batch_idx = x.new_zeros(x.shape[0]).long()

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                v = pyro.sample("v", prior)
            
            # v -> z prior: batch-blind decoder, batch enters as an additive shift on z_loc
            z_loc, z_scale = self.decoder_v_to_z(v)
            # The single b -> z edge. Routed through _shift() so n_batches=0 degrades
            # to the batch-blind model instead of raising IndexError (fix 7).
            # WAS: z_loc = z_loc + self.batch_shift(batch_idx)
            z_loc = z_loc + self._shift(self.batch_shift, batch_idx, x)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
        
            # z -> x reconstruction, not conditioned on batch
            mu = self.decoder_z_to_x(z)
            
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
        """Variational guide q(z|x) q(v|z,x).

        `batch_idx` IS ACCEPTED AND DELIBERATELY IGNORED -- do not "fix" this (finding 6,
        2026-08-24). This arm's guide is q(z|x): batch-blind by design. That is the whole
        thing that distinguishes Decipher-VZ-Add from Decipher-VZ2-Add, whose guide is
        q(z|x,b). Adding `z_loc = z_loc + self.batch_shift(batch_idx)` below would turn
        this model into model 2.

        The parameter exists only so that svi.step(x, batch_idx) can pass the same
        arguments to model() and guide(); Pyro requires matching signatures.

        Because the guide never touches batch_shift, this arm's imputation and R2 were
        already correct -- fix 1 applies to models 2 and 3 only.
        """
        pyro.module("decipher", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)

            z_loc, z_scale = self.encoder_x_to_z(x)
            z_scale = softplus(z_scale) + self._epsilon
            posterior_z = dist.Normal(z_loc, z_scale).to_event(1)
            z = pyro.sample("z", posterior_z)

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

    def compute_v_z_numpy(self, x: np.array, batch_idx=None):
        """Compute decipher_v, decipher_z and decipher_z_raw for the given counts.

        CHANGED 2026-08-24 -- fix 5. Previously returned only (v, z), where z was
        `encoder_x_to_z(x)` with no shift term at all, because this arm's guide is
        batch-blind. Models 2 and 3 meanwhile exported the batch-CORRECTED coordinate
        under the same column name. Per §2.2 models 1 and 2 are the same model written
        differently, so `decipher_z` was reporting opposite quantities across arms and
        was not comparable between them (finding 5). Since decipher_z drives Leiden ->
        clusters -> trajectories -> decipher_time -> rho (§2b), that inconsistency
        reached the headline metric.

        NOTE this changes THIS arm's decipher_z, and therefore its Leiden clusters and
        its rho. Models 2 and 3 already exported the corrected coordinate, so their
        decipher_z is unchanged and only gains decipher_z_raw alongside it. That is the
        intended effect of putting the three arms on one definition.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).
        batch_idx : torch.LongTensor, optional
            Per-cell batch codes from get_batch_idx(). Without it every cell is treated
            as batch 0, which makes decipher_z and decipher_z_raw identical.

        Returns
        -------
        v : (n_cells, dim_v)
            The decipher v space.
        z : (n_cells, dim_z)
            Batch-corrected z: every cell moved into one shared frame. Use for Leiden,
            trajectories, integration metrics and biology plots.
        z_raw : (n_cells, dim_z)
            The coordinate the guide actually produces. In this arm the guide is
            batch-blind, so encoder_x_to_z has itself learned to reproduce
            decoder_v_to_z(v) + batch_shift[b] -- the batch effect is in there
            implicitly. Use for anything feeding decoder_z_to_x.
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)

        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long)

        x = torch.log1p(x)
        z_loc, _ = self.encoder_x_to_z(x)
        # The guide adds no shift here (q(z|x) is batch-blind), so the encoder's own
        # output IS the raw, batch-containing coordinate.
        z_raw = z_loc
        z = self._to_common_frame(z_raw, self.batch_shift, batch_idx, x)

        # encoder_zx_to_v is fed z_raw, which is what it saw during training.
        # WAS: zx = torch.cat([z_loc, x], dim=-1)
        zx = torch.cat([z_raw, x], dim=-1)
        v_loc, _ = self.encoder_zx_to_v(zx)
        return v_loc.detach().numpy(), z.detach().numpy(), z_raw.detach().numpy()

    def impute_gene_expression_numpy(self, x):
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        z_loc, _, _, _ = self.guide(x)
        mu = self.decoder_z_to_x(z_loc)
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
