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

# NEW IMPORT: BatchCorrectedVToZ replaces the standard v->z decoder when batch correction is enabled
from decipher.tools._decipher.module import BatchCorrectedVToZ, ConditionalDenseNN


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
    prior: str = "normal"

    # NEW FIELDS: batch correction hyperparameters (mirror of attention-z-to-x config)
    use_batch_correction: bool = False
    n_batches: int = 0
    batch_emb_dim: int = 64
    n_heads: int = 4
    batch_combination_mode: str = "concat"
    batch_correction_dropout: float = 0.1

    _initialized_from_adata: bool = False

    # NEW PARAM: batch_key — when provided, sets n_batches and enables batch correction
    def initialize_from_adata(self, adata, batch_key=None):
        self.dim_genes = adata.shape[1]
        self.n_cells = adata.shape[0]
        # NEW: count the number of unique batches and flip the flag if a batch key is given
        if batch_key is not None:
            self.n_batches = int(adata.obs[batch_key].astype("category").cat.categories.size)
            self.use_batch_correction = True
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

        # NEW: swap in BatchCorrectedVToZ when batch correction is enabled;
        # this makes z batch-aware by having v attend over batch prototype embeddings.
        # The z->x decoder remains a plain ConditionalDenseNN (x needs no further batch correction).
        if config.use_batch_correction:
            self.decoder_v_to_z = BatchCorrectedVToZ(
                dim_v=self.config.dim_v,
                dim_z=self.config.dim_z,
                n_batches=self.config.n_batches,
                batch_emb_dim=self.config.batch_emb_dim,
                hidden_dims=list(self.config.layers_v_to_z),
                n_heads=self.config.n_heads,
                dropout=self.config.batch_correction_dropout,
                combination_mode=self.config.batch_combination_mode,
            )
        else:
            # ORIGINAL LINE (preserved): standard v->z decoder with no batch awareness
            self.decoder_v_to_z = ConditionalDenseNN(
                self.config.dim_v, self.config.layers_v_to_z, [self.config.dim_z] * 2
            )
        self.decoder_z_to_x = ConditionalDenseNN(
            self.config.dim_z, config.layers_z_to_x, [self.config.dim_genes]
        )
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes, [128], [self.config.dim_z] * 2
        )
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.config.dim_genes + self.config.dim_z,
            [128],
            [self.config.dim_v, self.config.dim_v],
        )

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    # NEW PARAM: batch_index — integer tensor of shape (batch_size,) identifying each cell's batch
    def model(self, x, context=None, batch_index=None):
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

            # NEW: when batch correction is on, v attends over all batch embeddings to produce
            # a batch-aware z. Otherwise, the original context-conditioned path is used.
            # OLD: z_loc, z_scale = self.decoder_v_to_z(v, context=context)
            if self.config.use_batch_correction:
                z_loc, z_scale = self.decoder_v_to_z(v, batch_index)
            else:
                z_loc, z_scale = self.decoder_v_to_z(v, context=context)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # z->x is unchanged: z carries the batch information, so x needs no correction here
            mu = self.decoder_z_to_x(z, context=context)
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

    # NEW PARAM: batch_index added to match model signature (SVI passes the same args to both).
    # The guide itself stays batch-unaware — the encoder produces batch-unaware posteriors,
    # and the ELBO's KL term reconciles them with the batch-aware generative prior.
    def guide(self, x, context=None, batch_index=None):
        pyro.module("decipher", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)

            z_loc, z_scale = self.encoder_x_to_z(x, context=context)
            z_scale = softplus(z_scale) + self._epsilon
            posterior_z = dist.Normal(z_loc, z_scale).to_event(1)
            z = pyro.sample("z", posterior_z)

            zx = torch.cat([z, x], dim=-1)
            v_loc, v_scale = self.encoder_zx_to_v(zx, context=context)
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

    def compute_v_z_numpy(self, x: np.array):
        """Compute decipher_v and decipher_z for a given input.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).

        Returns
        -------
        v : np.ndarray
            Decipher components v of shape (n_cells, dim_v).
        z : np.ndarray
            Decipher latent z of shape (n_cells, dim_z).
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)

        x = torch.log1p(x)
        z_loc, _ = self.encoder_x_to_z(x)
        zx = torch.cat([z_loc, x], dim=-1)
        v_loc, _ = self.encoder_zx_to_v(zx)
        return v_loc.detach().numpy(), z_loc.detach().numpy()

    def impute_gene_expression_numpy(self, x):
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        z_loc, _, _, _ = self.guide(x)
        mu = self.decoder_z_to_x(z_loc)
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
