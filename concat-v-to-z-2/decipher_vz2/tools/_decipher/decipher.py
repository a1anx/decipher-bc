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

from decipher_vz2.tools._decipher.module import ConditionalDenseNN


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
    dim_batch_embedding: int = 8
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
        
        # ---- batch embedding ----
        self.batch_emb = torch.nn.Embedding(
            num_embeddings=config.n_batches, 
            embedding_dim=config.dim_batch_embedding
        )
        # -------------------------
        
        # 2. Encoder
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes + self.config.dim_batch_embedding, [128], [self.config.dim_z] * 2
)
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.config.dim_genes + self.config.dim_z,
            [128],
            [self.config.dim_v, self.config.dim_v],
        )
        
        # 3. Decoder
        ## v -> z (now batch-conditioned, was pure biology)
        self.decoder_v_to_z = ConditionalDenseNN(
            input_dim=self.config.dim_v + self.config.dim_batch_embedding,
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
            
            # v -> z prior, now conditioned on batch
            v_combined = torch.cat([v, batch_vec], dim=-1)
            z_loc, z_scale = self.decoder_v_to_z(v_combined)
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
        pyro.module("decipher", self)
        if batch_idx is None:                                   # same fallback as model()
            batch_idx = x.new_zeros(x.shape[0]).long()
        batch_vec = self.batch_emb(batch_idx)
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)
            # encoder_x_to_z takes in x and batch embedding
            z_loc, z_scale = self.encoder_x_to_z(torch.cat([x, batch_vec], dim=-1))
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
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
            
        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long)
        batch_vec = self.batch_emb(batch_idx)

        x = torch.log1p(x)
        z_loc, _ = self.encoder_x_to_z(torch.cat([x, batch_vec], dim=-1))   # widened
        zx = torch.cat([z_loc, x], dim=-1)                                   # unchanged
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
