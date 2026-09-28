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

from decipher.tools._decipher.module import ConditionalDenseNN


@dataclass(unsafe_hash=True)
class DecipherConfig:
    dim_z: int = 10
    dim_v: int = 2
    layers_v_to_z: Sequence = (64,)
    layers_z_to_x: Sequence = tuple()

    # Define the embedding table 
    n_batches: int = None # Starts unitialized
    batch_emb_dim: int = 8 # Width of each embedding vector


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

    _initialized_from_adata: bool = False

    # EDIT: Add batch_key = None for batch correction
    def initialize_from_adata(self, adata, batch_key = None):
        self.dim_genes = adata.shape[1]
        self.n_cells = adata.shape[0]
        # Counts unique batches and stores in config, so __init__ knows how many rows to allocate in lookup table
        if batch_key is not None:
            self.n_batches = adata.obs[batch_key].nunique()
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

        self.decoder_v_to_z = ConditionalDenseNN(
            self.config.dim_v, self.config.layers_v_to_z, [self.config.dim_z] * 2
        )
        # EDIT: Applies Simple Contatneation of Batch Embedding
        # #Changes size of first linear layer inside that network to dim_z + batch_emb_dim
        self.decoder_z_to_x = ConditionalDenseNN(
            self.config.dim_z, config.layers_z_to_x, [self.config.dim_genes],
            context_dim = config.batch_emb_dim if config.n_batches is not None else 0 #EDIT: Make context_dim dimensional
        )
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes, [128], [self.config.dim_z] * 2
        )
        self.encoder_zx_to_v = ConditionalDenseNN(
            self.config.dim_genes + self.config.dim_z,
            [128],
            [self.config.dim_v, self.config.dim_v],
        )
        # Creates learnable lookup table, where each batch gets its own row
        # self allows PyTorch to track its parameters and include in the gradient updates
        self.batch_embedding = nn.Embedding(config.n_batches, config.batch_emb_dim) if config.n_batches is not None else None
        #EDIT: Make embedding table optional

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    #Change: Added batch_labels (integer tensor that allows correct lookup of batch), removed context = None
    def model(self, x, batch_labels=None): #EDIT: batch_labels=None --> default passes on (genes,)
        pyro.module("decipher", self)

        self.theta = pyro.param(
            "theta",
            x.new_ones(self.config.dim_genes),
            constraint=constraints.positive,
        )
        # Converts raw integers into matrix of shape (n_cells, batch_embd_dim)
        batch_emb = self.batch_embedding(batch_labels) if self.batch_embedding is not None else None
        #EDIT: Avoids batch injetion is self.batch_embedding is None

        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            with poutine.scale(scale=self.config.beta):
                if self.config.prior == "normal":
                    prior = dist.Normal(0, x.new_ones(self.config.dim_v)).to_event(1)
                elif self.config.prior == "gamma":
                    prior = dist.Gamma(0.3, x.new_ones(self.config.dim_v) * 0.8).to_event(1)
                else:
                    raise ValueError("Invalid prior, must be normal or gamma")
                v = pyro.sample("v", prior)

            z_loc, z_scale = self.decoder_v_to_z(v) # Removed context = context since old parameter is gone
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            mu = self.decoder_z_to_x(z, context=batch_emb)
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

    #Change: Added batch_labels (integer tensor that allows correct lookup of batch), removed context = None
    def guide(self, x, batch_labels=None): #EDIT: batch_labels=None --> default passes on (genes,)
        pyro.module("decipher", self)
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)

            z_loc, z_scale = self.encoder_x_to_z(x) # Removed context = context since old parameter is gone
            z_scale = softplus(z_scale) + self._epsilon
            posterior_z = dist.Normal(z_loc, z_scale).to_event(1)
            z = pyro.sample("z", posterior_z)

            zx = torch.cat([z, x], dim=-1)
            v_loc, v_scale = self.encoder_zx_to_v(zx) # Removed context = context since old parameter is gone
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

    # Add batch_labels as a parameter
    def impute_gene_expression_numpy(self, x, batch_labels=None): #EDIT: batch_labels=None --> default passes on (genes,)
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        #Convert numpy array into PyTorch integer tensor (64-bit is required for embedding lookups)
        if type(batch_labels) == np.ndarray:
            batch_labels = torch.tensor(batch_labels,dtype=torch.long)
        
        #Need to compute this explicitly because we're calling encoder directly in the next line
        x_log = torch.log1p(x)
        #Because of added batch_labels parameter, need to call encoder explicitly
        z_loc, _ = self.encoder_x_to_z(x_log)
        #Look up learned batch vectors and then convert to matrix of shape (n_cells, batch_emb_dim) 
        batch_emb = self.batch_embedding(batch_labels) if self.batch_embedding is not None else None
        #EDIT: Skip batch injection if self.batch_embedding is None 
        # Run decoder with batch context injected
        mu = self.decoder_z_to_x(z_loc, context=batch_emb)
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
