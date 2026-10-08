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

from decipher_zx.tools._decipher.module import ConditionalDenseNN


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
        # CHANGED 2026-08-25 -- B5 + B6 (handoff Part 7). model4 = decipher_zx.
        #
        # WAS: self.batch_shift = self._make_batch_shift(config)
        #
        # B6 deletes the b -> z generative edge outright. Batch now reaches the counts
        # through decoder_z_to_x instead (B5), so p(z|v) is batch-blind and z is no longer
        # REQUIRED to carry the batch effect -- that requirement was the whole reason
        # batch_shift existed. Keeping both would make contradictory demands on z:
        # p(z|v,b) says z should be batch-shifted, p(x|z,b) says it need not be. Both fit,
        # nothing selects between them (§6.5).
        #
        # Consequence for the export: with no table, _to_common_frame returns z_raw
        # untouched, so decipher_z == decipher_z_raw in this arm. That is correct, not a
        # regression -- nothing was added to z, so there is nothing to subtract.
        # -------------------------

        # 2. Encoder
        # Batch-blind by design: this arm's guide is q(z|x). See guide() below.
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
        ## z -> x (reconstruction), NOW CONDITIONED ON BATCH
        # CHANGED 2026-08-25 -- B5 (handoff Part 7). This is the p(x|z,b) change.
        #
        # WAS: ConditionalDenseNN(input_dim=dim_z, hidden_dims=..., output_dims=[dim_genes])
        #      with no context_dim -- the -add forks had dropped the argument that native
        #      Decipher already used.
        #
        # With layers_z_to_x = () this is a single Linear(dim_z + n_batches, dim_genes),
        # weight (dim_genes, dim_z + n_batches). ConditionalDenseNN concatenates context
        # FIRST (module.py:89 does torch.cat([context, h])), so weight[:, :n_batches] is
        # n_batches free gene-space vectors -- one per batch. For the sweep config that is
        # (200, 5): 1000 batch parameters reaching all 200 gene dimensions, against
        # batch_shift's 15 parameters reaching at most 3 (§6.5).
        #
        # decoder_z_to_x emits pre-softmax logits, so a per-batch additive offset here
        # becomes a per-batch per-gene MULTIPLICATIVE scaling after softmax -- the standard
        # model of a real scRNA-seq batch effect.
        #
        # context_dim=0 when n_batches=0 makes this degrade to the batch-blind decoder.
        self.decoder_z_to_x = ConditionalDenseNN(
            input_dim=self.config.dim_z,
            hidden_dims=config.layers_z_to_x,
            output_dims=[self.config.dim_genes],
            context_dim=self.config.n_batches,
        )
        

        self._epsilon = 1e-5

        self.theta = None

    @property
    def device(self):
        return self.dummy_param.device

    # ---- batch context helper (2026-08-25, B5/B6/B7) ----
    #
    # REPLACES the three batch-shift helpers this arm inherited from model1-add:
    # _make_batch_shift, _shift and _to_common_frame. All three existed to manage an
    # nn.Embedding table added to z_loc. B6 deletes that table, so all three are dead.
    # Batch is now supplied to a network as a CONCATENATED ONE-HOT (B8) rather than an
    # added vector -- see _batch_context below and the note on decoder_z_to_x.
    #
    # Why one-hot rather than nn.Embedding (B8): concatenation reaches BOTH output heads,
    # because a single final Linear emits loc and scale and they are split afterwards
    # (module.py:104); adding a vector reaches only the head you add it to. At n_batches=5
    # a one-hot gives the first layer 5 dedicated input slots, which is strictly no less
    # expressive than an nn.Embedding(5, d) feeding the same layer, and cheaper.

    def _batch_context(self, batch_idx, like):
        """One-hot batch context, shape (n_cells, n_batches), or None when there are none.

        Returned as None -- not zeros -- when n_batches == 0, because ConditionalDenseNN
        built with context_dim=0 never reads its `context` argument. That makes the whole
        arm degrade to the batch-blind model instead of raising (the same concern fix 7
        handled for the old lookup table).
        """
        if self.config.n_batches == 0:
            return None
        if batch_idx is None:
            batch_idx = like.new_zeros(like.shape[0]).long()
        return torch.nn.functional.one_hot(
            batch_idx, num_classes=self.config.n_batches
        ).to(like.dtype)

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
            
            # v -> z prior: p(z|v). CHANGED 2026-08-25 -- B6 (handoff Part 7).
            #
            # WAS: z_loc = z_loc + self._shift(self.batch_shift, batch_idx, x)
            #
            # The b -> z generative edge is gone. This is now identical to base Decipher's
            # z prior. Batch reaches the counts at decoder_z_to_x instead (B5), which means
            # z is only PERMITTED to carry batch, no longer REQUIRED to (§6.1).
            z_loc, z_scale = self.decoder_v_to_z(v)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            # z -> x reconstruction, NOW CONDITIONED ON BATCH.
            # CHANGED 2026-08-25 -- B5. WAS: mu = self.decoder_z_to_x(z)
            #
            # This is the p(x|z,b) edge, and the only place b enters this arm at all.
            # context=None when n_batches=0, which ConditionalDenseNN ignores because it
            # was built with context_dim=0.
            mu = self.decoder_z_to_x(z, context=self._batch_context(batch_idx, x))
            
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
        2026-08-24; still true for model4). This arm's guide is q(z|x): batch-blind by
        design. That is the whole thing that distinguishes decipher_zx (model4) from
        decipher_zx2 (model5), whose guide is q(z|x,b) via a concatenated one-hot.
        Adding batch to encoder_x_to_z below would turn this model into model5.

        The parameter exists only so that svi.step(x, batch_idx) can pass the same
        arguments to model() and guide(); Pyro requires matching signatures.

        NOTE (2026-08-25, B5): unlike model1-add, a batch-blind guide no longer makes the
        imputation path batch-blind. decoder_z_to_x now takes b, so
        impute_gene_expression_numpy MUST be given batch_idx -- see the note there. This is
        finding 1 arriving in this arm for the first time, by a different route.
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

        CHANGED 2026-08-25 -- B6/B9 (handoff Part 7). In this arm decipher_z and
        decipher_z_raw are now IDENTICAL, and that is correct rather than a regression.

        model1-add defined decipher_z as z_raw minus the cell's batch_shift row plus the
        mean row. B6 deletes that table, so there is no row to subtract -- and nothing was
        ever added to z in the first place, because batch now reaches the counts through
        decoder_z_to_x (B5) rather than through z. Nothing added, nothing to subtract.

        Both columns are still returned so that every arm exports the same two names and
        downstream code needs no per-arm branching. In models 5 and 6 they genuinely
        differ; here they do not.

        The uniform definition across all three new arms is:

            decipher_z = the encoder's output with the batch context held identical for
                         every cell.

        model4 satisfies it trivially, since its guide has no batch context at all.
        Models 5 and 6 satisfy it by averaging the encoder over all n_batches one-hots.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input data of shape (n_cells, n_genes).
        batch_idx : torch.LongTensor, optional
            Accepted for signature parity with models 5 and 6, and ignored here -- the
            guide is q(z|x), so nothing in this method depends on b.

        Returns
        -------
        v : (n_cells, dim_v)
            The decipher v space.
        z : (n_cells, dim_z)
            Equal to z_raw in this arm. Use for Leiden, trajectories, integration metrics
            and biology plots.
        z_raw : (n_cells, dim_z)
            The coordinate the guide produces. Use for anything feeding decoder_z_to_x --
            and note decoder_z_to_x now also needs the batch context, see
            impute_gene_expression_numpy.
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)

        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long)

        x = torch.log1p(x)
        z_loc, _ = self.encoder_x_to_z(x)
        # The guide is q(z|x) and there is no batch table (B6), so the encoder's own
        # output is both the raw and the common-frame coordinate.
        # WAS: z = self._to_common_frame(z_raw, self.batch_shift, batch_idx, x)
        z_raw = z_loc
        z = z_raw

        # encoder_zx_to_v is fed z_raw, which is what it saw during training.
        # WAS: zx = torch.cat([z_loc, x], dim=-1)
        zx = torch.cat([z_raw, x], dim=-1)
        v_loc, _ = self.encoder_zx_to_v(zx)
        return v_loc.detach().numpy(), z.detach().numpy(), z_raw.detach().numpy()

    def impute_gene_expression_numpy(self, x, batch_idx=None):
        """Reconstruct counts. CHANGED 2026-08-25 -- B5 brings finding 1 to this arm.

        decoder_z_to_x now takes b, so reconstructing without batch_idx puts EVERY cell
        through batch 0's decoder columns. In model1-add this method needed no batch_idx
        because the decoder was batch-blind; that is no longer true. Callers must pass
        codes from get_batch_idx(adata, config) -- reconstruction_r2_log1p and
        decipher_gene_imputation both do.

        batch_idx=None still falls back to all-zeros, which is correct only for a
        single-batch dataset.
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        z_loc, _, _, _ = self.guide(x)
        mu = self.decoder_z_to_x(z_loc, context=self._batch_context(batch_idx, z_loc))
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
