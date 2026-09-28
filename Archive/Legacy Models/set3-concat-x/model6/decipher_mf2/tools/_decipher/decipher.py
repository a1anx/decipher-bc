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

from decipher_mf2.tools._decipher.module import ConditionalDenseNN


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
        
        # ---- batch conditioning ----
        # REPLACED 2026-08-25 -- B5 + B6 + B7 (handoff Part 7).
        #
        # WAS: two nn.Embedding tables added to z_loc, one per edge --
        #     self.batch_shift_prior = self._make_batch_shift(config)   # model()
        #     self.batch_shift_post  = self._make_batch_shift(config)   # guide()
        #
        # BOTH are deleted, for two different reasons:
        #
        #  B6 removes batch_shift_prior. Batch now reaches the counts through
        #    decoder_z_to_x, so p(z|v) is batch-blind and z is only PERMITTED to carry
        #    batch, not REQUIRED to. Keeping p(z|v,b) alongside p(x|z,b) would make
        #    contradictory demands on z and nothing would select between them (§6.5).
        #
        #  B7 removes batch_shift_post, replacing it with a one-hot CONCATENATED into
        #    encoder_x_to_z below. The reason is the log1p in guide(): a constant offset
        #    in count space arrives at the encoder as a CELL-DEPENDENT amount, because
        #    log1p(c + d) - log1p(c) depends on c. Measured on the sweep data, a 0.5-count
        #    offset shows up as 0.406 for a count-0 cell and 0.080 for a count-5 cell --
        #    5.1x -- and adata.X runs 0..6. A constant added to z_loc cannot undo
        #    something that is not constant; a concatenated context can, because it enters
        #    before the ReLU and 128 hidden units and so changes which units fire.
        #    (Part 7 resolves §6.5's "batch_shift_post survives" against §6.6 in favour
        #    of §6.6.)
        #
        # Neither table exists in this arm any more, so decipher_z_raw - decipher_z is no
        # longer a per-batch constant -- see compute_v_z_numpy and Part 7's B9.
        # -------------------------

        # 2. Encoder. q(z|x,b): batch enters as a CONCATENATED ONE-HOT (B7/B8).
        # ConditionalDenseNN joins the context onto the input at the first layer
        # (module.py:88-89), so b reaches BOTH output heads -- z_loc and z_scale come out
        # of one final Linear and are split afterwards (module.py:104). The old additive
        # table reached z_loc only. Batch-dependent scale is safe on the encoder; §6.7
        # warns against it only on the prior, and B6 has removed the prior's batch edge
        # entirely.
        self.encoder_x_to_z = ConditionalDenseNN(
            self.config.dim_genes, [128], [self.config.dim_z] * 2,
            context_dim=self.config.n_batches,
        )
        # mean field: v's encoder only takes in x, not concatenated with z; never touched batch
        # RENAMED 2026-08-24 -- fix 7. This network was still called `encoder_zx_to_v`,
        # inherited from the structured-guide variants, but in the mean-field model it
        # takes dim_genes inputs and is only ever called on the counts. The name said it
        # consumed z, which is exactly the thing that distinguishes this arm. Renaming
        # changes its state_dict keys, which is already the case here via fix 2.
        # WAS: self.encoder_zx_to_v = ConditionalDenseNN(
        self.encoder_x_to_v = ConditionalDenseNN(
            self.config.dim_genes,
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
        # WAS: no context_dim -- the -add forks had dropped the argument native Decipher
        # already used.
        #
        # With layers_z_to_x = () this is a single Linear(dim_z + n_batches, dim_genes).
        # ConditionalDenseNN concatenates context FIRST (module.py:89), so
        # weight[:, :n_batches] is n_batches free gene-space vectors, one per batch --
        # (200, 5) for the sweep config. That is 1000 batch parameters reaching all 200
        # gene dimensions, against batch_shift's 15 reaching at most 3 (§6.5). Because
        # decoder_z_to_x emits pre-softmax logits, a per-batch additive offset here is a
        # per-batch per-gene MULTIPLICATIVE scaling after softmax.
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
            
            # v -> z prior: batch-blind decoder, batch enters as an additive shift on z_loc
            z_loc, z_scale = self.decoder_v_to_z(v)
            # fix 2: this is the GENERATIVE edge b -> z, so it uses batch_shift_prior.
            # WAS: z_loc = z_loc + self.batch_shift(batch_idx)   # same object as guide()
            # CHANGED 2026-08-25 -- B6. The b -> z generative edge is deleted;
            # p(z|v) is now batch-blind, identical to base Decipher's z prior.
            # WAS: z_loc = z_loc + self._shift(self.batch_shift_prior, batch_idx, x)
            z_scale = softplus(z_scale)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
        
            # z -> x reconstruction, not conditioned on batch
            # CHANGED 2026-08-25 -- B5. p(x|z,b): the only place b enters the
            # generative model now. WAS: mu = self.decoder_z_to_x(z)
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
        pyro.module("decipher", self)
        if batch_idx is None:                                   # same fallback as model()
            batch_idx = x.new_zeros(x.shape[0]).long()
        with pyro.plate("batch", len(x)), poutine.scale(scale=1.0):
            x = torch.log1p(x)
            # encoder_x_to_z is batch-blind; batch shift added to z_loc after
            # CHANGED 2026-08-25 -- B7/B8. q(z|x,b) via a CONCATENATED ONE-HOT.
            #
            # WAS: z_loc, z_scale = self.encoder_x_to_z(x)
            #      z_loc = z_loc + self._shift(self.batch_shift_post, batch_idx, x)
            #
            # The additive table could only move z_loc by a constant per batch, but
            # the batch effect arrives here through log1p as a cell-dependent warp
            # (5.1x between a count-0 and a count-5 cell on this data). The context
            # enters at the first layer, so the network applies a genuinely different
            # transformation per batch, and it reaches z_scale as well as z_loc.
            z_loc, z_scale = self.encoder_x_to_z(
                x, context=self._batch_context(batch_idx, x)
            )
            z_scale = softplus(z_scale) + self._epsilon
            posterior_z = dist.Normal(z_loc, z_scale).to_event(1)
            z = pyro.sample("z", posterior_z)

            # WAS: v_loc, v_scale = self.encoder_zx_to_v(x)  # used to be concatenated zx
            # fix 7: renamed to encoder_x_to_v -- mean field, so v is a function of the
            # counts alone and never sees z or the batch.
            v_loc, v_scale = self.encoder_x_to_v(x)
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
        `encoder_x_to_z(x) + batch_shift[0]` because callers never passed batch_idx.
        Model 1 meanwhile exported `encoder_x_to_z(x)` with no shift term at all, so the
        three arms were reporting different quantities under the same column name and
        `decipher_z` was not comparable across them (finding 5). Since decipher_z drives
        Leiden -> clusters -> trajectories -> decipher_time -> rho (§2b), that
        inconsistency reached the headline metric. Both coordinates are now returned,
        defined identically in all three arms.

        Returns
        -------
        v : (n_cells, dim_v)
            The decipher v space. In this mean-field arm v is a function of the counts
            alone, so it is unaffected by batch_idx.
        z : (n_cells, dim_z)
            Batch-corrected z: every cell moved into one shared frame. Use for Leiden,
            trajectories, integration metrics and biology plots.
        z_raw : (n_cells, dim_z)
            The coordinate the guide actually produces, batch effect included. Use for
            anything feeding decoder_z_to_x -- the counts it reproduces still contain
            the batch effect.
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)

        if batch_idx is None:
            batch_idx = torch.zeros(x.shape[0], dtype=torch.long)

        x = torch.log1p(x)

        # CHANGED 2026-08-25 -- B9 (handoff Part 7).
        #
        # WAS: z_raw = encoder_x_to_z(x) + batch_shift_post[b]
        #      z     = z_raw - batch_shift_post[b] + batch_shift_post.weight.mean(0)
        #
        # B7 replaced the additive table with a concatenated one-hot, so there is no
        # per-batch row to subtract. The batch changes the whole calculation, not a final
        # offset. The replacement is to ask the encoder what it WOULD have produced had
        # the cell come from a reference batch:
        #
        #   z_raw = encoder_x_to_z(x, onehot(b_i))            each cell's own batch
        #   z     = mean over b of encoder_x_to_z(x, onehot(b))
        #
        # Averaging over ALL n_batches references rather than picking one is deliberate.
        # Under the old additive table the reference was free: it added one constant vector
        # to every cell, so all distances -- and therefore Leiden, rho, silhouette, iLISI --
        # were unchanged by the choice. Under concatenation a different reference is a
        # different FUNCTION, so cells move by different amounts and rho moves with them.
        # Averaging removes the choice instead of requiring it to be recorded and held
        # fixed across arms. It also keeps every forward pass on a real one-hot, i.e. an
        # input the encoder was actually trained on -- feeding it a uniform [1/n]*n context
        # would be finding 1 in a new place.
        #
        # This does NOT artificially clean z: batch still enters through x, which differs
        # by batch and is never modified. Holding the context constant removes only the
        # context's own contribution, so a model that failed to learn still shows batch
        # structure here.
        #
        # Consequence: decipher_z_raw - decipher_z is no longer a per-batch constant, so
        # verification step 6's subtraction does not recover the learned shift in this arm.
        # Read decoder_z_to_x.layers[0].weight[:, :n_batches] instead -- n_batches free
        # gene-space vectors, directly comparable to uns["gene_shift_matrix"].
        z_raw, _ = self.encoder_x_to_z(x, context=self._batch_context(batch_idx, x))

        if self.config.n_batches == 0:
            z = z_raw
        else:
            n = x.shape[0]
            acc = None
            for b in range(self.config.n_batches):
                ctx = torch.nn.functional.one_hot(
                    torch.full((n,), b, dtype=torch.long),
                    num_classes=self.config.n_batches,
                ).to(x.dtype)
                zb, _ = self.encoder_x_to_z(x, context=ctx)
                acc = zb if acc is None else acc + zb
            z = acc / self.config.n_batches

        # mean-field: v comes from the counts alone, z never enters.
        v_loc, _ = self.encoder_x_to_v(x)
        return v_loc.detach().numpy(), z.detach().numpy(), z_raw.detach().numpy()

    def impute_gene_expression_numpy(self, x, batch_idx=None):
        """Reconstruct counts from the guide's z.

        CHANGED 2026-08-24 -- fix 1. `batch_idx` was not a parameter, so the call to
        guide() below hit the all-zeros fallback and reconstructed EVERY cell as if it
        belonged to batch 0. Training used batch_shift_post[b]; scoring used
        batch_shift_post[0], so r2_overall and r2_per_gene_median were wrong for 4 of
        the 5 batches. Callers now pass the real codes via get_batch_idx().
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x, dtype=torch.float32)
        # WAS: z_loc, _, _, _ = self.guide(x)
        # guide() returns the raw (batch-containing) z, which is what decoder_z_to_x
        # expects -- the observed counts still contain the batch effect.
        z_loc, _, _, _ = self.guide(x, batch_idx)
        # CHANGED 2026-08-25 -- B5: decoder_z_to_x now takes b as well.
        # WAS: mu = self.decoder_z_to_x(z_loc)
        mu = self.decoder_z_to_x(z_loc, context=self._batch_context(batch_idx, z_loc))
        mu = softmax(mu, dim=-1)
        library_size = x.sum(axis=-1, keepdim=True)
        return (library_size * mu).detach().numpy()
