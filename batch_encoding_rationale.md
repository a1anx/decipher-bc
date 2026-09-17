# Batch encoding: one-hot vs embedding, and why `context=`

**Written 2026-09-17.** Answers two questions asked about models 4/5/6, plus the follow-up
"should we switch to an embedding table to futureproof?"

Every claim below was checked against the code on 2026-09-17. Objects are tagged
**[this repo — path/line]**, **[measured]** for something computed ad hoc, or
**[proposed]** for something that does not exist.

---

## 0. The three arm sets, verified

Worth establishing first, because `model1/2/3` and `model1-add/2-add/3-add` are genuinely
different arm sets and are easy to conflate.

| | **SET 1** — `model1`, `model2`, `model3` | **SET 2** — `model*-add` | **SET 3** — `model4`, `model5`, `model6` |
|---|---|---|---|
| prior `p(z\|v,·)` conditioned? | yes — `cat([v, batch_vec])` | yes — `z_loc + shift[b]` | **no** — identical to base Decipher |
| reconstruction `p(x\|z,·)` conditioned? | no | no | **yes** — `context=one_hot(b)` |
| batch representation | learned `nn.Embedding`, dim **8** (`dim_batch_embedding`) | learned `nn.Embedding`, dim `dim_z` | **fixed one-hot**, dim `n_batches`, no learned table |
| initialisation | **none specified → PyTorch default `Normal(0,1)`** | `init.zeros_` (fix 3) | n/a — no table to initialise |
| how it's injected | concatenated into network **input** | added onto `z_loc` **output** | concatenated via native `context=` |

Verified **[this repo]**:

- `model1/decipher_vz/tools/_decipher/decipher.py` L98-100 builds
  `nn.Embedding(n_batches, config.dim_batch_embedding)`, with `dim_batch_embedding: int = 8`
  at L39. L163 does `v_combined = torch.cat([v, batch_vec], dim=-1)`, and
  `decoder_v_to_z` is widened to `input_dim = dim_v + dim_batch_embedding` (L117).
- **No explicit init** on `batch_emb` in either `model1/decipher_vz` or `model3/decipher_mf` —
  grepped, nothing. So PyTorch's `nn.Embedding` default applies: `Normal(0, 1)`.
- **Within SET 1 the guide side differs**: `model1`'s `encoder_x_to_z` is *not* widened
  (`dim_genes`), `model3`'s *is* (`dim_genes + dim_batch_embedding`, L107).

---

## 1. Why one-hot instead of an embedding?

### The core argument, which needs no parameter counting

> With `n_batches = 5` there are only ever **5 distinct batch vectors**, whatever produces
> them. A one-hot concatenated at the first layer lets those 5 vectors be *anything*,
> independently of each other. An `nn.Embedding` of dimension `d ≥ 5` can also produce any
> 5 — but by composing two matrices instead of one. Identical function class, more
> parameters. At `d < 5` it is strictly worse, because the 5 batches are forced to share
> structure they do not have.
>
> **So an embedding can match a one-hot here. It cannot beat it.**

This holds at *both* injection sites — `decoder_z_to_x` and `encoder_x_to_z` — because the
argument is about how many independent batch vectors can reach the first hidden layer, not
about what happens afterwards.

### Three supporting reasons

**Embeddings earn their keep at large `n_batches`.** With 200 patients and `d = 10`, an
embedding is real compression that shares statistical strength across batches. At 5 batches
there is nothing to compress — and note SET 1's `dim_batch_embedding = 8` is *larger* than a
one-hot would be, i.e. 8 numbers encoding 5 categories.

**It removes an untuned hyperparameter.** `dim_batch_embedding = 8` was never swept. A
one-hot has no such knob; its width is determined by the data.

**It removes the initialisation pathology that finding 3 was about.** `nn.Embedding` defaults
to `Normal(0,1)` — measured per-dim std ≈ 1.02 **[measured, recorded in
`batch_shift_review_handoff.md` finding 3]** — against a true per-batch shift std of
0.058–0.173. That is why fix 3 zero-initialised the SET 2 tables. SET 1 still carries the
un-fixed default. A one-hot has no weights to initialise; the batch parameters are ordinary
`Linear` columns under PyTorch's standard scheme.

### And it made the diagnostic direct

In `decoder_z_to_x` with `layers_z_to_x = ()`, the layer is a single
`Linear(dim_z + n_batches, n_genes)` and `ConditionalDenseNN` concatenates context **first**
(`module.py` L89, `torch.cat([context, h])`). So:

```
weight[:, :n_batches]   is  (200, 5)  —  the per-batch gene-space offset, read directly
```

directly comparable to `uns["gene_shift_matrix"]`, also `(5, 200)`. With an embedding you
would have to compose `batch_emb.weight @ W_slice` to recover the same thing. This is what
`batch_effect_recovery` **[this repo — `tools/diagnostics.py` in models 4/5/6]** relies on.

---

## 2. Why the `context=` parameter?

**Native Decipher already threads `context=` through all four networks.** Verified
**[this repo — `decipher/tools/_decipher/decipher.py`]**:

```
L120   z_loc, z_scale = self.decoder_v_to_z(v,  context=context)
L124   mu            = self.decoder_z_to_x(z,   context=context)
L142   z_loc, z_scale = self.encoder_x_to_z(x,  context=context)
L148   v_loc, v_scale = self.encoder_zx_to_v(zx, context=context)
```

`ConditionalDenseNN` was written with a `context_dim` argument for exactly this. Upstream
constructs the networks with `context_dim = 0`, so the argument is never exercised — but
**the `-add` forks dropped the argument entirely.** Using `context=` *restores* the interface
the base implementation already had. It is not a new mechanism.

Three practical consequences:

- The injection happens where the class puts it — concatenated at the first layer
  (`module.py` L88-89), before the nonlinearity.
- The `n_batches = 0` degenerate case is handled by the existing `context_dim = 0` path,
  which makes `forward` ignore `context` entirely.
- No new module and no new parameter: `state_dict` keys stay inside the existing networks.

---

## 3. One nuance to have ready

If pressed on "identical function class", the exactness depends on the site:

- **`decoder_z_to_x`** — `layers_z_to_x = ()` means one `Linear`, and concatenating a one-hot
  is **exactly** adding a per-batch 200-vector. Verified **[measured]**:
  `max | decoder_z_to_x(z, context=onehot) − (z @ Wz.T + Wb[:, b] + bias) | = 2.4e-07`,
  which is float32 rounding.
- **`encoder_x_to_z`** — a 128-unit hidden layer and a ReLU sit between the concat and the
  output, so it is **not** equivalent to adding a vector. Changing `b` changes which units
  fire. That is deliberate: the batch arrives there through `log1p` as a *cell-dependent*
  warp — a 0.5-count offset appears as 0.406 for a count-0 cell and 0.080 for a count-5 cell,
  **5.1×** **[measured]**, and `adata.X` runs 0–6 — and a constant cannot undo something that
  is not constant.

The "5 batches, 5 free vectors" argument holds at both sites. The "equals adding a vector"
claim holds only at the decoder.

---

## 4. Should we switch to an embedding table to futureproof?

**No.** It does not futureproof anything, and it costs something now.

### The switch is not work you are avoiding

Every batch vector in models 4/5/6 comes from **one method**, `_batch_context`, and the width
is declared in one or two lines per arm **[measured — counted 2026-09-17]**:

| arm | `context_dim=` declarations | `_batch_context(` call sites |
|---|---|---|
| `model4` | 1 | 3 |
| `model5` | 2 | 5 |
| `model6` | 2 | 5 |

Switching later means: add `self.batch_emb = nn.Embedding(n_batches, d)`, change
`context_dim=self.config.n_batches` to `context_dim=d`, and change `_batch_context`'s body
from `one_hot(...)` to `self.batch_emb(batch_idx)`. About three lines per arm, and **zero
call sites change.** That is not a migration worth pre-paying for.

### And switching now costs three things

**Current results become uncomparable.** At `d ≥ 5` the embedding is the same function class,
but a different parameterisation and initialisation means a different optimisation trajectory
and different numbers. All 84 runs would need re-running to know whether anything moved — and
the current numbers are already *at the ceiling*: `sil_decipher_z` of 0.007 / −0.029 / 0.026
for ZX / ZX2 / MF2 against a `counts_nobatch` ceiling of −0.014 **[measured — 9.9 sweep]**.
No headroom to gain, a clean result to lose.

**It reintroduces finding 3.** `nn.Embedding` defaults to `Normal(0,1)`. You would have to
remember to zero-init or rescale. One more thing to get wrong.

**It adds back an untuned knob.** `dim_batch_embedding` would need justifying or sweeping, on
a question where at 5 batches the answer provably cannot beat the one-hot.

### When you *would* switch

Make it a function of `n_batches` on the actual target data, not a hedge.

- **Parameter count.** A one-hot into `decoder_z_to_x` costs `n_batches × n_genes`
  parameters. At 5 × 200 that is 1,000 — nothing. At 200 patients × 2,000 genes it is
  400,000, and an embedding with `d = 10` cuts it to roughly 22,000. *That* is the crossover.
- **Unequal cell counts.** An embedding pools statistical strength across batches, so a batch
  with 20 cells borrows from the others. A one-hot gives that batch `n_genes` free parameters
  and 20 cells to fit them. That is when sharing actually buys something.

Neither applies at 5 balanced batches of 500 cells.

### The futureproofing that would actually matter

For real data the binding constraints are probably not the encoding:

- **Batches at test time that were not in training.** Neither a one-hot nor an embedding has a
  row for an unseen batch. That needs a decision — a reference batch at inference, or a
  held-out-batch protocol — and it is a larger design question than the encoding. Note B9's
  averaged-reference rule for `decipher_z` already assumes every batch can be enumerated at
  inference time.
- **Continuous or multiple covariates** — library prep date, chemistry version, donor age.
  `ConditionalDenseNN`'s `context` accepts any vector, so the mechanism generalises; *what
  goes in it* is the design work.

### You have not lost the option

SET 1 (`model1`, `model2`, `model3`) is a working embedding implementation already in the
repo. If it is needed later there is a reference implementation to copy from.

---

## 5. The framing worth leading with

`nn.Embedding` **is** a one-hot times a matrix. Verified **[measured]**:

```
emb(2)                   = [0.916, 0.784, 0.382]
onehot(2) @ emb.weight   = [0.916, 0.784, 0.382]
difference               = 0.0
```

At 5 batches the choice is not between two different mechanisms. It is whether to insert a
redundant matrix in front of the one you already have.
