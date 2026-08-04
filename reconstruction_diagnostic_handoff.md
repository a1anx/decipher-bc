# Reconstruction diagnostic — implementation handoff

## Context

From the 8/3 meeting with Josh, three action items were confirmed (in order):

1. **Check reconstruction curves** — are the models even fitting?
2. UMAP X space colored by batch — should be separated by batch; if not, something is wrong with the simulated data generation process.
3. Small sigmas' pseudotimes colored by batch.

Follow-ups conditional on the above:
- If the model is learning the batch sampling pattern → try a truly random batch, rather than sampling from Gaussian.
- If not → try contrastive loss to force similarity between batches in V (e.g., InfoNCE loss).

**This document covers action item 1 only.**

---

## Why reconstruction, not just loss

Loss decreasing is necessary but not sufficient. The ELBO Decipher optimizes is:

```
ELBO = E_q[log p(x | z)]   -   KL[q(z|x) || p(z|v)]   -   β · KL[q(v|x,z) || p(v)]
       └─ reconstruction ┘   └────── KL on z ──────┘        └────── KL on v ──────┘
```

Total loss can go down while the reconstruction term stays terrible (posterior collapse: KL → 0, decoder ignores z, everything reconstructs to the marginal mean). Loss curve looks fine, model is broken.

Both Josh (May 14) and Dr. Azizi echoed this: convergence is not fit. Dr. Azizi's exact ask was "the simplest thing of like the decoded expression versus actual expression — the R square should look really high."

---

## What we're building

Two new package functions, exposed via Decipher's `tl` / `pl` submodule convention:

- `dc.tl.reconstruction_r2_log1p(decipher, adata)` — computes R² on log1p scale, returns a result dict.
- `dc.pl.reconstruction_r2_log1p(decipher, adata)` — three-panel diagnostic plot: loss curves, x_hat vs x scatter, per-gene R² histogram.

Plus a small edit to `decipher_train` to capture per-epoch train ELBO (currently only printed, not returned).

---

## Design decisions (confirmed)

| Decision | Choice | Why |
|---|---|---|
| Loss curve capture | Store on decipher object as `decipher.train_losses_` and `decipher.val_losses_` | No return-signature change; curves live with the model that produced them |
| Silent try/except in training loop | **Leave as-is, do not remove** | User explicitly requested no changes to try/except |
| Which packages to patch | All four: `native-decipher`, `model1` (`decipher_vz`), `model2` (`decipher_vz2`), `model3` (`decipher_mf`) | Sweep loop uses all four; consistency required |
| R² scale | log1p | Standard for scRNA-seq VAE reconstruction plots; robust to library-size confound and high-count outliers |
| pl input signature | `pl.reconstruction_r2_log1p(decipher, adata)` — pl calls tl internally, caches on decipher | pl needs loss curves from decipher anyway; caching avoids recompute on plot tweaks; departs from `pl(adata)` convention but avoids duplicating curves into `.uns` |
| Cache invalidation | Check `id(adata)` mismatch, or `recompute=True` kwarg | Catches the realistic failure mode (different adata passed); doesn't catch in-place mutation of same object |
| First-pass scope | Native decipher, sigma = 1.0, single seed | Sanity check the diagnostic wires up cleanly before touching the sweep |

---

## What to look for in the output

**Loss curves.** Both train ELBO and val NLL should be descending and flattening. If val NLL still dropping at last epoch → undertrained. If val rising while train drops → overfitting. If both plateau high very early → fitting failure.

**Scatter (x_hat vs x, log1p).** Points hugging y = x → good fit. Horizontal band regardless of x → posterior collapse (decoder ignores z). Per-gene horizontal bars → decoder produces per-gene means with no cell-specific variation.

**Per-gene R² histogram.** Mass at high R² (>0.5) with small tail → healthy. Distribution centered near zero, or bimodal with a chunk near zero → many genes not being fit.

**For the σ = 10 puzzle specifically.** If native decipher at σ = 1.0 shows healthy R² (median per-gene R² > 0.3, curves flattened), then fitting is fine and the σ = 10 degradation isn't a fitting failure — it's a batch-effect or simulation-difficulty issue. Move on to action items 2 and 3.

**Caveat on the sim's R² ceiling.** With simulated data generated from `z' ~ N(z, σ²)` passed through a random neural network, some ceiling on R² is baked in by the injection noise. Don't expect R² = 1 even in principle. What matters: (a) does R² rise during training, (b) is it comparable across the four models, (c) does R² *collapse* at high σ.

---

## The patches (against `native-decipher`; swap package name for the other three)

### Patch 1 — `native-decipher/decipher/tools/decipher.py`

Two edits inside `decipher_train`. Try/except left alone.

**Edit A** — after `decipher.to(device)`:

```python
    decipher = Decipher(
        config=decipher_config,
    )
    decipher.to(device)
    decipher.train_losses_ = []      # NEW: per-epoch train ELBO (per-obs)
    decipher.val_losses_ = []        # NEW: per-epoch val NLL (per-obs)
```

**Edit B** — append train ELBO at the end of the epoch's batch loop, right after the `for xc in dataloader_train:` block closes and before `decipher.eval()`:

```python
        decipher.train_losses_.append(train_elbo / train_elbo_n_obs)  # NEW
```

**Edit C** — append val NLL right after the local list append:

```python
        val_losses.append(val_nll)
        decipher.val_losses_.append(val_nll)   # NEW
```

Consequence: if the try/except fires mid-epoch, `train_losses_` won't get that epoch appended (the `return` happens before). `len(decipher.train_losses_)` will be one short of the epoch at which failure occurred — actually useful as a signal of silent death.

### Patch 2 — new file `native-decipher/decipher/tools/diagnostics.py`

```python
"""Post-training reconstruction diagnostics."""

import numpy as np

from decipher.tools._decipher.data import get_dense_X


def reconstruction_r2_log1p(decipher, adata):
    """Compute reconstruction R^2 on log1p scale.

    Parameters
    ----------
    decipher : Decipher
        A trained decipher model.
    adata : sc.AnnData
        The data the model was trained on (or a held-out subset).

    Returns
    -------
    result : dict
        Keys:
          'x_log'        : (n_cells, n_genes) observed log1p counts
          'x_hat_log'    : (n_cells, n_genes) reconstructed log1p expression
          'r2_overall'   : float, pooled across all cell by gene entries
          'r2_per_gene'  : (n_genes,) R^2 per gene across cells; NaN for
                           zero-variance genes
    """
    x = get_dense_X(adata)
    decipher.eval()

    # impute_gene_expression_numpy returns library_size * mu on counts scale
    # (see Decipher.impute_gene_expression_numpy). log1p comparison against
    # raw counts is the correct scale.
    x_hat = decipher.impute_gene_expression_numpy(x)

    x_log = np.log1p(x.astype(float))
    x_hat_log = np.log1p(np.clip(x_hat, 0, None))

    ss_res = ((x_log - x_hat_log) ** 2).sum()
    ss_tot = ((x_log - x_log.mean()) ** 2).sum()
    r2_overall = float(1 - ss_res / ss_tot)

    ss_res_g = ((x_log - x_hat_log) ** 2).sum(axis=0)
    ss_tot_g = ((x_log - x_log.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_per_gene = 1 - ss_res_g / ss_tot_g
    r2_per_gene = np.where(ss_tot_g > 0, r2_per_gene, np.nan)

    return {
        "x_log": x_log,
        "x_hat_log": x_hat_log,
        "r2_overall": r2_overall,
        "r2_per_gene": r2_per_gene,
    }
```

### Patch 3 — expose in `native-decipher/decipher/tools/__init__.py`

Add:

```python
from decipher.tools.diagnostics import reconstruction_r2_log1p
```

### Patch 4 — new file `native-decipher/decipher/plot/diagnostics.py`

```python
"""Plotting for post-training diagnostics."""

import numpy as np
from matplotlib import pyplot as plt

from decipher.tools.diagnostics import reconstruction_r2_log1p as _tl_reconstruction_r2_log1p


def reconstruction_r2_log1p(
    decipher,
    adata,
    recompute=False,
    n_scatter_points=20_000,
    scatter_alpha=0.2,
    scatter_size=1,
    n_hist_bins=30,
    figsize=(15, 4),
    scatter_seed=0,
):
    """Three-panel reconstruction diagnostic: loss curves, x_hat vs x
    scatter, per-gene R^2 histogram.

    On first call, computes the diagnostic via
    dc.tl.reconstruction_r2_log1p and caches it on the decipher object as
    `decipher.reconstruction_r2_result_`. Subsequent calls reuse the
    cache unless `adata` has changed (checked via id(adata)) or
    `recompute=True` is passed.

    Loss curves are read from `decipher.train_losses_` and
    `decipher.val_losses_`, which are populated by decipher_train.

    Parameters
    ----------
    decipher : Decipher
        A trained decipher model.
    adata : sc.AnnData
        The data the model was trained on.
    recompute : bool, default False
        If True, recompute the diagnostic even if a cache is present.
    n_scatter_points : int
        Number of (cell, gene) entries to subsample for the scatter panel.
    scatter_alpha, scatter_size : float
        Matplotlib scatter styling.
    n_hist_bins : int
        Bins for the per-gene R^2 histogram.
    figsize : tuple
        Passed to plt.subplots.
    scatter_seed : int
        Seed for the scatter subsample. Fixed by default so reruns match.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (3,)
    """
    # Cache check
    cached_adata_id = getattr(decipher, "_reconstruction_r2_adata_id", None)
    if recompute or cached_adata_id != id(adata) or not hasattr(
        decipher, "reconstruction_r2_result_"
    ):
        decipher.reconstruction_r2_result_ = _tl_reconstruction_r2_log1p(decipher, adata)
        decipher._reconstruction_r2_adata_id = id(adata)
    result = decipher.reconstruction_r2_result_

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # (1) Loss curves
    ax = axes[0]
    train_losses = getattr(decipher, "train_losses_", [])
    val_losses = getattr(decipher, "val_losses_", [])
    if train_losses:
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="train ELBO (per obs)")
    if val_losses:
        epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(epochs, val_losses, label="val NLL (per obs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training curves")
    if train_losses or val_losses:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no loss curves available",
                ha="center", va="center", transform=ax.transAxes)

    # (2) Reconstruction scatter
    ax = axes[1]
    x_log = result["x_log"]
    x_hat_log = result["x_hat_log"]
    flat_x = x_log.ravel()
    flat_xhat = x_hat_log.ravel()
    rng = np.random.default_rng(scatter_seed)
    idx = rng.choice(
        flat_x.size, size=min(n_scatter_points, flat_x.size), replace=False
    )
    ax.scatter(flat_x[idx], flat_xhat[idx], s=scatter_size, alpha=scatter_alpha)
    lo = float(min(flat_x[idx].min(), flat_xhat[idx].min()))
    hi = float(max(flat_x[idx].max(), flat_xhat[idx].max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("observed log1p(x)")
    ax.set_ylabel("reconstructed log1p(x_hat)")
    ax.set_title(f"reconstruction (overall R^2 = {result['r2_overall']:.3f})")
    ax.legend()

    # (3) Per-gene R^2 histogram
    ax = axes[2]
    r2_per_gene = result["r2_per_gene"]
    valid = r2_per_gene[~np.isnan(r2_per_gene)]
    ax.hist(valid, bins=n_hist_bins)
    median_r2 = float(np.nanmedian(r2_per_gene))
    ax.axvline(
        median_r2, color="r", linestyle="--",
        label=f"median = {median_r2:.3f}",
    )
    ax.set_xlabel("per-gene R^2 (log1p scale)")
    ax.set_ylabel("number of genes")
    ax.set_title("per-gene reconstruction quality")
    ax.legend()

    fig.tight_layout()
    return fig, axes
```

### Patch 5 — expose in `native-decipher/decipher/plot/__init__.py`

Add:

```python
from decipher.plot.diagnostics import reconstruction_r2_log1p
```

---

## Apply to the other three packages

Same five patches. Only difference: replace `decipher` with the variant package name in every import path.

**model1** (`decipher_vz`):

```python
# In diagnostics.py
from decipher_vz.tools._decipher.data import get_dense_X

# In plot/diagnostics.py
from decipher_vz.tools.diagnostics import reconstruction_r2_log1p as _tl_reconstruction_r2_log1p
```

**model2** (`decipher_vz2`):

```python
from decipher_vz2.tools._decipher.data import get_dense_X
from decipher_vz2.tools.diagnostics import reconstruction_r2_log1p as _tl_reconstruction_r2_log1p
```

**model3** (`decipher_mf`):

```python
from decipher_mf.tools._decipher.data import get_dense_X
from decipher_mf.tools.diagnostics import reconstruction_r2_log1p as _tl_reconstruction_r2_log1p
```

**Two things to verify per variant before applying Patch 1:**

1. The `for xc in dataloader_train:` loop with the same try/except shape. If a variant already differs here, the edit needs to slot into whatever's there.
2. The `Decipher` object supports arbitrary attribute assignment (i.e., no `__slots__`). Standard for `nn.Module` subclasses; spot-check `model3` since the mean-field encoder is the most architecturally different from native.

---

## Reinstall

After edits, from repo root:

```bash
pip install -e native-decipher
pip install -e model1
pip install -e model2
pip install -e model3
```

Should be a no-op if editable installs are already in place. If imports of the new modules fail, one of them was installed non-editably.

---

## Notebook usage after patches land

```python
import decipher as dc
from decipher.tools._decipher import DecipherConfig

# ... load simulated adata at sigma = 1.0 ...

cfg = DecipherConfig(seed=0, ...)   # match sweep config
decipher, val_losses = dc.tl.decipher_train(adata, cfg)

# Lazy: pl runs tl on first call, caches result
fig, axes = dc.pl.reconstruction_r2_log1p(decipher, adata)

# Or explicit if you also want the result dict
result = dc.tl.reconstruction_r2_log1p(decipher, adata)
print(f"overall R^2: {result['r2_overall']:.3f}")
fig, axes = dc.pl.reconstruction_r2_log1p(decipher, adata)   # uses cache
```

---

## Scale of `impute_gene_expression_numpy` (confirmed)

Inspected the source in `native-decipher/decipher/tools/_decipher/decipher.py`:

```python
def impute_gene_expression_numpy(self, x):
    if type(x) == np.ndarray:
        x = torch.tensor(x, dtype=torch.float32)
    z_loc, _, _, _ = self.guide(x)
    mu = self.decoder_z_to_x(z_loc)
    mu = softmax(mu, dim=-1)
    library_size = x.sum(axis=-1, keepdim=True)
    return (library_size * mu).detach().numpy()
```

Return value is `library_size * mu` on counts scale, so `log1p(x)` vs `log1p(x_hat)` is the right comparison — no scale correction needed.

**Note on determinism.** Despite the `pyro.sample("z", ...)` and `pyro.sample("v", ...)` calls inside `guide`, the returned `x_hat` is deterministic for a given trained model and fixed input. `guide` returns `z_loc` (the encoder mean, a deterministic function of the input), and `impute_gene_expression_numpy` uses `z_loc` — not the sampled `z` — to compute `mu`. The `pyro.sample` calls build the Pyro trace as a side effect, but their sampled values are discarded. So repeated calls with the same input give the same R². Seed-to-seed variance across the sigma sweep is entirely from training (initialization, minibatch shuffling, SVI reparameterization noise), not from inference.

**Python scoping in `impute_gene_expression_numpy`.** `self.guide(x)` does `x = torch.log1p(x)` internally, but that assignment is local to `guide`. The outer `x` (used to compute `library_size`) stays as raw counts. Correct behavior, just confusing to read.

Remove the "verify scale" note from the `diagnostics.py` docstring — no verification needed.

---

## First-pass scope

Run the patched pipeline on **native decipher, sigma = 1.0, single seed** from the existing sweep. Report back:

1. Overall log1p R²
2. Median per-gene R²
3. Whether the loss curves flattened
4. Anything unexpected in the scatter panel (e.g., horizontal banding suggesting posterior collapse)

That determines whether we (a) accept the diagnostic and roll it out across the sweep or (b) discover the models genuinely aren't fitting and pivot to why.

---

## Not in scope for this handoff

- Action items 2 (UMAP X space by batch) and 3 (small-σ pseudotimes by batch) from the 8/3 meeting.
- Removing the silent try/except in `decipher_train` (user requested keep as-is).
- Extending the diagnostic across the full sigma sweep (do after first-pass validation).
- Rolling into a comparison figure across the four models.
