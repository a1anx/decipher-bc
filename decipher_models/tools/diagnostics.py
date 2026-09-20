"""Post-training reconstruction diagnostics."""

import numpy as np

from decipher_models.tools._decipher.data import get_batch_idx, get_dense_X


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

    # Reconstruct each cell in its own batch. Without this the guide and
    # impute_gene_expression_numpy fall back to "every cell is batch 0", and the reconstruction
    # is scored against counts that carry each cell's own batch effect.
    batch_idx = get_batch_idx(adata, decipher.config)

    # impute_gene_expression_numpy returns library_size * mu on counts scale,
    # so log1p(x) vs log1p(x_hat) is the correct comparison (no scale correction).
    x_hat = decipher.impute_gene_expression_numpy(x, batch_idx=batch_idx)

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


def batch_effect_recovery(decipher, adata):
    """Compare the batch effect the model LEARNED against the one in the data.

    Under "concat_x" there is no additive batch table, so `decipher_z_raw - decipher_z` is not
    a per-batch constant and cannot be read as the learned shift. This is the replacement.

    Why not just read decoder_z_to_x's batch columns and correlate them against
    uns["gene_shift_matrix"]? Two mismatches stack, and neither is a bug:

      1. Softmax shift-invariance. mu = softmax(decoder_z_to_x(...)), so adding a constant to
         all n_genes logits of one batch changes nothing. Each batch column is identified only
         up to an additive constant, so a raw correlation against the truth is not well defined.
      2. Different spaces. The decoder's offset is additive in PRE-SOFTMAX LOGITS, which after
         softmax and the library-size multiply is multiplicative per gene. The simulator's
         gene_shift is additive in `pre_x`, before round / min-subtract / clip.

    Both go away by comparing all the way downstream, in the space the counts actually live in:
    per-batch mean log1p expression, centred across batches. That is the same quantity on both
    sides, so the comparison is well posed no matter which functional form produced it.

    Returns
    -------
    dict with
        observed  : (n_batches, n_genes) centred per-batch mean of log1p(counts)
        predicted : (n_batches, n_genes) same, from the model's reconstruction
        r         : Pearson correlation between the two, flattened
        magnitude_ratio : ||predicted|| / ||observed||. 1.0 means the model reproduces the
                          batch effect at full strength; below 1 means it is shrinking it.
        per_batch_r     : (n_batches,) the same correlation computed per batch
    """
    x = get_dense_X(adata)
    batch_idx = get_batch_idx(adata, decipher.config)
    if batch_idx is None:
        raise ValueError(
            "batch_effect_recovery needs a model trained with batches "
            "(n_batches > 0 and a batch_key)."
        )
    codes = np.asarray(batch_idx)
    n_batches = decipher.config.n_batches

    decipher.eval()
    x_hat = decipher.impute_gene_expression_numpy(x, batch_idx=batch_idx)

    obs_log = np.log1p(x.astype(float))
    pred_log = np.log1p(np.clip(x_hat, 0, None))

    def per_batch_mean(m):
        out = np.vstack([m[codes == b].mean(axis=0) for b in range(n_batches)])
        return out - out.mean(axis=0, keepdims=True)  # centre across batches

    observed = per_batch_mean(obs_log)
    predicted = per_batch_mean(pred_log)

    r = float(np.corrcoef(observed.ravel(), predicted.ravel())[0, 1])
    denom = np.linalg.norm(observed)
    magnitude_ratio = float(np.linalg.norm(predicted) / denom) if denom > 0 else float("nan")
    per_batch_r = np.array(
        [float(np.corrcoef(observed[b], predicted[b])[0, 1]) for b in range(n_batches)]
    )

    return {
        "observed": observed,
        "predicted": predicted,
        "r": r,
        "magnitude_ratio": magnitude_ratio,
        "per_batch_r": per_batch_r,
    }
