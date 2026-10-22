"""Post-training reconstruction diagnostics."""

import numpy as np

from decipher_mf.tools._decipher.data import get_dense_X


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

    # impute_gene_expression_numpy returns library_size * mu on counts scale,
    # so log1p(x) vs log1p(x_hat) is the correct comparison (no scale correction).
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
