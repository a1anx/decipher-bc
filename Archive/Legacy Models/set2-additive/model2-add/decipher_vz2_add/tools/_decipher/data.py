import logging
import os
import time

import numpy as np
import randomname
import torch
import torch.distributions
import torch.nn.functional
import torch.utils.data

from decipher_vz2_add.tools._decipher import Decipher, DecipherConfig
from decipher_vz2_add.utils import DECIPHER_GLOBALS, create_decipher_uns_key

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s",
    level=logging.INFO,
)


def get_dense_X(adata):
    if isinstance(adata.X, np.ndarray):
        return adata.X
    else:
        return adata.X.toarray()


def get_random_name(seed=None):
    name = randomname.generate(
        ["a/algorithms", "a/food", "a/physics"],
        ["a/colors", "a/emotions"],
        ["n/algorithms", "n/food", "a/physics"],
        seed=seed,
    )
    datetime_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{datetime_str}-{name}"


def decipher_save_model(adata, model, overwrite=False):
    create_decipher_uns_key(adata)

    if "run_id_history" not in adata.uns["decipher"]:
        adata.uns["decipher"]["run_id_history"] = []

    if "run_id" not in adata.uns["decipher"] or not overwrite:
        adata.uns["decipher"]["run_id"] = get_random_name()
        adata.uns["decipher"]["run_id_history"].append(adata.uns["decipher"]["run_id"])
        logging.info(f"Saving decipher model with run_id {adata.uns['decipher']['run_id']}.")
    else:
        logging.info("Overwriting existing decipher model.")

    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    os.makedirs(full_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(full_path, "decipher_model.pt"))
    adata.uns["decipher"]["config"] = model.config.to_dict()


def decipher_load_model(adata):
    """Load a decipher model whose name is stored in the given AnnData.

    `adata.uns["decipher"]["run_id"]` must be set to the name of the decipher model to load.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.

    Returns
    -------
    model : Decipher
        The decipher model.
    """
    create_decipher_uns_key(adata)
    if "run_id" not in adata.uns["decipher"]:
        raise ValueError("No decipher model has been saved for this AnnData object.")

    model_config = DecipherConfig(**adata.uns["decipher"]["config"])
    model = Decipher(model_config)
    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    model.load_state_dict(torch.load(os.path.join(full_path, "decipher_model.pt")))
    model.eval()
    return model


def make_data_loader_from_adata(adata, batch_size=64, context_discrete_keys=None, **kwargs):
    """Create a PyTorch DataLoader from an AnnData object."""
    genes = torch.FloatTensor(get_dense_X(adata))
    params = [genes]
    
    if context_discrete_keys is None:
        context_discrete_keys = []

    for key in context_discrete_keys:
        # 1. Convert the obs column to categorical codes (integers)
        # 2. Keep it as .long() - DO NOT one-hot encode it here
        t = torch.tensor(adata.obs[key].astype("category").cat.codes.values).long()
        params.append(t)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*params),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )


def get_batch_idx(adata, config):
    """Integer batch codes per cell, matching what the training dataloader produced.

    ADDED 2026-08-24 -- fix 1 (batch_shift_review_handoff.md). The obs -> codes
    conversion previously existed only inside make_data_loader_from_adata, so every
    post-training caller silently fell back to "all cells are batch 0". Keeping the
    conversion in one place is what stops the two paths from drifting apart again.

    Returns None when the model was trained without batches, so callers can pass it
    straight through to the batch-blind path.
    """
    if config.n_batches == 0 or config.batch_key is None:
        return None
    if config.batch_key not in adata.obs:
        raise KeyError(
            f"Model was trained with batch_key={config.batch_key!r}, but that column is "
            f"not present in adata.obs. Available: {list(adata.obs.columns)}"
        )

    # Same conversion as make_data_loader_from_adata above.
    codes = np.asarray(adata.obs[config.batch_key].astype("category").cat.codes.values)

    # Fail loudly rather than silently mis-assigning shifts. A -1 means the column had
    # values outside its own categories (NaN); a code >= n_batches means the categories
    # were ordered differently than at training time.
    if codes.min() < 0:
        raise ValueError(
            f"adata.obs[{config.batch_key!r}] has unassigned (NaN) categories, which "
            f"produce code -1. Every cell needs a batch label."
        )
    if codes.max() >= config.n_batches:
        raise ValueError(
            f"adata.obs[{config.batch_key!r}] produced batch code {codes.max()}, but the "
            f"model was trained with n_batches={config.n_batches}. The category ordering "
            f"does not match training."
        )
    return torch.tensor(codes).long()


def get_decoder_z(adata, key="decipher_z_raw"):
    """Return a stored z coordinate in the frame `decoder_z_to_x` was trained on.

    ADDED 2026-08-24 -- fix 5 follow-up (batch_shift_review_handoff.md).

    USE THIS INSTEAD OF READING adata.obsm DIRECTLY whenever you are about to feed a
    stored z into decoder_z_to_x.

    Why this function has to exist: decipher_rotate_space applies a per-axis sign flip
    to decipher_z and decipher_z_raw *in place*, and stores the pre-flip arrays as
    <key>_not_rotated. The flip is cosmetic -- it is a reflection, so it leaves every
    distance-based metric (silhouette, iLISI, graph connectivity, the Leiden neighbour
    graph) and every rank correlation unchanged, which is why nothing downstream ever
    noticed. But decoder_z_to_x has no reflection invariance:
    decoder_z_to_x(z * sign) != decoder_z_to_x(z). Decoding the post-flip array evaluates
    the decoder on a coordinate it never saw during training.

    This was harmless until fix 5, because nothing in this package ever decoded a stored
    coordinate -- impute_gene_expression_numpy re-derives z from the counts by running
    guide(x, batch_idx). Fix 5 introduced the first consumer that reads a z out of obsm
    and decodes it, so the frame suddenly matters.

    It fails silently and intermittently, which is the dangerous combination: the sign
    correction is data-dependent, so some runs flip and some do not, and a flipped run
    still returns a finite, plausible-looking number. Measured R^2 error from decoding
    the post-flip array on toy runs: 0.688 (vz_add), 0.170 (vz2_add), 0.006 (mf_add).

    Base Decipher has no batch_shift and therefore exports no decipher_z_raw; its
    decipher_z IS the raw coordinate, so fall back to it. This makes the function safe to
    call on all four arms, including the untouched control.

    Parameters
    ----------
    adata : sc.AnnData
        Must carry the decipher embeddings in .obsm.
    key : str, default "decipher_z_raw"
        Which coordinate to fetch. Use "decipher_z_raw" for anything that feeds the
        decoder; "decipher_z" is the batch-corrected coordinate, for metrics and plots.

    Returns
    -------
    np.ndarray of shape (n_cells, dim_z)
    """
    for candidate in (f"{key}_not_rotated", key, "decipher_z_not_rotated", "decipher_z"):
        if candidate in adata.obsm:
            return np.asarray(adata.obsm[candidate])
    raise KeyError(
        f"No usable z coordinate for {key!r}. adata.obsm has: {list(adata.obsm)}. "
        f"Run decipher_train (or decipher_rotate_space) first."
    )
    
    
    
