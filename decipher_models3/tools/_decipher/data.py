import logging
import os
import time

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import randomname
import torch
import torch.distributions
import torch.nn.functional
import torch.utils.data

from decipher_models3.tools._decipher import Decipher, DecipherConfig
from decipher_models3.utils import DECIPHER_GLOBALS, create_decipher_uns_key

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


def decipher_load_model(adata, batch_levels: Optional[Sequence[str]] = None):
    """Load a decipher model whose name is stored in the given AnnData.

    `adata.uns["decipher"]["run_id"]` must be set to the name of the decipher model to load.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix.
    batch_levels : list of str, optional
        Only for a checkpoint saved before the config recorded `batch_levels`: the batch labels
        of the FULL training AnnData, `list(full_adata.obs[<batch_key>].cat.categories)`. They
        are stored in `adata.uns["decipher"]["config"]`, so later loads need no argument. Never
        taken from `adata`'s own categories, which miss any level absent from a subset.

    Returns
    -------
    model : Decipher
        The decipher model.
    """
    create_decipher_uns_key(adata)
    if "run_id" not in adata.uns["decipher"]:
        raise ValueError("No decipher model has been saved for this AnnData object.")

    saved = adata.uns["decipher"]["config"]
    if batch_levels is not None:
        saved["batch_levels"] = _checked_batch_levels(saved, batch_levels)
    model_config = DecipherConfig(**saved)
    if model_config.n_batches > 0 and model_config.batch_levels is None:
        raise ValueError(
            "This checkpoint predates the batch-levels fix: its config has no `batch_levels`, so "
            "batch codes cannot be built safely. Pass "
            f"`batch_levels=list(full_adata.obs[{model_config.batch_key!r}].cat.categories)` "
            "(the levels of the full training AnnData, not a subset) to `decipher_load_model`, "
            "or retrain."
        )
    model = Decipher(model_config)
    model_run_id = adata.uns["decipher"]["run_id"]
    save_folder = DECIPHER_GLOBALS["save_folder"]
    full_path = os.path.join(save_folder, model_run_id)
    state_dict = torch.load(os.path.join(full_path, "decipher_model.pt"))
    _check_not_one_hot_concat_x(model, state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _checked_batch_levels(saved_config: dict, batch_levels: Sequence[str]) -> list:
    """`batch_levels` as strings, if they fit the saved config's n_batches and levels."""
    levels = [str(level) for level in batch_levels]
    n_batches = saved_config.get("n_batches", 0)
    if len(levels) != n_batches:
        raise ValueError(
            f"batch_levels has {len(levels)} levels, but the model was trained with "
            f"n_batches={n_batches}."
        )
    stored = saved_config.get("batch_levels")
    if stored is not None and [str(level) for level in stored] != levels:
        raise ValueError(
            f"batch_levels {levels} differ from the levels saved with the model {list(stored)}."
        )
    return levels


def _check_not_one_hot_concat_x(model: Decipher, state_dict: dict) -> None:
    """Raise when a concat_x checkpoint predates the shared batch embedding.

    `decipher_models` / `decipher_models2` built concat_x (model4-6) with a fixed one-hot batch
    context and no `batch_emb` table, so their state_dicts lack `batch_emb.weight` and their
    first-layer widths differ. Without this check `load_state_dict` fails with a list of shape
    mismatches that does not say why.
    """
    if model.config.batch_embedding_mode != "concat_x" or model.batch_emb is None:
        return
    if "batch_emb.weight" not in state_dict:
        raise ValueError(
            "This checkpoint is a one-hot concat_x model (model4-6) from decipher_models or "
            "decipher_models2: it has no `batch_emb.weight`, while decipher_models3's concat_x "
            "arms use a learned shared batch embedding. Load it with `decipher_models` "
            "(or `decipher_m5` for model5), the package that trained it."
        )


def batch_codes(values, batch_levels: Sequence[str], batch_key: Optional[str] = None) -> np.ndarray:
    """Integer code (int64) of each batch label: its position in `batch_levels`.

    `batch_levels` are the full training AnnData's levels (`DecipherConfig.batch_levels`). The
    labels are compared as strings, as the levels are stored. Raises ValueError on a label that
    is not a level, or NaN: its code would be -1, which silently indexes the last embedding row.
    """
    labels = pd.Series(np.asarray(values, dtype=object))
    labels = labels.where(labels.isna(), labels.astype(str))
    codes = pd.Categorical(labels, categories=list(batch_levels)).codes.astype(np.int64)
    if (codes < 0).any():
        column = "batch labels" if batch_key is None else f"adata.obs[{batch_key!r}]"
        offending = list(pd.unique(labels[codes < 0]))
        raise ValueError(
            f"{column} has labels {offending} that are not training batch levels "
            f"{list(batch_levels)} (NaN counts as unknown). Every cell needs a known batch label."
        )
    return codes


def make_data_loader_from_adata(
    adata,
    batch_size: int = 64,
    context_levels: Optional[dict] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader from an AnnData object.

    `context_levels` maps each discrete obs key to its levels (for the batch key, the full
    training AnnData's `DecipherConfig.batch_levels`); each key becomes a long tensor of codes
    built against those levels, never against `adata`'s own categories.
    """
    genes = torch.FloatTensor(get_dense_X(adata))
    params = [genes]

    for key, levels in (context_levels or {}).items():
        # Keep the codes as .long() - DO NOT one-hot encode them here
        params.append(torch.from_numpy(batch_codes(adata.obs[key], levels, key)))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*params),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )


def get_batch_idx(adata, config) -> Optional[torch.Tensor]:
    """Integer batch codes per cell, matching what the training dataloader produced.

    Codes are positions in `config.batch_levels` (the full training AnnData's levels), so a
    subset that lacks some batch still gets the training codes.

    Returns None when the model was trained without batches, so callers can pass it straight
    through to the batch-blind path.
    """
    if config.n_batches == 0 or config.batch_key is None:
        return None
    if config.batch_key not in adata.obs:
        raise KeyError(
            f"Model was trained with batch_key={config.batch_key!r}, but that column is "
            f"not present in adata.obs. Available: {list(adata.obs.columns)}"
        )
    if config.batch_levels is None:
        raise ValueError(
            "config.batch_levels is None, so batch codes cannot be built. Load the model with "
            "`decipher_load_model(adata, batch_levels=...)` or set it from the full training "
            "AnnData."
        )
    codes = batch_codes(adata.obs[config.batch_key], config.batch_levels, config.batch_key)
    return torch.from_numpy(codes)


def get_decoder_z(adata, key="decipher_z_raw"):
    """Return a stored z coordinate in the frame `decoder_z_to_x` was trained on.

    USE THIS INSTEAD OF READING adata.obsm DIRECTLY whenever you are about to feed a stored z
    into decoder_z_to_x.

    Why this function has to exist: decipher_rotate_space applies a per-axis sign flip to
    decipher_z and decipher_z_raw *in place*, and stores the pre-flip arrays as
    <key>_not_rotated. The flip is cosmetic -- it is a reflection, so it leaves every
    distance-based metric (silhouette, iLISI, graph connectivity, the Leiden neighbour graph)
    and every rank correlation unchanged, which is why nothing downstream ever noticed. But
    decoder_z_to_x has no reflection invariance: decoder_z_to_x(z * sign) != decoder_z_to_x(z).
    Decoding the post-flip array evaluates the decoder on a coordinate it never saw in training.

    It fails silently and intermittently, which is the dangerous combination: the sign
    correction is data-dependent, so some runs flip and some do not, and a flipped run still
    returns a finite, plausible-looking number.

    A model trained with batch_conditioning="none" exports no decipher_z_raw; its decipher_z IS
    the raw coordinate, so fall back to it.

    Parameters
    ----------
    adata : sc.AnnData
        Must carry the decipher embeddings in .obsm.
    key : str, default "decipher_z_raw"
        Which coordinate to fetch. Use "decipher_z_raw" for anything that feeds the decoder;
        "decipher_z" is the batch-corrected coordinate, for metrics and plots.

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
