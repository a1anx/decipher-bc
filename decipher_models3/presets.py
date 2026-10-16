"""The seven model arms of the batch-conditioning comparison, as DecipherConfig flag sets.

ADDED 2026-10-16. `decipher_models3` is a copy of `decipher_models2` (itself a copy of
`decipher_models`, with the positional train/val split) that carries all seven presets again.
Flags must stay identical to `decipher_models.presets`, so each run pairs with its
`decipher_models` twin. There is no `additive` family here: model4-6 use a learned batch
embedding in place of the one-hot batch vector.

    batch_embedding_mode  concat_z | concat_x               -- which family: Set 1 | 3
    batch_conditioning    decoder_only | decoder_encoder    -- is the guide conditioned: step 1 | 2
    mean_field_v          False | True                     -- mean-field encoder:       step 3

`DecipherConfig.__post_init__` validates these values at construction, so a renamed flag raises
rather than silently selecting a different model.
"""

PRESETS = {
    # baseline -- no batch conditioning anywhere
    "native": dict(batch_conditioning="none", mean_field_v=False, batch_embedding_mode="concat_z"),
    # ---- Set 1: concat -> z. Learned embedding concatenated into decoder_v_to_z's input. ----
    "model1": dict(
        batch_conditioning="decoder_only", mean_field_v=False, batch_embedding_mode="concat_z"
    ),
    "model2": dict(
        batch_conditioning="decoder_encoder", mean_field_v=False, batch_embedding_mode="concat_z"
    ),
    "model3": dict(
        batch_conditioning="decoder_encoder", mean_field_v=True, batch_embedding_mode="concat_z"
    ),
    # ---- Set 3: concat -> x. b -> z edge deleted; batch enters the reconstruction. ----
    "model4": dict(
        batch_conditioning="decoder_only", mean_field_v=False, batch_embedding_mode="concat_x"
    ),
    "model5": dict(
        batch_conditioning="decoder_encoder", mean_field_v=False, batch_embedding_mode="concat_x"
    ),
    "model6": dict(
        batch_conditioning="decoder_encoder", mean_field_v=True, batch_embedding_mode="concat_x"
    ),
}

# The standalone package each preset was migrated from. Kept for provenance.
LEGACY_PACKAGES = {
    "native": "decipher",
    "model1": "decipher_vz",
    "model2": "decipher_vz2",
    "model3": "decipher_mf",
    "model4": None,  # embedding replaces one-hot; no standalone package
    "model5": None,  # embedding replaces one-hot; no standalone package
    "model6": None,  # embedding replaces one-hot; no standalone package
}
