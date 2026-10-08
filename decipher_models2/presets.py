"""The two model arms of the split-fix study, as DecipherConfig flag sets.

ADDED 2026-10-08. `decipher_models2` is a copy of `decipher_models` that keeps only `native` and
`model5` (see Claude Files/plans/1007_splitfix-relay/). Flags must stay identical to
`decipher_models.presets`, so each run pairs with its `decipher_models` twin.

    batch_embedding_mode  concat_z | additive | concat_x   -- which family: Set 1 | 2 | 3
    batch_conditioning    decoder_only | decoder_encoder   -- is the guide conditioned: step 1 | 2
    mean_field_v          False | True                     -- mean-field encoder:       step 3

`DecipherConfig.__post_init__` validates these values at construction, so a renamed flag raises
rather than silently selecting a different model. It still accepts all three
`batch_embedding_mode` values; only the presets are trimmed.
"""

PRESETS = {
    # baseline -- no batch conditioning anywhere
    "native": dict(batch_conditioning="none", mean_field_v=False, batch_embedding_mode="concat_z"),
    # Set 3: concat -> x. b -> z edge deleted; batch enters the reconstruction as one-hot.
    "model5": dict(
        batch_conditioning="decoder_encoder", mean_field_v=False, batch_embedding_mode="concat_x"
    ),
}

# The standalone package each preset was migrated from. Kept for provenance.
LEGACY_PACKAGES = {
    "native": "decipher",
    "model5": "decipher_zx2",
}
