"""The two model arms of `decipher_m5`, as DecipherConfig flag sets.

ADDED 2026-10-08. `decipher_m5` was copied from `decipher_models2` and trimmed to the one-hot
`concat_x` batch path, so `batch_embedding_mode` and `dim_batch_embedding` no longer exist.

    batch_conditioning    none | decoder_encoder   -- native | model5
    mean_field_v          False                    -- full (non-mean-field) v encoder

`DecipherConfig.__post_init__` validates `batch_conditioning` at construction, so any other value
(including the dropped "decoder_only") raises rather than silently selecting a different model.
"""

PRESETS = {
    # baseline -- no batch conditioning anywhere
    "native": dict(batch_conditioning="none", mean_field_v=False),
    # b -> z edge deleted; batch enters encoder_x_to_z and decoder_z_to_x as a learned per-batch
    # row (tables batch_ctx_enc / batch_ctx_dec) added at the first layer.
    "model5": dict(batch_conditioning="decoder_encoder", mean_field_v=False),
}

# The standalone package each preset was migrated from. Kept for provenance.
LEGACY_PACKAGES = {
    "native": "decipher",
    "model5": "decipher_zx2",
}
