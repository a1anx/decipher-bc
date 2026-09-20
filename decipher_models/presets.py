"""The ten model arms of the batch-conditioning comparison, as DecipherConfig flag sets.

ADDED 2026-09-18. Single source of truth. This table previously existed byte-identically in
`tests/test_variant_equivalence.py` and in the bifurcation sweep pipeline, with nothing keeping
them in sync -- and a drifted copy does not fail, it trains a different architecture than the
results are labeled with.

Ten package names collapse to a product of three flags, because the 1 -> 2 -> 3 progression is the
same step in every family (see `batch-conditioning-comparison.md` section 6):

    batch_embedding_mode  concat_z | additive | concat_x   -- which family: Set 1 | 2 | 3
    batch_conditioning    decoder_only | decoder_encoder   -- is the guide conditioned: step 1 | 2
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
    # ---- Set 2: add -> z. Zero-init embedding added onto z_loc after a batch-blind network. ----
    "model1-add": dict(
        batch_conditioning="decoder_only", mean_field_v=False, batch_embedding_mode="additive"
    ),
    "model2-add": dict(
        batch_conditioning="decoder_encoder", mean_field_v=False, batch_embedding_mode="additive"
    ),
    "model3-add": dict(
        batch_conditioning="decoder_encoder", mean_field_v=True, batch_embedding_mode="additive"
    ),
    # ---- Set 3: concat -> x. b -> z edge deleted; batch enters the reconstruction as one-hot. ----
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

# The standalone package each preset was migrated from. Kept for provenance, and used by
# tests/test_variant_equivalence.py to check the unified model is numerically identical to the
# variant it replaced.
LEGACY_PACKAGES = {
    "native": "decipher",
    "model1": "decipher_vz",
    "model2": "decipher_vz2",
    "model3": "decipher_mf",
    "model1-add": "decipher_vz_add",
    "model2-add": "decipher_vz2_add",
    "model3-add": "decipher_mf_add",
    "model4": "decipher_zx",
    "model5": "decipher_zx2",
    "model6": "decipher_mf2",
}
