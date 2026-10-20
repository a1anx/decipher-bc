"""The model arms of the batch-conditioning comparison, as DecipherConfig flag sets.

ADDED 2026-10-16. `decipher_models3` is a copy of `decipher_models2` (itself a copy of
`decipher_models`, with the positional train/val split) that carries all seven presets again.
Their flags must stay identical to `decipher_models.presets`, so each run pairs with its
`decipher_models` twin. Later presets (model4r, model5r) exist only here. There is no `additive`
family here: model4-6 use a learned batch embedding in place of the one-hot batch vector.

    batch_embedding_mode  concat_z | concat_x               -- which family: Set 1 | 3
    batch_conditioning    decoder_only | decoder_encoder    -- is the guide conditioned: step 1 | 2
    mean_field_v          False | True                     -- mean-field encoder:       step 3
    recon_batch           concat | residual                -- concat_x decoder: h([z; e_b]) |
                                                              W z + c + r([z; e_b])  (model4r/5r)
                          | residual_in_span               -- W(z + ρ([z; e_b])) + c (model7i)
                          | concat_plus_residual_z         -- h([z; e_b]) + r_z(z)   (model5rz)
    prior_batch           none | concat | residual         -- prior: f(v) | f([v; e_b]) |
                                                              f(v) + t([v; e_b])  (model7/7i)
                                                              (unset: derived from the mode)

`DecipherConfig.__post_init__` validates these values at construction, so a renamed flag raises
rather than silently selecting a different model.

Presets added after the seven (batch-invariance plan, 2026-10-16) are a base arm plus training
penalty flags: `native_mmd`, `model2_mmd`, `model5_mmd` (MMD invariance penalty on v) and `model8`
(model5 plus the latent cycle-consistency penalty on v). They build the same network as their base
arm; only the training loss differs. `model8v` is model8 with `prior="vamp"` (VampPrior on v),
which adds learned pseudo-inputs.
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
    # ---- 1b: model4/5 with a state-dependent decoder correction outside span(W). ----
    # logits = W z + c + r([z; e_b]); W has no e_b columns, r's output layer starts at zero.
    "model4r": dict(
        batch_conditioning="decoder_only",
        mean_field_v=False,
        batch_embedding_mode="concat_x",
        recon_batch="residual",
    ),
    "model5r": dict(
        batch_conditioning="decoder_encoder",
        mean_field_v=False,
        batch_embedding_mode="concat_x",
        recon_batch="residual",
    ),
    # ---- model7: MrVI-style anchored residual prior on model5r. ----
    # z_loc = f(v) + t([v; e_b]) (t's output layer zero), penalty lambda_r * mean ||t||^2;
    # decoder W z + c + r([z; e_b]) (1b); guide d_z([log1p x; e_b]).
    "model7": dict(
        batch_conditioning="decoder_encoder",
        mean_field_v=False,
        batch_embedding_mode="concat_x",
        recon_batch="residual",
        prior_batch="residual",
        lambda_r=1.0,
    ),
    # model7i: model7 with the decoder inside span(W), W(z + ρ([z; e_b])) + c (MrVI's decoder).
    "model7i": dict(
        batch_conditioning="decoder_encoder",
        mean_field_v=False,
        batch_embedding_mode="concat_x",
        recon_batch="residual_in_span",
        prior_batch="residual",
        lambda_r=1.0,
    ),
}

# ---- Training-penalty presets: a base arm plus a penalty flag (no new parameters). ----
# 1a: MMD invariance penalty on v, weight lambda_inv (the Oetjen d3 driver's --lambda-inv
# overrides it).
_MMD = dict(v_invariance="mmd", lambda_inv=1.0)
PRESETS["native_mmd"] = {**PRESETS["native"], **_MMD}
PRESETS["model2_mmd"] = {**PRESETS["model2"], **_MMD}
PRESETS["model5_mmd"] = {**PRESETS["model5"], **_MMD}
# model8: latent cycle-consistency of v under a swapped batch (sysVI) on model5, weight lambda_c
# (the driver's --lambda-c overrides it).
PRESETS["model8"] = {**PRESETS["model5"], "cycle_v": True, "lambda_c": 1.0}
# model8v: model8 with a VampPrior on v (K = 100 pseudo-inputs, batches round-robin). It adds the
# pseudo-inputs as parameters, so unlike the presets above its state_dict has one extra key.
PRESETS["model8v"] = {**PRESETS["model8"], "prior": "vamp", "n_pseudo_inputs": 100}

# ---- Controls on the residual result (batch-invariance follow-up, 2026-10-17). ----
# model5rz: model5 + a batch-blind r_z(z) (model5r's width, zero output layer) and model5r's N(0, 1)
# table, so model5rz vs model5 is capacity alone and model5r vs model5rz is e_b inside r.
PRESETS["model5rz"] = {
    **PRESETS["model5"],
    "recon_batch": "concat_plus_residual_z",
    "batch_emb_init": "normal",
}
# model5r0: model5r with a zero table (separates the N(0, 1) init from the decoder).
PRESETS["model5r0"] = {**PRESETS["model5r"], "batch_emb_init": "zero"}
# model7iw / model7im: model7i with ρ widened to 4500 (about model5r's r parameter count at
# G = 2000) and 512 (a mid-size point). t keeps width 32.
PRESETS["model7iw"] = {**PRESETS["model7i"], "residual_hidden": 4500}
PRESETS["model7im"] = {**PRESETS["model7i"], "residual_hidden": 512}
# The penalties on model5r: MMD on v (lambda_inv) and the latent cycle (lambda_c), as for model5.
PRESETS["model5r_mmd"] = {**PRESETS["model5r"], **_MMD}
PRESETS["model8r"] = {**PRESETS["model5r"], "cycle_v": True, "lambda_c": 1.0}

# The standalone package each preset was migrated from. Kept for provenance.
LEGACY_PACKAGES = {
    "native": "decipher",
    "model1": "decipher_vz",
    "model2": "decipher_vz2",
    "model3": "decipher_mf",
    "model4": None,  # embedding replaces one-hot; no standalone package
    "model5": None,  # embedding replaces one-hot; no standalone package
    "model6": None,  # embedding replaces one-hot; no standalone package
    "model4r": None,  # new in decipher_models3 (batch-invariance plan, 1b)
    "model5r": None,  # new in decipher_models3 (batch-invariance plan, 1b)
    "model7": None,  # new in decipher_models3 (batch-invariance plan, model7)
    "model7i": None,  # new in decipher_models3 (batch-invariance plan, model7i)
    "model5rz": None,  # new in decipher_models3 (batch-invariance follow-up, A1)
    "model5r0": None,  # new in decipher_models3 (batch-invariance follow-up, A1)
    "model7iw": None,  # new in decipher_models3 (batch-invariance follow-up, A1)
    "model7im": None,  # new in decipher_models3 (batch-invariance follow-up, A1)
    "model5r_mmd": None,  # new in decipher_models3 (batch-invariance follow-up, B1)
    "model8r": None,  # new in decipher_models3 (batch-invariance follow-up, B1)
}
