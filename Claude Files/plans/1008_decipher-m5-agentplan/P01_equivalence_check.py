"""Check that `decipher_m5` builds the same native/model5 models as `decipher_models2`.

For each preset, build the untrained model in both packages from the same synthetic AnnData and
seed (the construction path `decipher_train` uses), then compare bit-for-bit: state_dict keys and
tensors, eval-mode guide z_loc/z_scale, decoder_z_to_x output, imputed counts,
compute_v_z_numpy (v, z, z_raw) and the ELBO under a fixed Pyro seed.

Run: .venv/bin/python "Claude Files/plans/1008_decipher-m5-agentplan/P01_equivalence_check.py"
Exits 1 if anything differs.
"""

import importlib
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import Trace_ELBO

BATCH_KEY = "batch"


def make_adata(n_cells: int = 200, n_genes: int = 50, n_batches: int = 3, seed: int = 0):
    """Synthetic integer counts with a categorical batch column and unique obs names."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=rng.gamma(2.0, 2.0, size=n_genes), size=(n_cells, n_genes))
    obs = pd.DataFrame(
        {BATCH_KEY: pd.Categorical([f"b{i % n_batches}" for i in range(n_cells)])},
        index=[f"cell{i}" for i in range(n_cells)],
    )
    return ad.AnnData(X=counts.astype(np.float32), obs=obs)


def build_model(package: str, preset: str, adata, seed: int = 0):
    """Build an untrained Decipher from `package` exactly as `decipher_train` does.

    Returns (model, x, batch_idx) with x the dense training counts and batch_idx their codes.
    """
    tools = importlib.import_module(f"{package}.tools.decipher")
    data = importlib.import_module(f"{package}.tools._decipher.data")
    presets = importlib.import_module(f"{package}.presets").PRESETS
    cls = importlib.import_module(f"{package}.tools._decipher")

    adata = adata.copy()
    config = cls.DecipherConfig(**presets[preset], seed=seed)
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
    tools._make_train_val_split(adata, config.val_frac, config.seed)
    adata_train = adata[adata.obs["decipher_split"] == "train", :]
    config.initialize_from_adata(adata_train, batch_key=BATCH_KEY)
    pyro.clear_param_store()
    pyro.util.set_rng_seed(config.seed)
    model = cls.Decipher(config=config)
    model.eval()

    x = torch.tensor(data.get_dense_X(adata_train), dtype=torch.float32)
    batch_idx = data.get_batch_idx(adata_train, config)
    return model, x, batch_idx


def model_outputs(model, x, batch_idx, seed: int = 0) -> dict:
    """Every eval-mode output compared between packages, keyed by name."""
    out = {}
    with torch.no_grad():
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        z_loc, _, z_scale, _ = model.guide(x, batch_idx)
        out["guide z_loc"] = z_loc
        out["guide z_scale"] = z_scale
        out["decoder_z_to_x"] = model.decoder_z_to_x(
            z_loc, context=model._recon_context(batch_idx, z_loc)
        )
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        out["impute"] = torch.as_tensor(model.impute_gene_expression_numpy(x, batch_idx))
        v, z, z_raw = model.compute_v_z_numpy(x, batch_idx)
        out["v"], out["z"], out["z_raw"] = map(torch.as_tensor, (v, z, z_raw))
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        out["elbo"] = torch.tensor(Trace_ELBO().loss(model.model, model.guide, x, batch_idx))
    return out


def compare(preset: str, ref: str = "decipher_models2", new: str = "decipher_m5") -> bool:
    """Print one line per check for `preset`; return True when everything is identical."""
    adata = make_adata()
    m_ref, x_ref, b_ref = build_model(ref, preset, adata)
    m_new, x_new, b_new = build_model(new, preset, adata)
    checks = {
        "inputs x": torch.equal(x_ref, x_new),
        "inputs batch_idx": torch.equal(b_ref, b_new),
    }

    sd_ref, sd_new = m_ref.state_dict(), m_new.state_dict()
    checks["state_dict keys"] = list(sd_ref) == list(sd_new)
    checks["state_dict tensors"] = checks["state_dict keys"] and all(
        torch.equal(sd_ref[k], sd_new[k]) for k in sd_ref
    )

    o_ref, o_new = model_outputs(m_ref, x_ref, b_ref), model_outputs(m_new, x_new, b_new)
    for name in o_ref:
        checks[name] = torch.equal(o_ref[name], o_new[name])

    print(f"[{preset}] n_batches={m_new.config.n_batches}, {len(sd_new)} state_dict keys")
    for name, ok in checks.items():
        print(f"  {name:20s} {'identical' if ok else 'DIFFERENT'}")
    return all(checks.values())


def main() -> int:
    results = {preset: compare(preset) for preset in ("native", "model5")}
    for preset, ok in results.items():
        print(f"{preset}: {'IDENTICAL' if ok else 'DIFFERENT'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
