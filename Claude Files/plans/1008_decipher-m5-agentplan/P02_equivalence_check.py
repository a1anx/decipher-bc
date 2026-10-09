"""Check that `decipher_m5`'s table-based model5 computes the same function as the one-hot model5.

For each reference package (`decipher_models2`, `decipher_models`): build the untrained one-hot
model5, convert its state_dict with `remap_model5_state_dict`, load it into a `decipher_m5` model5
and compare every eval-mode output with assert_close(rtol=1e-5, atol=1e-6): guide z_loc/z_scale,
decoder_z_to_x logits, imputed counts, compute_v_z_numpy (v, z, z_raw) and the ELBO under a fixed
Pyro seed. Also checks: native still bit-identical to `decipher_models2` (P01's compare), the
table equals the old one-hot weight columns, the matched init bound, batch_idx=None -> batch 0,
n_batches=0 -> no tables, an old checkpoint loading end to end through `decipher_load_model`
(after an h5ad round trip), and a negative control (permuted table rows must differ).

Run: .venv/bin/python "Claude Files/plans/1008_decipher-m5-agentplan/P02_equivalence_check.py"
Exits 1 if any check fails.
"""

import importlib
import math
import sys
import tempfile
from pathlib import Path

import anndata as ad
import pyro
import torch
from pyro.infer import Trace_ELBO
from torch.nn.functional import one_hot

sys.path.insert(0, str(Path(__file__).resolve().parent))
P01 = importlib.import_module("P01_equivalence_check")
make_adata, build_model = P01.make_adata, P01.build_model

from decipher_m5.presets import PRESETS  # noqa: E402
from decipher_m5.tools._decipher import (  # noqa: E402
    Decipher,
    DecipherConfig,
    decipher_load_model,
    remap_model5_state_dict,
)

RTOL, ATOL = 1e-5, 1e-6
NEW = "decipher_m5"
REFS = ("decipher_models2", "decipher_models")
RESULTS: dict = {}


def record(name: str, fn) -> None:
    """Run one check; a raised AssertionError marks it failed. Prints one line."""
    try:
        detail = fn()
        RESULTS[name] = True
        print(f"  ok    {name}" + (f"  ({detail})" if detail else ""))
    except AssertionError as err:
        RESULTS[name] = False
        print(f"  FAIL  {name}: {err}")


def decode_logits(model, z, batch_idx):
    """decoder_z_to_x logits through public attributes only, for either layout."""
    if getattr(model, "batch_ctx_dec", None) is not None:  # decipher_m5 model5
        return model.decoder_z_to_x(z, offset=model.batch_ctx_dec(batch_idx))
    if model.decoder_z_to_x.context_dim > 0:  # one-hot model5
        return model.decoder_z_to_x(z, context=one_hot(batch_idx, model.config.n_batches).float())
    return model.decoder_z_to_x(z)  # native


def model_outputs(model, x, batch_idx, seed: int = 0) -> dict:
    """Every eval-mode output compared between packages, keyed by name."""
    out = {}
    with torch.no_grad():
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        z_loc, _, z_scale, _ = model.guide(x, batch_idx)
        out["guide z_loc"], out["guide z_scale"] = z_loc, z_scale
        out["decoder_z_to_x"] = decode_logits(model, z_loc, batch_idx)
        out["impute"] = torch.as_tensor(model.impute_gene_expression_numpy(x, batch_idx))
        v, z, z_raw = model.compute_v_z_numpy(x, batch_idx)
        out["v"], out["z"], out["z_raw"] = map(torch.as_tensor, (v, z, z_raw))
        pyro.clear_param_store()
        pyro.util.set_rng_seed(seed)
        out["elbo"] = torch.tensor(Trace_ELBO().loss(model.model, model.guide, x, batch_idx))
    return out


def errors(a: torch.Tensor, b: torch.Tensor) -> str:
    diff = (a.double() - b.double()).abs()
    rel = diff / b.double().abs().clamp_min(1e-12)
    return f"max abs {diff.max().item():.2e}, max rel {rel.max().item():.2e}"


def assert_outputs_close(section: str, ref: dict, new: dict) -> None:
    for name in ref:

        def close(n=name):
            torch.testing.assert_close(new[n], ref[n], rtol=RTOL, atol=ATOL)
            return errors(new[n], ref[n])

        record(f"{section}: {name}", close)


def remapped_model5(ref_pkg: str, adata):
    """(reference one-hot model5, decipher_m5 model5 holding its remapped weights, x, batch_idx)."""
    m_ref, x, batch_idx = build_model(ref_pkg, "model5", adata)
    m_new, x_new, b_new = build_model(NEW, "model5", adata)
    assert torch.equal(x, x_new) and torch.equal(batch_idx, b_new), "inputs differ"
    m_new.load_state_dict(remap_model5_state_dict(m_ref.state_dict()))  # strict
    return m_ref, m_new, x, batch_idx


def check_native() -> None:
    print("[native] vs decipher_models2 (P01 compare, bit-identical)")
    record("native bit-identical", lambda: None if P01.compare("native") else _fail("differs"))


def _fail(msg: str):
    raise AssertionError(msg)


def check_model5(ref_pkg: str, adata) -> None:
    print(f"[model5] {ref_pkg} one-hot -> remap -> {NEW}")
    m_ref, m_new, x, batch_idx = remapped_model5(ref_pkg, adata)
    n_batches = m_new.config.n_batches

    def table_equals_columns():
        old_w = m_ref.decoder_z_to_x.layers[0].weight[:, :n_batches]
        assert torch.equal(m_new.batch_ctx_dec.weight.T, old_w), "dec table != old columns"
        old_w = m_ref.encoder_x_to_z.layers[0].weight[:, :n_batches]
        assert torch.equal(m_new.batch_ctx_enc.weight.T, old_w), "enc table != old columns"
        return f"B={n_batches}, dec table {tuple(m_new.batch_ctx_dec.weight.shape)}"

    record(f"{ref_pkg} remap: tables == old one-hot columns", table_equals_columns)
    out_ref, out_new = model_outputs(m_ref, x, batch_idx), model_outputs(m_new, x, batch_idx)
    assert_outputs_close(f"{ref_pkg} remap", out_ref, out_new)


def check_matched_init(adata) -> None:
    print("[model5] matched init: max |w| vs U(+-1/sqrt(B + in_dim)) bound")
    m_new, _, _ = build_model(NEW, "model5", adata)
    n_batches = m_new.config.n_batches
    for side, table, net in (
        ("enc", m_new.batch_ctx_enc, m_new.encoder_x_to_z),
        ("dec", m_new.batch_ctx_dec, m_new.decoder_z_to_x),
    ):
        first = net.layers[0]
        bound = 1 / math.sqrt(n_batches + first.in_features)
        for pname, p in (("table", table.weight), ("weight", first.weight), ("bias", first.bias)):

            def within(p=p, bound=bound):
                m = p.abs().max().item()
                assert 0.5 * bound < m <= bound, f"max |w| {m:.4f} outside (bound/2, {bound:.4f}]"
                return f"max |w| {m:.4f} <= bound {bound:.4f}"

            record(f"init {side} {pname}", within)


def check_edges(adata) -> None:
    print("[model5] edges")
    _, m_new, x, batch_idx = remapped_model5("decipher_models2", adata)

    def none_is_batch0():
        with torch.no_grad():
            a = m_new.compute_v_z_numpy(x, None)
            b = m_new.compute_v_z_numpy(x, torch.zeros(len(x), dtype=torch.long))
        assert all((p == q).all() for p, q in zip(a, b)), "batch_idx=None != batch 0"

    record("batch_idx=None == batch 0", none_is_batch0)

    def no_batches_no_tables():
        config = DecipherConfig(**PRESETS["model5"])
        config.initialize_from_adata(adata, batch_key="no_such_column")  # -> n_batches = 0
        m = Decipher(config).eval()
        assert m.batch_ctx_enc is None and m.batch_ctx_dec is None, "tables built for B=0"
        assert not any(k.startswith("batch_ctx") for k in m.state_dict())
        v, z, z_raw = m.compute_v_z_numpy(adata.X)
        assert (z == z_raw).all(), "B=0 model is not batch-blind"

    record("n_batches=0 -> no tables, batch-blind", no_batches_no_tables)

    def permuted_rows_differ():
        m_bad = remapped_model5("decipher_models2", adata)[1]
        with torch.no_grad():
            m_bad.batch_ctx_enc.weight.copy_(m_bad.batch_ctx_enc.weight.roll(1, dims=0))
        z_ok = model_outputs(m_new, x, batch_idx)["guide z_loc"]
        z_bad = model_outputs(m_bad, x, batch_idx)["guide z_loc"]
        assert not torch.allclose(z_ok, z_bad, rtol=RTOL, atol=ATOL), "control did not differ"

    record("negative control (rolled enc rows) differs", permuted_rows_differ)


def check_old_checkpoint_load(ref_pkg: str, adata) -> None:
    print(f"[model5] {ref_pkg} checkpoint -> h5ad round trip -> {NEW} decipher_load_model")
    m_ref, x, batch_idx = build_model(ref_pkg, "model5", adata)
    ref_data = importlib.import_module(f"{ref_pkg}.tools._decipher.data")
    ref_utils = importlib.import_module(f"{ref_pkg}.utils")
    new_utils = importlib.import_module(f"{NEW}.utils")
    old_ref, old_new = (
        ref_utils.DECIPHER_GLOBALS["save_folder"],
        new_utils.DECIPHER_GLOBALS["save_folder"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        ref_utils.DECIPHER_GLOBALS["save_folder"] = tmp
        new_utils.DECIPHER_GLOBALS["save_folder"] = tmp
        try:
            saved = adata.copy()
            ref_data.decipher_save_model(saved, m_ref)
            saved.write_h5ad(Path(tmp) / "saved.h5ad")
            loaded = decipher_load_model(ad.read_h5ad(Path(tmp) / "saved.h5ad"))
        finally:
            ref_utils.DECIPHER_GLOBALS["save_folder"] = old_ref
            new_utils.DECIPHER_GLOBALS["save_folder"] = old_new
    record(
        f"{ref_pkg} load: has batch tables",
        lambda: None if loaded.batch_ctx_enc is not None else _fail("no tables"),
    )
    out_ref, out_new = model_outputs(m_ref, x, batch_idx), model_outputs(loaded, x, batch_idx)
    assert_outputs_close(f"{ref_pkg} load", out_ref, out_new)


def main() -> int:
    adata = make_adata()
    check_native()
    for ref_pkg in REFS:
        check_model5(ref_pkg, adata)
    check_matched_init(adata)
    check_edges(adata)
    for ref_pkg in REFS:
        check_old_checkpoint_load(ref_pkg, adata)
    n_fail = sum(not ok for ok in RESULTS.values())
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
