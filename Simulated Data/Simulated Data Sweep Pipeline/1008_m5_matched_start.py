"""Matched-start check: decipher_m5 model5 trained from decipher_models2's exact starting point.

ADDED 2026-10-08. `decipher_m5` builds model5's batch input as two `nn.Embedding` tables, so the
same decipherseed draws different starting weights than `decipher_models2` and a same-seed run is
an independent training run (`sweep1008_m5`). Here each `decipher_m5` run instead starts from the
`decipher_models2` start: inside `decipher_train`, the `decipher_models2` model5 is built at the
same RNG state, its weights are remapped into the `decipher_m5` model
(`remap_model5_state_dict`), and the RNG is left where `decipher_models2` leaves it, so the
minibatch order and the Pyro samples follow the same stream too. If the two packages compute the
same math, the runs should match the `sweep1008_splitfix` runs up to float-rounding drift.

Sigma 2, seed 3, decipherseed 1-3, through the split-fix driver's `run_one` (wandb, CSV row,
figures, h5ad). Outputs go to `1008/m5_matched_start/` (passed as the pipeline's `run_date`
dir), then each run's z/v is compared with its `decipher_models2` twin.

    python "Simulated Data/Simulated Data Sweep Pipeline/1008_m5_matched_start.py" [--no-wandb]
    python ... --compare-only
"""

import argparse
import importlib
import os
import sys
from multiprocessing import get_context

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
driver = importlib.import_module("1008_sweep_splitfix_bifurcation")

TAG = "sweep1008_m5_matched"
RUN_DIR = os.path.join("1008", "m5_matched_start")
SIGMA, SEED, DSEEDS = 2.0, 3, [1, 2, 3]
SWEEPS = os.path.join(_HERE, "..", "Sweeps and Results", "shift_sigma_sweep_bifurcation")
OUT_DIR = os.path.join(SWEEPS, RUN_DIR)
REF_CSV = os.path.join(SWEEPS, "1008", "sweep1008_splitfix_sweep_log.csv")


def matched_decipher(config):
    """A decipher_m5 model5 holding decipher_models2's start at the current RNG state."""
    import pyro

    d2 = importlib.import_module("decipher_models2.tools._decipher")
    d5 = importlib.import_module("decipher_m5.tools._decipher")
    presets2 = importlib.import_module("decipher_models2.presets").PRESETS
    if config.batch_conditioning != "decoder_encoder":
        raise ValueError("matched start is defined for model5 only")
    start = pyro.util.get_rng_state()
    ref = d2.Decipher(config=d2.DecipherConfig(**{**config.to_dict(), **presets2["model5"]}))
    after_ref = pyro.util.get_rng_state()
    pyro.util.set_rng_state(start)
    model = d5.Decipher(config=config)
    model.load_state_dict(d5.remap_model5_state_dict(ref.state_dict()))
    pyro.util.set_rng_state(after_ref)
    return model


def _init_worker():
    driver._init_worker()
    # Only decipher_train's construction is swapped; decipher_load_model (data.py) is untouched.
    importlib.import_module("decipher_m5.tools.decipher").Decipher = matched_decipher


def todo_jobs(use_wandb):
    return [
        j
        for j in driver.jobs(
            notebook_tag=TAG, use_wandb=use_wandb, run_date=RUN_DIR, package="decipher_m5"
        )
        if j["model"] == "model5" and j["shift_sigma"] == SIGMA and j["seed"] == SEED
    ]


def compare():
    """Per decipherseed: ledger metrics and max |Δ| of decipher_z / decipher_v vs the twin."""
    import anndata as ad

    ref = pd.read_csv(REF_CSV)
    new = pd.read_csv(os.path.join(OUT_DIR, f"{TAG}_sweep_log.csv"))
    rows = []
    for d in DSEEDS:
        sel = dict(model="model5", shift_sigma=SIGMA, seed=SEED, decipher_seed=d)
        r = ref.loc[(ref[list(sel)] == pd.Series(sel)).all(axis=1)].iloc[0]
        n = new.loc[(new[list(sel)] == pd.Series(sel)).all(axis=1)].iloc[0]
        a, b = ad.read_h5ad(r["trained_h5ad"]), ad.read_h5ad(n["trained_h5ad"])
        assert (a.obs_names == b.obs_names).all()
        row = {"decipher_seed": d}
        for col in ("rho", "branch_asw", "r2_overall", "n_epochs"):
            row[f"{col}_models2"], row[f"{col}_m5"] = r[col], n[col]
        for key in ("decipher_z", "decipher_v"):
            za, zb = np.asarray(a.obsm[key]), np.asarray(b.obsm[key])
            row[f"max_abs_diff_{key}"] = float(np.abs(za - zb).max())
            row[f"range_{key}"] = float(np.ptp(za))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "compare_vs_models2.csv"), index=False)
    print(out.T.to_string())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    if not args.compare_only:
        os.makedirs(OUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUT_DIR, f"{TAG}_sweep_log.csv")
        jobs = todo_jobs(use_wandb=not args.no_wandb)
        print(f"{len(jobs)} jobs -> {OUT_DIR}", flush=True)
        rows = []
        with get_context("spawn").Pool(len(jobs), initializer=_init_worker) as pool:
            for rec in pool.imap_unordered(driver.run_one, jobs):
                rows.append(rec)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                status = rec["error"] or f"rho={rec['rho']:.3f} n_epochs={rec['n_epochs']}"
                print(f"decipherseed={rec['decipher_seed']} {rec['seconds']}s {status}", flush=True)
        if any(pd.notna(r["error"]) for r in rows):
            print("FAILED runs; see the CSV", flush=True)
            return 1
    compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
