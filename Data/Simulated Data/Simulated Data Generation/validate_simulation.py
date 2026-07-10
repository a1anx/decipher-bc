"""
Phase 0 simulation validator for Decipher-BC.

Runs the four data-generation checks that must pass before a simulated
AnnData is trustworthy to train on, and prints a pass/fail gate table.

The checks operate on the concatenated AnnData -- the object

    Check 1  Shift present in counts        systematic batch signal exists
    Check 2  Shared latent_t                one trajectory, not five
    Check 3  Orthogonal shift on z dim-1    non-absorbable, metric stays valid
    Check 4  Shift survives z -> X          all magnitudes, incl. the smallest

Check 4 optionally takes `simulate_fn` (your simulate_simple2) to run an exact
paired-mu mechanistic backstop that localizes any failure as cancellation
(propagation through W) vs. quantization (round/clip erasing small shifts).
"""


from simulation_functions import simulate_simple2
from make_simulated_adata import shift_magnitudes_simple
import numpy as np


def _norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _counts(adata, counts_layer):
    X = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return X.astype(float)


def _mean_vec_noise_floor(base, rng):
    """Split-half noise floor for a mean expression vector.
    Conservative by ~sqrt(2): halves use n/2 cells vs the n-cell between-batch
    comparison, so exceeding this floor is strong evidence of real signal."""
    n = len(base)
    h = n // 2
    perm = rng.permutation(n)
    return _norm(base[perm[:h]].mean(0) - base[perm[h:2 * h]].mean(0))


def validate_simulation(adata, simulate_fn=None, *, batch_key="batch",
                        latent_t_key="latent_t", counts_layer="counts",
                        z_time_key="latent_z0", z_shift_key="latent_z1",
                        noise_multiple=2.0, corr_min=0.95, verbose=True):
    """Run the four Phase 0 checks and print a gate table.

    Returns dict: {"passed": bool, "checks": [ {name, status, message}, ... ]}.
    status is one of "PASS", "FAIL", "SKIP".
    """
    rng = np.random.default_rng(0)
    obs = adata.obs
    results = []

    # ---- resolve batch structure -------------------------------------------
    shift_by_batch = obs.groupby(batch_key, observed=True)["shift"].first()
    batches = list(shift_by_batch.index)
    baseline = [b for b in batches if abs(float(shift_by_batch[b])) < 1e-12]
    shifted = sorted([b for b in batches if b not in baseline],
                     key=lambda b: abs(float(shift_by_batch[b])))
    active_types = set(obs["shift_type"].unique()) - {"none"}
    active_type = sorted(active_types)[0] if active_types else "none"

    C = _counts(adata, counts_layer)
    idx = {b: (obs[batch_key].astype(str).values == str(b)) for b in batches}

    if not shifted or not baseline:
        results.append(("Structure", "SKIP",
                        "No baseline+shifted batches found; nothing to validate."))
        return _report(results, active_type, batches, verbose)

    base_C = C[idx[baseline[0]]]
    floor = _mean_vec_noise_floor(base_C, rng)
    base_mean = base_C.mean(0)
    sep = {b: _norm(C[idx[b]].mean(0) - base_mean) for b in shifted}

    # ---- Check 1: shift present in counts -----------------------------------
    max_b = max(sep, key=sep.get)
    ratio1 = sep[max_b] / floor if floor > 0 else np.inf
    if ratio1 > noise_multiple:
        results.append(("Shift present in counts", "PASS",
                        f"largest-shift batch '{max_b}' separates {sep[max_b]:.3f} "
                        f"from baseline = {ratio1:.1f}x noise floor ({floor:.3f})."))
    else:
        results.append(("Shift present in counts", "FAIL",
                        f"largest separation {sep[max_b]:.3f} ~ noise floor "
                        f"({floor:.3f}); batch signal absent from counts. "
                        f"Likely a per-batch normalization cancelling the shift."))

    # ---- Check 2: shared latent_t distribution ------------------------------
    means = np.array([obs.loc[idx[b], latent_t_key].mean() for b in batches])
    stds = np.array([obs.loc[idx[b], latent_t_key].std() for b in batches])
    mean_spread = float(means.max() - means.min())
    std_spread = float(stds.max() - stds.min())
    if mean_spread < 0.05 and std_spread < 0.05:
        results.append(("Shared latent_t", "PASS",
                        f"one distribution across batches (mean spread "
                        f"{mean_spread:.3f}, std spread {std_spread:.3f}) -> "
                        f"single shared trajectory."))
    else:
        results.append(("Shared latent_t", "FAIL",
                        f"latent_t differs across batches (mean spread "
                        f"{mean_spread:.3f}, std spread {std_spread:.3f}); batches "
                        f"may encode different biology -> shared-trajectory "
                        f"assumption broken, correction target ill-defined."))

    # ---- Check 3: orthogonal shift on z dim-1 -------------------------------
    # Sampling- & noise-invariant: since z0 = t + noise, the per-batch quantity
    # (z0_mean - t_mean) is ~0 for every batch UNLESS the shift leaked onto the
    # time axis, in which case it becomes shift_b. This subtracts out both the
    # latent_t sampling spread and the (unknown) noise level. sigma is estimated
    # from the data as std(z0 - t), so no sim parameter needs to be passed in.
    if z_time_key in obs and z_shift_key in obs:
        z0 = obs[z_time_key].values.astype(float)
        z1 = obs[z_shift_key].values.astype(float)
        t = obs[latent_t_key].values.astype(float)
        corr_z0t = float(np.corrcoef(z0, t)[0, 1])
        corr_z1t = float(np.corrcoef(z1, t)[0, 1])

        sigma_hat = float(np.std(z0 - t))            # noise level, data-estimated
        min_n = min(int(idx[b].sum()) for b in batches)
        tol3 = max(0.02, 5 * sigma_hat / np.sqrt(min_n))

        # dim-0 contamination: does the time axis pick up the batch shift?
        d = {b: float(obs.loc[idx[b], z_time_key].mean()
                      - obs.loc[idx[b], latent_t_key].mean()) for b in batches}
        dim0_contam = max(d.values()) - min(d.values())
        # dim-1 carries the intended shift?
        resid = max(abs(float(obs.loc[idx[b], z_shift_key].mean())
                        - float(shift_by_batch[b])) for b in batches)

        if dim0_contam > tol3:
            results.append(("Orthogonal shift on z1", "FAIL",
                            f"shift contaminated the TIME axis: (z0_mean - t_mean) "
                            f"varies by {dim0_contam:.3f} across batches (> {tol3:.3f}). "
                            f"Shift is collinear with time -> absorbable as pseudotime, "
                            f"and latent_t entangled with batch (metric invalid)."))
        elif corr_z0t <= corr_z1t:
            results.append(("Orthogonal shift on z1", "FAIL",
                            f"z dim-0 is not the time axis: corr(z0,t)={corr_z0t:.3f} "
                            f"<= corr(z1,t)={corr_z1t:.3f}. Dimensions may be swapped."))
        elif resid > tol3:
            results.append(("Orthogonal shift on z1", "FAIL",
                            f"z dim-1 does not carry the intended shift "
                            f"(max |z1_mean - shift| = {resid:.3f} > {tol3:.3f})."))
        else:
            results.append(("Orthogonal shift on z1", "PASS",
                            f"shift on z dim-1, orthogonal to time (time-axis "
                            f"contamination {dim0_contam:.3f} < {tol3:.3f}; "
                            f"corr(z0,t)={corr_z0t:.3f} vs corr(z1,t)={corr_z1t:.3f}; "
                            f"dim-1 tracks shift, resid {resid:.3f}). "
                            f"[sigma_hat~{sigma_hat:.2f}]"))
    else:
        results.append(("Orthogonal shift on z1", "SKIP",
                        f"'{z_time_key}'/'{z_shift_key}' not in obs; cannot verify "
                        f"orthogonality from ground-truth z."))

    # ---- Check 4: shift survives z -> X (all magnitudes) --------------------
    smallest = shifted[0]
    ratio_small = sep[smallest] / floor if floor > 0 else np.inf
    sep_ordered = [sep[b] for b in shifted]
    monotonic = all(a <= b + 1e-9 for a, b in zip(sep_ordered, sep_ordered[1:]))
    survived = ratio_small > noise_multiple

    pair_msg = ""
    if simulate_fn is not None:
        pair_msg = _paired_mu(simulate_fn, active_type, float(shift_by_batch[max_b]))

    if survived and monotonic:
        results.append(("Survives z -> X", "PASS",
                        f"all magnitudes survive the count transform "
                        f"(separations {[round(s,2) for s in sep_ordered]} "
                        f"monotonic; smallest shift '{smallest}' = "
                        f"{ratio_small:.1f}x noise).{pair_msg}"))
    elif not survived:
        results.append(("Survives z -> X", "FAIL",
                        f"smallest shift '{smallest}' lost after count transform "
                        f"(separation {sep[smallest]:.3f} ~ noise {floor:.3f}); "
                        f"round/clip likely erasing small shifts at this "
                        f"magnitude.{pair_msg}"))
    else:
        results.append(("Survives z -> X", "FAIL",
                        f"batch separation NOT monotonic in shift "
                        f"({[round(s,2) for s in sep_ordered]}); count transform "
                        f"distorting relative magnitudes.{pair_msg}"))

    return _report(results, active_type, batches, verbose)


def _paired_mu(simulate_fn, active_type, shift):
    """Exact paired-mu backstop: same cells, vary only the shift.
    For a linear decoder the per-cell difference must be the CONSTANT shift*W[1,:],
    so cross-cell std ~ 0 (no cancellation) and mean norm > 0 (propagates)."""
    try:
        p0 = simulate_fn(shift_type="none", shift=0.0, seed=0, cell_seed=0)
        ps = simulate_fn(shift_type=active_type, shift=shift, seed=0, cell_seed=0)
        X0 = p0.X.toarray() if hasattr(p0.X, "toarray") else np.asarray(p0.X, float)
        Xs = ps.X.toarray() if hasattr(ps.X, "toarray") else np.asarray(ps.X, float)
        diff = Xs - X0
        per_cell_std = float(diff.std(0).max())
        mean_norm = _norm(diff.mean(0))
        if mean_norm > 1e-8 and per_cell_std < 1e-6:
            return (f" [paired-mu: pure constant offset, zero cancellation "
                    f"(cross-cell std {per_cell_std:.1e}, |offset| {mean_norm:.3f})]")
        if mean_norm <= 1e-8:
            return (f" [paired-mu FAIL: shift did NOT propagate through W "
                    f"(|offset| {mean_norm:.1e}) -> algebraic cancellation]")
        return (f" [paired-mu WARN: offset not constant across cells "
                f"(cross-cell std {per_cell_std:.1e}); decoder non-linear or "
                f"noise not shared]")
    except Exception as e:
        return f" [paired-mu SKIP: simulate_fn call failed ({type(e).__name__})]"


def _report(results, active_type, batches, verbose):
    passed = all(s == "PASS" for _, s, _ in results if s != "SKIP")
    if verbose:
        w = 60
        print("=" * w)
        print(f"  SIMULATION VALIDATION  -  shift_type='{active_type}', "
              f"{len(batches)} batches")
        print("=" * w)
        for i, (name, status, msg) in enumerate(results, 1):
            tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[skip]"}[status]
            print(f"  {tag} {i}. {name}")
            for line in _wrap(msg, w - 10):
                print(f"          {line}")
        print("-" * w)
        gate = "PASS -- safe to train" if passed else "FAIL -- do NOT train"
        print(f"  GATE: {gate}")
        print("=" * w)
    return {"passed": passed,
            "checks": [{"name": n, "status": s, "message": m}
                       for n, s, m in results]}


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Self-test harness. In your project, instead of this, just call:
    #
    #     from validate_simulation import validate_simulation
    #     adata = shift_magnitudes_simple("vz", np.array([0.1,0.2,0.3]), mag=1.0)
    #     validate_simulation(adata, simulate_fn=simulate_simple2)
    #
    # The block below reproduces the four test scenarios (correct / per-batch
    # bug / tiny-shift / collinear) against a local copy of the generator.
    # -----------------------------------------------------------------------

    shifts = np.array([0.10, 0.20, 0.30, 0.4])

    print("\n########## SCENARIO A: correct vz data (expect GATE PASS) ##########\n")
    good = shift_magnitudes_simple("vz", shifts, mag=1.0)
    validate_simulation(good, simulate_fn=simulate_simple2)


    print("\n######## SCENARIO C: tiny shift + strong noise (survival) ##########\n")
    tiny = shift_magnitudes_simple("vz", np.array([0.02, 0.05, 0.08, 0.1]), mag=1.0, sigma=1.0)
    validate_simulation(tiny, simulate_fn=simulate_simple2)
