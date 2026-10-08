"""Per-batch flip: reflect one batch's decipher_v across the line through its two arm tips.

Extracted from notebook "9.11 vspace_batch_flip.ipynb" (section "Reflect across the line through
the two arm tips"). One batch can land in a mirrored frame in `obsm["decipher_v"]`; that is a
reflection (det -1), so a rotation or `v -> -v` cannot fix it. Two choices of mirror line:
  --line branch-point (default): the perpendicular bisector of the flipped batch's branch point
      (mean v of cells near `uns["branching_t"]`) and the other batches' branch point, so the fork
      lands exactly on the others' fork.
  --line tips: the line through each arm's tip (the cell with the largest `latent_t` on branch
      +1 / -1); the tips stay fixed and the trunk swings to the other side. Uses `branch_id` / `latent_t` (simulated data only).

Usage (from anywhere):
    python "0929_tip_line_reflect.py" [--h5ad PATH] [--flip batch03] [--line branch-point|tips]
                                      [--tip-q CUTOFF]
                                      [--fig out.png] [--save-h5ad out.h5ad]

Does not modify the input h5ad. `--save-h5ad` writes a copy with `obsm["decipher_v_aligned"]`.
"""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_H5AD = (
    ROOT
    / "Simulated Data/Sweeps and Results/shift_sigma_sweep_bifurcation/0918/trained"
    / "sweep0918_ds3_sigma2.0_seed3_model5_bif_genes_decipherseed_1.h5ad"
)


def arm_tips(V, mask, branch_id, latent_t, tip_q=None):
    """Tip of branch +1 and branch -1 within `mask`.

    tip_q=None: v of the single cell with the largest latent_t (the very end of the arm).
    Otherwise: mean v of the cells with latent_t > tip_q.
    """
    tips = []
    for s in (1, -1):
        arm = mask & (branch_id == s)
        if tip_q is None:
            tips.append(V[np.flatnonzero(arm)[np.argmax(latent_t[arm])]])
        else:
            tips.append(V[arm & (latent_t > tip_q)].mean(0))
    return tips


def reflect_across_line(V, m, p, q):
    """Reflect rows V[m] across the line through points p and q.

    Point-plus-direction form, so a vertical line doesn't break it.
    """
    V = V.copy()
    d = (q - p) / np.linalg.norm(q - p)
    X = V[m] - p
    V[m] = p + 2 * np.outer(X @ d, d) - X  # projection onto line, doubled, minus the point
    return V


def branch_point(V, mask, latent_t, branching_t, width=0.05):
    """Mean v of the cells within `width` of the fork pseudotime, within `mask`."""
    return V[mask & (np.abs(latent_t - branching_t) < width)].mean(0)


def bisector(b, b_ref):
    """Two points on the perpendicular bisector of b and b_ref: reflecting across it maps b to b_ref."""
    mid = (b + b_ref) / 2
    return mid, mid + np.array([-(b_ref - b)[1], (b_ref - b)[0]])


def overlap(V, m):
    """Median distance from each flipped cell to its nearest cell in the other batches (lower = better)."""
    return np.median(cKDTree(V[~m]).query(V[m])[0])


def flip_batch(a, flip="batch03", line="branch-point", tip_q=None, key="decipher_v"):
    """Return (V_flipped, p, q) for batch `flip` of AnnData `a`; p, q are two points on the mirror.

    line="branch-point": mirror is the perpendicular bisector of the flipped batch's branch point
    and the other batches' branch point (`uns["branching_t"]`), so the two land on each other.
    line="tips": mirror passes through the two arm tips (see `arm_tips`).
    """
    V = a.obsm[key]
    m = (a.obs["batch"] == flip).values
    t = a.obs["latent_t"].values
    if line == "branch-point":
        bt = float(a.uns["branching_t"])
        p, q = bisector(branch_point(V, m, t, bt), branch_point(V, ~m, t, bt))
    else:
        p, q = arm_tips(V, m, a.obs["branch_id"].values, t, tip_q)
    return reflect_across_line(V, m, p, q), p, q


def plot(V0, V1, m, p, q, flip, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, V, name in zip(axes, (V0, V1), ("original", "reflected")):
        ax.scatter(*V[~m].T, s=3, color="0.75")
        ax.scatter(*V[m].T, s=3, color="C3", label=flip)
        ax.axline(p, q, color="k", ls="--", lw=1)  # infinite mirror line, not just the p-q segment
        ax.set_title(f"{name}  (overlap {overlap(V, m):.3f})")
        ax.set_aspect("equal")
    axes[0].legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    ap.add_argument("--flip", default="batch03", help="batch to reflect")
    ap.add_argument(
        "--line",
        choices=("branch-point", "tips"),
        default="branch-point",
        help="mirror line: bisector of the two branch points (default) or arm-tip line",
    )
    ap.add_argument(
        "--tip-q",
        type=float,
        default=None,
        help="average cells with latent_t > this for each tip (default: the single\n"
        "cell with the largest latent_t on each arm)",
    )
    ap.add_argument("--fig", type=Path, help="save before/after scatter here")
    ap.add_argument("--save-h5ad", type=Path, help="write a copy with obsm['decipher_v_aligned']")
    args = ap.parse_args()

    a = ad.read_h5ad(args.h5ad)
    a.obs_names_make_unique()
    V0 = a.obsm["decipher_v"].copy()
    m = (a.obs["batch"] == args.flip).values
    V1, p, q = flip_batch(a, args.flip, args.line, args.tip_q)

    slope, icpt = np.polyfit([p[0], q[0]], [p[1], q[1]], 1)
    print(f"mirror points {np.round(p, 2)} {np.round(q, 2)}   y = {slope:.3f} x + {icpt:.3f}")
    print(f"overlap original {overlap(V0, m):.3f} -> reflected {overlap(V1, m):.3f}")

    if args.fig:
        plot(V0, V1, m, p, q, args.flip, args.fig)
    if args.save_h5ad:
        a.obsm["decipher_v_aligned"] = V1
        a.write_h5ad(args.save_h5ad)


if __name__ == "__main__":
    main()
