"""Plot the CSV written by 0929_track_flip_during_training.py.

Three stacked panels over training epochs: Procrustes det sign per batch (one row per batch, red =
mirrored relative to the majority at the end), sign ambiguity per batch (0 = clear, 1 = the batch
is nearly a line in v so the sign is undecidable), and each batch's distance to the other batches.

Usage:  python 0929_plot_flip_tracking.py [CSV]   (default: the batch03 sigma=2 seed=3 model5 run)
Writes <csv stem>.png next to the CSV.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
csv = Path(sys.argv[1]) if len(sys.argv) > 1 else next(HERE.glob("0929_flip_tracking_*.csv"))
d = pd.read_csv(csv)
batches = [c[4:] for c in d.columns if c.startswith("det_")]
final = d.iloc[-1]
maj = 1 if sum(final[f"det_{b}"] for b in batches) >= 0 else -1

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
det = np.array([d[f"det_{b}"] for b in batches])
axes[0].imshow(
    (det != maj).astype(float),
    aspect="auto",
    cmap="Reds",
    vmin=0,
    vmax=1.3,
    extent=[d.epoch.min() - 0.5, d.epoch.max() + 0.5, len(batches) - 0.5, -0.5],
    interpolation="nearest",
)
axes[0].set_yticks(range(len(batches)), batches)
axes[0].set_title("det sign vs final majority (red = mirrored)")
for b in batches:
    axes[1].plot(d.epoch, d[f"ambig_{b}"], label=b, lw=1)
    axes[2].plot(d.epoch, d[f"overlap_{b}"], label=b, lw=1)
axes[1].set_title("sign ambiguity (0 clear, 1 undecidable)")
axes[2].set_title("median distance to nearest cell in the other batches")
axes[2].set_xlabel("epoch")
axes[2].legend(ncol=5, fontsize=8)
fig.tight_layout()
out = csv.with_suffix(".png")
fig.savefig(out, dpi=130)
print(out)
