"""Control sweep for the split-fix test: the 126 split-fix jobs through the UNMODIFIED 0918 path.

ADDED 2026-10-08. A thin wrapper around 0918_sweep_3models_bifurcation.py (left untouched). It
narrows that driver's grid to the arms native:genes and model5:genes, bifurcation on, and tag
`sweep1008_control`, then runs the driver's own main(), run_one() and _init_worker() as they are
(so the by-name train/val split bug is still present: expect ~1450/1050). The jobs are the same
126 as 1008_sweep_splitfix_bifurcation.py, so the two ledgers join 1:1 on KEY.

Run it as a file on disk (the spawn pool re-imports __main__ from its path):
    python "Simulated Data/Simulated Data Sweep Pipeline/1008_sweep_control_bifurcation.py" --dry-run
    nohup python ... --workers 8 > sweep1008_control.log 2>&1 &
Flags are the 0918 driver's (--workers, --limit, --no-wandb, --tag, --sigmas, ...). Resumes from
`<MMDD>/sweep1008_control_sweep_log.csv`.
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

driver = importlib.import_module("0918_sweep_3models_bifurcation")

CONTROL_TAG = "sweep1008_control"
CONTROL_ARMS = [arm for arm in driver.ARMS if arm[2] == "genes"]  # native:genes, model5:genes
CONTROL_BIFURCATIONS = {"bif": driver.BIFURCATIONS["bif"]}


def narrow_driver_grid():
    """Point the 0918 driver's grid at the control cells (jobs() reads these module globals)."""
    driver.ARMS = CONTROL_ARMS
    driver.BIFURCATIONS = CONTROL_BIFURCATIONS
    driver.NOTEBOOK_TAG = CONTROL_TAG


def main():
    narrow_driver_grid()
    return driver.main()


if __name__ == "__main__":
    sys.exit(main())
