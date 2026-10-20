"""Per-epoch run logging that never breaks training: an always-on CSV log, a checkpoint path and a
run summary JSON.

`CsvLog` writes one row per epoch and is trimmed on resume, so an epoch rerun after a crash is not
logged twice. `decipher_train(..., run_dir=...)` uses it; see its docstring for the resume rules.
"""

import csv
import json
import os
from typing import Optional

METRICS_CSV = "metrics.csv"
CHECKPOINT = "train_checkpoint.pt"
RUN_SUMMARY = "run_summary.json"


class CsvLog:
    """Append-only CSV of dict rows, header taken from the first row's keys. Rows kept in memory."""

    def __init__(self, path: str):
        self.path = path
        self.rows: list[dict] = []
        self.header: Optional[list[str]] = None

    def log(self, row: dict) -> None:
        if self.header is None:
            self.header = list(row)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.header).writeheader()
        extra = set(row) - set(self.header)
        if extra:
            raise ValueError(f"row has keys {sorted(extra)} not in the CSV header {self.header}")
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.header).writerow(row)
        self.rows.append(dict(row))

    def resume(self, n_rows: int) -> None:
        """Keep the first `n_rows` rows of the file on disk (those a checkpoint covers) and rewrite
        it. Values read back are strings, as csv gives them. A missing file with `n_rows == 0` is
        an empty log; a file shorter than `n_rows` raises."""
        if not os.path.exists(self.path):
            if n_rows == 0:
                return
            raise ValueError(f"cannot resume {n_rows} rows: {self.path} does not exist")
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            header, rows = reader.fieldnames, list(reader)
        if len(rows) < n_rows:
            raise ValueError(f"{self.path} has {len(rows)} rows, the checkpoint covers {n_rows}")
        self.header, self.rows = header, rows[:n_rows]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, self.header)
            writer.writeheader()
            writer.writerows(self.rows)


def checkpoint_path(run_dir: str) -> str:
    return os.path.join(run_dir, CHECKPOINT)


def write_run_summary(run_dir: str, **fields) -> dict:
    """Merge `fields` into `run_dir/run_summary.json` and return the merged summary."""
    path = os.path.join(run_dir, RUN_SUMMARY)
    summary = {}
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
    summary.update(fields)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
