"""`decipher_models3.run_logging`: CsvLog and the run summary JSON."""

import csv
import json

import pytest

from decipher_models3.run_logging import CsvLog, write_run_summary


def test_run_summary_merges_fields_across_calls(tmp_path):
    write_run_summary(str(tmp_path), best_epoch=3)
    merged = write_run_summary(str(tmp_path), n_epochs_trained=5)
    assert merged == {"best_epoch": 3, "n_epochs_trained": 5}
    assert json.loads((tmp_path / "run_summary.json").read_text()) == merged


# --- CsvLog --------------------------------------------------------------------------------------


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_csv_header_is_written_once(tmp_path):
    log = CsvLog(str(tmp_path / "m.csv"))
    log.log({"epoch": 0, "loss": 1.5})
    log.log({"epoch": 1, "loss": None})
    assert read_rows(log.path) == [["epoch", "loss"], ["0", "1.5"], ["1", ""]]
    assert log.rows == [{"epoch": 0, "loss": 1.5}, {"epoch": 1, "loss": None}]


def test_csv_extra_key_raises(tmp_path):
    log = CsvLog(str(tmp_path / "m.csv"))
    log.log({"epoch": 0})
    with pytest.raises(ValueError, match=r"\['lr'\] not in the CSV header"):
        log.log({"epoch": 1, "lr": 0.1})


def test_csv_resume_trims_to_the_checkpointed_rows(tmp_path):
    path = str(tmp_path / "m.csv")
    first = CsvLog(path)
    for epoch in range(3):
        first.log({"epoch": epoch})
    resumed = CsvLog(path)
    resumed.resume(2)
    resumed.log({"epoch": 2})
    assert read_rows(path) == [["epoch"], ["0"], ["1"], ["2"]]
    assert resumed.rows == [{"epoch": "0"}, {"epoch": "1"}, {"epoch": 2}]


def test_csv_resume_of_zero_rows_on_a_missing_file_is_empty(tmp_path):
    log = CsvLog(str(tmp_path / "m.csv"))
    log.resume(0)
    assert log.rows == []


def test_csv_resume_beyond_the_file_raises(tmp_path):
    log = CsvLog(str(tmp_path / "m.csv"))
    log.log({"epoch": 0})
    with pytest.raises(ValueError, match="has 1 rows, the checkpoint covers 2"):
        CsvLog(log.path).resume(2)
