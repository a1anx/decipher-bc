"""No-training, no-network check that the split-fix driver's post-training wandb config update
never touches a key already in the run's config (wandb raises ConfigError on a changed value)."""

import importlib
import os
import sys
from types import SimpleNamespace

from decipher_models2.tools._decipher import DecipherConfig

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Simulated Data",
        "Simulated Data Sweep Pipeline",
    ),
)

sweep = importlib.import_module("1008_sweep_splitfix_bifurcation")


def _initial_config_keys():
    captured = {}
    fake_wandb = SimpleNamespace(init=lambda **kwargs: captured.update(kwargs))
    sweep._wandb_init(sweep.jobs(run_date="1008")[0], fake_wandb)
    # the pipeline later merges DecipherConfig into the same config (it has n_epochs = max)
    return set(captured["config"]) | set(DecipherConfig().to_dict())


def test_result_config_keys_do_not_overlap_initial_config_keys():
    record = {col: 0 for col in sweep.WANDB_RESULT_KEYS}
    assert set(sweep._wandb_result_config(record)) & _initial_config_keys() == set()


def test_result_config_keeps_the_trained_epoch_count_under_its_own_key():
    record = dict(n_train=2250, n_val=250, n_epochs=145, git_sha="abc", git_dirty=True)
    assert sweep._wandb_result_config(record) == dict(
        n_train=2250, n_val=250, n_epochs_trained=145, git_sha="abc", git_dirty=True
    )
