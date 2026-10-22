"""`--preset` of `Real Data/Oetjen Bone Marrow/1015_oetjen6_m5_decipher.py`: the batch-shift
diagnostic on hand-built tensors for both presets, and the unknown-preset error. No training."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1015_oetjen6_m5_decipher.py"
)
_spec = importlib.util.spec_from_file_location("oetjen6_m5_preset", SCRIPT)
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)

LEVELS = ["A", "B"]


def _embedding(rows: list[list[float]]) -> nn.Embedding:
    emb = nn.Embedding(len(rows), len(rows[0]))
    emb.weight.data = torch.tensor(rows)
    return emb


def _model5() -> SimpleNamespace:
    return SimpleNamespace(batch_ctx_dec=_embedding([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]))


def _model5e(layers_z_to_x: tuple = ()) -> SimpleNamespace:
    # dim_z = 2, d_e = 2, G = 3: decoder weight G x (dim_z + d_e); only the d_e columns matter
    weight = torch.tensor([[9.0, 9.0, 1.0, 0.0], [9.0, 9.0, 0.0, 2.0], [9.0, 9.0, 0.0, 0.0]])
    layer = SimpleNamespace(weight=nn.Parameter(weight))
    return SimpleNamespace(
        config=SimpleNamespace(dim_z=2, layers_z_to_x=layers_z_to_x),
        decoder_z_to_x=SimpleNamespace(layers=[layer]),
        batch_emb=_embedding([[3.0, 2.0], [0.0, 0.0]]),
    )


def test_model5_shift_norm_is_the_decoder_table_row_norm():
    assert m5.batch_shift_norms(_model5(), "model5", LEVELS) == {"A": 5.0, "B": 2.0}


def test_model5e_shift_norm_is_the_norm_of_W_e_times_e_b():
    # e_A = (3, 2) -> shift (3, 4, 0), norm 5; e_B = 0 -> shift 0
    assert m5.batch_shift_norms(_model5e(), "model5e", LEVELS) == {"A": 5.0, "B": 0.0}


def test_model5e_embedding_norms():
    assert m5.batch_emb_norms(_model5e(), LEVELS) == pytest.approx(
        {"A": 13**0.5, "B": 0.0}, abs=1e-4
    )


def test_model5e_shift_raises_when_the_decoder_has_hidden_layers():
    with pytest.raises(ValueError, match="layers_z_to_x"):
        m5.batch_shift_norms(_model5e(layers_z_to_x=(16,)), "model5e", LEVELS)


def test_shift_norm_raises_on_an_unknown_preset():
    with pytest.raises(ValueError, match="model9"):
        m5.batch_shift_norms(_model5(), "model9", LEVELS)


def test_shift_norm_raises_when_levels_and_rows_differ():
    with pytest.raises(ValueError, match="batch levels"):
        m5.batch_shift_norms(_model5(), "model5", ["A"])


def test_cli_rejects_an_unknown_preset(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--seed", "0", "--preset", "model9"])
    with pytest.raises(SystemExit) as exc:
        m5.main()
    assert exc.value.code == 2
    assert "model9" in capsys.readouterr().err
