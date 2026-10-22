"""`Real Data/Oetjen Bone Marrow/1016_oetjen_d3_decipher.py`: the concat_x batch-shift diagnostic
on hand-built tensors, its hidden-layer error, no shift for concat_z arms, no norms without a table
(native), the residual-decoder shift norm and its batch-centred version (model4r/5r/7) on a
hand-built r, the in-span shift norms ||W ρ|| (model7i) on a hand-built ρ and W, the default output
folder per preset, `PRESET_CHOICES` == `PRESETS`, the λ tag, the
out-dir for plain, scan and final runs and their errors, the
penalty-history meta, `--lambda-inv` on the MMD presets and
`--lambda-r` on model7/model7i. And
`1016_oetjen_d3_queue.py`: job-file parsing and the frozen-tree command with src, out-dir and log
in the main checkout. No training."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from decipher_models3.presets import PRESETS
from decipher_models3.tools._decipher import DecipherConfig

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Real Data"
    / "Oetjen Bone Marrow"
    / "1016_oetjen_d3_decipher.py"
)
_spec = importlib.util.spec_from_file_location("oetjen_d3_driver", SCRIPT)
d3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d3)

LEVELS = ["A", "B"]


def _model(mode: str = "concat_x", layers_z_to_x: tuple = ()) -> SimpleNamespace:
    # L = 2, D = 2, G = 3: decoder weight G x (L + D); only the D columns matter
    weight = torch.tensor([[9.0, 9.0, 1.0, 0.0], [9.0, 9.0, 0.0, 2.0], [9.0, 9.0, 0.0, 0.0]])
    emb = nn.Embedding(2, 2)
    emb.weight.data = torch.tensor([[3.0, 2.0], [0.0, 0.0]])
    return SimpleNamespace(
        config=SimpleNamespace(
            dim_z=2, layers_z_to_x=layers_z_to_x, batch_embedding_mode=mode, recon_batch="concat"
        ),
        decoder_z_to_x=SimpleNamespace(layers=[SimpleNamespace(weight=nn.Parameter(weight))]),
        batch_emb=emb,
    )


def test_concat_x_shift_norm_is_the_norm_of_W_e_times_e_b():
    # e_A = (3, 2) -> shift (3, 4, 0), norm 5; e_B = 0 -> shift 0
    assert d3.batch_shift_norms(_model(), LEVELS) == {"A": 5.0, "B": 0.0}


def test_concat_x_shift_raises_when_the_decoder_has_hidden_layers():
    with pytest.raises(ValueError, match="layers_z_to_x"):
        d3.concat_x_shift_norms(_model(layers_z_to_x=(16,)), LEVELS)


def test_concat_x_shift_raises_when_levels_and_rows_differ():
    with pytest.raises(ValueError, match="batch levels"):
        d3.concat_x_shift_norms(_model(), ["A"])


@pytest.mark.parametrize("preset", ["model1", "model2", "model3"])
def test_concat_z_presets_have_no_shift(preset):
    mode = PRESETS[preset]["batch_embedding_mode"]
    assert d3.batch_shift_norms(_model(mode=mode), LEVELS) is None


@pytest.mark.parametrize("preset", ["model4", "model6"])
def test_concat_x_presets_have_a_shift(preset):
    mode = PRESETS[preset]["batch_embedding_mode"]
    assert d3.batch_shift_norms(_model(mode=mode), LEVELS) == {"A": 5.0, "B": 0.0}


def _residual_model() -> SimpleNamespace:
    # L = 1, D = 1, G = 2. r([z; e]) = W2 relu([z; e]) with W1 = I, W2 = diag(3, 4), so
    # r = (3 relu(z), 4 relu(e)). e_A = 1, e_B = 0.
    r = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        r[0].weight.copy_(torch.eye(2))
        r[0].bias.zero_()
        r[2].weight.copy_(torch.diag(torch.tensor([3.0, 4.0])))
        r[2].bias.zero_()
    emb = nn.Embedding(2, 1)
    emb.weight.data = torch.tensor([[1.0], [0.0]])
    return SimpleNamespace(
        config=SimpleNamespace(
            dim_z=1, layers_z_to_x=(), batch_embedding_mode="concat_x", recon_batch="residual"
        ),
        decoder_residual=r,
        batch_emb=emb,
    )


_RESIDUAL_Z = torch.tensor([[1.0], [-1.0]])


def test_residual_shift_norm_is_the_mean_over_cells_of_the_norm_of_r_z_e_b():
    # A: r = (3, 4) -> 5 and (0, 4) -> 4, mean 4.5; B: (3, 0) -> 3 and (0, 0) -> 0, mean 1.5
    assert d3.batch_shift_norms(_residual_model(), LEVELS, _RESIDUAL_Z.numpy()) == {
        "A": 4.5,
        "B": 1.5,
    }


def test_residual_centered_shift_norm_drops_the_batch_independent_part_of_r():
    # batch mean per cell: (3 relu(z), 2); r - mean: A (0, 2), B (0, -2) for both cells -> 2, 2
    assert d3.batch_shift_norms_centered(_residual_model(), LEVELS, _RESIDUAL_Z.numpy()) == {
        "A": 2.0,
        "B": 2.0,
    }


def test_residual_centered_shift_norm_without_z_raises():
    with pytest.raises(ValueError, match="needs z"):
        d3.batch_shift_norms_centered(_residual_model(), LEVELS)


@pytest.mark.parametrize("model", [_model(mode="concat_z"), _model()], ids=["concat_z", "concat_x"])
def test_concat_arms_have_no_centered_shift_norm(model):
    assert d3.batch_shift_norms_centered(model, LEVELS, _RESIDUAL_Z.numpy()) is None


def test_residual_shift_norm_without_z_raises():
    with pytest.raises(ValueError, match="needs z"):
        d3.batch_shift_norms(_residual_model(), LEVELS)


def test_residual_shift_raises_when_levels_and_rows_differ():
    with pytest.raises(ValueError, match="batch levels"):
        d3.residual_shift_norms(_residual_model(), ["A"], _RESIDUAL_Z)


def _in_span_model(layers_z_to_x: tuple = ()) -> SimpleNamespace:
    # L = 1, D = 1, G = 2. ρ([z; e]) = W2 relu([z; e]) with W1 = I, W2 = (1, 1), so
    # ρ = relu(z) + relu(e); W = (3, 4)^T with bias c = (100, -100), which the shift must ignore.
    # e_A = 1, e_B = 0. decoder_residual is a decoy: the in-span norm must not use it.
    rho_in_span = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    a = nn.Linear(1, 2)
    decoy = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
    with torch.no_grad():
        rho_in_span[0].weight.copy_(torch.eye(2))
        rho_in_span[0].bias.zero_()
        rho_in_span[2].weight.copy_(torch.tensor([[1.0, 1.0]]))
        rho_in_span[2].bias.zero_()
        a.weight.copy_(torch.tensor([[3.0], [4.0]]))
        a.bias.copy_(torch.tensor([100.0, -100.0]))
    emb = nn.Embedding(2, 1)
    emb.weight.data = torch.tensor([[1.0], [0.0]])
    return SimpleNamespace(
        config=SimpleNamespace(
            dim_z=1,
            layers_z_to_x=layers_z_to_x,
            batch_embedding_mode="concat_x",
            recon_batch="residual_in_span",
        ),
        decoder_residual_in_span=rho_in_span,
        decoder_residual=decoy,
        decoder_z_to_x=SimpleNamespace(layers=[a]),
        batch_emb=emb,
    )


def test_in_span_shift_norm_is_the_mean_over_cells_of_the_norm_of_W_rho():
    # A: ρ = 2 -> W ρ = (6, 8), norm 10; ρ = 1 -> 5; mean 7.5.
    # B: ρ = 1 -> 5; ρ = 0 -> 0; mean 2.5.
    assert d3.batch_shift_norms(_in_span_model(), LEVELS, _RESIDUAL_Z.numpy()) == {
        "A": 7.5,
        "B": 2.5,
    }


def test_in_span_centered_shift_norm_is_W_times_rho_minus_its_batch_mean():
    # batch mean of ρ per cell: 1.5 and 0.5; ρ - mean = +-0.5 for both cells -> ||W 0.5|| = 2.5
    assert d3.batch_shift_norms_centered(_in_span_model(), LEVELS, _RESIDUAL_Z.numpy()) == {
        "A": 2.5,
        "B": 2.5,
    }


def test_in_span_shift_raises_when_the_decoder_has_hidden_layers():
    with pytest.raises(ValueError, match="layers_z_to_x"):
        d3.batch_shift_norms(_in_span_model(layers_z_to_x=(16,)), LEVELS, _RESIDUAL_Z.numpy())


@pytest.mark.parametrize("norms", [d3.batch_shift_norms, d3.batch_shift_norms_centered])
def test_in_span_shift_norm_without_z_raises(norms):
    with pytest.raises(ValueError, match="needs z"):
        norms(_in_span_model(), LEVELS)


def test_in_span_shift_raises_when_levels_and_rows_differ():
    with pytest.raises(ValueError, match="batch levels"):
        d3.residual_shift_norms(_in_span_model(), ["A"], _RESIDUAL_Z)


@pytest.mark.parametrize("preset", d3.PRESET_CHOICES)
def test_default_out_dir_names_dataset_and_preset(preset):
    assert d3.default_out_dir("oetjen5", preset) == SCRIPT.parent / f"1016_oetjen5_d3_{preset}"


def test_preset_choices_are_the_presets_keys():
    assert d3.PRESET_CHOICES == tuple(PRESETS)


@pytest.mark.parametrize(
    "lambdas, tag",
    [
        ({"lambda_c": 0.1}, "lc0p1"),
        ({"lambda_c": 1.0}, "lc1"),
        ({"lambda_c": 10.0}, "lc10"),
        ({"lambda_inv": 0.05}, "li0p05"),
        ({"lambda_r": 2.5}, "lr2p5"),
        ({}, None),
    ],
)
def test_lambda_tag_is_prefix_plus_g_format_with_dot_as_p(lambdas, tag):
    assert d3.lambda_tag(lambdas) == tag


ROOT = Path("/repo/Oetjen")


def _names(lambdas: dict, scan: bool = False, final: bool = False) -> dict:
    return d3.run_names("oetjen5", "model8", 1, lambdas, scan=scan, final=final, root=ROOT)


def test_run_names_without_scan_or_final_keep_the_suite_names():
    assert _names({}) == {
        "mode": None,
        "out_dir": ROOT / "1016_oetjen5_d3_model8",
        "lambda_tag": None,
    }


def test_run_names_for_a_scan_carry_the_lambda_tag():
    assert _names({"lambda_c": 0.1}, scan=True) == {
        "mode": "scan",
        "out_dir": ROOT / "1016_oetjen5_d3_model8_lc0p1",
        "lambda_tag": "lc0p1",
    }


def test_run_names_for_a_final_run_are_untagged_with_tag_final():
    assert _names({"lambda_c": 1.0}, final=True) == {
        "mode": "final",
        "out_dir": ROOT / "1016_oetjen5_d3_model8",
        "lambda_tag": "lc1",
    }


@pytest.mark.parametrize("lambdas", [{}, {"lambda_c": 1.0, "lambda_inv": 1.0}])
def test_scan_without_exactly_one_lambda_raises(lambdas):
    with pytest.raises(ValueError, match="exactly one"):
        _names(lambdas, scan=True)


def test_scan_with_final_raises():
    with pytest.raises(ValueError, match="exclusive"):
        _names({"lambda_c": 1.0}, scan=True, final=True)


def test_lambda_flags_become_config_field_overrides():
    args = d3.build_parser().parse_args(["--seed", "0", "--preset", "native", "--lambda-c", "0.1"])
    assert d3.lambda_overrides(args) == {"lambda_c": 0.1}


@pytest.mark.parametrize(
    "model",
    [_model(mode="concat_z"), _model(), _residual_model(), _in_span_model()],
    ids=["z", "x", "residual", "in_span"],
)
def test_a_model_without_a_table_has_no_norms(model):
    # native builds no batch_emb; nor does a batch arm with n_batches == 0
    model.batch_emb = None
    assert (
        d3.batch_emb_norms(model, LEVELS),
        d3.batch_shift_norms(model, LEVELS),
        d3.batch_shift_norms_centered(model, LEVELS),
    ) == (None, None, None)


# --- batch-inv P04: penalty history meta and the MMD presets -------------------------------


def _trained(penalty_history: dict) -> SimpleNamespace:
    return SimpleNamespace(
        train_losses_=[5.0, 4.0], val_losses_=[7.0, 6.0], penalty_history_=penalty_history
    )


@pytest.mark.parametrize("history", [{}, {"mmd_v": [0.3, 0.1]}])
def test_meta_penalty_history_is_the_model_history(history):
    assert d3.penalty_meta(_trained(history)) == {"penalty_history": history}


@pytest.mark.parametrize("preset", ["native_mmd", "model2_mmd", "model5_mmd"])
def test_lambda_inv_flag_overrides_the_mmd_preset_weight(preset):
    args = d3.build_parser().parse_args(["--seed", "0", "--preset", preset, "--lambda-inv", "10"])
    config = DecipherConfig(**{**PRESETS[preset], **d3.lambda_overrides(args)})
    assert (config.v_invariance, config.lambda_inv) == ("mmd", 10.0)


@pytest.mark.parametrize("preset", ["model7", "model7i"])
def test_lambda_r_flag_overrides_the_model7_preset_weight(preset):
    args = d3.build_parser().parse_args(["--seed", "0", "--preset", preset, "--lambda-r", "10"])
    config = DecipherConfig(**{**PRESETS[preset], **d3.lambda_overrides(args)})
    assert (config.prior_batch, config.lambda_r) == ("residual", 10.0)


_qspec = importlib.util.spec_from_file_location(
    "oetjen_d3_queue", SCRIPT.with_name("1016_oetjen_d3_queue.py")
)
queue = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(queue)
REPO = Path("/main")
RUNTREE = Path("/scratch/runtree_t")
OETJEN = REPO / "Real Data" / "Oetjen Bone Marrow"


def test_queue_runs_the_runtree_driver_with_src_and_out_dir_in_the_main_checkout():
    cmd, log = queue.job_command(
        ["--preset", "model5", "--seed", "2", "--lambda-c", "1", "--scan"], RUNTREE, REPO
    )
    assert (cmd, log) == (
        [
            str(REPO / ".venv" / "bin" / "python"),
            str(RUNTREE / "Real Data" / "Oetjen Bone Marrow" / "1016_oetjen_d3_decipher.py"),
            *["--preset", "model5", "--seed", "2", "--lambda-c", "1", "--scan"],
            *["--src", str(OETJEN / "1016_oetjen5" / "oetjen5_shared.h5ad")],
            *["--out-dir", str(OETJEN / "1016_oetjen5_d3_model5_lc1")],
        ],
        OETJEN / "1016_oetjen5_d3_model5_lc1" / "logs" / "seed2_lc1.log",
    )


def test_queue_resolves_a_job_out_dir_against_the_main_checkout():
    cmd, log = queue.job_command(
        ["--preset", "native", "--seed", "0", "--out-dir", "x"], RUNTREE, REPO
    )
    assert (cmd[-2:], log) == (["--out-dir", str(REPO / "x")], REPO / "x" / "logs" / "seed0.log")


def test_queue_reads_one_job_per_line_skipping_blanks_and_comments():
    text = "# scan\n--preset model5 --seed 0 --lambda-c 0.1 --scan\n\n  --preset native --seed 1\n"
    assert queue.read_jobs(text) == [
        ["--preset", "model5", "--seed", "0", "--lambda-c", "0.1", "--scan"],
        ["--preset", "native", "--seed", "1"],
    ]


# ---- batch-fu P02: model5rz norms, --tagged-final, new presets ----


def _rz_model() -> SimpleNamespace:
    # model5rz: the _model() concat_x decoder (e_b columns kept) plus r_z(z) = Linear(L=2 -> G=3)
    # on z only; passing [z; e_b] to r_z would fail on the shape
    model = _model()
    model.config.recon_batch = "concat_plus_residual_z"
    model.decoder_residual_z = nn.Linear(2, 3)
    return model


_RZ_Z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]).numpy()


def test_model5rz_shift_norm_is_the_norm_of_W_e_times_e_b_as_for_model5():
    assert d3.batch_shift_norms(_rz_model(), LEVELS, _RZ_Z) == {"A": 5.0, "B": 0.0}


def test_model5rz_centered_shift_norm_is_zero_because_r_z_has_no_e_b():
    assert d3.batch_shift_norms_centered(_rz_model(), LEVELS, _RZ_Z) == {"A": 0.0, "B": 0.0}


def test_model5rz_centered_shift_norm_fails_loudly_when_r_z_carries_a_batch_part(monkeypatch):
    model = _rz_model()
    real = d3._residual_per_batch

    def leaky(decipher, levels, z):
        r_z, emb = real(decipher, levels, z)
        return (lambda e: r_z(e) + e.sum()), emb

    monkeypatch.setattr(d3, "_residual_per_batch", leaky)
    with pytest.raises(AssertionError, match="carries a batch part"):
        d3.batch_shift_norms_centered(model, LEVELS, _RZ_Z)


def test_model5rz_centered_shift_norm_is_exactly_zero_for_five_batches_and_large_r_z():
    # sum_b r_z / B leaves a float residue for B = 5 at |r_z| ~ 1e3; the rz path must not
    model = _rz_model()
    model.batch_emb = nn.Embedding(5, model.batch_emb.weight.shape[1])
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.decoder_residual_z.weight.copy_(300 * torch.randn(3, 2, generator=gen))
    z = (10 * torch.randn(200, 2, generator=gen)).numpy()
    levels = ["P", "G", "T", "W", "R"]
    assert d3.batch_shift_norms_centered(model, levels, z) == dict.fromkeys(levels, 0.0)


@pytest.mark.parametrize(
    "lambdas, tag",
    [
        ({"lambda_inv": 100.0}, "li100"),
        ({"lambda_c": 1000.0}, "lc1000"),
        ({"lambda_c": 10000.0}, "lc10000"),
    ],
)
def test_lambda_tag_of_large_values_has_no_exponent(lambdas, tag):
    assert d3.lambda_tag(lambdas) == tag


def _tagged(lambdas: dict, final: bool = True) -> dict:
    return d3.run_names(
        "oetjen5", "model5r_mmd", 2, lambdas, scan=False, final=final, root=ROOT, tagged_final=True
    )


def test_tagged_final_names_carry_the_lambda_tag():
    assert _tagged({"lambda_inv": 100.0}) == {
        "mode": "final",
        "out_dir": ROOT / "1016_oetjen5_d3_model5r_mmd_li100_final",
        "lambda_tag": "li100",
        "tagged_final": "li100",
    }


@pytest.mark.parametrize(
    "lambdas, final",
    [
        ({"lambda_inv": 100.0}, False),
        ({}, True),
        ({"lambda_inv": 100.0, "lambda_c": 1.0}, True),
    ],
    ids=["without_final", "without_lambda", "two_lambdas"],
)
def test_tagged_final_without_final_or_with_other_than_one_lambda_raises(lambdas, final):
    with pytest.raises(ValueError, match="--tagged-final"):
        _tagged(lambdas, final=final)


def test_tagged_final_flag_is_off_by_default_and_parsed_when_given():
    base = ["--seed", "0", "--preset", "model5r_mmd", "--lambda-inv", "100", "--final"]
    parser = d3.build_parser()
    assert (
        parser.parse_args(base).tagged_final,
        parser.parse_args([*base, "--tagged-final"]).tagged_final,
    ) == (
        False,
        True,
    )


def test_queue_names_a_tagged_final_out_dir_and_log_with_the_lambda_tag():
    job = ["--preset", "model8r", "--seed", "1", "--lambda-c", "1000", "--final", "--tagged-final"]
    cmd, log = queue.job_command(job, RUNTREE, REPO)
    out = OETJEN / "1016_oetjen5_d3_model8r_lc1000_final"
    assert (cmd[-2:], log) == (["--out-dir", str(out)], out / "logs" / "seed1_lc1000.log")


@pytest.mark.parametrize(
    "preset, flag, field",
    [("model5r_mmd", "--lambda-inv", "lambda_inv"), ("model8r", "--lambda-c", "lambda_c")],
)
def test_lambda_flags_override_the_new_penalty_presets(preset, flag, field):
    args = d3.build_parser().parse_args(["--seed", "0", "--preset", preset, flag, "100"])
    config = DecipherConfig(**{**PRESETS[preset], **d3.lambda_overrides(args)})
    assert getattr(config, field) == 100.0
