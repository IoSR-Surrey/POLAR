import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from models.listener_based.ipd.model import IPD
from propagation.model import SingleSourceBinauralFreeFieldPropagation
from utils.miscellaneous import target_angle_for_listeners
from utils.sofa_functions import load_sofa

FS = 48000
IR_LEN = 256
N_FFT = 4096
SIGNAL_LEN = 1024


@pytest.fixture
def sofaile():
    return load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")


@pytest.fixture
def lstn_pos():
    return torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]])


@pytest.fixture
def target_angles(lstn_pos):
    target_pos = torch.tensor([2.0, 0.0])
    return target_angle_for_listeners(lstn_pos, target_pos).unsqueeze(-1)


@pytest.fixture
def single_target_angle():
    return torch.tensor([[0.0]])


@pytest.fixture
def single_listener_model(sofaile, single_target_angle) -> IPD:
    return IPD(FS, sofaile, single_target_angle)


@pytest.fixture
def model(sofaile, target_angles) -> IPD:
    return IPD(FS, sofaile, target_angles)


def _mono_reference(batch: int, signal_len: int = SIGNAL_LEN) -> torch.Tensor:
    return torch.randn(batch, 1, signal_len)


def _binaural_from_reference(model: IPD, reference_signals: torch.Tensor) -> torch.Tensor:
    return model.compute_binaural_reference_signals(reference_signals)


def test_init_sets_expected_attributes(model: IPD, target_angles) -> None:
    assert model.model_type == "listener_based"
    assert model.fs == FS
    assert model.n_fft == N_FFT
    assert model.erb_weighting is False
    assert model.use_mse is False
    assert torch.equal(model.target_angles, target_angles)
    assert model.target_hrirs.shape == (len(target_angles), 2, IR_LEN)


def test_erb_frequency_bin_weights_are_normalised(model: IPD) -> None:
    weights = model.erb_frequency_bin_weights

    assert weights.shape == (N_FFT // 2 + 1,)
    assert weights.sum().item() == pytest.approx(1.0)
    assert torch.all(weights >= 0.0)


def test_freq_bin_mask_respects_band_limits(model: IPD) -> None:
    masked_frequencies = model.fft_frequencies[model.freq_bin_mask]

    assert masked_frequencies.min().item() >= model.f_min
    assert masked_frequencies.max().item() <= model.f_max
    assert masked_frequencies.numel() > 0


def test_mask_empty_raises(sofaile, single_target_angle) -> None:
    with pytest.raises(ValueError, match="No FFT bins"):
        IPD(FS, sofaile, single_target_angle, f_min=8000.0, f_max=200.0)


def test_compute_binaural_reference_signals_shape(model: IPD) -> None:
    n_stim = 2
    n_listeners = len(model.target_angles)
    reference_signals = _mono_reference(n_stim)
    binaural_reference = model.compute_binaural_reference_signals(reference_signals)

    assert binaural_reference.shape == (
        n_stim * n_listeners,
        2,
        SIGNAL_LEN + IR_LEN - 1,
    )


def test_forward_output_shape(model: IPD) -> None:
    n_stim = 2
    n_listeners = len(model.target_angles)
    reference_signals = _mono_reference(n_stim)
    binaural_signals = _binaural_from_reference(model, reference_signals)
    result = model(binaural_signals, reference_signals)

    assert result["output"].shape == (n_stim * n_listeners,)


def test_forward_returns_expected_keys(single_listener_model: IPD) -> None:
    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals)
    result = single_listener_model(binaural_signals, reference_signals)

    assert set(result.keys()) == {"output", "kwargs"}
    assert set(result["kwargs"].keys()) == {"binaural_reference_signals"}


def test_identical_binaural_signals_have_zero_ipd(single_listener_model: IPD) -> None:
    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals)
    result = single_listener_model(binaural_signals, reference_signals)

    assert result["output"].item() == pytest.approx(0.0, abs=1e-5)


def test_mismatched_binaural_signals_have_positive_ipd(single_listener_model: IPD) -> None:
    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals)
    delayed = torch.roll(binaural_signals, shifts=3, dims=-1)
    delayed[:, 1, :] = binaural_signals[:, 1, :]
    result = single_listener_model(delayed, reference_signals)

    assert result["output"].item() > 0.0


def test_ipd_is_non_negative(model: IPD) -> None:
    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(model, reference_signals)
    result = model(binaural_signals, reference_signals)

    assert torch.all(result["output"] >= 0.0)


def test_ipd_bounded_by_pi(model: IPD) -> None:
    torch.manual_seed(0)
    reference_signals = _mono_reference(1)
    binaural_signals = torch.randn_like(_binaural_from_reference(model, reference_signals))
    result = model(binaural_signals, reference_signals)

    assert torch.all(result["output"] <= float(torch.pi) + 1e-5)


def test_erb_weighting_changes_score(sofaile, single_target_angle) -> None:
    torch.manual_seed(0)
    model_off = IPD(FS, sofaile, single_target_angle, erb_weighting=False)
    model_on = IPD(FS, sofaile, single_target_angle, erb_weighting=True)

    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(model_off, reference_signals)
    noisy = binaural_signals + 0.1 * torch.randn_like(binaural_signals)

    out_off = model_off(noisy, reference_signals)["output"]
    out_on = model_on(noisy, reference_signals)["output"]

    assert out_off.shape == out_on.shape
    assert not torch.allclose(out_off, out_on)


def test_use_mse_changes_score(sofaile, single_target_angle) -> None:
    torch.manual_seed(0)
    model_mae = IPD(FS, sofaile, single_target_angle, use_mse=False)
    model_mse = IPD(FS, sofaile, single_target_angle, use_mse=True)

    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(model_mae, reference_signals)
    noisy = binaural_signals + 0.1 * torch.randn_like(binaural_signals)

    out_mae = model_mae(noisy, reference_signals)["output"]
    out_mse = model_mse(noisy, reference_signals)["output"]

    assert not torch.allclose(out_mae, out_mse)


def test_ipd_error_matches_forward(model: IPD) -> None:
    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(model, reference_signals)
    binaural_reference = _binaural_from_reference(model, reference_signals)

    direct = model.interaural_phase_difference_error(binaural_signals, binaural_reference)
    via_forward = model(binaural_signals, reference_signals)["output"]

    assert torch.allclose(direct, via_forward)


def test_reference_rendered_through_target_hrirs_matches_propagation(sofaile, lstn_pos) -> None:
    target_pos = torch.tensor([[2.0, 0.0]])
    target_angles = target_angle_for_listeners(lstn_pos, target_pos.squeeze(0)).unsqueeze(-1)
    model = IPD(FS, sofaile, target_angles)

    stim = torch.ones(1, 1, SIGNAL_LEN)
    prop = SingleSourceBinauralFreeFieldPropagation(lstn_pos, target_pos, sofaile, fs=FS)
    propagated = prop(stim, torch.ones(1, 1, 1)).squeeze(0)

    rendered = model.compute_binaural_reference_signals(stim)

    min_len = min(propagated.shape[-1], rendered.shape[-1])
    assert torch.allclose(
        propagated[..., :min_len],
        rendered[..., :min_len],
        atol=1e-4,
    )


def test_forward_is_differentiable(single_listener_model: IPD) -> None:
    reference_signals = _mono_reference(1, signal_len=256)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals).detach()
    binaural_signals = binaural_signals + 0.01 * torch.randn_like(binaural_signals)
    binaural_signals.requires_grad_(True)

    result = single_listener_model(binaural_signals, reference_signals)
    result["output"].sum().backward()

    assert binaural_signals.grad is not None
    assert torch.any(binaural_signals.grad != 0.0)


def test_allows_different_time_lengths(single_listener_model: IPD) -> None:
    reference_signals = _mono_reference(1, signal_len=128)
    binaural_reference = _binaural_from_reference(single_listener_model, reference_signals)
    longer = torch.nn.functional.pad(binaural_reference, (0, 64))

    result = single_listener_model.interaural_phase_difference_error(longer, binaural_reference)

    assert result.shape == (1,)
    assert torch.all(result >= 0.0)
