import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from models.listener_based.LSD.model import LSD
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
def single_listener_model(sofaile, single_target_angle) -> LSD:
    return LSD(FS, sofaile, single_target_angle)


@pytest.fixture
def model(sofaile, target_angles) -> LSD:
    return LSD(FS, sofaile, target_angles)


def _mono_reference(batch: int, signal_len: int = SIGNAL_LEN) -> torch.Tensor:
    return torch.randn(batch, 1, signal_len)


def _reference_for_model(model: LSD, batch: int = 1, signal_len: int = SIGNAL_LEN) -> torch.Tensor:
    return _mono_reference(batch, signal_len)


def _binaural_from_reference(model: LSD, reference_signals: torch.Tensor) -> torch.Tensor:
    return model.compute_binaural_reference_signals(reference_signals)


def test_init_sets_expected_attributes(model: LSD, target_angles) -> None:
    assert model.model_type == "listener_based"
    assert model.fs == FS
    assert model.n_fft == N_FFT
    assert model.erb_weighting is True
    assert torch.equal(model.target_angles, target_angles)
    assert model.target_hrirs.shape == (len(target_angles), 2, IR_LEN)


def test_erb_frequency_bin_weights_are_normalised(model: LSD) -> None:
    weights = model.erb_frequency_bin_weights

    assert weights.shape == (N_FFT // 2 + 1,)
    assert weights.sum().item() == pytest.approx(1.0)
    assert torch.all(weights >= 0.0)


def test_freq_bin_mask_respects_band_limits(model: LSD) -> None:
    masked_frequencies = model.fft_frequencies[model.freq_bin_mask]

    assert masked_frequencies.min().item() >= model.f_min
    assert masked_frequencies.max().item() <= model.f_max
    assert masked_frequencies.numel() > 0


def test_compute_binaural_reference_signals_shape(model: LSD) -> None:
    n_stim = 2
    n_listeners = len(model.target_angles)
    reference_signals = _mono_reference(n_stim)
    binaural_reference = model.compute_binaural_reference_signals(reference_signals)

    assert binaural_reference.shape == (
        n_stim * n_listeners,
        2,
        SIGNAL_LEN + IR_LEN - 1,
    )


def test_forward_output_shape(model: LSD) -> None:
    n_stim = 2
    n_listeners = len(model.target_angles)
    reference_signals = _mono_reference(n_stim)
    binaural_signals = _binaural_from_reference(model, reference_signals)
    result = model(binaural_signals, reference_signals)

    assert result["output"].shape == (n_stim * n_listeners,)


def test_forward_returns_expected_keys(single_listener_model: LSD) -> None:
    reference_signals = _reference_for_model(single_listener_model)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals)
    result = single_listener_model(binaural_signals, reference_signals)

    assert set(result.keys()) == {"output", "kwargs"}
    assert set(result["kwargs"].keys()) == {"binaural_reference_signals"}


def test_identical_binaural_signals_have_zero_lsd(single_listener_model: LSD) -> None:
    reference_signals = _reference_for_model(single_listener_model)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals)
    result = single_listener_model(binaural_signals, reference_signals)

    assert result["output"].item() == pytest.approx(0.0, abs=1e-5)


def test_mismatched_binaural_signals_have_positive_lsd(single_listener_model: LSD) -> None:
    reference_signals = _reference_for_model(single_listener_model)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals)
    noisy_binaural = binaural_signals + 0.1 * torch.randn_like(binaural_signals)
    result = single_listener_model(noisy_binaural, reference_signals)

    assert result["output"].item() > 0.0


def test_lsd_is_non_negative(model: LSD) -> None:
    reference_signals = _mono_reference(1)
    binaural_signals = _binaural_from_reference(model, reference_signals)
    result = model(binaural_signals, reference_signals)

    assert torch.all(result["output"] >= 0.0)


def test_erb_weighting_can_be_disabled(sofaile, target_angles) -> None:
    model = LSD(FS, sofaile, target_angles, erb_weighting=False)
    reference_signals = _reference_for_model(model)
    binaural_signals = _binaural_from_reference(model, reference_signals)

    result = model(binaural_signals, reference_signals)

    assert result["output"].shape == (len(target_angles),)
    assert torch.all(result["output"] >= 0.0)


def test_log_spectral_distortion_matches_forward(model: LSD) -> None:
    reference_signals = _reference_for_model(model)
    binaural_signals = _binaural_from_reference(model, reference_signals)
    binaural_reference = _binaural_from_reference(model, reference_signals)

    direct = model.log_spectral_distortion(binaural_signals, binaural_reference)
    via_forward = model(binaural_signals, reference_signals)["output"]

    assert torch.allclose(direct, via_forward)


def test_reference_rendered_through_target_hrirs_matches_propagation(sofaile, lstn_pos) -> None:
    target_pos = torch.tensor([[2.0, 0.0]])
    target_angles = target_angle_for_listeners(lstn_pos, target_pos.squeeze(0)).unsqueeze(-1)
    model = LSD(FS, sofaile, target_angles)

    stim = torch.ones(1, 1, SIGNAL_LEN)
    prop = SingleSourceBinauralFreeFieldPropagation(lstn_pos, target_pos, sofaile, fs=FS)
    propagated = prop(stim, torch.ones(1, 1, 1)).squeeze(0)

    reference_signals = stim
    rendered = model.compute_binaural_reference_signals(reference_signals)

    min_len = min(propagated.shape[-1], rendered.shape[-1])
    assert torch.allclose(
        propagated[..., :min_len],
        rendered[..., :min_len],
        atol=1e-4,
    )


def test_forward_is_differentiable(single_listener_model: LSD) -> None:
    reference_signals = _reference_for_model(single_listener_model, signal_len=256)
    binaural_signals = _binaural_from_reference(single_listener_model, reference_signals).detach()
    binaural_signals = binaural_signals + 0.01 * torch.randn_like(binaural_signals)
    binaural_signals.requires_grad_(True)

    result = single_listener_model(binaural_signals, reference_signals)
    result["output"].sum().backward()

    assert binaural_signals.grad is not None
    assert torch.any(binaural_signals.grad != 0.0)


def test_long_signals_use_fft_length_at_least_sig_len(single_listener_model: LSD) -> None:
    sig_len = N_FFT + 500
    binaural_a = torch.zeros(1, 2, sig_len)
    binaural_b = torch.zeros(1, 2, sig_len)
    binaural_a[..., -16:] = 1.0

    same = single_listener_model.log_spectral_distortion(binaural_a, binaural_a)
    diff = single_listener_model.log_spectral_distortion(binaural_a, binaural_b)

    assert same.item() == pytest.approx(0.0, abs=1e-5)
    assert diff.item() > 0.0
