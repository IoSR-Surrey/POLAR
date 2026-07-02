import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from utils.dsp import hilbert_transform
from utils.miscellaneous import load_npy_to_torch
from utils.peripheral_processing import (
    compute_iacc,
    compute_ild,
    compute_itd,
    estimate_interaural_cues,
    rectify_signals,
)

FS = 48000
N_BANDS = 24
LAG_VALUES = torch.arange(-0.001 * FS, 0.001 * FS + 1, 1, dtype=torch.float32) / FS


@pytest.fixture
def gtfb():
    return load_npy_to_torch("data/banks/gtfb_48k.npy")


def _synthetic_gtfb_signals(batch: int = 1, n_bands: int = N_BANDS, n_samples: int = 512) -> torch.Tensor:
    return torch.randn(batch, n_bands, 2, n_samples)


def test_rectify_signals_splits_low_and_high_bands() -> None:
    gtfb_signals = _synthetic_gtfb_signals()
    split_band = 11

    rectified = rectify_signals(gtfb_signals, split_band=split_band)
    lf_expected = torch.max(
        gtfb_signals[:, :split_band], torch.zeros_like(gtfb_signals[:, :split_band])
    )
    hf_expected = torch.abs(hilbert_transform(gtfb_signals[:, split_band:]))

    assert rectified.shape == gtfb_signals.shape
    assert torch.allclose(rectified[:, :split_band], lf_expected)
    assert torch.allclose(rectified[:, split_band:], hf_expected)
    assert not torch.allclose(rectified[:, :split_band], gtfb_signals[:, :split_band])
    assert not torch.allclose(rectified[:, split_band:], gtfb_signals[:, split_band:])


def test_compute_ild_positive_when_left_is_louder() -> None:
    rectified = torch.zeros(1, N_BANDS, 2, 64)
    rectified[:, :, 0, :] = 2.0
    rectified[:, :, 1, :] = 1.0

    ild = compute_ild(rectified)

    assert ild.shape == (1, N_BANDS)
    assert torch.all(ild > 0.0)


def test_compute_ild_negative_when_right_is_louder() -> None:
    rectified = torch.zeros(1, N_BANDS, 2, 64)
    rectified[:, :, 0, :] = 1.0
    rectified[:, :, 1, :] = 2.0

    ild = compute_ild(rectified)

    assert torch.all(ild < 0.0)


def test_compute_iacc_peak_near_zero_for_coincident_ears() -> None:
    rectified = torch.zeros(1, N_BANDS, 2, 128)
    rectified[:, :, 0, 32] = 1.0
    rectified[:, :, 1, 32] = 1.0

    iacc = compute_iacc(rectified, LAG_VALUES)
    peak_lag_index = iacc[0, 0].argmax().item()
    centre_lag_index = LAG_VALUES.shape[0] // 2

    assert abs(peak_lag_index - centre_lag_index) <= 2


def test_compute_itd_sign_follows_documented_convention() -> None:
    rectified = torch.zeros(1, N_BANDS, 2, 128)
    rectified[:, :, 0, 20] = 1.0
    rectified[:, :, 1, 40] = 1.0

    iacc = compute_iacc(rectified, LAG_VALUES)
    itd = compute_itd(iacc, tau_iacc=1e-2, lag_values=LAG_VALUES)

    assert itd.shape == (1, N_BANDS)
    assert torch.all(itd > 0.0)


def test_estimate_interaural_cues_output_keys_and_shapes(gtfb) -> None:
    binaural = torch.randn(2, 2, 1024)
    cues = estimate_interaural_cues(binaural, gtfb, LAG_VALUES)

    assert set(cues.keys()) == {"itd", "ild", "gtfb_signals", "rectified_signals", "iacc"}
    assert cues["itd"].shape == (2, N_BANDS)
    assert cues["ild"].shape == (2, N_BANDS)
    assert cues["gtfb_signals"].shape[0] == 2
    assert cues["gtfb_signals"].shape[2] == 2


def test_estimate_interaural_cues_matches_matlab_reference(gtfb) -> None:
    ref_gtfb_left = load_npy_to_torch("data/test_data/test_data_llado2025/gtfb_left.npy")
    ref_gtfb_right = load_npy_to_torch("data/test_data/test_data_llado2025/gtfb_right.npy")
    ref_itd = load_npy_to_torch("data/test_data/test_data_llado2025/itd.npy")
    ref_ild = load_npy_to_torch("data/test_data/test_data_llado2025/ild.npy")
    input_data = load_npy_to_torch("data/test_data/test_data_llado2025/input.npy")

    cues = estimate_interaural_cues(input_data, gtfb, LAG_VALUES)

    assert torch.allclose(cues["gtfb_signals"][:, :, 0, :4831], ref_gtfb_left)
    assert torch.allclose(cues["gtfb_signals"][:, :, 1, :4831], ref_gtfb_right)
    assert torch.allclose(cues["itd"], -1 * ref_itd, atol=1e-3)
    assert torch.allclose(cues["ild"], -1 * ref_ild, atol=1e-1)
