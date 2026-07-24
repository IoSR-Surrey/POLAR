import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from utils.IKO_functions import compute_real_sh_matrix, num_hoa_channels

IKO_ORDER = 3
N_CHANNELS = num_hoa_channels(IKO_ORDER)
N_SPEAKERS = 20
DEVICE = "cpu"
DTYPE = torch.float32
SPH_TO_DRIVER_NPY = ROOT / "data" / "IKO" / "sph_to_driver.npy"


def load_iko_brir_optimisation_module():
    module_path = ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py"
    spec = importlib.util.spec_from_file_location("iko_brir_optimisation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iko_brir = load_iko_brir_optimisation_module()


def hoa_flat_index_to_nm(order, channel_idx):
    idx = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if idx == channel_idx:
                return n, m
            idx += 1
    raise IndexError(f"channel_idx {channel_idx} out of range for order {order}")


def one_hot_hoa_weights(n_channels, channel_idx, device=DEVICE, dtype=DTYPE):
    weights = torch.zeros(n_channels, device=device, dtype=dtype)
    weights[channel_idx] = 1.0
    return weights


def driver_energy_coefficient_of_variation(energies):
    energies = np.asarray(energies, dtype=np.float64)
    return float(energies.std() / (energies.mean() + 1e-12))


def normalised_driver_energy_profile(energies):
    energies = np.asarray(energies, dtype=np.float64)
    return energies / (np.linalg.norm(energies) + 1e-12)


def make_synthetic_sph_to_driver(n_channels, n_speakers, n_taps):
    """Idealised SH-to-driver matrix for fingerprint logic checks."""
    sph_to_driver = torch.zeros(n_channels, n_speakers, n_taps, dtype=DTYPE)
    shared_ir = torch.ones(n_taps, dtype=DTYPE)
    for speaker_idx in range(n_speakers):
        sph_to_driver[0, speaker_idx] = shared_ir

    dipole_ir = torch.zeros(n_taps, dtype=DTYPE)
    dipole_ir[0] = 1.0
    for speaker_idx in range(n_speakers):
        gain = 1.0 if speaker_idx < n_speakers // 2 else 0.1
        sph_to_driver[1, speaker_idx] = dipole_ir * gain
        sph_to_driver[3, speaker_idx] = dipole_ir * (0.1 if speaker_idx < n_speakers // 2 else 1.0)
    return sph_to_driver


@pytest.fixture
def synthetic_sph_to_driver():
    return make_synthetic_sph_to_driver(N_CHANNELS, N_SPEAKERS, 32)


@pytest.fixture
def real_sph_to_driver():
    if not SPH_TO_DRIVER_NPY.is_file():
        pytest.skip(f"Missing measured data: {SPH_TO_DRIVER_NPY}")
    return iko_brir.load_sph_to_driver(SPH_TO_DRIVER_NPY, N_CHANNELS, N_SPEAKERS, DEVICE, DTYPE)


@pytest.mark.parametrize("channel_idx", range(N_CHANNELS))
def test_one_hot_weight_selects_sph_to_driver_row(synthetic_sph_to_driver, channel_idx) -> None:
    weights = one_hot_hoa_weights(N_CHANNELS, channel_idx)
    speaker_ir = iko_brir.combine_speaker_impulse_responses(weights, synthetic_sph_to_driver)
    expected = synthetic_sph_to_driver[channel_idx]
    assert torch.allclose(speaker_ir, expected)


def test_one_hot_real_data_matches_sph_to_driver_rows(real_sph_to_driver) -> None:
    for channel_idx in range(N_CHANNELS):
        weights = one_hot_hoa_weights(N_CHANNELS, channel_idx)
        speaker_ir = iko_brir.combine_speaker_impulse_responses(weights, real_sph_to_driver)
        expected = real_sph_to_driver[channel_idx]
        assert torch.allclose(speaker_ir, expected)


def test_omni_fingerprint_is_uniform_on_synthetic_data(synthetic_sph_to_driver) -> None:
    energies = iko_brir.fingerprint_driver_energies(synthetic_sph_to_driver, 0).detach().cpu().numpy()
    assert driver_energy_coefficient_of_variation(energies) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(energies) / np.min(energies)) == pytest.approx(1.0, rel=1e-6)


def test_first_order_fingerprint_is_nonuniform_on_synthetic_data(synthetic_sph_to_driver) -> None:
    omni_cv = driver_energy_coefficient_of_variation(iko_brir.fingerprint_driver_energies(synthetic_sph_to_driver, 0))
    dipole_cv = driver_energy_coefficient_of_variation(iko_brir.fingerprint_driver_energies(synthetic_sph_to_driver, 1))
    assert dipole_cv > omni_cv + 0.1


def test_opposing_first_order_dipoles_differ_on_synthetic_data(synthetic_sph_to_driver) -> None:
    profile_left = normalised_driver_energy_profile(iko_brir.fingerprint_driver_energies(synthetic_sph_to_driver, 1))
    profile_right = normalised_driver_energy_profile(iko_brir.fingerprint_driver_energies(synthetic_sph_to_driver, 3))
    correlation = float(profile_left @ profile_right)
    assert correlation < 0.5


def test_omni_fingerprint_is_uniform_on_measured_data(real_sph_to_driver) -> None:
    energies = iko_brir.fingerprint_driver_energies(real_sph_to_driver, 0).detach().cpu().numpy()
    assert driver_energy_coefficient_of_variation(energies) < 0.2
    assert float(np.max(energies) / np.min(energies)) < 2.0


@pytest.mark.parametrize("channel_idx", [1, 3])
def test_first_order_fingerprint_is_nonuniform_on_measured_data(real_sph_to_driver, channel_idx) -> None:
    energies = iko_brir.fingerprint_driver_energies(real_sph_to_driver, channel_idx).detach().cpu().numpy()
    assert driver_energy_coefficient_of_variation(energies) > 0.5


def test_omni_is_more_uniform_than_first_order_on_measured_data(real_sph_to_driver) -> None:
    omni_cv = driver_energy_coefficient_of_variation(iko_brir.fingerprint_driver_energies(real_sph_to_driver, 0))
    for channel_idx in (1, 3):
        dipole_cv = driver_energy_coefficient_of_variation(iko_brir.fingerprint_driver_energies(real_sph_to_driver, channel_idx))
        assert omni_cv < dipole_cv / 3.0


def test_first_order_horizontal_dipoles_have_distinct_fingerprints(real_sph_to_driver) -> None:
    profile_m_neg1 = normalised_driver_energy_profile(iko_brir.fingerprint_driver_energies(real_sph_to_driver, 1))
    profile_m_pos1 = normalised_driver_energy_profile(iko_brir.fingerprint_driver_energies(real_sph_to_driver, 3))
    correlation = float(profile_m_neg1 @ profile_m_pos1)
    assert correlation < 0.8


def test_measured_one_hot_fingerprints_are_not_all_identical(real_sph_to_driver) -> None:
    profiles = np.stack([
        normalised_driver_energy_profile(iko_brir.fingerprint_driver_energies(real_sph_to_driver, channel_idx))
        for channel_idx in range(N_CHANNELS)
    ])
    correlation = profiles @ profiles.T
    off_diagonal = correlation[~np.eye(N_CHANNELS, dtype=bool)]
    assert float(np.min(off_diagonal)) < 0.99


def test_fingerprint_summary_plot_writes_file(real_sph_to_driver, tmp_path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharey=True)
    for channel_idx, ax in enumerate(axes.ravel()):
        n, m = hoa_flat_index_to_nm(IKO_ORDER, channel_idx)
        energies = iko_brir.fingerprint_driver_energies(real_sph_to_driver, channel_idx).detach().cpu().numpy()
        ax.bar(np.arange(N_SPEAKERS), energies)
        ax.set_title(f"ACN{channel_idx} n={n} m={m:+d}", fontsize=8)
        ax.set_xticks([])
    fig.suptitle("One-hot driver excitation fingerprints")
    fig.tight_layout()
    output_path = tmp_path / "fingerprint_summary.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    assert output_path.is_file()


def test_physical_frame_omni_plus_acn3_peaks_at_iko_back() -> None:
    """omni+ACN3 matches listening: cardioid-like peak at IKO back (π)."""
    weights = torch.zeros(N_CHANNELS, dtype=DTYPE)
    weights[0] = 1.0
    weights[3] = 1.0
    azimuths, pattern = iko_brir.beam_pattern_over_circle(IKO_ORDER, weights, num_points=360)
    peak_az = float(azimuths[np.argmax(np.abs(pattern))])
    assert peak_az == pytest.approx(np.pi, abs=np.deg2rad(1.0))


def test_physical_frame_omni_plus_acn1_peaks_at_iko_right() -> None:
    """omni+ACN1 matches listening: cardioid-like peak at IKO right (3π/2)."""
    weights = torch.zeros(N_CHANNELS, dtype=DTYPE)
    weights[0] = 1.0
    weights[1] = 1.0
    azimuths, pattern = iko_brir.beam_pattern_over_circle(IKO_ORDER, weights, num_points=360)
    peak_az = float(azimuths[np.argmax(np.abs(pattern))])
    assert peak_az == pytest.approx(1.5 * np.pi, abs=np.deg2rad(1.0))


def test_directivity_lobe_uses_physical_frame_listener_direction() -> None:
    """Listener at IKO front prefers omni-ACN3 (physical front) over omni+ACN3 (back)."""
    listener_az = 0.0  # IKO front
    reflector_az = np.pi / 2.0
    sh_matrix_full, listener_mask, reflector_mask = iko_brir.build_pattern_sampling(
        IKO_ORDER, listener_az, reflector_az, 5.0, 360, DEVICE, DTYPE,
    )
    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            IKO_ORDER,
            iko_brir.physical_azimuths_to_math_sh(np.array([listener_az, reflector_az])),
            np.zeros(2),
        )
    ).to(dtype=DTYPE)

    weights_front = torch.zeros(N_CHANNELS, dtype=DTYPE)
    weights_front[0] = 1.0
    weights_front[3] = -1.0
    _, _, _, gains_front, lobe_front, _ = iko_brir.compute_directivity_loss(
        weights_front, sh_matrix_full, listener_mask, reflector_mask, sh_matrix,
    )
    weights_back = weights_front.clone()
    weights_back[3] = 1.0
    _, _, _, gains_back, lobe_back, _ = iko_brir.compute_directivity_loss(
        weights_back, sh_matrix_full, listener_mask, reflector_mask, sh_matrix,
    )
    assert float(gains_front[0]) > float(gains_back[0])
    assert float(lobe_front) > float(lobe_back)
