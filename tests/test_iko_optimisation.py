import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.IKO_functions import compute_real_sh_matrix, num_hoa_channels
from utils.sofa_functions import load_sofa


def load_iko_optimisation_module():
    module_path = ROOT / "scripts" / "IKO" / "IKO_optimisation.py"
    spec = importlib.util.spec_from_file_location("iko_optimisation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iko = load_iko_optimisation_module()

FS = 48000
IKO_ORDER = 3
DEVICE = "cpu"
DTYPE = torch.float32


@pytest.fixture
def sofaile():
    return load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")


@pytest.fixture
def listener_positions():
    return torch.tensor([[0.0, 0.0]], dtype=DTYPE)


@pytest.fixture
def geometry(listener_positions):
    return iko.setup_geometry(
        d=2.0,
        iko_angle_deg=30.0,
        reflector_angle_deg=-30.0,
        listener_positions=listener_positions,
        fs=FS,
    )


def impulse_hrir(length: int = 32) -> torch.Tensor:
    hrir = torch.zeros(1, 2, length, dtype=DTYPE)
    hrir[0, 0, 0] = 1.0
    hrir[0, 1, 0] = 1.0
    return hrir


def test_setup_geometry_returns_expected_fields(geometry) -> None:
    expected_keys = {
        "iko_position",
        "reflector_position",
        "sh_azimuths_rad",
        "sh_elevations_rad",
        "iko_azimuth_deg",
        "reflector_azimuth_deg",
        "delay_samples",
    }
    assert expected_keys.issubset(geometry.keys())
    assert geometry["sh_azimuths_rad"].shape == (2,)
    assert geometry["sh_elevations_rad"].shape == (2,)
    assert geometry["delay_samples"] > 0


def test_setup_geometry_reflection_delay_matches_path_length_difference(listener_positions) -> None:
    geometry = iko.setup_geometry(2.0, 30.0, -30.0, listener_positions, FS)
    iko_position = geometry["iko_position"]
    reflector_position = geometry["reflector_position"]

    direct_dist = torch.norm(iko_position - listener_positions, dim=-1)
    reflect_dist = (
        torch.norm(reflector_position - iko_position, dim=-1)
        + torch.norm(reflector_position - listener_positions, dim=-1)
    )
    expected_delay = int(torch.round((reflect_dist - direct_dist) / 343.0 * FS).item())
    assert geometry["delay_samples"] == expected_delay


def test_sh_weights_to_nm_covers_all_channels() -> None:
    order = 3
    weights = torch.arange(float(num_hoa_channels(order)), dtype=DTYPE)
    nm_values = iko.sh_weights_to_nm(order, weights)

    assert len(nm_values) == (order + 1) ** 2
    for idx, (n, m, value) in enumerate(nm_values):
        assert -n <= m <= n
        assert value.item() == pytest.approx(float(idx))


def test_omni_initialisation_sets_only_zero_order_weight() -> None:
    weight_model = nn.Linear(1, num_hoa_channels(IKO_ORDER), bias=False, dtype=DTYPE)
    weight_model.weight.data.zero_()
    weight_model.weight.data[0, 0] = 1.0

    weights = weight_model(torch.ones(1, 1, dtype=DTYPE)).reshape(-1)
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1:].abs().max().item() == pytest.approx(0.0)


def test_render_binaural_path_delay_aligns_output_lengths() -> None:
    signal = torch.ones(1, 16, dtype=DTYPE)
    hrir = impulse_hrir(length=8)
    delay_samples = 10

    direct = iko.render_binaural_path(signal, hrir, 1.0, 0, delay_samples, DEVICE, DTYPE)
    reflected = iko.render_binaural_path(signal, hrir, 1.0, delay_samples, 0, DEVICE, DTYPE)

    assert direct.shape == reflected.shape
    assert direct.shape[0] == 1
    assert direct.shape[1] == 2


def test_render_binaural_path_reflection_delay_shifts_impulse_peak() -> None:
    signal = torch.zeros(1, 32, dtype=DTYPE)
    signal[0, 0] = 1.0
    hrir = impulse_hrir(length=4)
    delay_samples = 8

    direct = iko.render_binaural_path(signal, hrir, 1.0, 0, delay_samples, DEVICE, DTYPE)
    reflected = iko.render_binaural_path(signal, hrir, 1.0, delay_samples, 0, DEVICE, DTYPE)

    direct_peak = int(direct[0, 0].argmax().item())
    reflect_peak = int(reflected[0, 0].argmax().item())
    assert reflect_peak - direct_peak == delay_samples


def test_render_binaural_sum_applies_directional_gains_from_sh_matrix(geometry) -> None:
    signal = torch.randn(1, 32, dtype=DTYPE)
    hrir = impulse_hrir(length=8)
    weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    weights[0] = 2.0
    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            IKO_ORDER,
            geometry["sh_azimuths_rad"].numpy(),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(dtype=DTYPE)
    expected_directional_weights = sh_matrix @ weights

    binaural, directional_weights, iko_signal, reflector_signal = iko.render_binaural_sum(
        signal,
        hrir,
        hrir,
        weights,
        sh_matrix,
        geometry["delay_samples"],
        reflector_attenuation=0.8,
        device=DEVICE,
        dtype=DTYPE,
    )

    assert binaural.shape[0] == 1
    assert binaural.shape[1] == 2
    assert iko_signal.shape == binaural.shape
    assert reflector_signal.shape == binaural.shape
    torch.testing.assert_close(directional_weights, expected_directional_weights)


def test_render_binaural_sum_reflector_attenuation_scales_reflected_path(geometry) -> None:
    signal = torch.ones(1, 32, dtype=DTYPE)
    hrir = impulse_hrir(length=4)
    weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    weights[0] = 1.0
    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            IKO_ORDER,
            geometry["sh_azimuths_rad"].numpy(),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(dtype=DTYPE)

    full_attenuation, _, _, _ = iko.render_binaural_sum(
        signal, hrir, hrir, weights, sh_matrix, geometry["delay_samples"], 1.0, DEVICE, DTYPE
    )
    partial_attenuation, _, _, _ = iko.render_binaural_sum(
        signal, hrir, hrir, weights, sh_matrix, geometry["delay_samples"], 0.5, DEVICE, DTYPE
    )

    assert not torch.allclose(full_attenuation, partial_attenuation)


def test_get_binaural_hrir_shape(sofaile, listener_positions, geometry) -> None:
    hrir = iko.get_binaural_hrir(
        listener_positions,
        geometry["iko_position"],
        sofaile,
        DEVICE,
        DTYPE,
    )
    assert hrir.shape[0] == 1
    assert hrir.shape[1] == 2
    assert hrir.shape[2] == 256


def test_beam_pattern_over_circle_is_flat_for_omni_weights() -> None:
    weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    weights[0] = 1.0
    _, pattern = iko.beam_pattern_over_circle(IKO_ORDER, weights, num_points=180)
    assert float(np.std(pattern)) == pytest.approx(0.0, abs=1e-10)


def test_build_pattern_sampling_marks_listener_and_reflector_directions(geometry) -> None:
    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])
    sh_matrix_full, lobe_mask = iko.build_pattern_sampling(
        IKO_ORDER, listener_az, reflector_az, 20.0, 360, DEVICE, DTYPE,
    )

    assert sh_matrix_full.shape == (360, num_hoa_channels(IKO_ORDER))
    assert lobe_mask.shape == (360,)
    assert lobe_mask.any()
    assert (~lobe_mask).any()


def test_compute_directivity_loss_matches_manual_components(geometry) -> None:
    weights = torch.tensor([1.0, 0.2, -0.1] + [0.0] * 13, dtype=DTYPE)
    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])
    sh_matrix_full, lobe_mask = iko.build_pattern_sampling(
        IKO_ORDER, listener_az, reflector_az, 20.0, 360, DEVICE, DTYPE,
    )
    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            IKO_ORDER,
            geometry["sh_azimuths_rad"].numpy(),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(dtype=DTYPE)

    sidelobe_loss, directivity_loss, gains, lobe_energy = iko.compute_directivity_loss(
        weights, sh_matrix_full, lobe_mask, sh_matrix,
    )
    pattern = sh_matrix_full @ weights
    expected_sidelobe = pattern[~lobe_mask].pow(2).mean()
    expected_lobe = pattern[lobe_mask].pow(2).mean()
    expected_ratio = expected_sidelobe / (expected_lobe + 1e-6)

    torch.testing.assert_close(gains, sh_matrix @ weights)
    torch.testing.assert_close(sidelobe_loss, expected_sidelobe)
    torch.testing.assert_close(lobe_energy, expected_lobe)
    torch.testing.assert_close(directivity_loss, expected_ratio)


def test_expand_vertical_elevations_deg_mirrors_positive_angles() -> None:
    elevations = iko.expand_vertical_elevations_deg([30.0, 60.0, 90.0])
    np.testing.assert_allclose(elevations, [-90.0, -60.0, -30.0, 30.0, 60.0, 90.0])


def test_compute_vertical_radiation_loss_matches_manual_mean() -> None:
    weights = torch.tensor([1.0, 0.2, -0.1] + [0.0] * 13, dtype=DTYPE)
    sh_matrix_vertical = iko.build_vertical_sampling(IKO_ORDER, np.array([30.0, -30.0]), 24, DEVICE, DTYPE)
    pattern = sh_matrix_vertical @ weights
    expected = pattern.pow(2).mean()
    torch.testing.assert_close(iko.compute_vertical_radiation_loss(weights, sh_matrix_vertical), expected)


def test_vertical_radiation_loss_is_higher_for_vertical_than_horizontal_mode() -> None:
    sh_matrix_vertical = iko.build_vertical_sampling(IKO_ORDER, np.array([60.0, -60.0, 90.0, -90.0]), 72, DEVICE, DTYPE)
    horizontal_weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    horizontal_weights[3] = 1.0  # n=1, m=+1
    vertical_weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    vertical_weights[2] = 1.0  # n=1, m=0

    horizontal_loss = iko.compute_vertical_radiation_loss(horizontal_weights, sh_matrix_vertical)
    vertical_loss = iko.compute_vertical_radiation_loss(vertical_weights, sh_matrix_vertical)
    assert vertical_loss.item() > horizontal_loss.item()


def test_plot_beam_pattern_overlay_writes_file(tmp_path, geometry) -> None:
    weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    weights[0] = 1.0
    weights[4] = 0.5
    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])

    iko.plot_beam_pattern_overlay(
        IKO_ORDER,
        weights,
        listener_az,
        reflector_az,
        tmp_path,
        stage="final",
    )

    assert (tmp_path / "beam_pattern_overlay_final.png").exists()


def test_setup_geometry_sh_azimuths_are_iko_relative_to_look_direction(listener_positions) -> None:
    """Listener/reflector SH azimuths are measured from the IKO look direction, not world +x."""
    geometry = iko.setup_geometry(2.0, 30.0, -30.0, listener_positions, FS, iko_look_direction_deg=180.0)
    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])
    assert listener_az == pytest.approx(np.deg2rad(30.0), abs=0.02)
    assert reflector_az == pytest.approx(np.deg2rad(90.0), abs=0.02)


def test_beam_pattern_overlay_peak_matches_main_plot(geometry) -> None:
    """Overlay and main beam plots must use the same azimuth convention."""
    weights = torch.zeros(num_hoa_channels(IKO_ORDER), dtype=DTYPE)
    weights[0] = 1.0
    weights[1] = -1.0

    azimuths, pattern = iko.beam_pattern_over_circle(IKO_ORDER, weights)
    main_peak_plot_az = float(iko._beam_pattern_plot_azimuths(azimuths)[np.argmax(np.abs(pattern))])

    az_samples = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    sh_matrix = compute_real_sh_matrix(IKO_ORDER, az_samples, np.zeros(360))
    overlay_pattern = sh_matrix @ weights.numpy()
    overlay_peak_plot_az = float(
        iko._beam_pattern_plot_azimuths(az_samples)[np.argmax(np.abs(overlay_pattern))]
    )

    assert main_peak_plot_az == pytest.approx(overlay_peak_plot_az)


def test_weight_model_gradients_flow_through_render_binaural_sum(geometry) -> None:
    signal = torch.randn(1, 32, dtype=DTYPE, requires_grad=False)
    hrir = impulse_hrir(length=8)
    weight_model = nn.Linear(1, num_hoa_channels(IKO_ORDER), bias=False, dtype=DTYPE)
    weight_model.weight.data.zero_()
    weight_model.weight.data[0, 0] = 1.0
    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            IKO_ORDER,
            geometry["sh_azimuths_rad"].numpy(),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(dtype=DTYPE)

    weights = weight_model(torch.ones(1, 1, dtype=DTYPE)).reshape(-1)
    binaural, _, _, _ = iko.render_binaural_sum(
        signal, hrir, hrir, weights, sh_matrix, geometry["delay_samples"], 0.8, DEVICE, DTYPE
    )
    loss = binaural.pow(2).mean()
    loss.backward()

    assert weight_model.weight.grad is not None
    assert weight_model.weight.grad.abs().sum().item() > 0.0
