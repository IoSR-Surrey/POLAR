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
    module_path = ROOT / "scripts" / "IKO_optimisation.py"
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

    binaural, directional_weights = iko.render_binaural_sum(
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

    full_attenuation, _ = iko.render_binaural_sum(
        signal, hrir, hrir, weights, sh_matrix, geometry["delay_samples"], 1.0, DEVICE, DTYPE
    )
    partial_attenuation, _ = iko.render_binaural_sum(
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
    binaural, _ = iko.render_binaural_sum(
        signal, hrir, hrir, weights, sh_matrix, geometry["delay_samples"], 0.8, DEVICE, DTYPE
    )
    loss = binaural.pow(2).mean()
    loss.backward()

    assert weight_model.weight.grad is not None
    assert weight_model.weight.grad.abs().sum().item() > 0.0
