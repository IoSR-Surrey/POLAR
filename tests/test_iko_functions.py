import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pytest

from utils.IKO_functions import (
    compute_beam_pattern,
    compute_real_sh_matrix,
    example_horizontal_beam,
    num_hoa_channels,
)


@pytest.mark.parametrize("order, expected_channels", [(0, 1), (1, 4), (2, 9), (3, 16)])
def test_num_hoa_channels(order: int, expected_channels: int) -> None:
    assert num_hoa_channels(order) == expected_channels


def test_compute_real_sh_matrix_shape_and_dtype() -> None:
    azimuths = np.array([0.0, np.pi / 2, np.pi])
    elevations = np.zeros(3)
    Y = compute_real_sh_matrix(order=3, azimuths=azimuths, elevations=elevations)

    assert Y.shape == (3, 16)
    assert Y.dtype == np.float64


def test_compute_real_sh_matrix_rejects_mismatched_azimuth_elevation_shapes() -> None:
    with pytest.raises(AssertionError):
        compute_real_sh_matrix(
            order=1,
            azimuths=np.array([0.0, 1.0]),
            elevations=np.array([0.0]),
        )


def test_omnidirectional_weight_gives_constant_horizontal_pattern() -> None:
    order = 3
    num_points = 72
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    elevations = np.zeros(num_points)
    weights = np.zeros(num_hoa_channels(order))
    weights[0] = 1.0

    pattern = compute_beam_pattern(order, weights, azimuths, elevations)

    assert pattern.shape == (num_points,)
    assert np.std(pattern) == pytest.approx(0.0, abs=1e-12)


def test_single_basis_vector_reproduces_sh_column() -> None:
    order = 2
    azimuths = np.array([0.3, 1.1, 2.4])
    elevations = np.array([0.0, 0.1, -0.2])
    Y = compute_real_sh_matrix(order, azimuths, elevations)

    for idx in range(Y.shape[1]):
        weights = np.zeros(Y.shape[1])
        weights[idx] = 1.0
        pattern = compute_beam_pattern(order, weights, azimuths, elevations)
        np.testing.assert_allclose(pattern, Y[:, idx], rtol=1e-12, atol=1e-12)


def test_compute_beam_pattern_matches_matrix_multiplication() -> None:
    order = 3
    azimuths = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    elevations = np.zeros_like(azimuths)
    weights = np.linspace(-0.5, 0.5, num_hoa_channels(order))
    Y = compute_real_sh_matrix(order, azimuths, elevations)

    pattern = compute_beam_pattern(order, weights, azimuths, elevations)
    np.testing.assert_allclose(pattern, Y @ weights, rtol=1e-12, atol=1e-12)


def test_compute_beam_pattern_rejects_wrong_weight_length() -> None:
    azimuths = np.array([0.0])
    elevations = np.array([0.0])
    with pytest.raises(AssertionError):
        compute_beam_pattern(order=3, weights=np.ones(8), azimuths=azimuths, elevations=elevations)


def test_example_horizontal_beam_peaks_at_target_azimuth() -> None:
    target_azimuth = np.deg2rad(45.0)
    azimuths, elevations, pattern = example_horizontal_beam(
        order=3,
        num_points=360,
        target_azimuth=target_azimuth,
        target_elevation=0.0,
    )

    assert elevations.shape == azimuths.shape
    peak_idx = int(np.argmax(pattern))
    peak_azimuth = azimuths[peak_idx]
    angular_error = np.abs((peak_azimuth - target_azimuth + np.pi) % (2 * np.pi) - np.pi)
    assert angular_error == pytest.approx(0.0, abs=np.deg2rad(2.0))


def test_zero_elevation_horizontal_slice_is_real_valued() -> None:
    azimuths = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    elevations = np.zeros_like(azimuths)
    Y = compute_real_sh_matrix(order=3, azimuths=azimuths, elevations=elevations)

    assert np.all(np.isfinite(Y))
    assert np.max(np.abs(Y.imag)) == 0.0 if np.iscomplexobj(Y) else True
