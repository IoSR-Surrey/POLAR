import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "listening"))
os.chdir(ROOT)

import numpy as np
import pytest

import play_icld_ictd as player  # type: ignore[import-not-found]  # noqa: E402  (added to sys.path above)


def test_constant_power_gains_are_normalised() -> None:
    for icld_db in [-20.0, -6.0, 0.0, 6.0, 20.0]:
        left_gain, right_gain = player.constant_power_gains(icld_db)
        assert left_gain**2 + right_gain**2 == pytest.approx(1.0)


def test_positive_icld_makes_left_louder() -> None:
    left_gain, right_gain = player.constant_power_gains(6.0)
    assert left_gain > right_gain
    assert 20.0 * np.log10(left_gain / right_gain) == pytest.approx(6.0)


def test_zero_icld_is_centred() -> None:
    left_gain, right_gain = player.constant_power_gains(0.0)
    assert left_gain == pytest.approx(right_gain)


def test_fractional_delay_shifts_impulse_later() -> None:
    signal = np.zeros(64)
    signal[10] = 1.0
    delayed = player.fractional_delay(signal, 5.0)
    assert int(np.argmax(delayed)) == 15


def test_positive_ictd_delays_right_channel() -> None:
    mono = np.zeros(128)
    mono[20] = 1.0
    stereo = player.render_stereo(mono, 0.0, 0.001, fs=1000)
    left_peak = int(np.argmax(np.abs(stereo[:, 0])))
    right_peak = int(np.argmax(np.abs(stereo[:, 1])))
    assert left_peak == 20
    assert right_peak == 21


def test_negative_ictd_delays_left_channel() -> None:
    mono = np.zeros(128)
    mono[20] = 1.0
    stereo = player.render_stereo(mono, 0.0, -0.001, fs=1000)
    left_peak = int(np.argmax(np.abs(stereo[:, 0])))
    right_peak = int(np.argmax(np.abs(stereo[:, 1])))
    assert left_peak == 21
    assert right_peak == 20


def test_render_stereo_preserves_icld_ratio_after_normalisation() -> None:
    mono = np.random.default_rng(0).standard_normal(2048)
    stereo = player.render_stereo(mono, 6.0, 0.0, fs=48000)
    normalised = player.normalize_for_playback(stereo)
    left_rms = np.sqrt(np.mean(normalised[:, 0].astype(np.float64) ** 2))
    right_rms = np.sqrt(np.mean(normalised[:, 1].astype(np.float64) ** 2))
    assert 20.0 * np.log10(left_rms / right_rms) == pytest.approx(6.0, abs=1e-3)


def test_render_material_without_filters_matches_render_stereo() -> None:
    mono = np.random.default_rng(1).standard_normal(256)
    expected = player.render_stereo(mono, 3.0, 0.0005, fs=8000)
    got = player.render_material(mono, None, 3.0, 0.0005, fs=8000)
    assert np.allclose(got, expected)


def test_render_material_identity_filter_matches_no_filter() -> None:
    mono = np.random.default_rng(2).standard_normal(256)
    identity = np.zeros((2, 5))
    identity[:, 0] = 1.0
    with_filter = player.render_material(mono, identity, 0.0, 0.0, fs=8000)
    without = player.render_material(mono, None, 0.0, 0.0, fs=8000)
    n = min(len(with_filter), len(without))
    assert np.allclose(with_filter[:n], without[:n], atol=1e-6)


def test_render_material_applies_filter_before_cues() -> None:
    mono = np.zeros(16)
    mono[0] = 1.0
    # Left filter passes through; right filter is silent.
    filters = np.zeros((2, 4))
    filters[0, 0] = 1.0
    stereo = player.render_material(mono, filters, 0.0, 0.0, fs=8000)
    assert np.max(np.abs(stereo[:, 0])) > 0.0
    assert np.max(np.abs(stereo[:, 1])) == pytest.approx(0.0)


def test_load_stereo_filters_from_npy(tmp_path) -> None:
    filters = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
    path = tmp_path / "spk_filters.npy"
    np.save(path, filters)
    loaded = player.load_stereo_filters(path)
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded, filters[0])


def test_load_stereo_filters_rejects_non_stereo(tmp_path) -> None:
    path = tmp_path / "spk_filters.npy"
    np.save(path, np.zeros((4, 3)))
    with pytest.raises(ValueError):
        player.load_stereo_filters(path)


def test_select_material_index_in_range() -> None:
    assert player.select_material_index(0, 10) == 0
    assert player.select_material_index(9, 10) == 9


def test_select_material_index_out_of_range_returns_none() -> None:
    assert player.select_material_index(9, 5) is None
    assert player.select_material_index(-1, 10) is None
