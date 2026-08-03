import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "listening"))
os.chdir(ROOT)

import numpy as np
import pytest
from scipy.io import wavfile

import create_vbap_stimuli as vbap  # type: ignore[import-not-found]  # noqa: E402
import localisation_test as lt  # type: ignore[import-not-found]  # noqa: E402


# Speakers at ±30 deg (same layout as config/hyperpameters.json).
SPK_POS = np.array([
    [np.sqrt(3.0), 1.0],
    [np.sqrt(3.0), -1.0],
], dtype=np.float64)


def test_speaker_angles_match_layout() -> None:
    angles = vbap.speaker_angles_deg(SPK_POS)
    assert angles[0] == pytest.approx(30.0, abs=1e-6)
    assert angles[1] == pytest.approx(-30.0, abs=1e-6)


def test_vbap_gains_center_are_equal_power() -> None:
    gains = vbap.vbap_gains_2d(SPK_POS, 0.0)
    assert gains.shape == (2,)
    assert gains[0] == pytest.approx(gains[1], abs=1e-6)
    assert np.linalg.norm(gains) == pytest.approx(1.0, abs=1e-6)


def test_vbap_gains_at_left_speaker() -> None:
    gains = vbap.vbap_gains_2d(SPK_POS, 30.0)
    assert gains[0] == pytest.approx(1.0, abs=1e-6)
    assert gains[1] == pytest.approx(0.0, abs=1e-6)


def test_vbap_gains_at_right_speaker() -> None:
    gains = vbap.vbap_gains_2d(SPK_POS, -30.0)
    assert gains[0] == pytest.approx(0.0, abs=1e-6)
    assert gains[1] == pytest.approx(1.0, abs=1e-6)


def test_vbap_gains_outside_arc_uses_nearest() -> None:
    gains = vbap.vbap_gains_2d(SPK_POS, 90.0)
    assert gains[0] == pytest.approx(1.0, abs=1e-6)
    assert gains[1] == pytest.approx(0.0, abs=1e-6)


def test_apply_vbap_gains_scales_channels() -> None:
    mono = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    gains = np.array([0.8, 0.6], dtype=np.float64)
    out = vbap.apply_vbap_gains(mono, gains)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], mono * 0.8)
    np.testing.assert_allclose(out[:, 1], mono * 0.6)


def test_export_vbap_stimuli_writes_parseable_wavs(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    fs = 8000
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    mono = (np.random.default_rng(0).standard_normal(fs) * 0.1).astype(np.float32)
    wavfile.write(tracks_dir / "cello_1.wav", fs, mono)
    tracks = [("cello_1", mono.astype(np.float64))]

    out_dir = tmp_path / "stimuli"
    written = vbap.export_vbap_stimuli(
        tracks, out_dir, SPK_POS, target_angles=[0.0, -30.0], fs=fs
    )
    assert len(written) == 2
    assert written[0].name == "vbap_0_cello_1.wav"
    assert written[1].name == "vbap_-30_cello_1.wav"
    fields = lt.parse_stimulus_filename(written[1].name)
    assert fields["optimisation_type"] == "vbap"
    assert fields["target_angle"] == pytest.approx(-30.0)
    audio = lt.load_multichannel_wav(written[1], fs)
    assert audio.shape[1] == 2
    # At -30 deg, almost all energy should be on the right channel (index 1).
    assert np.max(np.abs(audio[:, 1])) > np.max(np.abs(audio[:, 0])) * 10
