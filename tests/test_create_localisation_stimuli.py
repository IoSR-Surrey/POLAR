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

import create_localisation_stimuli as cls  # type: ignore[import-not-found]  # noqa: E402
import localisation_test as lt  # type: ignore[import-not-found]  # noqa: E402


def test_format_target_angle() -> None:
    assert cls.format_target_angle(30) == "+30"
    assert cls.format_target_angle(-15.5) == "-15.5"
    assert cls.format_target_angle(0.0) == "0"
    assert cls.format_target_angle(-0.0) == "-0"


def test_normalize_audio_hits_target_rms() -> None:
    audio = np.array([[0.1, -0.2], [0.05, 0.3]], dtype=np.float64)
    out = cls.normalize_audio(audio, target_rms=0.1)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(0.1, abs=1e-6)
    assert out[1, 1] / out[0, 1] == pytest.approx(audio[1, 1] / audio[0, 1], abs=1e-5)


def test_format_stimulus_filename_roundtrips_parser() -> None:
    name = cls.format_stimulus_filename("2026-08-02-135216", 30, "cello_1")
    assert name == "2026-08-02-135216_+30_cello_1.wav"
    fields = lt.parse_stimulus_filename(name)
    assert fields["optimisation_type"] == "2026-08-02-135216"
    assert fields["target_angle"] == pytest.approx(30.0)
    assert fields["program_material"] == "cello_1"


def test_format_stimulus_filename_sanitizes_underscores() -> None:
    name = cls.format_stimulus_filename("2026-08-02_135216", 0, "cello_1")
    assert name == "2026-08-02-135216_0_cello_1.wav"


def test_infer_optimisation_type_uses_run_timestamp(tmp_path) -> None:
    run = tmp_path / "cagrad" / "2026-01-01_120000"
    run.mkdir(parents=True)
    assert cls.infer_optimisation_type(run) == "2026-01-01-120000"


def test_export_stimuli_writes_parseable_wavs(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    fs = 8000
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()
    mono = (np.random.default_rng(0).standard_normal(fs) * 0.1).astype(np.float32)
    wavfile.write(tracks_dir / "cello_1.wav", fs, mono)

    filters = np.zeros((2, 8), dtype=np.float64)
    filters[:, 0] = 1.0
    jobs = [{
        "output_dir": tmp_path / "cagrad" / "2026-01-01_120000",
        "spk_filters": filters,
        "optimisation_type": "2026-01-01-120000",
        "target_angle": -30.0,
    }]
    tracks = cls.load_audio_tracks(tracks_dir, fs)
    out_dir = tmp_path / "stimuli"
    written = cls.export_stimuli(jobs, tracks, out_dir, fs=fs)
    assert len(written) == 1
    assert written[0].name == "2026-01-01-120000_-30_cello_1.wav"
    fields = lt.parse_stimulus_filename(written[0].name)
    assert fields["target_angle"] == pytest.approx(-30.0)
    audio = lt.load_multichannel_wav(written[0], fs)
    assert audio.shape[1] == 2
    assert audio.shape[0] >= fs
