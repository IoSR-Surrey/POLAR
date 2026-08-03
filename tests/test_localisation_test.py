import csv
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

import localisation_test as lt  # type: ignore[import-not-found]  # noqa: E402  (added to sys.path above)


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def test_parse_stimulus_filename_basic() -> None:
    fields = lt.parse_stimulus_filename("2026-08-02-135216_+30.wav")
    assert fields["optimisation_type"] == "2026-08-02-135216"
    assert fields["target_angle"] == pytest.approx(30.0)
    assert fields["program_material"] == ""


def test_parse_stimulus_filename_with_program_material() -> None:
    fields = lt.parse_stimulus_filename("2026-08-02-135216_+30_cello_1.wav")
    assert fields["target_angle"] == pytest.approx(30.0)
    assert fields["program_material"] == "cello_1"


def test_parse_stimulus_filename_negative_angle() -> None:
    fields = lt.parse_stimulus_filename("2026-08-02-135216_-15.wav")
    assert fields["target_angle"] == pytest.approx(-15.0)


def test_parse_stimulus_filename_negative_zero() -> None:
    fields = lt.parse_stimulus_filename("polar1_-0_cello_1.wav")
    assert fields["optimisation_type"] == "polar1"
    assert fields["target_angle"] == pytest.approx(0.0)
    assert fields["program_material"] == "cello_1"
    # Keep the sign bit so -0 can be distinguished from +0 if needed.
    import math
    assert math.copysign(1.0, fields["target_angle"]) < 0


def test_parse_stimulus_filename_vbap() -> None:
    fields = lt.parse_stimulus_filename("vbap_+15_cello_1.wav")
    assert fields["optimisation_type"] == "vbap"
    assert fields["target_angle"] == pytest.approx(15.0)
    assert fields["program_material"] == "cello_1"


def test_parse_stimulus_filename_wrong_field_count() -> None:
    with pytest.raises(ValueError):
        lt.parse_stimulus_filename("cagrad.wav")


def test_parse_stimulus_filename_bad_angle() -> None:
    with pytest.raises(ValueError):
        lt.parse_stimulus_filename("cagrad_left.wav")


def test_discover_stimuli_loads_valid_wavs(tmp_path) -> None:
    fs = 8000
    silence = np.zeros((fs, 2), dtype=np.float32)
    for name in [
        "2026-08-02-135216_+30.wav",
        "2026-08-02-135216_-30_cello_1.wav",
        "notes.txt",
    ]:
        if name.endswith(".wav"):
            wavfile.write(tmp_path / name, fs, silence)
        else:
            (tmp_path / name).write_text("ignore me")

    stimuli = lt.discover_stimuli(tmp_path)
    names = sorted(s["filename"] for s in stimuli)
    assert names == [
        "2026-08-02-135216_+30.wav",
        "2026-08-02-135216_-30_cello_1.wav",
    ]


def test_discover_stimuli_skips_malformed(tmp_path) -> None:
    fs = 8000
    silence = np.zeros((fs, 1), dtype=np.float32)
    wavfile.write(tmp_path / "bad_name.wav", fs, silence)
    wavfile.write(tmp_path / "2026-08-02-135216_0.wav", fs, silence)
    stimuli = lt.discover_stimuli(tmp_path)
    assert [s["filename"] for s in stimuli] == ["2026-08-02-135216_0.wav"]


def test_build_results_filename_from_config() -> None:
    name = lt.build_results_filename("VL0", "P03", timestamp="20260802-185500")
    assert name == "localisation_results_VL0_P03_20260802-185500.csv"


def test_default_results_path_uses_responses_dir() -> None:
    path = lt.default_results_path("seat A", "P/1", timestamp="20260802-185500")
    assert path.parent == lt.DEFAULT_RESULTS_DIR
    assert path.name == "localisation_results_seat-A_P-1_20260802-185500.csv"


def test_resolve_results_path_respects_override() -> None:
    state = {
        "seat": "VL0",
        "participant": "P03",
        "results_path": "C:/tmp/custom.csv",
        "results_overridden": True,
    }
    assert lt.resolve_results_path(state) == Path("C:/tmp/custom.csv")


def test_expand_stimuli_with_repetitions() -> None:
    stimuli = [{"filename": "a.wav"}, {"filename": "b.wav"}]
    expanded = lt.expand_stimuli_with_repetitions(stimuli, 3)
    assert len(expanded) == 6
    assert [s["filename"] for s in expanded] == ["a.wav", "b.wav", "a.wav", "b.wav", "a.wav", "b.wav"]


def test_expand_stimuli_with_repetitions_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        lt.expand_stimuli_with_repetitions([{"filename": "a.wav"}], 0)


# --------------------------------------------------------------------------- #
# Angle geometry
# --------------------------------------------------------------------------- #
def test_wrap180_wraps_into_range() -> None:
    assert lt.wrap180(190.0) == pytest.approx(-170.0)
    assert lt.wrap180(-190.0) == pytest.approx(170.0)
    assert lt.wrap180(0.0) == pytest.approx(0.0)
    assert lt.wrap180(360.0) == pytest.approx(0.0)


def test_signed_error_wraps_across_boundary() -> None:
    assert lt.signed_error(170.0, -170.0) == pytest.approx(-20.0)
    assert lt.signed_error(-170.0, 170.0) == pytest.approx(20.0)
    assert lt.signed_error(30.0, 20.0) == pytest.approx(10.0)


def test_click_to_azimuth_cardinpoints() -> None:
    cx, cy = 400, 400
    # Front = straight up on screen (smaller y).
    assert lt.click_to_azimuth(cx, cy - 100, cx, cy) == pytest.approx(0.0)
    # Left (+90) = screen left (smaller x).
    assert lt.click_to_azimuth(cx - 100, cy, cx, cy) == pytest.approx(90.0)
    # Right (-90) = screen right (larger x).
    assert lt.click_to_azimuth(cx + 100, cy, cx, cy) == pytest.approx(-90.0)
    # Back (180) = straight down.
    assert abs(lt.click_to_azimuth(cx, cy + 100, cx, cy)) == pytest.approx(180.0)


def test_click_to_azimuth_centre_returns_none() -> None:
    assert lt.click_to_azimuth(400, 400, 400, 400) is None


def test_azimuth_screen_roundtrip() -> None:
    cx, cy, r = 400, 400, 300
    for angle in [-150.0, -90.0, -30.0, 0.0, 45.0, 90.0, 150.0]:
        x, y = lt.azimuth_to_screen(angle, cx, cy, r)
        recovered = lt.click_to_azimuth(x, y, cx, cy)
        assert lt.signed_error(recovered, angle) == pytest.approx(0.0, abs=0.5)


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def test_normalize_for_playback_hits_target_rms() -> None:
    audio = np.array([[0.1, -0.2], [0.05, 0.3]], dtype=np.float64)
    out = lt.normalize_for_playback(audio, target_rms=0.1)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(0.1, abs=1e-6)
    # Inter-channel ratio preserved.
    assert out[1, 1] / out[0, 1] == pytest.approx(audio[1, 1] / audio[0, 1], abs=1e-5)


def test_load_multichannel_wav_preserves_channels(tmp_path) -> None:
    fs = 8000
    data = (np.random.default_rng(0).standard_normal((fs, 4)) * 0.1).astype(np.float32)
    path = tmp_path / "2026-08-02-135216_0.wav"
    wavfile.write(path, fs, data)
    loaded = lt.load_multichannel_wav(path, fs)
    assert loaded.shape == (fs, 4)


def test_load_multichannel_wav_resamples(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    fs = 8000
    data = (np.random.default_rng(1).standard_normal((fs, 2)) * 0.1).astype(np.float32)
    path = tmp_path / "2026-08-02-135216_0.wav"
    wavfile.write(path, fs, data)
    loaded = lt.load_multichannel_wav(path, 16000)
    assert loaded.shape[1] == 2
    assert loaded.shape[0] == pytest.approx(fs * 2, rel=0.01)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_compute_group_stats_values() -> None:
    stats = lt.compute_group_stats([10.0, -10.0, 10.0, -10.0])
    assert stats["n"] == 4
    assert stats["mean_signed_error"] == pytest.approx(0.0)
    assert stats["mean_abs_error"] == pytest.approx(10.0)
    assert stats["rmse"] == pytest.approx(10.0)


def test_compute_group_stats_empty() -> None:
    stats = lt.compute_group_stats([])
    assert stats["n"] == 0
    assert np.isnan(stats["mean_abs_error"])


def test_aggregate_stats_groups_by_type() -> None:
    rows = [
        {"signed_error": "4", "optimisation_type": "cagrad", "seat_id": "VL0"},
        {"signed_error": "-6", "optimisation_type": "cagrad", "seat_id": "VL0"},
        {"signed_error": "22", "optimisation_type": "lsd", "seat_id": "VL0"},
        {"signed_error": "-18", "optimisation_type": "lsd", "seat_id": "VL0"},
    ]
    result = lt.aggregate_stats(rows)
    assert set(result["by_optimisation_type"].keys()) == {"cagrad", "lsd"}
    assert result["by_optimisation_type"]["lsd"]["mean_abs_error"] == pytest.approx(20.0)
    assert "anova_abs_error_by_type" in result
    assert "cagrad" in result["bias_ttest_by_type"]


# --------------------------------------------------------------------------- #
# CSV round-trip + analysis
# --------------------------------------------------------------------------- #
def test_append_trial_row_writes_header_once(tmp_path) -> None:
    results = tmp_path / "results.csv"
    row = {col: 0 for col in lt.CSV_COLUMNS}
    row.update({"optimisation_type": "cagrad", "seat_id": "VL0", "signed_error": 5.0})
    lt.append_trial_row(results, row)
    lt.append_trial_row(results, row)
    with open(results, newline="", encoding="utf-8") as f:
        lines = list(csv.reader(f))
    assert lines[0] == lt.CSV_COLUMNS
    assert len(lines) == 3  # header + two rows


def test_analyse_csv_writes_outputs(tmp_path) -> None:
    results = tmp_path / "results.csv"
    rng = np.random.default_rng(0)
    for i in range(12):
        opt = "cagrad" if i % 2 == 0 else "lsd"
        target = float(rng.choice([-30.0, 0.0, 30.0]))
        reported = target + float(rng.normal(0, 5))
        err = lt.signed_error(reported, target)
        material = "cello_1" if i % 2 == 0 else "brass_2"
        row = {
            "timestamp": "2026-01-01T00:00:00",
            "seat_id": "VL0",
            "participant_id": "P03",
            "filename": f"2026-08-02-135216_{int(target)}_{material}.wav",
            "optimisation_type": opt,
            "program_material": material,
            "target_angle": target,
            "reported_angle": round(reported, 2),
            "signed_error": round(err, 2),
            "abs_error": round(abs(err), 2),
            "n_replays": 0,
            "trial_index": i,
            "n_trials": 12,
        }
        lt.append_trial_row(results, row)

    analysis = lt.analyse_csv(results)
    assert analysis["stats_path"].exists()
    assert analysis["scatter_path"].exists()
    assert analysis["error_path"].exists()
    assert analysis["stats"]["overall"]["n"] == 12
