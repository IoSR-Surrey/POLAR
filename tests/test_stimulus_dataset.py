import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pytest
import torch

from datasets.stimulus_dataset.dataset import (
    BandpassWhiteNoiseDataset,
    get_stim_loader,
    _erb_bandwidth,
    _hz_to_erb_rate,
)

FS = 48000
N_BANDS = 8
SAMPLES_PER_BAND = 2
STIMULUS_LENGTH_MS = 500
RAMP_TIME_MS = 10


@pytest.fixture
def dataset():
    return BandpassWhiteNoiseDataset(
        fs=FS,
        n_bands=N_BANDS,
        n_samples_per_band=SAMPLES_PER_BAND,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
    )


def _power_spectrum(signal, fs):
    n_fft = len(signal)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    power = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
    return freqs, power


def _mean_power_in_range(freqs, power, f_low, f_high):
    mask = (freqs >= f_low) & (freqs <= f_high)
    assert mask.any(), f"No FFT bins in [{f_low}, {f_high}] Hz"
    return power[mask].mean()


def _first_sample_for_band(dataset, band_idx):
    return dataset[band_idx * dataset.n_samples_per_band]


def test_dataset_length_and_shape(dataset) -> None:
    assert len(dataset) == N_BANDS * SAMPLES_PER_BAND
    assert dataset.stimulus.shape == (len(dataset), 1, dataset.n_samples)
    assert dataset[0].shape == (1, dataset.n_samples)
    assert dataset[0].dtype == torch.float32


def test_stimulus_is_unit_rms(dataset) -> None:
    for idx in range(len(dataset)):
        stim = dataset[idx, 0].numpy()
        rms = np.sqrt(np.mean(stim ** 2))
        assert rms == pytest.approx(1.0, rel=1e-4)


def test_ramp_attenuates_onset_and_offset(dataset) -> None:
    # Use a mid-frequency band so the ramp is not confounded by very slow oscillations.
    stim = _first_sample_for_band(dataset, dataset.n_bands // 2)[0].numpy()
    assert abs(stim[0]) < 1e-4
    assert abs(stim[-1]) < 1e-4

    ramp = dataset.ramp_samples
    onset_power = np.mean(stim[: ramp // 2] ** 2)
    steady_power = np.mean(stim[ramp : -ramp] ** 2)
    assert onset_power < 0.1 * steady_power


def test_samples_within_band_differ(dataset) -> None:
    band_0_samples = [dataset[i] for i in range(SAMPLES_PER_BAND)]
    assert not torch.allclose(band_0_samples[0], band_0_samples[1])


def test_different_bands_produce_different_signals(dataset) -> None:
    band_0 = _first_sample_for_band(dataset, 0)
    band_1 = _first_sample_for_band(dataset, 1)
    assert not torch.allclose(band_0, band_1)


def test_generation_is_reproducible() -> None:
    kwargs = dict(
        fs=FS,
        n_bands=N_BANDS,
        n_samples_per_band=1,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
    )
    dataset_a = BandpassWhiteNoiseDataset(**kwargs)
    dataset_b = BandpassWhiteNoiseDataset(**kwargs)
    assert torch.equal(dataset_a.stimulus, dataset_b.stimulus)


def test_raises_when_ramp_exceeds_half_stimulus_length() -> None:
    with pytest.raises(ValueError, match="must exceed twice ramp_time_ms"):
        BandpassWhiteNoiseDataset(
            fs=FS,
            n_bands=4,
            n_samples_per_band=1,
            stimulus_length_ms=10,
            ramp_time_ms=6,
        )


def test_each_band_attenuates_out_of_band_frequencies(dataset) -> None:
    low_freqs, high_freqs = dataset._band_edges()

    for band_idx in range(dataset.n_bands):
        stim = _first_sample_for_band(dataset, band_idx)[0].numpy()
        freqs, power = _power_spectrum(stim, dataset.fs)

        low = low_freqs[band_idx]
        high = high_freqs[band_idx]
        bandwidth = high - low
        margin = max(bandwidth * 0.25, 50.0)

        in_band_power = _mean_power_in_range(freqs, power, low, high)

        below_high = max(low - margin, dataset.f_min)
        if below_high > dataset.f_min:
            below_band_power = _mean_power_in_range(freqs, power, dataset.f_min, below_high)
            assert in_band_power > 30.0 * below_band_power

        above_low = min(high + margin, dataset.f_max)
        if above_low < dataset.f_max:
            above_band_power = _mean_power_in_range(freqs, power, above_low, dataset.f_max)
            assert in_band_power > 30.0 * above_band_power


def test_each_band_energy_is_concentrated_in_passband(dataset) -> None:
    low_freqs, high_freqs = dataset._band_edges()

    for band_idx in range(dataset.n_bands):
        stim = _first_sample_for_band(dataset, band_idx)[0].numpy()
        freqs, power = _power_spectrum(stim, dataset.fs)

        low = low_freqs[band_idx]
        high = high_freqs[band_idx]
        in_band_mask = (freqs >= low) & (freqs <= high)

        in_band_energy = power[in_band_mask].sum()
        total_energy = power.sum()
        # Lowest band has a narrow absolute bandwidth, so finite-length FFT spreads energy.
        min_fraction = 0.55 if band_idx == 0 else 0.75
        assert in_band_energy / total_energy > min_fraction


def test_adjacent_bands_have_low_spectral_crosstalk(dataset) -> None:
    low_freqs, high_freqs = dataset._band_edges()
    centre_freqs = 0.5 * (low_freqs + high_freqs)

    for band_idx in range(dataset.n_bands):
        stim = _first_sample_for_band(dataset, band_idx)[0].numpy()
        freqs, power = _power_spectrum(stim, dataset.fs)

        own_low = low_freqs[band_idx]
        own_high = high_freqs[band_idx]
        own_bandwidth = own_high - own_low
        own_power = _mean_power_in_range(
            freqs, power, own_low - 0.1 * own_bandwidth, own_high + 0.1 * own_bandwidth
        )

        for other_idx in range(dataset.n_bands):
            if other_idx == band_idx:
                continue

            other_bandwidth = _erb_bandwidth(centre_freqs[other_idx])
            other_low = centre_freqs[other_idx] - other_bandwidth / 2
            other_high = centre_freqs[other_idx] + other_bandwidth / 2
            other_power = _mean_power_in_range(freqs, power, other_low, other_high)

            assert own_power > 10.0 * other_power


def test_get_stim_loader_batches_stimuli() -> None:
    batch_size = 4
    loader, dataset = get_stim_loader(
        fs=FS,
        n_bands=N_BANDS,
        n_samples_per_band=SAMPLES_PER_BAND,
        batch_size=batch_size,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
        shuffle=False,
        drop_last=True,
    )

    stim_batch = next(iter(loader))
    assert stim_batch.shape == (batch_size, 1, dataset.n_samples)
    assert stim_batch.dtype == torch.float32
    assert len(loader) == len(dataset) // batch_size


@pytest.fixture
def linear_dataset():
    return BandpassWhiteNoiseDataset(
        fs=FS,
        n_bands=N_BANDS,
        n_samples_per_band=SAMPLES_PER_BAND,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
        band_mode="linear",
    )


def test_linear_band_edges_cover_frequency_range(linear_dataset) -> None:
    low_freqs, high_freqs = linear_dataset._band_edges()

    assert low_freqs[0] == pytest.approx(linear_dataset.f_min)
    assert high_freqs[-1] == pytest.approx(linear_dataset.f_max)
    assert np.allclose(high_freqs[:-1], low_freqs[1:])

    bandwidths = high_freqs - low_freqs
    assert np.allclose(bandwidths, bandwidths[0])


def test_linear_bands_attenuate_out_of_band_frequencies(linear_dataset) -> None:
    low_freqs, high_freqs = linear_dataset._band_edges()

    for band_idx in range(linear_dataset.n_bands):
        stim = _first_sample_for_band(linear_dataset, band_idx)[0].numpy()
        freqs, power = _power_spectrum(stim, linear_dataset.fs)

        low = low_freqs[band_idx]
        high = high_freqs[band_idx]
        bandwidth = high - low
        margin = max(bandwidth * 0.25, 50.0)

        in_band_power = _mean_power_in_range(freqs, power, low, high)

        below_high = max(low - margin, linear_dataset.f_min)
        if below_high > linear_dataset.f_min:
            below_band_power = _mean_power_in_range(freqs, power, linear_dataset.f_min, below_high)
            assert in_band_power > 30.0 * below_band_power

        above_low = min(high + margin, linear_dataset.f_max)
        if above_low < linear_dataset.f_max:
            above_band_power = _mean_power_in_range(freqs, power, above_low, linear_dataset.f_max)
            assert in_band_power > 30.0 * above_band_power


def test_invalid_band_mode_raises() -> None:
    with pytest.raises(ValueError, match="band_mode must be 'erb', 'linear', 'log', or 'linear_erb'"):
        BandpassWhiteNoiseDataset(
            fs=FS,
            n_bands=4,
            n_samples_per_band=1,
            band_mode="octave",
        )


def test_invalid_erb_factor_raises() -> None:
    with pytest.raises(ValueError, match="erb_factor must be > 0"):
        BandpassWhiteNoiseDataset(
            fs=FS,
            n_bands=LINEAR_ERB_N_BANDS,
            n_samples_per_band=1,
            band_mode="linear_erb",
            erb_factor=0.0,
        )


def test_linear_erb_raises_when_spectrum_not_covered() -> None:
    with pytest.raises(ValueError, match="does not cover .* without holes"):
        BandpassWhiteNoiseDataset(
            fs=FS,
            n_bands=8,
            n_samples_per_band=1,
            stimulus_length_ms=STIMULUS_LENGTH_MS,
            ramp_time_ms=RAMP_TIME_MS,
            band_mode="linear_erb",
            erb_factor=1.0,
        )


LINEAR_ERB_N_BANDS = 42


@pytest.fixture
def linear_erb_dataset():
    return BandpassWhiteNoiseDataset(
        fs=FS,
        n_bands=LINEAR_ERB_N_BANDS,
        n_samples_per_band=SAMPLES_PER_BAND,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
        band_mode="linear_erb",
        erb_factor=1.0,
    )


def test_linear_erb_anchors_are_erb_rate_spaced(linear_erb_dataset) -> None:
    erb_low, erb_high, erb_anchors, anchor_hz = linear_erb_dataset._linear_erb_anchor_centres()
    expected = np.linspace(erb_low, erb_high, linear_erb_dataset.n_bands)
    assert np.allclose(erb_anchors, expected)
    assert anchor_hz[0] == pytest.approx(linear_erb_dataset.f_min)
    assert anchor_hz[-1] == pytest.approx(linear_erb_dataset.f_max)


def test_linear_erb_samples_centres_around_anchors(linear_erb_dataset) -> None:
    erb_low, erb_high, erb_anchors, _ = linear_erb_dataset._linear_erb_anchor_centres()
    delta = erb_anchors[1] - erb_anchors[0]
    for band_idx in range(linear_erb_dataset.n_bands):
        lo = max(erb_anchors[band_idx] - 0.5 * delta, erb_low)
        hi = min(erb_anchors[band_idx] + 0.5 * delta, erb_high)
        for sample_idx in range(linear_erb_dataset.n_samples_per_band):
            dataset_idx = band_idx * linear_erb_dataset.n_samples_per_band + sample_idx
            erb_c = _hz_to_erb_rate(linear_erb_dataset.centre_freqs[dataset_idx])
            assert lo <= erb_c <= hi


def test_linear_erb_samples_within_band_have_different_centres(linear_erb_dataset) -> None:
    centres = [
        linear_erb_dataset.centre_freqs[i]
        for i in range(linear_erb_dataset.n_samples_per_band)
    ]
    assert centres[0] != centres[1]


def test_linear_erb_bandwidth_is_erb_over_factor(linear_erb_dataset) -> None:
    low_freqs, high_freqs = linear_erb_dataset._band_edges()
    _, _, _, anchor_centres = linear_erb_dataset._linear_erb_anchor_centres()
    expected_bw = _erb_bandwidth(anchor_centres) / linear_erb_dataset.erb_factor
    assert np.allclose(high_freqs - low_freqs, expected_bw)


def test_linear_erb_factor_narrows_bandwidth() -> None:
    # Same n_bands / range so anchors match; factor=2 needs denser grid for coverage,
    # so compare bandwidths at identical centres without constructing a factor=2 dataset.
    wide = BandpassWhiteNoiseDataset(
        fs=FS, n_bands=LINEAR_ERB_N_BANDS, n_samples_per_band=1, band_mode="linear_erb", erb_factor=1.0,
        stimulus_length_ms=STIMULUS_LENGTH_MS, ramp_time_ms=RAMP_TIME_MS,
    )
    low_w, high_w = wide._band_edges()
    _, _, _, centres = wide._linear_erb_anchor_centres()
    low_n, high_n = wide._edges_from_centres(centres, bandwidth_scale=2.0)
    assert np.allclose(high_n - low_n, 0.5 * (high_w - low_w))


def test_linear_erb_anchor_bands_cover_spectrum_without_holes() -> None:
    dataset = BandpassWhiteNoiseDataset(
        fs=FS,
        n_bands=48,
        n_samples_per_band=1,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
        f_min=20.0,
        f_max=20000.0,
        band_mode="linear_erb",
        erb_factor=1.0,
    )
    low_freqs, high_freqs = dataset._band_edges()
    assert low_freqs[0] <= dataset.f_min
    assert high_freqs[-1] >= dataset.f_max
    assert np.all(high_freqs[:-1] >= low_freqs[1:])

@pytest.fixture
def log_dataset():
    return BandpassWhiteNoiseDataset(
        fs=FS,
        n_bands=N_BANDS,
        n_samples_per_band=SAMPLES_PER_BAND,
        stimulus_length_ms=STIMULUS_LENGTH_MS,
        ramp_time_ms=RAMP_TIME_MS,
        band_mode="log",
    )


def test_log_band_edges_cover_frequency_range(log_dataset) -> None:
    low_freqs, high_freqs = log_dataset._band_edges()

    assert low_freqs[0] == pytest.approx(log_dataset.f_min)
    assert high_freqs[-1] == pytest.approx(log_dataset.f_max)
    assert np.allclose(high_freqs[:-1], low_freqs[1:])

    bandwidths = high_freqs - low_freqs
    assert np.all(bandwidths[1:] > bandwidths[:-1])

    ratios = high_freqs / low_freqs
    assert np.allclose(ratios, ratios[0])


def test_log_bands_attenuate_out_of_band_frequencies(log_dataset) -> None:
    low_freqs, high_freqs = log_dataset._band_edges()

    for band_idx in range(log_dataset.n_bands):
        stim = _first_sample_for_band(log_dataset, band_idx)[0].numpy()
        freqs, power = _power_spectrum(stim, log_dataset.fs)

        low = low_freqs[band_idx]
        high = high_freqs[band_idx]
        bandwidth = high - low
        margin = max(bandwidth * 0.25, 50.0)

        in_band_power = _mean_power_in_range(freqs, power, low, high)

        below_high = max(low - margin, log_dataset.f_min)
        if below_high > log_dataset.f_min:
            below_band_power = _mean_power_in_range(freqs, power, log_dataset.f_min, below_high)
            assert in_band_power > 30.0 * below_band_power

        above_low = min(high + margin, log_dataset.f_max)
        if above_low < log_dataset.f_max:
            above_band_power = _mean_power_in_range(freqs, power, above_low, log_dataset.f_max)
            assert in_band_power > 30.0 * above_band_power


def test_bandpass_dataset_plot_returns_two_axes() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotting.XY_plots import bandpass_dataset_plot

    dataset = BandpassWhiteNoiseDataset(
        fs=FS,
        n_bands=12,
        n_samples_per_band=2,
        stimulus_length_ms=5,
        ramp_time_ms=1,
        band_mode="linear_erb",
        erb_factor=0.25,
    )
    fig, axes = bandpass_dataset_plot(dataset)
    assert len(axes) == 2
    plt.close(fig)
