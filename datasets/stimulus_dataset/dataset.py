import torch
from torch.utils.data import Dataset
import numpy as np
from scipy import signal


def _hz_to_erb_rate(f):
    return 21.4 * np.log10(1 + 0.00437 * f)


def _erb_rate_to_hz(erb):
    return (10 ** (erb / 21.4) - 1) / 0.00437


def _erb_bandwidth(fc):
    return 0.108 * fc + 24.7


def _raised_cosine_ramp(n_samples, ramp_samples):
    envelope = np.ones(n_samples, dtype=np.float64)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
    envelope[:ramp_samples] = ramp
    envelope[-ramp_samples:] = ramp[::-1]
    return envelope


class BandpassWhiteNoiseDataset(Dataset):
    def __init__(self, fs, n_bands, n_samples_per_band, stimulus_length_ms = 5, ramp_time_ms = 1, f_min=20.0, f_max=20000.0):

        self.fs = fs
        self.n_bands = n_bands
        self.n_samples_per_band = n_samples_per_band
        self.stimulus_length_ms = stimulus_length_ms
        self.ramp_time_ms = ramp_time_ms
        self.f_min = f_min
        self.f_max = min(f_max, fs / 2 - 1)

        self.n_samples = int(round(stimulus_length_ms * fs / 1000))
        self.ramp_samples = int(round(ramp_time_ms * fs / 1000))

        if self.n_samples <= 2 * self.ramp_samples:
            raise ValueError(
                f"stimulus_length_ms ({stimulus_length_ms}) must exceed twice ramp_time_ms ({ramp_time_ms})."
            )

        self.stimulus = self.generate_stimulus()

    def __len__(self):
        return self.n_bands * self.n_samples_per_band

    def __getitem__(self, idx):
        return self.stimulus[idx]

    def _band_edges(self):
        erb_low = _hz_to_erb_rate(self.f_min)
        erb_high = _hz_to_erb_rate(self.f_max)
        erb_centres = np.linspace(erb_low, erb_high, self.n_bands)
        centre_freqs = _erb_rate_to_hz(erb_centres)
        bandwidths = _erb_bandwidth(centre_freqs)
        low_freqs = np.maximum(centre_freqs - bandwidths / 2, 1.0)
        high_freqs = np.minimum(centre_freqs + bandwidths / 2, self.fs / 2 - 1)
        return low_freqs, high_freqs

    def _bandpass_white_noise(self, low_freq, high_freq, seed):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(self.n_samples)
        sos = signal.butter(4, [low_freq, high_freq], btype="band", fs=self.fs, output="sos")
        filtered = signal.sosfiltfilt(sos, noise)
        filtered *= _raised_cosine_ramp(self.n_samples, self.ramp_samples)
        rms = np.sqrt(np.mean(filtered ** 2))
        if rms > 0:
            filtered /= rms
        return filtered.astype(np.float32)

    def generate_stimulus(self):
        low_freqs, high_freqs = self._band_edges()
        stimuli = np.zeros((len(self), 1, self.n_samples), dtype=np.float32)

        for band_idx in range(self.n_bands):
            for sample_idx in range(self.n_samples_per_band):
                dataset_idx = band_idx * self.n_samples_per_band + sample_idx
                seed = band_idx * self.n_samples_per_band + sample_idx
                stimuli[dataset_idx, 0] = self._bandpass_white_noise(
                    low_freqs[band_idx], high_freqs[band_idx], seed
                )

        return torch.from_numpy(stimuli)
