import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
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
    def __init__(
        self,
        fs,
        n_bands,
        n_samples_per_band,
        stimulus_length_ms=5,
        ramp_time_ms=1,
        f_min=20.0,
        f_max=20000.0,
        band_mode="erb",
        erb_factor=1.0,
    ):
        if band_mode not in ("erb", "linear", "log", "linear_erb"):
            raise ValueError(
                f"band_mode must be 'erb', 'linear', 'log', or 'linear_erb', got {band_mode!r}."
            )
        if erb_factor <= 0:
            raise ValueError(f"erb_factor must be > 0, got {erb_factor!r}.")

        self.fs = fs
        self.n_bands = n_bands
        self.n_samples_per_band = n_samples_per_band
        self.stimulus_length_ms = stimulus_length_ms
        self.ramp_time_ms = ramp_time_ms
        self.f_min = f_min
        self.f_max = min(f_max, fs / 2 - 1)
        self.band_mode = band_mode
        self.erb_factor = erb_factor

        self.n_samples = int(round(stimulus_length_ms * fs / 1000))
        self.ramp_samples = int(round(ramp_time_ms * fs / 1000))

        if self.n_samples <= 2 * self.ramp_samples:
            raise ValueError(
                f"stimulus_length_ms ({stimulus_length_ms}) must exceed twice ramp_time_ms ({ramp_time_ms})."
            )

        if self.band_mode == "linear_erb":
            self._assert_linear_erb_spectrum_covered()

        self.stimulus = self.generate_stimulus()

    def _assert_linear_erb_spectrum_covered(self):
        """Raise if linear_erb anchor bands leave holes in [f_min, f_max]."""
        _, _, _, centres = self._linear_erb_anchor_centres()
        bandwidths = _erb_bandwidth(centres) / self.erb_factor
        low = centres - bandwidths / 2
        high = centres + bandwidths / 2

        holes = []
        if low[0] > self.f_min:
            holes.append(f"[{self.f_min:.1f}, {low[0]:.1f}] Hz below first band")
        if high[-1] < self.f_max:
            holes.append(f"[{high[-1]:.1f}, {self.f_max:.1f}] Hz above last band")
        gap_mask = high[:-1] < low[1:]
        for i in np.flatnonzero(gap_mask):
            holes.append(f"[{high[i]:.1f}, {low[i + 1]:.1f}] Hz between bands {i} and {i + 1}")

        if holes:
            erb_span = _hz_to_erb_rate(self.f_max) - _hz_to_erb_rate(self.f_min)
            min_bands = int(np.ceil(erb_span * self.erb_factor)) + 1
            raise ValueError(
                f"band_mode='linear_erb' with n_bands={self.n_bands} and "
                f"erb_factor={self.erb_factor} does not cover [{self.f_min}, {self.f_max}] Hz "
                f"without holes ({'; '.join(holes[:3])}{'...' if len(holes) > 3 else ''}). "
                f"Use at least n_bands={min_bands}."
            )

    def __len__(self):
        return self.n_bands * self.n_samples_per_band

    def __getitem__(self, idx):
        return self.stimulus[idx]

    def _edges_from_centres(self, centre_freqs, bandwidth_scale=1.0):
        bandwidths = _erb_bandwidth(centre_freqs) / bandwidth_scale
        low_freqs = np.maximum(centre_freqs - bandwidths / 2, 1.0)
        high_freqs = np.minimum(centre_freqs + bandwidths / 2, self.fs / 2 - 1)
        return low_freqs, high_freqs

    def _erb_band_edges(self):
        erb_low = _hz_to_erb_rate(self.f_min)
        erb_high = _hz_to_erb_rate(self.f_max)
        erb_centres = np.linspace(erb_low, erb_high, self.n_bands)
        centre_freqs = _erb_rate_to_hz(erb_centres)
        return self._edges_from_centres(centre_freqs, bandwidth_scale=1.0)

    def _linear_erb_anchor_centres(self):
        """ERB-rate anchors covering [f_min, f_max]."""
        erb_low = _hz_to_erb_rate(self.f_min)
        erb_high = _hz_to_erb_rate(self.f_max)
        erb_anchors = np.linspace(erb_low, erb_high, self.n_bands)
        return erb_low, erb_high, erb_anchors, _erb_rate_to_hz(erb_anchors)

    def _linear_erb_sample_centre(self, band_idx, sample_idx):
        """Sample f_c uniformly in ERB-rate between midpoints to neighbouring anchors."""
        erb_low, erb_high, erb_anchors, _ = self._linear_erb_anchor_centres()
        if self.n_bands == 1:
            lo, hi = erb_low, erb_high
        else:
            delta = erb_anchors[1] - erb_anchors[0]
            lo = max(erb_anchors[band_idx] - 0.5 * delta, erb_low)
            hi = min(erb_anchors[band_idx] + 0.5 * delta, erb_high)
        rng = np.random.default_rng(10_000_003 + band_idx * self.n_samples_per_band + sample_idx)
        erb_c = float(rng.uniform(lo, hi)) if hi > lo else float(erb_anchors[band_idx])
        return float(_erb_rate_to_hz(erb_c))

    def _linear_erb_band_edges(self):
        """Nominal edges at ERB anchors (actual stimuli use sampled centres)."""
        _, _, _, anchor_centres = self._linear_erb_anchor_centres()
        return self._edges_from_centres(anchor_centres, bandwidth_scale=self.erb_factor)

    def _linear_band_edges(self):
        edges = np.linspace(self.f_min, self.f_max, self.n_bands + 1)
        return edges[:-1], edges[1:]

    def _log_band_edges(self):
        edges = np.geomspace(self.f_min, self.f_max, self.n_bands + 1)
        return edges[:-1], edges[1:]

    def _band_edges(self):
        if self.band_mode == "erb":
            return self._erb_band_edges()
        if self.band_mode == "linear_erb":
            return self._linear_erb_band_edges()
        if self.band_mode == "linear":
            return self._linear_band_edges()
        return self._log_band_edges()

    def _bandpass_white_noise(self, low_freq, high_freq, seed):
        # Filter a long buffer then crop: short clips (e.g. 5 ms) cannot realise
        # low-frequency / narrow ERB bands if filtfilt is run on n_samples alone.
        n_long = max(
            self.n_samples * 8,
            int(self.fs / max(low_freq, 1.0) * 10),
            4096,
        )
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(n_long)
        sos = signal.butter(4, [low_freq, high_freq], btype="band", fs=self.fs, output="sos")
        filtered = signal.sosfiltfilt(sos, noise)
        start = (n_long - self.n_samples) // 2
        filtered = filtered[start : start + self.n_samples]
        filtered *= _raised_cosine_ramp(self.n_samples, self.ramp_samples)
        rms = np.sqrt(np.mean(filtered ** 2))
        if rms > 0:
            filtered /= rms
        return filtered.astype(np.float32)

    def generate_stimulus(self):
        stimuli = np.zeros((len(self), 1, self.n_samples), dtype=np.float32)
        centre_freqs = np.zeros(len(self), dtype=np.float64)

        if self.band_mode == "linear_erb":
            for band_idx in range(self.n_bands):
                for sample_idx in range(self.n_samples_per_band):
                    dataset_idx = band_idx * self.n_samples_per_band + sample_idx
                    fc = self._linear_erb_sample_centre(band_idx, sample_idx)
                    centre_freqs[dataset_idx] = fc
                    bw = _erb_bandwidth(fc) / self.erb_factor
                    low = max(fc - bw / 2, 1.0)
                    high = min(fc + bw / 2, self.fs / 2 - 1)
                    stimuli[dataset_idx, 0] = self._bandpass_white_noise(low, high, dataset_idx)
        else:
            low_freqs, high_freqs = self._band_edges()
            for band_idx in range(self.n_bands):
                for sample_idx in range(self.n_samples_per_band):
                    dataset_idx = band_idx * self.n_samples_per_band + sample_idx
                    centre_freqs[dataset_idx] = 0.5 * (low_freqs[band_idx] + high_freqs[band_idx])
                    stimuli[dataset_idx, 0] = self._bandpass_white_noise(
                        low_freqs[band_idx], high_freqs[band_idx], dataset_idx
                    )

        self.centre_freqs = centre_freqs
        return torch.from_numpy(stimuli)


class MockStimulusDataset(Dataset):
    """Single-sample stimulus dataset for quick wiring checks (impulse or white noise)."""

    def __init__(
        self,
        fs,
        stimulus_length_ms=5,
        stimulus_type="impulse",
        seed=0,
    ):
        if stimulus_type not in ("impulse", "noise"):
            raise ValueError(f"stimulus_type must be 'impulse' or 'noise', got {stimulus_type!r}.")

        self.fs = fs
        self.stimulus_length_ms = stimulus_length_ms
        self.stimulus_type = stimulus_type
        self.seed = seed
        self.n_samples = int(round(stimulus_length_ms * fs / 1000))
        if self.n_samples < 1:
            raise ValueError(f"stimulus_length_ms ({stimulus_length_ms}) yields fewer than 1 sample at fs={fs}.")

        self.stimulus = self.generate_stimulus()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.stimulus[idx]

    def generate_stimulus(self):
        stim = np.zeros((1, 1, self.n_samples), dtype=np.float32)
        if self.stimulus_type == "impulse":
            stim[0, 0, 0] = 1.0
        else:
            rng = np.random.default_rng(self.seed)
            noise = rng.standard_normal(self.n_samples).astype(np.float32)
            rms = np.sqrt(np.mean(noise ** 2))
            if rms > 0:
                noise /= rms
            stim[0, 0] = noise
        return torch.from_numpy(stim)


def get_stim_loader(
    fs,
    n_bands,
    n_samples_per_band,
    batch_size,
    stimulus_length_ms=5,
    ramp_time_ms=1,
    f_min=20.0,
    f_max=20000.0,
    band_mode="erb",
    erb_factor=1.0,
    dataset_type="bandpass",
    stimulus_type="impulse",
    seed=0,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):
    # Pre-generated stimuli are indexed in the main process; workers add no benefit on Windows.
    if sys.platform == "win32" and num_workers > 0:
        num_workers = 0

    if dataset_type == "bandpass":
        dataset = BandpassWhiteNoiseDataset(
            fs=fs,
            n_bands=n_bands,
            n_samples_per_band=n_samples_per_band,
            stimulus_length_ms=stimulus_length_ms,
            ramp_time_ms=ramp_time_ms,
            f_min=f_min,
            f_max=f_max,
            band_mode=band_mode,
            erb_factor=erb_factor,
        )
    elif dataset_type == "mock":
        dataset = MockStimulusDataset(
            fs=fs,
            stimulus_length_ms=stimulus_length_ms,
            stimulus_type=stimulus_type,
            seed=seed,
        )
    else:
        raise ValueError(f"dataset_type must be 'bandpass' or 'mock', got {dataset_type!r}.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader, dataset
