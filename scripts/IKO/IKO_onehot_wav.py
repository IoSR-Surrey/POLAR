import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.IKO_functions import num_hoa_channels


def pink_noise(n, random_state=None):
    """Generate pink noise (1/f) of length n."""
    rng = np.random.default_rng(random_state)
    num_bins = n // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, num_bins)
    magnitudes = rng.standard_normal(num_bins)
    spectrum = magnitudes * np.exp(1j * phases)
    frequencies = np.arange(num_bins, dtype=float)
    frequencies[0] = 1.0
    spectrum /= np.sqrt(frequencies)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=n)
    signal -= signal.mean()
    signal /= signal.std()
    return signal


def make_pink_noise_burst(fs, burst_s, silence_s, seed):
    """Pink-noise burst with leading and trailing silence."""
    burst_len = int(round(burst_s * fs))
    silence_len = int(round(silence_s * fs))
    burst = pink_noise(burst_len, random_state=seed)
    return np.concatenate([
        np.zeros(silence_len, dtype=np.float64),
        burst,
        np.zeros(silence_len, dtype=np.float64),
    ])


def make_impulse_signal(n_samples, fs, impulse_ms=5.0, amplitude=1.0):
    """Return a silent signal with a unit impulse at the given time offset."""
    signal = np.zeros(n_samples, dtype=np.float64)
    idx = int(round(impulse_ms * 1e-3 * fs))
    if idx < 0 or idx >= n_samples:
        raise ValueError(
            f"impulse_ms={impulse_ms} places impulse at sample {idx}, "
            f"outside signal length {n_samples}"
        )
    signal[idx] = amplitude
    return signal


def mono_to_hoa_channels(mono, weights):
    """Scale a mono signal by HOA weights. Returns (T, L)."""
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return mono[:, np.newaxis] * w[np.newaxis, :]


def normalize_wav_audio(audio, target_peak=0.95):
    """Peak-normalise multi-channel audio for WAV export."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def one_hot_hoa_weight(order, channel_idx):
    """Return a one-hot HOA weight vector for the given channel index."""
    weights = np.zeros(num_hoa_channels(order), dtype=np.float64)
    weights[channel_idx] = 1.0
    return weights


def one_hot_hoa_label(order, channel_idx):
    """Return (n, m) label for a flat HOA channel index."""
    idx = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if idx == channel_idx:
                return n, m
            idx += 1
    raise IndexError(f"channel_idx {channel_idx} out of range for order {order}")


def save_onehot_hoa_wav(
    output_dir,
    order=3,
    fs=48000,
    duration_s=3.0,
    burst_s=0.25,
    silence_s=0.25,
    impulse_ms=0.0,
    seed=0,
    target_peak=0.95,
):
    """Save pink-noise and one-hot impulse HOA WAVs for every SH basis weight."""
    n_channels = num_hoa_channels(order)
    continuous_dir = Path(output_dir) / "wav" / "continuous"
    burst_dir = Path(output_dir) / "wav" / "burst"
    impulse_dir = Path(output_dir) / "wav" / "impulse"
    continuous_dir.mkdir(parents=True, exist_ok=True)
    burst_dir.mkdir(parents=True, exist_ok=True)
    impulse_dir.mkdir(parents=True, exist_ok=True)

    mono = pink_noise(int(round(duration_s * fs)), random_state=seed)
    mono_burst = make_pink_noise_burst(fs, burst_s, silence_s, seed)
    mono_impulse = make_impulse_signal(int(round(duration_s * fs)), fs, impulse_ms=impulse_ms)

    saved_paths = []
    channel_info = []
    for channel_idx in range(n_channels):
        n, m = one_hot_hoa_label(order, channel_idx)
        label = f"n{n}_m{m:+d}"
        channel_info.append({"channel_idx": channel_idx, "n": n, "m": m, "label": label})
        weights = one_hot_hoa_weight(order, channel_idx)

        hoa_audio = normalize_wav_audio(mono_to_hoa_channels(mono, weights), target_peak=target_peak)
        continuous_path = continuous_dir / f"hoa_onehot_{label}.wav"
        wavfile.write(continuous_path, fs, hoa_audio)

        hoa_burst = normalize_wav_audio(mono_to_hoa_channels(mono_burst, weights), target_peak=target_peak)
        burst_path = burst_dir / f"hoa_onehot_{label}.wav"
        wavfile.write(burst_path, fs, hoa_burst)

        hoa_impulse = normalize_wav_audio(mono_to_hoa_channels(mono_impulse, weights), target_peak=target_peak)
        impulse_path = impulse_dir / f"hoa_onehot_{label}.wav"
        wavfile.write(impulse_path, fs, hoa_impulse)

        saved_paths.extend([continuous_path, burst_path, impulse_path])
        print(
            f"Saved one-hot HOA WAVs for channel {channel_idx} ({label}): "
            f"continuous/{continuous_path.name}, burst/{burst_path.name}, impulse/{impulse_path.name}"
        )

    manifest = {
        "order": order,
        "fs": fs,
        "n_hoa_channels": n_channels,
        "duration_s": duration_s,
        "burst_s": burst_s,
        "silence_s": silence_s,
        "impulse_ms": impulse_ms,
        "seed": seed,
        "channels": channel_info,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return saved_paths, manifest_path, continuous_dir, burst_dir, impulse_dir


def main():
    with open("config/hyperpameters.json", "r") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    seed = config["env"].get("seed", 0)
    order = 3

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path("outputs") / "iko_onehot" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths, manifest_path, continuous_dir, burst_dir, impulse_dir = save_onehot_hoa_wav(
        output_dir, order=order, fs=fs, seed=seed, duration_s=3.0,
    )
    print(f"Saved {len(saved_paths) // 3} continuous WAVs to {continuous_dir}")
    print(f"Saved {len(saved_paths) // 3} burst WAVs to {burst_dir}")
    print(f"Saved {len(saved_paths) // 3} impulse WAVs to {impulse_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
