import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.functional as TF
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "IKO"))
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location("iko_optimisation", ROOT / "scripts" / "IKO" / "IKO_optimisation.py")
iko = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iko)

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


def load_weights(weights_arg, order):
    """Load HOA weights from a .npy/.json path or comma-separated values."""
    n_channels = num_hoa_channels(order)
    path = Path(weights_arg)
    if path.suffix == ".npy":
        weights = np.load(path)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        weights = np.asarray(data["weights"] if isinstance(data, dict) and "weights" in data else data, dtype=np.float64)
    else:
        weights = np.asarray([float(x.strip()) for x in weights_arg.split(",")], dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n_channels:
        raise ValueError(f"Expected {n_channels} HOA weights for order {order}, got {weights.shape[0]}.")
    return weights


def load_mono_wav(path, target_fs):
    fs, data = wavfile.read(path)
    audio = np.asarray(data, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        audio = audio / np.iinfo(data.dtype).max
    if fs != target_fs:
        audio = TF.resample(
            torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
            fs,
            target_fs,
        ).squeeze(0).numpy()
    return audio


def save_hoa_wav(
    weights,
    output_path,
    fs=48000,
    order=3,
    input_wav=None,
    duration_s=3.0,
    burst_s=0.25,
    silence_s=0.25,
    seed=0,
    target_peak=0.95,
    burst=False,
):
    """Render a mono source through HOA weights and save a multi-channel WAV."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_channels = num_hoa_channels(order)
    if weights.shape[0] != n_channels:
        raise ValueError(f"Expected {n_channels} HOA weights for order {order}, got {weights.shape[0]}.")

    if input_wav is not None:
        mono = load_mono_wav(input_wav, fs)
    elif burst:
        mono = make_pink_noise_burst(fs, burst_s, silence_s, seed)
    else:
        mono = pink_noise(int(round(duration_s * fs)), random_state=seed)

    hoa_audio = normalize_wav_audio(mono_to_hoa_channels(mono, weights), target_peak=target_peak)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, fs, hoa_audio)
    return output_path, hoa_audio.shape


def save_weights(output_dir, weights):
    npy_dir = Path(output_dir) / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    weights_path = npy_dir / "hoa_weights.npy"
    np.save(weights_path, np.asarray(weights, dtype=np.float64).reshape(-1))
    return weights_path


def plot_hoa_weights(order, weights, plots_dir):
    """Plot HOA weights on an (m, n) grid."""
    weights_t = torch.tensor(np.asarray(weights, dtype=np.float64).reshape(-1))
    w = weights_t.detach().cpu().numpy()
    vmax = max(float(np.max(np.abs(w))), 1e-6)

    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.full((order + 1, 2 * order + 1), np.nan)
    for n, m, value in iko.sh_weights_to_nm(order, weights_t):
        grid[n, m + order] = value

    im = ax.imshow(
        grid, origin="lower", aspect="equal", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[-order - 0.5, order + 0.5, -0.5, order + 0.5],
    )
    for n, m, value in iko.sh_weights_to_nm(order, weights_t):
        ax.text(m, n, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_xticks(list(range(-order, order + 1)))
    ax.set_yticks(list(range(order + 1)))
    ax.set_xlabel("m")
    ax.set_ylabel("n (order)")
    ax.set_title("HOA weights")
    ax.grid(True, color="white", linewidth=0.5, alpha=0.5)
    fig.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    plots_path = Path(plots_dir) / "hoa_weights.png"
    fig.savefig(plots_path, dpi=150)
    plt.close(fig)
    return plots_path


def plot_beam_pattern(order, weights, plots_dir, listener_az=None, reflector_az=None):
    """Save horizontal beam-pattern polar plot."""
    weights_t = torch.tensor(np.asarray(weights, dtype=np.float64).reshape(-1))
    azimuths, pattern = iko.beam_pattern_over_circle(order, weights_t)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    plot_azimuths = iko._beam_pattern_plot_azimuths(azimuths)
    ax.plot(plot_azimuths, pattern, label="Pattern")
    r_lim = max(float(np.max(np.abs(pattern))), 0.1) * 1.1
    if listener_az is not None:
        plot_listener_az = float(iko._beam_pattern_plot_azimuths(listener_az))
        ax.plot([plot_listener_az, plot_listener_az], [0, r_lim], color="blue", linestyle="--", linewidth=1.5, label="Listener")
    if reflector_az is not None:
        plot_reflector_az = float(iko._beam_pattern_plot_azimuths(reflector_az))
        ax.plot([plot_reflector_az, plot_reflector_az], [0, r_lim], color="gray", linestyle="--", linewidth=1.5, label="Reflector")
    ax.set_title("Beam pattern")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plots_path = Path(plots_dir) / "beam_pattern.png"
    fig.savefig(plots_path, dpi=150)
    plt.close(fig)
    return plots_path


def save_beam_pattern_plots(order, weights, output_dir, fs):
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    listener_positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    geometry = iko.setup_geometry(2.0, 30.0, -30.0, listener_positions, fs)
    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])

    weights_t = torch.tensor(np.asarray(weights, dtype=np.float64).reshape(-1))
    beam_pattern_path = plot_beam_pattern(order, weights, plots_dir, listener_az, reflector_az)
    iko.plot_beam_pattern_overlay(order, weights_t, listener_az, reflector_az, plots_dir, stage="hoa")
    weights_plot_path = plot_hoa_weights(order, weights, plots_dir)
    overlay_path = plots_dir / "beam_pattern_overlay_hoa.png"
    return beam_pattern_path, overlay_path, weights_plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Render HOA weights to a multi-channel WAV (16 tracks for order 3).",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="HOA weights as a .npy/.json file path or comma-separated values (16 floats for order 3).",
    )
    parser.add_argument("--output", type=str, default=None, help="Output WAV path.")
    parser.add_argument("--order", type=int, default=3, help="HOA order (default 3 -> 16 channels).")
    parser.add_argument("--input-wav", type=str, default=None, help="Optional mono WAV to pan through the weights.")
    parser.add_argument("--burst", action="store_true", help="Use a pink-noise burst instead of continuous pink noise.")
    parser.add_argument("--duration-s", type=float, default=3.0, help="Duration for continuous pink noise.")
    parser.add_argument("--burst-s", type=float, default=0.25, help="Burst length when --burst is set.")
    parser.add_argument("--silence-s", type=float, default=0.25, help="Leading/trailing silence when --burst is set.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config value).")
    args = parser.parse_args()

    with open("config/hyperpameters.json", "r") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    seed = config["env"].get("seed", 0) if args.seed is None else args.seed
    weights = load_weights(args.weights, args.order)

    if args.output is not None:
        output_path = Path(args.output)
        output_dir = output_path.parent
        wav_path = output_dir / "wav" / output_path.name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("outputs") / "iko_hoa_wav" / timestamp
        wav_path = output_dir / "wav" / "hoa.wav"

    output_path, shape = save_hoa_wav(
        weights,
        wav_path,
        fs=fs,
        order=args.order,
        input_wav=args.input_wav,
        duration_s=args.duration_s,
        burst_s=args.burst_s,
        silence_s=args.silence_s,
        seed=seed,
        burst=args.burst,
    )
    weights_path = save_weights(output_dir, weights)
    beam_pattern_path, overlay_path, weights_plot_path = save_beam_pattern_plots(
        args.order, weights, output_dir, fs,
    )

    manifest = {
        "order": args.order,
        "n_hoa_channels": num_hoa_channels(args.order),
        "fs": fs,
        "weights": weights.tolist(),
        "input_wav": args.input_wav,
        "burst": args.burst,
        "duration_s": args.duration_s,
        "burst_s": args.burst_s,
        "silence_s": args.silence_s,
        "seed": seed,
        "output_wav": str(output_path),
        "weights_npy": str(weights_path),
        "beam_pattern_plot": str(beam_pattern_path),
        "beam_pattern_overlay_plot": str(overlay_path),
        "hoa_weights_plot": str(weights_plot_path),
        "shape_samples_channels": [shape[0], shape[1]],
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Loaded {len(weights)} HOA weights: {weights}")
    print(f"Saved {shape[1]}-channel HOA WAV ({shape[0]} samples @ {fs} Hz) to {output_path}")
    print(f"Saved weights to {weights_path}")
    print(f"Saved beam pattern plots to {output_dir / 'plots'}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
