import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
import torchaudio.functional as TF
from scipy.io import wavfile


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iko_brir = _load_module("iko_brir_optimisation", ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py")

N_IKO_LOUDSPEAKERS = iko_brir.N_IKO_LOUDSPEAKERS
SPH_TO_DRIVER_NPY = iko_brir.SPH_TO_DRIVER_NPY


def white_noise(n, random_state=None):
    rng = np.random.default_rng(random_state)
    signal = rng.standard_normal(n)
    signal -= signal.mean()
    if signal.std() > 0:
        signal /= signal.std()
    return signal


def raised_cosine_ramp(n_samples, ramp_samples):
    envelope = np.ones(n_samples, dtype=np.float64)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_samples) / ramp_samples))
    envelope[:ramp_samples] = ramp
    envelope[-ramp_samples:] = ramp[::-1]
    return envelope


def apply_ramp(signal, fs, ramp_s):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    ramp_samples = int(round(ramp_s * fs))
    if ramp_samples <= 0:
        return signal
    if signal.shape[0] <= 2 * ramp_samples:
        raise ValueError(
            f"Signal length ({signal.shape[0]} samples) must exceed twice the ramp "
            f"({2 * ramp_samples} samples, {2 * ramp_s:.3f} s)."
        )
    return signal * raised_cosine_ramp(signal.shape[0], ramp_samples)


def load_hoa_weights(output_dir, stage):
    weights_path = Path(output_dir) / "npy" / f"{stage}_hoa_weights.npy"
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find HOA weights at {weights_path}")
    return np.load(weights_path).astype(np.float64).reshape(-1)


def speaker_ir_from_weights(weights, eigenbeam_to_loudspeaker, device, dtype):
    weights_t = torch.tensor(weights, device=device, dtype=dtype)
    return iko_brir.combine_speaker_impulse_responses(weights_t, eigenbeam_to_loudspeaker)


def render_driver_signals(stimulus, weights, eigenbeam_to_loudspeaker, device, dtype):
    speaker_ir = speaker_ir_from_weights(weights, eigenbeam_to_loudspeaker, device, dtype)
    stim = torch.tensor(stimulus, dtype=torch.float32).view(1, 1, -1)
    filters = speaker_ir.detach().cpu().to(dtype=torch.float32).unsqueeze(0)
    filtered = TF.convolve(stim, filters, mode="full")
    return np.asarray(filtered.squeeze(0).T, dtype=np.float64)


def normalize_wav(audio, target_peak=0.95):
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def save_driver_wav(path, audio, fs, target_peak=0.95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.shape[1] != N_IKO_LOUDSPEAKERS:
        raise ValueError(f"Expected {N_IKO_LOUDSPEAKERS} driver channels, got {audio.shape[1]}.")
    wavfile.write(path, fs, normalize_wav(audio, target_peak=target_peak))


def default_wav_path(output_dir, stage):
    return Path(output_dir) / "wav" / f"{stage}_driver_white_noise.wav"


def load_config_folders(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    folders = config.get("folders")
    if folders is None:
        raise ValueError("Config must contain a 'folders' key.")
    if isinstance(folders, str):
        folders = [folders]
    if not isinstance(folders, list) or not folders:
        raise ValueError("'folders' must be a non-empty list of paths.")
    output_dirs = []
    for folder in folders:
        path = Path(folder)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        else:
            path = path.resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Output directory does not exist: {path}")
        output_dirs.append(path)
    return output_dirs


def export_run(output_dir, stage, fs, device, dtype, eigenbeam_to_loudspeaker, mono, out_path=None, target_peak=0.95):
    weights = load_hoa_weights(output_dir, stage)
    driver_audio = render_driver_signals(mono, weights, eigenbeam_to_loudspeaker, device, dtype)
    wav_path = Path(out_path) if out_path is not None else default_wav_path(output_dir, stage)
    save_driver_wav(wav_path, driver_audio, fs, target_peak=target_peak)
    print(f"Loaded {stage} HOA weights from {output_dir}")
    print(f"Rendered driver signals: {driver_audio.shape} (samples, {N_IKO_LOUDSPEAKERS} channels) @ {fs} Hz")
    print(f"Saved WAV to {wav_path.resolve()}")
    return wav_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a multi-channel WAV of IKO driver signals from saved HOA weights "
        "and white noise."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "play_iko_config.json",
        help="JSON config with a list of IKO optimisation output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single IKO optimisation run folder containing npy/{stage}_hoa_weights.npy.",
    )
    parser.add_argument(
        "--stage",
        choices=("initial", "final"),
        default="final",
        help="Which saved HOA weights to use.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output WAV path for a single --output-dir run "
        "(default: <output-dir>/wav/{stage}_driver_white_noise.wav).",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="Sample rate in Hz. Defaults to config/hyperpameters.json env.fs.",
    )
    parser.add_argument("--duration-s", type=float, default=2.0, help="White-noise duration in seconds.")
    parser.add_argument("--ramp-s", type=float, default=0.05, help="Raised-cosine fade-in/out duration in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for white-noise generation.")
    parser.add_argument(
        "--sph-to-driver",
        type=Path,
        default=SPH_TO_DRIVER_NPY,
        help="Path to hoa_to_driver.npy.",
    )
    parser.add_argument("--peak", type=float, default=0.95, help="Target peak before writing the WAV.")
    args = parser.parse_args()
    if args.output_dir is None and (args.config is None or not args.config.exists()):
        parser.error("provide --output-dir or a valid --config file")
    if args.out is not None and args.output_dir is None:
        parser.error("--out requires --output-dir")
    return args


def main():
    args = parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output directory does not exist: {output_dir}")
        output_dirs = [output_dir]
    else:
        output_dirs = load_config_folders(args.config.resolve())

    with open(ROOT / "config" / "hyperpameters.json", encoding="utf-8") as f:
        config = json.load(f)
    fs = args.fs if args.fs is not None else int(config["env"]["fs"])
    device = config["env"]["device"]
    dtype = torch.float32

    first_weights = load_hoa_weights(output_dirs[0], args.stage)
    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        args.sph_to_driver.resolve(), first_weights.shape[0], N_IKO_LOUDSPEAKERS, device, dtype,
    )
    mono = white_noise(int(round(args.duration_s * fs)), random_state=args.seed)
    mono = apply_ramp(mono, fs, args.ramp_s)

    for output_dir in output_dirs:
        out_path = args.out.resolve() if args.out is not None and len(output_dirs) == 1 else None
        export_run(
            output_dir, args.stage, fs, device, dtype, eigenbeam_to_loudspeaker, mono,
            out_path=out_path, target_peak=args.peak,
        )


if __name__ == "__main__":
    main()
