"""Create multi-channel stimulus WAVs for localisation_test.py.

Loads spk_filters from optimisation run folders, convolves each with mono
audio tracks, and writes WAVs named:

  {optimisation_type}_{target_angle}[_{program_material}].wav

optimisation_type defaults to the run-folder timestamp with '_' -> '-'
(e.g. outputs/cagrad/2026-08-02_135216 -> 2026-08-02-135216).

Example:
  python scripts/listening/create_localisation_stimuli.py \\
    --output-dir outputs/cagrad/2026-08-02_135216 \\
    --out-dir data/localisation_stimuli
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from scipy.io import wavfile


def format_target_angle(angle):
    """Format target angle for filenames (+ = left).

    Ordinary zero is '0'. Negative zero (-0.0) is written as '-0' so a
    left/right-flipped centre condition can be distinguished in filenames.
    """
    angle = float(angle)
    if angle == 0.0:
        return "-0" if math.copysign(1.0, angle) < 0 else "0"
    return f"{angle:+g}"


def sanitize_filename_field(value, field_name):
    """Replace underscores so the value stays a single underscore-separated field."""
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string; got {value!r}.")
    return value.replace("_", "-")


def format_stimulus_filename(optimisation_type, target_angle, program_material=""):
    """Build a localisation_test-compatible stimulus filename (with .wav)."""
    opt_type = sanitize_filename_field(optimisation_type, "optimisation_type")
    angle_str = format_target_angle(target_angle)
    if "_" in angle_str:
        raise ValueError(f"target_angle formatted with underscore: {angle_str!r}.")
    parts = [opt_type, angle_str]
    material = str(program_material).strip()
    if material:
        parts.append(material)
    return "_".join(parts) + ".wav"


def infer_optimisation_type(output_dir):
    """Use the run folder name (timestamp), with '_' replaced by '-'."""
    name = Path(output_dir).resolve().name
    if not name:
        raise ValueError(f"Cannot infer optimisation_type from empty folder name: {output_dir}")
    return sanitize_filename_field(name, "optimisation_type")


def load_target_angle(output_dir):
    path = Path(output_dir) / "json" / "target_angle.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find target_angle at {path}. Pass --target-angle explicitly."
        )
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    return float(value)


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


def load_spk_filters(output_dir):
    """Load final spk_filters as (n_sources, n_taps) from a run folder."""
    filters_path = Path(output_dir) / "npy" / "spk_filters.npy"
    if not filters_path.exists():
        raise FileNotFoundError(f"Could not find spk_filters at {filters_path}")

    spk_filters = np.load(filters_path)
    manifest_path = Path(output_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if "spk_filters" in manifest and manifest["spk_filters"].get("skipped"):
            raise ValueError(
                f"spk_filters was skipped in manifest: {manifest['spk_filters'].get('reason')}"
            )

    if spk_filters.ndim != 3:
        raise ValueError(
            f"Expected spk_filters with shape (batch, n_sources, n_taps), got {spk_filters.shape}"
        )
    filters = spk_filters[0]
    if filters.ndim != 2 or filters.shape[1] < 1:
        raise ValueError(f"Expected per-source filters shaped (n_sources, n_taps), got {filters.shape}")
    return filters


def load_mono_wav(path, target_fs):
    fs, data = wavfile.read(path)
    audio = np.asarray(data, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        audio = audio / np.iinfo(data.dtype).max
    if fs != target_fs:
        import torch
        import torchaudio.functional as TF

        audio = TF.resample(
            torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
            fs,
            target_fs,
        ).squeeze(0).numpy()
    return audio


def apply_spk_filters(stimulus, spk_filters):
    """Convolve mono stimulus with each speaker filter; returns (T, n_sources)."""
    import torch
    import torchaudio.functional as TF

    stim = torch.tensor(stimulus, dtype=torch.float32).view(1, 1, -1)
    filters = torch.tensor(spk_filters, dtype=torch.float32).unsqueeze(0)
    channels = []
    for source_idx in range(filters.shape[1]):
        source_filter = filters[:, source_idx:source_idx + 1, :]
        filtered = TF.convolve(stim, source_filter, mode="full")
        channels.append(filtered.squeeze().numpy())
    min_len = min(len(ch) for ch in channels)
    return np.stack([ch[:min_len] for ch in channels], axis=1)


def load_audio_tracks(tracks_dir, fs, track_names=None):
    """Return list of (stem, mono_array) for WAVs in tracks_dir.

    If track_names is given, only those stems are loaded (order preserved).
    """
    tracks_dir = Path(tracks_dir)
    if not tracks_dir.is_dir():
        raise NotADirectoryError(f"Audio tracks directory does not exist: {tracks_dir}")

    if track_names is None:
        paths = sorted(tracks_dir.glob("*.wav"))
    else:
        paths = []
        for name in track_names:
            stem = Path(name).stem
            path = tracks_dir / f"{stem}.wav"
            if not path.exists():
                raise FileNotFoundError(f"Audio track not found: {path}")
            paths.append(path)

    if not paths:
        raise FileNotFoundError(f"No .wav files found in {tracks_dir}")

    playlist = []
    for path in paths:
        playlist.append((path.stem, load_mono_wav(path, fs)))
    return playlist


# ITU-R BS.1770 integrated loudness target.
DEFAULT_TARGET_LUFS = -30.0
DEFAULT_MAX_PEAK = 0.99


def normalize_audio(audio, fs, target_lufs=DEFAULT_TARGET_LUFS, max_peak=DEFAULT_MAX_PEAK):
    """Scale to target integrated LUFS, preserving inter-channel level.

    If the LUFS gain would push the absolute peak above max_peak, gain is
    reduced further so the file stays within that peak limit.
    """
    import pyloudnorm as pyln

    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return audio.astype(np.float32)
    data = audio.reshape(-1, 1) if audio.ndim == 1 else audio
    meter = pyln.Meter(int(fs))
    loudness = float(meter.integrated_loudness(data))
    if not np.isfinite(loudness):
        raise ValueError(f"Could not measure integrated loudness (got {loudness}).")
    gain = 10.0 ** ((float(target_lufs) - loudness) / 20.0)
    audio = audio * gain
    peak = float(np.max(np.abs(audio)))
    if peak > max_peak and peak > 0.0:
        audio = audio * (max_peak / peak)
    return audio.astype(np.float32)


def save_wav(path, audio, fs, target_lufs=DEFAULT_TARGET_LUFS, max_peak=DEFAULT_MAX_PEAK):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        path, fs, normalize_audio(audio, fs, target_lufs=target_lufs, max_peak=max_peak)
    )


def resolve_output_dirs(args):
    dirs = []
    if args.output_dir:
        for d in args.output_dir:
            path = d.resolve() if d.is_absolute() else (ROOT / d).resolve()
            if not path.is_dir():
                raise NotADirectoryError(f"Output directory does not exist: {path}")
            dirs.append(path)
    if args.config is not None and args.config.exists():
        dirs.extend(load_config_folders(args.config.resolve()))
    seen = set()
    unique = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def build_filter_jobs(output_dirs, optimisation_type, target_angle):
    """Collect per-run metadata needed to name and render stimuli."""
    jobs = []
    for output_dir in output_dirs:
        spk_filters = load_spk_filters(output_dir)
        if optimisation_type is not None:
            opt_type = sanitize_filename_field(optimisation_type, "optimisation_type")
        else:
            opt_type = infer_optimisation_type(output_dir)
        angle = float(target_angle) if target_angle is not None else load_target_angle(output_dir)
        jobs.append({
            "output_dir": output_dir,
            "spk_filters": spk_filters,
            "optimisation_type": opt_type,
            "target_angle": angle,
        })
    return jobs


def export_stimuli(
    jobs,
    tracks,
    out_dir,
    fs,
    lufs=DEFAULT_TARGET_LUFS,
    dry_run=False,
):
    """Render filter x track combinations and write named WAVs.

    Returns the list of written (or planned) Path objects.
    """
    out_dir = Path(out_dir)
    written = []
    for job in jobs:
        for material, mono in tracks:
            filename = format_stimulus_filename(
                job["optimisation_type"],
                job["target_angle"],
                program_material=material,
            )
            out_path = out_dir / filename
            n_sources, n_taps = job["spk_filters"].shape
            print(
                f"{'[dry-run] ' if dry_run else ''}"
                f"{filename}  <-  {job['output_dir'].name} "
                f"({n_sources} spk, {n_taps} taps) x {material}"
            )
            if not dry_run:
                filtered = apply_spk_filters(mono, job["spk_filters"])
                save_wav(out_path, filtered, fs, target_lufs=lufs)
            written.append(out_path)
    return written


def parse_track_names(tracks_arg):
    if tracks_arg is None:
        return None
    names = [part.strip() for part in tracks_arg.split(",") if part.strip()]
    if not names:
        raise ValueError("--tracks must list at least one track stem.")
    return names


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create multi-channel localisation-test stimulus WAVs by filtering "
            "audio tracks with saved spk_filters from optimisation runs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "localisation_stimuli_config.json",
        help="JSON config with a 'folders' list of optimisation run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        action="append",
        default=None,
        help="Optimisation run folder containing npy/spk_filters.npy. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "localisation_stimuli",
        help="Directory to write stimulus WAVs into.",
    )
    parser.add_argument(
        "--optimisation-type",
        type=str,
        default=None,
        help="Override optimisation_type for all runs (default: run-folder timestamp).",
    )
    parser.add_argument(
        "--target-angle",
        type=float,
        default=None,
        help="Override target angle in degrees for all runs (+ = left). "
        "Default: json/target_angle.json per run.",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=ROOT / "data" / "audio_tracks",
        help="Directory of mono program-material WAVs.",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="Comma-separated track stems to use (default: all .wav in --tracks-dir).",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="Sample rate in Hz. Defaults to config/hyperpameters.json env.fs.",
    )
    parser.add_argument(
        "--lufs", type=float, default=DEFAULT_TARGET_LUFS,
        help="Target integrated loudness in LUFS when writing WAVs (default: -30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned filenames without writing audio.",
    )
    args = parser.parse_args()
    if not args.output_dir and (args.config is None or not args.config.exists()):
        parser.error("provide --output-dir and/or a valid --config file with 'folders'")
    return args


def main():
    args = parse_args()

    if args.fs is not None:
        fs = args.fs
    else:
        config_path = ROOT / "config" / "hyperpameters.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                fs = int(json.load(f)["env"]["fs"])
        else:
            fs = 48000

    output_dirs = resolve_output_dirs(args)
    if not output_dirs:
        raise SystemExit("No optimisation run folders to process.")

    jobs = build_filter_jobs(output_dirs, args.optimisation_type, args.target_angle)
    tracks = load_audio_tracks(args.tracks_dir, fs, track_names=parse_track_names(args.tracks))

    print(f"fs={fs}")
    print(f"{len(jobs)} filter set(s) x {len(tracks)} track(s) -> {len(jobs) * len(tracks)} WAV(s)")
    print(f"Writing to {args.out_dir.resolve()}")

    written = export_stimuli(
        jobs,
        tracks,
        args.out_dir,
        fs=fs,
        lufs=args.lufs,
        dry_run=args.dry_run,
    )
    print(f"Done ({len(written)} file{'s' if len(written) != 1 else ''}).")


if __name__ == "__main__":
    main()
