"""Create multi-channel stimulus WAVs panned with 2D VBAP.

Applies constant-power VBAP gains to mono audio tracks so the virtual source
is steered to the requested target angle(s), using the localisation_test.py
filename convention:

  {optimisation_type}_{target_angle}[_{program_material}].wav

Angle convention: + = left (positive y). Speaker positions default to
config/hyperpameters.json geometry.speakers.

Example:
  python scripts/listening/create_vbap_stimuli.py \\
    --target-angle -30 --target-angle 0 --target-angle 30 \\
    --tracks cello_1,brass_1 \\
    --out-dir data/localisation_stimuli
"""
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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cls = _load_module(
    "create_localisation_stimuli",
    ROOT / "scripts" / "listening" / "create_localisation_stimuli.py",
)


def load_speakers_from_hyperparameters(config_path=None):
    path = Path(config_path) if config_path is not None else ROOT / "config" / "hyperpameters.json"
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    speakers = np.asarray(config["geometry"]["speakers"], dtype=np.float64)
    if speakers.ndim != 2 or speakers.shape[1] != 2:
        raise ValueError(f"Expected geometry.speakers shaped (n_sources, 2), got {speakers.shape}")
    return speakers


def load_speakers_from_output_dir(output_dir):
    path = Path(output_dir) / "npy" / "spk_pos.npy"
    if not path.exists():
        raise FileNotFoundError(f"Could not find spk_pos at {path}")
    spk_pos = np.load(path)
    if spk_pos.ndim == 1:
        spk_pos = spk_pos.reshape(-1, 2)
    spk_pos = np.asarray(spk_pos, dtype=np.float64)
    if spk_pos.ndim != 2 or spk_pos.shape[1] != 2:
        raise ValueError(f"Expected spk_pos shaped (n_sources, 2), got {spk_pos.shape}")
    return spk_pos


def speaker_angles_deg(spk_pos):
    """Azimuth of each speaker in degrees (+ = left), listener at origin."""
    spk_pos = np.asarray(spk_pos, dtype=np.float64).reshape(-1, 2)
    return np.rad2deg(np.arctan2(spk_pos[:, 1], spk_pos[:, 0]))


def unit_vector_from_angle_deg(angle_deg):
    """Unit direction for target azimuth: + = left (positive y)."""
    rad = np.deg2rad(float(angle_deg))
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)


def vbap_gains_2d(spk_pos, target_angle_deg, eps=1e-12):
    """2D VBAP gains for a target azimuth (degrees, + = left).

    Returns an array of shape (n_sources,) with non-negative, constant-power
    gains (Euclidean norm 1). If the target lies outside every speaker pair,
    the nearest speaker is used alone.
    """
    spk_pos = np.asarray(spk_pos, dtype=np.float64).reshape(-1, 2)
    n_sources = spk_pos.shape[0]
    if n_sources < 2:
        raise ValueError(f"VBAP needs at least 2 speakers, got {n_sources}.")

    norms = np.linalg.norm(spk_pos, axis=1, keepdims=True)
    if np.any(norms < eps):
        raise ValueError("Speaker positions must be non-zero (listener at origin).")
    unit_spk = spk_pos / norms
    p = unit_vector_from_angle_deg(target_angle_deg)

    angles = speaker_angles_deg(unit_spk)
    order = np.argsort(angles)
    # Consecutive pairs in angle order; wrap for >2 speakers (surround).
    n_pairs = n_sources if n_sources > 2 else 1
    pairs = [(order[i], order[(i + 1) % n_sources]) for i in range(n_pairs)]

    for i, j in pairs:
        L = np.column_stack([unit_spk[i], unit_spk[j]])
        if abs(np.linalg.det(L)) < eps:
            continue
        g_pair = np.linalg.solve(L, p)
        if np.all(g_pair >= -eps):
            g_pair = np.maximum(g_pair, 0.0)
            gains = np.zeros(n_sources, dtype=np.float64)
            gains[i] = g_pair[0]
            gains[j] = g_pair[1]
            norm = np.linalg.norm(gains)
            if norm > eps:
                gains /= norm
            return gains

    nearest = int(np.argmax(unit_spk @ p))
    gains = np.zeros(n_sources, dtype=np.float64)
    gains[nearest] = 1.0
    return gains


def apply_vbap_gains(mono, gains):
    """Scale a mono track by VBAP gains; returns (T, n_sources)."""
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)
    gains = np.asarray(gains, dtype=np.float64).reshape(-1)
    return mono[:, None] * gains[None, :]


def parse_target_angles(args):
    angles = []
    if args.target_angle:
        angles.extend(float(a) for a in args.target_angle)
    if args.target_angles:
        for part in args.target_angles.split(","):
            part = part.strip()
            if part:
                angles.append(float(part))
    if not angles:
        raise ValueError("Provide at least one target angle via --target-angle or --target-angles.")
    return angles


def export_vbap_stimuli(
    tracks,
    out_dir,
    spk_pos,
    target_angles,
    fs,
    optimisation_type="vbap",
    rms=None,
    dry_run=False,
):
    """Render angle x track combinations and write named WAVs."""
    if rms is None:
        rms = cls.DEFAULT_TARGET_RMS
    out_dir = Path(out_dir)
    opt_type = cls.sanitize_filename_field(optimisation_type, "optimisation_type")
    written = []
    for angle in target_angles:
        gains = vbap_gains_2d(spk_pos, angle)
        gain_str = ", ".join(f"{g:.4f}" for g in gains)
        for material, mono in tracks:
            filename = cls.format_stimulus_filename(
                opt_type, angle, program_material=material
            )
            out_path = out_dir / filename
            print(
                f"{'[dry-run] ' if dry_run else ''}"
                f"{filename}  <-  VBAP angle={cls.format_target_angle(angle)} "
                f"gains=[{gain_str}] x {material}"
            )
            if not dry_run:
                audio = apply_vbap_gains(mono, gains)
                cls.save_wav(out_path, audio, fs, target_rms=rms)
            written.append(out_path)
    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create multi-channel localisation-test stimulus WAVs by applying "
            "2D VBAP gains to audio tracks for one or more target angles."
        )
    )
    parser.add_argument(
        "--target-angle",
        type=float,
        action="append",
        default=None,
        help="Target azimuth in degrees (+ = left). Repeatable.",
    )
    parser.add_argument(
        "--target-angles",
        type=str,
        default=None,
        help="Comma-separated target azimuths in degrees (+ = left).",
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
        default="vbap",
        help="Filename optimisation_type field (default: vbap).",
    )
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=ROOT / "config" / "hyperpameters.json",
        help="JSON with geometry.speakers (used unless --output-dir / --spk-pos is set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional optimisation run folder; loads npy/spk_pos.npy for speaker layout.",
    )
    parser.add_argument(
        "--spk-pos",
        type=Path,
        default=None,
        help="Optional path to an (n_sources, 2) .npy of speaker positions.",
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
        "--rms", type=float, default=None,
        help="Target RMS when writing WAVs (default: create_localisation_stimuli.DEFAULT_TARGET_RMS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned filenames and gains without writing audio.",
    )
    return parser.parse_args()


def resolve_spk_pos(args):
    if args.spk_pos is not None:
        path = args.spk_pos.resolve() if args.spk_pos.is_absolute() else (ROOT / args.spk_pos).resolve()
        spk_pos = np.load(path)
        if spk_pos.ndim == 1:
            spk_pos = spk_pos.reshape(-1, 2)
        return np.asarray(spk_pos, dtype=np.float64)
    if args.output_dir is not None:
        path = args.output_dir.resolve() if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
        return load_speakers_from_output_dir(path)
    return load_speakers_from_hyperparameters(args.hyperparameters.resolve())


def main():
    args = parse_args()
    target_angles = parse_target_angles(args)

    if args.fs is not None:
        fs = args.fs
    else:
        with open(args.hyperparameters, encoding="utf-8") as f:
            fs = int(json.load(f)["env"]["fs"])

    spk_pos = resolve_spk_pos(args)
    angles = speaker_angles_deg(spk_pos)
    print(f"fs={fs}  speakers={len(spk_pos)}  speaker_angles_deg={np.round(angles, 3).tolist()}")
    print(f"target_angles={target_angles}")

    tracks = cls.load_audio_tracks(
        args.tracks_dir, fs, track_names=cls.parse_track_names(args.tracks)
    )
    n_out = len(target_angles) * len(tracks)
    print(f"{len(target_angles)} angle(s) x {len(tracks)} track(s) -> {n_out} WAV(s)")
    print(f"Writing to {args.out_dir.resolve()}")

    written = export_vbap_stimuli(
        tracks,
        args.out_dir,
        spk_pos,
        target_angles,
        fs=fs,
        optimisation_type=args.optimisation_type,
        rms=cls.DEFAULT_TARGET_RMS if args.rms is None else args.rms,
        dry_run=args.dry_run,
    )
    print(f"Done ({len(written)} file{'s' if len(written) != 1 else ''}).")


if __name__ == "__main__":
    main()
