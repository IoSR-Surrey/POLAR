"""Create 7-channel WAVs for localisation and webMUSHRA listening tests.

Loudspeaker layout (all at 2 m), 1-based channel order:
  1:+30  2:-30  3:+35  4:-35  5:+15  6:0  7:-15

Conditions:
  - reference / hidden-reference: dry mono on the loudspeaker at the target angle
  - anchor-low / anchor-mid: same as reference, Butterworth LPF at 3.5 / 7 kHz
  - vbap: 2D VBAP on the configured loudspeakers (default ch 1,2 = ±30°)
  - stereo-lambda*: spk_filters from stereo runs -> ch 1,2 (rest silent);
    optional flip_lr swaps filters 0<->1 for mirrored angles (no time reverse)
  - rodenas-4ch: 4-channel filters -> ch 1-4 (angles from config rodenas_angles_deg)

Example:
  python scripts/listening/create_7ch_listening_stimuli.py
  python scripts/listening/create_7ch_listening_stimuli.py --dry-run
"""
import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from scipy import signal
from scipy.io import wavfile

import importlib.util


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cls = _load_module(
    "create_localisation_stimuli",
    ROOT / "scripts" / "listening" / "create_localisation_stimuli.py",
)
vbap = _load_module(
    "create_vbap_stimuli",
    ROOT / "scripts" / "listening" / "create_vbap_stimuli.py",
)

N_CHANNELS = 7
DEFAULT_CONFIG = ROOT / "config" / "listening_tests_7ch.json"


def pink_noise(n, random_state=None):
    """Unit-variance pink noise (same method as play_filtered.pink_noise)."""
    rng = np.random.default_rng(random_state)
    num_bins = n // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, num_bins)
    magnitudes = rng.standard_normal(num_bins)
    spectrum = magnitudes * np.exp(1j * phases)
    frequencies = np.arange(num_bins, dtype=float)
    frequencies[0] = 1.0
    spectrum /= np.sqrt(frequencies)
    spectrum[0] = 0.0
    signal_out = np.fft.irfft(spectrum, n=n)
    signal_out -= signal_out.mean()
    signal_out /= signal_out.std()
    return signal_out


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def angle_to_channel(angle_deg, physical_speakers, tol_deg=0.6):
    """Map target/speaker azimuth to 0-based channel index."""
    angle = float(angle_deg)
    best = None
    best_err = None
    for spk in physical_speakers:
        err = abs(float(spk["angle_deg"]) - angle)
        if best_err is None or err < best_err:
            best_err = err
            best = int(spk["channel"]) - 1
    if best_err is None or best_err > tol_deg:
        raise ValueError(
            f"No physical loudspeaker within {tol_deg}° of {angle}° "
            f"(nearest error {best_err})."
        )
    return best


def apply_fade(audio, fs, fade_s):
    audio = np.asarray(audio, dtype=np.float64)
    n_fade = int(round(float(fade_s) * fs))
    if n_fade <= 0:
        return audio
    n_fade = min(n_fade, audio.shape[0] // 2)
    if n_fade <= 0:
        return audio
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float64)
    out = audio.copy()
    out[:n_fade] *= ramp
    out[-n_fade:] *= ramp[::-1]
    return out


def make_noise_mono(fs, duration_s, seed):
    n = int(round(float(duration_s) * fs))
    return pink_noise(n, random_state=seed).astype(np.float64)


def load_or_make_mono(stem, tracks_dir, fs, duration_s, fade_s, noise_seed):
    if stem == "noise":
        mono = make_noise_mono(fs, duration_s, noise_seed)
    else:
        path = Path(tracks_dir) / f"{stem}.wav"
        if not path.exists():
            raise FileNotFoundError(f"Audio track not found: {path}")
        mono = cls.load_mono_wav(path, fs)
        n = int(round(float(duration_s) * fs))
        if mono.shape[0] < n:
            raise ValueError(
                f"{path.name} is only {mono.shape[0] / fs:.2f}s; need {duration_s}s."
            )
        mono = mono[:n]
    return apply_fade(mono, fs, fade_s)


def empty_7ch(n_samples):
    return np.zeros((n_samples, N_CHANNELS), dtype=np.float64)


def place_on_channel(mono, channel_idx):
    audio = empty_7ch(len(mono))
    audio[:, channel_idx] = mono
    return audio


def match_length(audio, n_samples):
    """Crop or zero-pad to exactly n_samples (FIR 'full' conv is longer than mono)."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    n = int(n_samples)
    if audio.shape[0] == n:
        return audio
    if audio.shape[0] > n:
        return audio[:n]
    out = np.zeros((n, audio.shape[1]), dtype=np.float64)
    out[: audio.shape[0]] = audio
    return out


def lowpass_multichannel(audio, fs, cutoff_hz, order):
    """Zero-phase Butterworth LPF on every channel (anchors)."""
    audio = np.asarray(audio, dtype=np.float64)
    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff {cutoff_hz} Hz must be < Nyquist ({nyquist} Hz).")
    sos = signal.butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    out = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = signal.sosfiltfilt(sos, audio[:, ch])
    return out


def map_filters_to_7ch(filtered_nch, filter_angles_deg, physical_speakers, n_samples=None):
    """Pad/route (T, n_sources) into (T, 7) using nearest physical loudspeakers."""
    filtered_nch = np.asarray(filtered_nch, dtype=np.float64)
    if filtered_nch.ndim != 2:
        raise ValueError(f"Expected (T, n_sources), got {filtered_nch.shape}")
    if n_samples is not None:
        filtered_nch = match_length(filtered_nch, n_samples)
    n_sources = filtered_nch.shape[1]
    if len(filter_angles_deg) != n_sources:
        raise ValueError(
            f"filter_angles_deg length {len(filter_angles_deg)} != n_sources {n_sources}"
        )
    out = empty_7ch(filtered_nch.shape[0])
    used = set()
    for src_idx, angle in enumerate(filter_angles_deg):
        ch = angle_to_channel(angle, physical_speakers)
        if ch in used:
            raise ValueError(
                f"Two filter sources map to physical channel {ch + 1} "
                f"(angles include {angle})."
            )
        used.add(ch)
        out[:, ch] = filtered_nch[:, src_idx]
    return out


def speaker_angles_from_pos(spk_pos):
    spk_pos = np.asarray(spk_pos, dtype=np.float64).reshape(-1, 2)
    return np.rad2deg(np.arctan2(spk_pos[:, 1], spk_pos[:, 0]))


def load_run_filters(output_dir):
    output_dir = Path(output_dir)
    filters = cls.load_spk_filters(output_dir)
    spk_pos_path = output_dir / "npy" / "spk_pos.npy"
    if not spk_pos_path.exists():
        raise FileNotFoundError(f"Could not find spk_pos at {spk_pos_path}")
    angles = speaker_angles_from_pos(np.load(spk_pos_path))
    return filters, angles, output_dir


def format_angle_folder(angle):
    angle = float(angle)
    if angle == 0.0:
        return "0" if math.copysign(1.0, angle) > 0 else "-0"
    return f"{angle:+g}"


def write_condition_wav(path, audio, fs, lufs):
    cls.save_wav(path, audio, fs, target_lufs=lufs)


def parse_stereo_entry(value):
    """Return (run_path_str, flip_lr).

    Config values may be a path string, or {"run": path, "flip_lr": true}.
    flip_lr swaps speaker filters 0<->1 (mirror L/R); no time reversal.
    """
    if isinstance(value, str):
        return value, False
    if not isinstance(value, dict) or "run" not in value:
        raise ValueError(
            f"stereo_runs entry must be a path string or {{'run': ..., 'flip_lr': bool}}; got {value!r}"
        )
    return value["run"], bool(value.get("flip_lr", False))


def flip_lr_filters(spk_filters):
    """Swap filter 0 and 1 (speaker 1 <-> 2). No time reversal."""
    spk_filters = np.asarray(spk_filters)
    if spk_filters.ndim != 2 or spk_filters.shape[0] < 2:
        raise ValueError(
            f"flip_lr needs at least 2 speaker filters, got shape {spk_filters.shape}"
        )
    out = spk_filters.copy()
    out[[0, 1]] = out[[1, 0]]
    return out


def vbap_speaker_layout(physical_speakers, channels, distance_m=2.0):
    """Build (n, 2) speaker positions and 0-based channel indices for VBAP."""
    angle_by_ch = {int(s["channel"]): float(s["angle_deg"]) for s in physical_speakers}
    spk_pos = []
    channel_idxs = []
    for ch in channels:
        ch = int(ch)
        if ch not in angle_by_ch:
            raise ValueError(f"VBAP channel {ch} not in physical_speakers.")
        rad = np.deg2rad(angle_by_ch[ch])
        spk_pos.append([float(distance_m) * np.cos(rad), float(distance_m) * np.sin(rad)])
        channel_idxs.append(ch - 1)
    return np.asarray(spk_pos, dtype=np.float64), channel_idxs


def route_to_channels(nch_audio, channel_idxs):
    """Place (T, n_active) into (T, 7) at the given 0-based channel indices."""
    nch_audio = np.asarray(nch_audio, dtype=np.float64)
    if nch_audio.ndim != 2 or nch_audio.shape[1] != len(channel_idxs):
        raise ValueError(
            f"Expected (T, {len(channel_idxs)}), got {nch_audio.shape}"
        )
    out = empty_7ch(nch_audio.shape[0])
    for src_idx, ch in enumerate(channel_idxs):
        out[:, ch] = nch_audio[:, src_idx]
    return out


def build_jobs(config):
    """Return list of render jobs and list of skip notes."""
    physical = config["physical_speakers"]
    angles = [float(a) for a in config["angles_deg"]]
    tracks = list(config["tracks"])
    stereo_runs = config.get("stereo_runs") or {}
    rodenas_run = config.get("rodenas_run")
    rodenas_name = config.get("rodenas_condition", "rodenas-4ch")
    if "rodenas_angles_deg" in config:
        rodenas_angles = {float(a) for a in config["rodenas_angles_deg"]}
    else:
        rodenas_angles = set(angles)
    vbap_cfg = config.get("vbap") or {}
    vbap_enabled = bool(vbap_cfg.get("enabled", False))

    jobs = []
    skips = []

    for angle in angles:
        ch = angle_to_channel(angle, physical)
        for material in tracks:
            jobs.append({
                "kind": "reference",
                "condition": "reference",
                "angle": angle,
                "material": material,
                "channel_idx": ch,
            })
            jobs.append({
                "kind": "hidden_reference",
                "condition": "hidden-reference",
                "angle": angle,
                "material": material,
                "channel_idx": ch,
            })
            jobs.append({
                "kind": "anchor",
                "condition": "anchor-low",
                "angle": angle,
                "material": material,
                "channel_idx": ch,
                "cutoff_hz": float(config["anchor_low_hz"]),
            })
            jobs.append({
                "kind": "anchor",
                "condition": "anchor-mid",
                "angle": angle,
                "material": material,
                "channel_idx": ch,
                "cutoff_hz": float(config["anchor_mid_hz"]),
            })

        if vbap_enabled:
            vbap_name = vbap_cfg.get("condition", "vbap")
            for material in tracks:
                jobs.append({
                    "kind": "vbap",
                    "condition": vbap_name,
                    "angle": angle,
                    "material": material,
                })

        angle_key = str(int(angle)) if float(angle).is_integer() else str(angle)
        # also accept "+15" / "-15" keys
        stereo_for_angle = None
        for key in (angle_key, f"{angle:+g}", f"{angle:g}", str(angle)):
            if key in stereo_runs:
                stereo_for_angle = stereo_runs[key]
                break
        if stereo_for_angle is None:
            stereo_for_angle = {}

        for condition, entry in stereo_for_angle.items():
            if not entry:
                skips.append(f"stereo {condition} @ {angle}°: empty run path")
                continue
            run_path, flip_lr = parse_stereo_entry(entry)
            path = resolve_path(run_path)
            if not path.is_dir():
                skips.append(f"stereo {condition} @ {angle}°: missing {path}")
                continue
            for material in tracks:
                jobs.append({
                    "kind": "filtered",
                    "condition": condition,
                    "angle": angle,
                    "material": material,
                    "run_dir": path,
                    "flip_lr": flip_lr,
                })

        if not stereo_for_angle:
            skips.append(
                f"stereo @ {angle}°: no runs in config yet "
                "(add folders under stereo_runs when optimisations finish)"
            )

        if rodenas_run and float(angle) in rodenas_angles:
            path = resolve_path(rodenas_run)
            if not path.is_dir():
                skips.append(f"{rodenas_name} @ {angle}°: missing {path}")
            else:
                for material in tracks:
                    jobs.append({
                        "kind": "filtered",
                        "condition": rodenas_name,
                        "angle": angle,
                        "material": material,
                        "run_dir": path,
                        "flip_lr": False,
                    })

    return jobs, skips


def render_job(job, mono_cache, filter_cache, physical, fs, lpf_order, vbap_cfg=None):
    material = job["material"]
    mono = mono_cache[material]
    n_samples = len(mono)

    if job["kind"] in ("reference", "hidden_reference"):
        return place_on_channel(mono, job["channel_idx"])

    if job["kind"] == "anchor":
        ref = place_on_channel(mono, job["channel_idx"])
        # sosfiltfilt can change length slightly at the edges; force exact length.
        return match_length(
            lowpass_multichannel(ref, fs, job["cutoff_hz"], lpf_order),
            n_samples,
        )

    if job["kind"] == "vbap":
        vbap_cfg = vbap_cfg or {}
        channels = vbap_cfg.get("channels", [1, 2])
        distance_m = float(vbap_cfg.get("distance_m", 2.0))
        spk_pos, channel_idxs = vbap_speaker_layout(physical, channels, distance_m)
        gains = vbap.vbap_gains_2d(spk_pos, job["angle"])
        panned = vbap.apply_vbap_gains(mono, gains)
        return route_to_channels(match_length(panned, n_samples), channel_idxs)

    if job["kind"] == "filtered":
        run_dir = Path(job["run_dir"])
        key = str(run_dir)
        if key not in filter_cache:
            filter_cache[key] = load_run_filters(run_dir)
        spk_filters, filter_angles, _ = filter_cache[key]
        if job.get("flip_lr"):
            spk_filters = flip_lr_filters(spk_filters)
        filtered = cls.apply_spk_filters(mono, spk_filters)
        return map_filters_to_7ch(
            filtered, filter_angles, physical, n_samples=n_samples,
        )

    raise ValueError(f"Unknown job kind: {job['kind']}")


def localisation_path(out_dir, condition, angle, material):
    filename = cls.format_stimulus_filename(condition, angle, program_material=material)
    return Path(out_dir) / "localisation" / filename


def mushra_path(out_dir, condition, angle, material):
    angle_folder = format_angle_folder(angle)
    return Path(out_dir) / "mushra" / angle_folder / material / f"{condition}.wav"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 7-channel localisation / webMUSHRA stimulus WAVs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON config (default: config/listening_tests_7ch.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output root from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned files without writing audio.",
    )
    parser.add_argument(
        "--lufs",
        type=float,
        default=cls.DEFAULT_TARGET_LUFS,
        help="Target integrated loudness in LUFS when writing WAVs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve() if args.config.is_absolute() else (ROOT / args.config).resolve()
    config = load_config(config_path)

    out_dir = resolve_path(args.out_dir if args.out_dir is not None else config["out_dir"])
    fs = int(config.get("fs", 48000))
    duration_s = float(config.get("duration_s", 8.0))
    fade_s = float(config.get("fade_s", 0.05))
    lpf_order = int(config.get("lpf_order", 4))
    tracks_dir = resolve_path(config.get("tracks_dir", "data/audio_tracks"))
    physical = config["physical_speakers"]
    noise_seed = int(config.get("noise_seed", 0))
    vbap_cfg = config.get("vbap") or {}

    jobs, skips = build_jobs(config)

    print(f"config: {config_path}")
    print(f"out_dir: {out_dir}")
    print(f"fs={fs}  duration_s={duration_s}  fade_s={fade_s}  lpf_order={lpf_order}")
    print(f"{len(jobs)} file(s) to write")
    for note in skips:
        print(f"skip: {note}")

    mono_cache = {}
    for material in config["tracks"]:
        if args.dry_run and material != "noise":
            # still verify tracks exist
            path = tracks_dir / f"{material}.wav"
            if material != "noise" and not path.exists():
                raise FileNotFoundError(f"Audio track not found: {path}")
            mono_cache[material] = None
        else:
            mono_cache[material] = load_or_make_mono(
                material, tracks_dir, fs, duration_s, fade_s, noise_seed,
            )

    filter_cache = {}
    written = []
    for job in jobs:
        loc_path = localisation_path(out_dir, job["condition"], job["angle"], job["material"])
        mush_path = mushra_path(out_dir, job["condition"], job["angle"], job["material"])
        print(
            f"{'[dry-run] ' if args.dry_run else ''}"
            f"{job['condition']}  angle={cls.format_target_angle(job['angle'])}  "
            f"material={job['material']}"
        )
        if args.dry_run:
            written.append(loc_path)
            continue

        audio = render_job(
            job, mono_cache, filter_cache, physical, fs, lpf_order, vbap_cfg=vbap_cfg,
        )
        write_condition_wav(loc_path, audio, fs, args.lufs)
        mush_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(loc_path, mush_path)
        written.append(loc_path)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": str(config_path),
        "out_dir": str(out_dir),
        "fs": fs,
        "duration_s": duration_s,
        "fade_s": fade_s,
        "n_channels": N_CHANNELS,
        "physical_speakers": physical,
        "channel_map_note": (
            "WAV column i is loudspeaker channel i+1 "
            "(1:+30, 2:-30, 3:+35, 4:-35, 5:+15, 6:0, 7:-15)."
        ),
        "anchors": {
            "filter": f"Butterworth sosfiltfilt order={lpf_order}",
            "anchor-low_hz": config["anchor_low_hz"],
            "anchor-mid_hz": config["anchor_mid_hz"],
        },
        "vbap": vbap_cfg,
        "skips": skips,
        "n_files": len(written),
        "localisation_dir": str(out_dir / "localisation"),
        "mushra_dir": str(out_dir / "mushra"),
        "naming": {
            "localisation": "{condition}_{angle}_{material}.wav",
            "mushra": "mushra/{angle}/{material}/{condition}.wav",
        },
    }
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest: {manifest_path}")

    print(f"Done ({len(written)} file{'s' if len(written) != 1 else ''}).")


if __name__ == "__main__":
    main()
