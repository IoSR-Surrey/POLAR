"""Match perceived loudness of 7ch listening-test conditions with gain sliders.

Loads the condition WAVs from a reference folder (default:
data/listening_tests_7ch/mushra/0/noise/), lets you preview each with a dB
slider, saves the gains to JSON, then applies those same per-condition gains
to every matching WAV under data/listening_tests_7ch/ (mushra + localisation).

Example:
  python scripts/listening/match_condition_loudness.py
  python scripts/listening/match_condition_loudness.py --apply
  python scripts/listening/match_condition_loudness.py --list-devices
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

DEFAULT_SOURCE_DIR = ROOT / "data" / "listening_tests_7ch" / "mushra" / "0" / "noise"
DEFAULT_LIBRARY_DIR = ROOT / "data" / "listening_tests_7ch"
DEFAULT_GAINS_PATH = ROOT / "data" / "listening_tests_7ch" / "condition_gains_db.json"
DEFAULT_GAIN_RANGE_DB = 12.0


def load_wav(path):
    fs, data = wavfile.read(path)
    audio = np.asarray(data, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    return int(fs), audio


def write_wav(path, fs, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        print(f"  warning: peak {peak:.3f} > 0.99 in {path.name} (possible clipping)")
    wavfile.write(path, int(fs), np.asarray(audio, dtype=np.float32))


def db_to_lin(gain_db):
    return 10.0 ** (float(gain_db) / 20.0)


def list_condition_wavs(source_dir):
    source_dir = Path(source_dir)
    paths = sorted(source_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No WAV files found in {source_dir}")
    return paths


def condition_from_localisation_stem(stem, conditions):
    """Match longest condition name prefix in localisation filenames."""
    for cond in sorted(conditions, key=len, reverse=True):
        if stem == cond or stem.startswith(cond + "_"):
            return cond
    return None


def find_library_targets(library_dir, conditions):
    """Return [(path, condition), ...] for mushra + localisation WAVs."""
    library_dir = Path(library_dir)
    conditions = list(conditions)
    targets = []

    mushra_dir = library_dir / "mushra"
    if mushra_dir.is_dir():
        for path in sorted(mushra_dir.rglob("*.wav")):
            cond = path.stem
            if cond in conditions:
                targets.append((path, cond))

    loc_dir = library_dir / "localisation"
    if loc_dir.is_dir():
        for path in sorted(loc_dir.glob("*.wav")):
            cond = condition_from_localisation_stem(path.stem, conditions)
            if cond is not None:
                targets.append((path, cond))

    return targets


def load_gains_file(path):
    path = Path(path)
    if not path.exists():
        return {
            "gains_db": {},
            "applied_gains_db": {},
            "source_dir": None,
            "library_dir": None,
        }
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("gains_db", {})
    data.setdefault("applied_gains_db", {})
    return data


def save_gains_file(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote gains: {path}")


def apply_gains_to_library(gains_path, library_dir=None, dry_run=False):
    """Apply saved condition gains to all library WAVs (undoing any previous apply)."""
    gains_path = Path(gains_path)
    data = load_gains_file(gains_path)
    gains_db = {str(k): float(v) for k, v in data.get("gains_db", {}).items()}
    if not gains_db:
        raise ValueError(f"No gains_db found in {gains_path}. Run the GUI and Save first.")

    library_dir = Path(
        library_dir
        if library_dir is not None
        else data.get("library_dir") or DEFAULT_LIBRARY_DIR
    )
    if not library_dir.is_absolute():
        library_dir = (ROOT / library_dir).resolve()

    applied_prev = {str(k): float(v) for k, v in data.get("applied_gains_db", {}).items()}
    targets = find_library_targets(library_dir, gains_db.keys())
    if not targets:
        raise FileNotFoundError(f"No matching WAVs under {library_dir}")

    print(f"library: {library_dir}")
    print(f"gains file: {gains_path}")
    print(f"{len(targets)} file(s) to update")
    for cond, g in sorted(gains_db.items()):
        prev = applied_prev.get(cond, 0.0)
        delta = g - prev
        print(f"  {cond}: target={g:+.2f} dB  (prev applied={prev:+.2f}, delta={delta:+.2f})")

    updated = 0
    for path, cond in targets:
        target_db = gains_db[cond]
        prev_db = applied_prev.get(cond, 0.0)
        delta_db = target_db - prev_db
        scale = db_to_lin(delta_db)
        print(
            f"{'[dry-run] ' if dry_run else ''}"
            f"{path.relative_to(library_dir)}  {cond}  x{scale:.4f} ({delta_db:+.2f} dB)"
        )
        if dry_run:
            continue
        fs, audio = load_wav(path)
        write_wav(path, fs, audio * scale)
        updated += 1

    if not dry_run:
        data["applied_gains_db"] = dict(gains_db)
        data["library_dir"] = str(library_dir)
        data["applied"] = datetime.now().isoformat(timespec="seconds")
        save_gains_file(gains_path, data)
        print(f"Updated {updated} file(s).")
    else:
        print(f"Dry-run only ({len(targets)} file(s)).")


def select_output_device(device_index, min_channels):
    devices = sd.query_devices()
    if device_index is not None:
        if device_index < 0 or device_index >= len(devices):
            raise ValueError(f"Device index {device_index} is out of range.")
        if devices[device_index]["max_output_channels"] < min_channels:
            raise RuntimeError(
                f"Device {device_index} ({devices[device_index]['name']}) has only "
                f"{devices[device_index]['max_output_channels']} output channels; "
                f"need at least {min_channels}."
            )
        return device_index

    default_output = sd.default.device[1]
    if default_output is not None and devices[default_output]["max_output_channels"] >= min_channels:
        return default_output
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] >= min_channels:
            return idx
    raise RuntimeError(f"No output device with at least {min_channels} channels was found.")


def downmix_for_device(audio, out_channels):
    """Fit (T, C) audio to out_channels (sum+scale if device has fewer channels)."""
    audio = np.asarray(audio, dtype=np.float64)
    n_ch = audio.shape[1]
    if out_channels >= n_ch:
        if out_channels == n_ch:
            return audio
        out = np.zeros((audio.shape[0], out_channels), dtype=np.float64)
        out[:, :n_ch] = audio
        return out
    # Fewer device channels than file: energy-preserving sum into mono, then copy.
    mono = audio.sum(axis=1, keepdims=True) / np.sqrt(n_ch)
    return np.repeat(mono, out_channels, axis=1)


class LoopPlayer:
    """Continuously loops multi-channel audio with a live linear gain."""

    def __init__(self, fs, n_channels, device_index):
        self.fs = int(fs)
        self.n_channels = int(n_channels)
        self.device_index = device_index
        self.audio = np.zeros((1, self.n_channels), dtype=np.float32)
        self.gain = 1.0
        self.playing = False
        self.position = 0
        self.stream = None

    def set_audio(self, audio):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected (T, {self.n_channels}), got {getattr(audio, 'shape', None)}"
            )
        self.audio = audio
        self.position = 0

    def set_gain(self, gain_lin):
        self.gain = float(gain_lin)

    def _callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if not self.playing or len(self.audio) == 0:
            outdata.fill(0)
            return
        n = len(self.audio)
        pos = self.position
        end = pos + frames
        g = self.gain
        if end <= n:
            outdata[:] = self.audio[pos:end] * g
            self.position = end % n
        else:
            first = n - pos
            outdata[:first] = self.audio[pos:] * g
            rem = frames - first
            outdata[first:] = self.audio[:rem] * g
            self.position = rem % n

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.n_channels,
            dtype="float32",
            callback=self._callback,
            device=(None, self.device_index),
        )
        self.stream.start()

    def stop(self):
        self.playing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def run_gui(source_dir, gains_path, library_dir, device_index, gain_range_db, out_channels):
    import tkinter as tk
    from tkinter import messagebox, ttk

    source_dir = Path(source_dir)
    gains_path = Path(gains_path)
    library_dir = Path(library_dir)

    paths = list_condition_wavs(source_dir)
    loaded = {}
    fs0 = None
    n_file_ch = None
    for path in paths:
        fs, audio = load_wav(path)
        if fs0 is None:
            fs0 = fs
            n_file_ch = audio.shape[1]
        elif fs != fs0:
            raise ValueError(f"Sample-rate mismatch: {path.name} is {fs}, expected {fs0}")
        elif audio.shape[1] != n_file_ch:
            raise ValueError(
                f"Channel-count mismatch: {path.name} has {audio.shape[1]}, expected {n_file_ch}"
            )
        loaded[path.stem] = audio

    conditions = [p.stem for p in paths]
    existing = load_gains_file(gains_path)
    initial_gains = {
        cond: float(existing.get("gains_db", {}).get(cond, 0.0)) for cond in conditions
    }

    if out_channels is None:
        # Prefer full multichannel if available; otherwise stereo downmix.
        try:
            device_index = select_output_device(device_index, n_file_ch)
            play_channels = n_file_ch
        except RuntimeError:
            device_index = select_output_device(device_index, 2)
            play_channels = 2
            print(
                f"No {n_file_ch}-channel device found; monitoring via "
                f"{play_channels}-channel downmix."
            )
    else:
        play_channels = int(out_channels)
        device_index = select_output_device(device_index, play_channels)

    play_cache = {
        cond: downmix_for_device(audio, play_channels) for cond, audio in loaded.items()
    }
    player = LoopPlayer(fs0, play_channels, device_index)
    player.start()

    root = tk.Tk()
    root.title("Match condition loudness")

    status = tk.StringVar(value=f"source: {source_dir}  |  device #{device_index} ({play_channels} ch)")
    current = tk.StringVar(value=conditions[0])
    gain_vars = {cond: tk.DoubleVar(value=initial_gains[cond]) for cond in conditions}

    def sync_player_gain():
        cond = current.get()
        player.set_gain(db_to_lin(gain_vars[cond].get()))

    def play_condition(cond):
        current.set(cond)
        player.set_audio(play_cache[cond])
        sync_player_gain()
        player.playing = True
        status.set(f"playing: {cond}  ({gain_vars[cond].get():+.2f} dB)")

    def stop_playback():
        player.playing = False
        status.set("stopped")

    def on_gain_change(cond, _value=None):
        if current.get() == cond and player.playing:
            sync_player_gain()
            status.set(f"playing: {cond}  ({gain_vars[cond].get():+.2f} dB)")

    def collect_gains():
        return {cond: round(float(gain_vars[cond].get()), 3) for cond in conditions}

    def save_gains():
        data = load_gains_file(gains_path)
        data["gains_db"] = collect_gains()
        data["source_dir"] = str(source_dir)
        data["library_dir"] = str(library_dir)
        data["fs"] = fs0
        data["n_channels"] = n_file_ch
        data["updated"] = datetime.now().isoformat(timespec="seconds")
        data.setdefault("applied_gains_db", {})
        save_gains_file(gains_path, data)
        status.set(f"saved gains -> {gains_path.name}")
        messagebox.showinfo("Saved", f"Gains written to:\n{gains_path}")

    def save_and_apply():
        save_gains()
        if not messagebox.askyesno(
            "Apply gains",
            f"Apply these gains to all matching WAVs under:\n{library_dir}\n\n"
            "Previously applied gains (if any) will be corrected to the new values.",
        ):
            return
        try:
            apply_gains_to_library(gains_path, library_dir=library_dir, dry_run=False)
        except Exception as exc:
            messagebox.showerror("Apply failed", str(exc))
            return
        status.set("applied gains to library")
        messagebox.showinfo("Applied", f"Updated WAVs under:\n{library_dir}")

    def reset_gains():
        for cond in conditions:
            gain_vars[cond].set(0.0)
        sync_player_gain()
        status.set("gains reset to 0 dB")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(frm, text="Condition gains (dB)", font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
    )

    for i, cond in enumerate(conditions):
        row = i + 1
        ttk.Button(frm, text="Play", width=6, command=lambda c=cond: play_condition(c)).grid(
            row=row, column=0, padx=(0, 6), pady=2, sticky="w"
        )
        ttk.Label(frm, text=cond, width=22).grid(row=row, column=1, sticky="w")
        scale = ttk.Scale(
            frm,
            from_=-float(gain_range_db),
            to=float(gain_range_db),
            orient="horizontal",
            variable=gain_vars[cond],
            command=lambda v, c=cond: on_gain_change(c, v),
        )
        scale.grid(row=row, column=2, sticky="ew", padx=6)
        val_lbl = ttk.Label(frm, width=8)
        val_lbl.grid(row=row, column=3, sticky="e")

        def _update_label(cond=cond, label=val_lbl, var=gain_vars[cond]):
            label.config(text=f"{var.get():+.2f}")

        gain_vars[cond].trace_add("write", lambda *_a, c=cond: (_update_label(c), on_gain_change(c)))
        _update_label()

    frm.columnconfigure(2, weight=1)

    btns = ttk.Frame(frm)
    btns.grid(row=len(conditions) + 1, column=0, columnspan=4, sticky="ew", pady=(12, 4))
    ttk.Button(btns, text="Stop", command=stop_playback).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Reset to 0 dB", command=reset_gains).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Save gains", command=save_gains).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Save & apply to all", command=save_and_apply).pack(side="left")

    ttk.Label(frm, textvariable=status, wraplength=640).grid(
        row=len(conditions) + 2, column=0, columnspan=4, sticky="w", pady=(8, 0)
    )

    def on_close():
        stop_playback()
        player.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    play_condition(conditions[0])
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Slider UI to match condition loudness, then apply gains to all 7ch stimuli."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Folder of condition WAVs to preview (default: mushra/0/noise).",
    )
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=DEFAULT_LIBRARY_DIR,
        help="Root containing mushra/ and localisation/ to update on apply.",
    )
    parser.add_argument(
        "--gains",
        type=Path,
        default=DEFAULT_GAINS_PATH,
        help="JSON path for saved condition gains (dB).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply saved gains to the library (no GUI).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --apply, print planned gain changes without writing.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice output device index.",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=None,
        help="Force monitor channel count (e.g. 2 = stereo downmix). Default: native if possible.",
    )
    parser.add_argument(
        "--gain-range-db",
        type=float,
        default=DEFAULT_GAIN_RANGE_DB,
        help="Slider range +/- dB (default: 12).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sounddevice devices and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    source_dir = args.source_dir if args.source_dir.is_absolute() else (ROOT / args.source_dir)
    library_dir = args.library_dir if args.library_dir.is_absolute() else (ROOT / args.library_dir)
    gains_path = args.gains if args.gains.is_absolute() else (ROOT / args.gains)

    if args.apply:
        apply_gains_to_library(gains_path, library_dir=library_dir, dry_run=args.dry_run)
        return

    run_gui(
        source_dir=source_dir.resolve(),
        gains_path=gains_path.resolve(),
        library_dir=library_dir.resolve(),
        device_index=args.device,
        gain_range_db=args.gain_range_db,
        out_channels=args.out_channels,
    )


if __name__ == "__main__":
    main()
