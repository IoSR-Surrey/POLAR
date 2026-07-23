import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pygame
import sounddevice as sd
from scipy.io import wavfile


def load_config_fs(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return int(config["env"]["fs"])


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


def resolve_npy_dir(folder):
    folder = folder.resolve()
    if (folder / "npy").is_dir():
        return folder / "npy"
    if folder.name == "npy" or any(folder.glob("*_binaural_signal.npy")):
        return folder
    raise FileNotFoundError(
        f"Could not find binaural signals under {folder}. "
        "Expected a run folder with npy/, or a folder containing *_binaural_signal.npy."
    )


def load_binaural_npy(path):
    data = np.load(path)
    if data.ndim == 3 and data.shape[0] == 1 and data.shape[1] == 2:
        return np.asarray(data[0].T, dtype=np.float64)
    if data.ndim == 2 and data.shape[0] == 2:
        return np.asarray(data.T, dtype=np.float64)
    if data.ndim == 2 and data.shape[1] == 2:
        return np.asarray(data, dtype=np.float64)
    raise ValueError(
        f"Expected binaural array with shape (1, 2, T), (2, T), or (T, 2); got {data.shape}."
    )


def load_binaural_sets(output_dirs):
    binaural_sets = []
    for output_dir in output_dirs:
        npy_dir = resolve_npy_dir(output_dir)
        signals = []
        for path in sorted(npy_dir.glob("*_binaural_signal.npy")):
            label = path.stem.replace("_binaural_signal", "")
            audio = load_binaural_npy(path)
            signals.append({"label": label, "path": path, "audio": audio})
        if not signals:
            raise FileNotFoundError(f"No binaural signals found in {npy_dir}")
        try:
            folder_label = str(output_dir.relative_to(ROOT))
        except ValueError:
            folder_label = str(output_dir)
        binaural_sets.append({"label": folder_label, "signals": signals})
        print(f"Loaded {len(signals)} binaural signal(s) from {output_dir}:")
        for signal in signals:
            print(f"  {signal['label']}: {signal['path'].name}, shape={signal['audio'].shape}")
    return binaural_sets


def normalize_for_playback(audio, target_peak=0.12):
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def select_output_device(device_index=None):
    devices = sd.query_devices()
    if device_index is not None:
        if device_index < 0 or device_index >= len(devices):
            raise ValueError(f"Device index {device_index} is out of range.")
        if devices[device_index]["max_output_channels"] < 2:
            raise RuntimeError(
                f"Device {device_index} ({devices[device_index]['name']}) has only "
                f"{devices[device_index]['max_output_channels']} output channels; need at least 2."
            )
        return device_index

    default_output = sd.default.device[1]
    if default_output is not None and devices[default_output]["max_output_channels"] >= 2:
        return default_output

    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] >= 2:
            return idx

    raise RuntimeError("No stereo output device was found.")


def list_devices():
    print(sd.query_devices())


def save_wav(path, audio, fs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, fs, normalize_for_playback(audio))


class LoopPlayer:
    def __init__(self, audio, fs, device_index):
        self.audio = audio
        self.fs = fs
        self.device_index = device_index
        self.playing = False
        self.position = 0
        self.stream = None

    def _callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if not self.playing:
            outdata.fill(0)
            return

        n_samples = len(self.audio)
        pos = self.position
        for i in range(frames):
            outdata[i] = self.audio[pos]
            pos += 1
            if pos >= n_samples:
                pos = 0
        self.position = pos

    def start(self):
        devices = sd.query_devices()
        sd.default.device = (None, self.device_index)
        print(f"Using output device {self.device_index}: {devices[self.device_index]['name']}")
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.audio.shape[1],
            dtype="float32",
            callback=self._callback,
            device=(None, self.device_index),
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        sd.stop()

    def toggle(self):
        self.playing = not self.playing

    def set_audio(self, audio):
        self.audio = audio
        self.position = 0


def run_playback_loop(binaural_sets, fs, device_index, target_peak):
    devices = sd.query_devices()
    device_name = devices[device_index]["name"]

    folder_idx = 0
    signal_idx = 0

    def current_signal():
        return binaural_sets[folder_idx]["signals"][signal_idx]

    def current_audio():
        return normalize_for_playback(current_signal()["audio"], target_peak=target_peak)

    player = LoopPlayer(current_audio(), fs, device_index)
    player.start()

    def select_signal(new_idx):
        nonlocal signal_idx
        signal_idx = new_idx % len(binaural_sets[folder_idx]["signals"])
        player.set_audio(current_audio())

    def select_signal_by_label(label):
        nonlocal signal_idx
        signals = binaural_sets[folder_idx]["signals"]
        for idx, signal in enumerate(signals):
            if signal["label"] == label:
                signal_idx = idx
                player.set_audio(current_audio())
                return True
        return False

    def select_folder(new_idx):
        nonlocal folder_idx, signal_idx
        folder_idx = new_idx % len(binaural_sets)
        signal_idx = signal_idx % len(binaural_sets[folder_idx]["signals"])
        player.set_audio(current_audio())

    pygame.init()
    screen = pygame.display.set_mode((760, 360))
    pygame.display.set_caption("Binaural playback")
    font = pygame.font.SysFont(None, 28)

    running = True
    clock = pygame.time.Clock()
    while running:
        signal = current_signal()
        binaural_set = binaural_sets[folder_idx]
        duration_s = signal["audio"].shape[0] / fs
        screen.fill((25, 25, 25))
        status = "Playing — Space to pause" if player.playing else "Paused — Space to play"
        for i, line in enumerate([
            status,
            f"Signal: {signal['label']} ({signal_idx + 1}/{len(binaural_set['signals'])})",
            f"File: {signal['path'].name}",
            "Left/Right to change signal",
            f"Folder: {binaural_set['label']} ({folder_idx + 1}/{len(binaural_sets)})",
            "Up/Down to change folder   I for initial",
            f"Duration: {duration_s:.3f} s   Device {device_index}: {device_name}   ESC to quit",
        ]):
            screen.blit(font.render(line, True, (220, 220, 220)), (20, 20 + 32 * i))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not getattr(event, "repeat", 0):
                    player.toggle()
                elif event.key == pygame.K_i and not getattr(event, "repeat", 0):
                    if not select_signal_by_label("initial"):
                        print("No initial binaural signal found in the current folder.")
                elif event.key == pygame.K_LEFT and not getattr(event, "repeat", 0):
                    select_signal(signal_idx - 1)
                elif event.key == pygame.K_RIGHT and not getattr(event, "repeat", 0):
                    select_signal(signal_idx + 1)
                elif event.key == pygame.K_UP and not getattr(event, "repeat", 0):
                    select_folder(folder_idx - 1)
                elif event.key == pygame.K_DOWN and not getattr(event, "repeat", 0):
                    select_folder(folder_idx + 1)

        clock.tick(30)

    player.stop()
    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play saved binaural signals from optimisation run npy folders."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "play_binaural_config.json",
        help="JSON config with a list of output folders to cycle through.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to a single run folder or npy folder containing *_binaural_signal.npy files.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="Sample rate in Hz. Defaults to config/hyperpameters.json env.fs.",
    )
    parser.add_argument(
        "--peak",
        type=float,
        default=0.12,
        help="Target peak after normalisation before playback.",
    )
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument(
        "--save-wav",
        type=Path,
        default=None,
        help="Optional path to save the normalised stereo WAV for the first loaded signal.",
    )
    parser.add_argument("--no-play", action="store_true", help="Skip playback (use with --save-wav).")
    args = parser.parse_args()
    if args.output_dir is None and (args.config is None or not args.config.exists()):
        parser.error("provide --output-dir or a valid --config file")
    return args


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output directory does not exist: {output_dir}")
        output_dirs = [output_dir]
    else:
        output_dirs = load_config_folders(args.config.resolve())

    fs = args.fs if args.fs is not None else load_config_fs(ROOT / "config" / "hyperpameters.json")
    binaural_sets = load_binaural_sets(output_dirs)

    first_signal = binaural_sets[0]["signals"][0]
    print(
        f"First signal shape: {first_signal['audio'].shape} "
        f"(samples, channels), fs={fs} Hz"
    )

    if args.save_wav is not None:
        save_wav(args.save_wav, first_signal["audio"], fs)
        print(f"Saved WAV to {args.save_wav.resolve()}")

    if args.no_play:
        return

    output_device = select_output_device(args.device)
    run_playback_loop(binaural_sets, fs, output_device, args.peak)


if __name__ == "__main__":
    main()
