import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pygame
import sounddevice as sd
import torch
import torchaudio.functional as TF
from scipy.io import wavfile


def pink_noise(n, random_state=None):
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


def make_pink_noise_burst(fs, burst_s, silence_s, seed, target_peak=0.1):
    burst_len = int(round(burst_s * fs))
    silence_len = int(round(silence_s * fs))
    burst = pink_noise(burst_len, random_state=seed)
    peak = np.max(np.abs(burst))
    if peak > 0:
        burst = burst * (target_peak / peak)
    return np.concatenate([
        np.zeros(silence_len, dtype=np.float64),
        burst,
        np.zeros(silence_len, dtype=np.float64),
    ])


def load_spk_filters(output_dir):
    filters_path = output_dir / "npy" / "spk_filters.npy"
    if not filters_path.exists():
        raise FileNotFoundError(f"Could not find spk_filters at {filters_path}")

    spk_filters = np.load(filters_path)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        if "spk_filters" in manifest and manifest["spk_filters"].get("skipped"):
            raise ValueError(f"spk_filters was skipped in manifest: {manifest['spk_filters'].get('reason')}")

    if spk_filters.ndim != 3:
        raise ValueError(f"Expected spk_filters with shape (batch, n_sources, n_taps), got {spk_filters.shape}")

    filters = spk_filters[0]
    if filters.ndim != 2 or filters.shape[1] < 1:
        raise ValueError(f"Expected per-source filters shaped (n_sources, n_taps), got {filters.shape}")

    return filters


def apply_spk_filters(stimulus, spk_filters):
    stim = torch.tensor(stimulus, dtype=torch.float32).view(1, 1, -1)
    filters = torch.tensor(spk_filters, dtype=torch.float32).unsqueeze(0)
    channels = []
    for source_idx in range(filters.shape[1]):
        source_filter = filters[:, source_idx:source_idx + 1, :]
        filtered = TF.convolve(stim, source_filter, mode="full")
        channels.append(filtered.squeeze().numpy())
    min_len = min(len(ch) for ch in channels)
    return np.stack([ch[:min_len] for ch in channels], axis=1)


def normalize_for_playback(audio, target_peak=0.9):
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def select_output_device(n_channels, device_index=None):
    devices = sd.query_devices()
    if device_index is not None:
        if device_index < 0 or device_index >= len(devices):
            raise ValueError(f"Device index {device_index} is out of range.")
        if devices[device_index]["max_output_channels"] < n_channels:
            raise RuntimeError(
                f"Device {device_index} ({devices[device_index]['name']}) has only "
                f"{devices[device_index]['max_output_channels']} output channels; "
                f"need at least {n_channels}."
            )
        return device_index

    default_output = sd.default.device[1]
    if default_output is not None and devices[default_output]["max_output_channels"] >= n_channels:
        return default_output

    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] >= n_channels:
            return idx

    raise RuntimeError(f"No output device with at least {n_channels} channels was found.")


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


def run_playback_loop(audio, fs, device_index):
    devices = sd.query_devices()
    device_name = devices[device_index]["name"]

    pygame.init()
    screen = pygame.display.set_mode((640, 200))
    pygame.display.set_caption("Filtered pink-noise burst")
    font = pygame.font.SysFont(None, 28)

    player = LoopPlayer(normalize_for_playback(audio), fs, device_index)
    player.start()
    n_channels = player.audio.shape[1]

    running = True
    clock = pygame.time.Clock()
    while running:
        screen.fill((25, 25, 25))
        status = "Playing — Space to pause" if player.playing else "Paused — Space to play"
        for i, line in enumerate([
            status,
            f"Device {device_index}: {device_name}",
            f"{n_channels} channels   ESC to quit",
        ]):
            screen.blit(font.render(line, True, (220, 220, 220)), (20, 24 + 36 * i))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not getattr(event, "repeat", 0):
                    player.toggle()

        clock.tick(30)

    player.stop()
    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a pink-noise burst through loudspeakers filtered by saved spk_filters."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to an optimisation output run folder.")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz.")
    parser.add_argument("--burst-s", type=float, default=0.5, help="Pink-noise burst duration in seconds.")
    parser.add_argument("--silence-s", type=float, default=0.5, help="Silence before and after the burst in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pink-noise generation.")
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--save-wav", type=Path, default=None, help="Optional path to save the filtered multi-channel WAV.")
    parser.add_argument("--no-play", action="store_true", help="Skip playback (use with --save-wav).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")

    spk_filters = load_spk_filters(output_dir)
    n_sources, n_taps = spk_filters.shape
    print(f"Loaded spk_filters from {output_dir} with shape ({n_sources}, {n_taps})")

    stimulus = make_pink_noise_burst(args.fs, args.burst_s, args.silence_s, args.seed)
    filtered_audio = apply_spk_filters(stimulus, spk_filters)
    print(f"Filtered audio shape: {filtered_audio.shape} (samples, channels)")

    if args.save_wav is not None:
        save_wav(args.save_wav, filtered_audio, args.fs)
        print(f"Saved filtered burst to {args.save_wav}")

    if args.no_play:
        return

    output_device = select_output_device(n_sources, args.device)
    run_playback_loop(filtered_audio, args.fs, output_device)


if __name__ == "__main__":
    main()
