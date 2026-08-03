import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pygame
import sounddevice as sd
import torch
import torchaudio.functional as TF
from scipy.io import wavfile


ICLD_STEP_DB = 1.0
ICLD_LIMIT_DB = 20.0
ICTD_STEP_MS = 0.1
ICTD_LIMIT_MS = 10.0


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
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def make_white_noise(fs, duration_s=3.0, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(int(round(duration_s * fs)))
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak
    return noise


def build_material_playlist(audio_tracks_dir, target_fs):
    tracks_dir = Path(audio_tracks_dir)
    if not tracks_dir.is_dir():
        raise NotADirectoryError(f"Audio tracks directory does not exist: {tracks_dir}")
    playlist = [("white noise", make_white_noise(target_fs))]
    for path in sorted(tracks_dir.glob("*.wav")):
        playlist.append((path.stem, load_mono_wav(path, target_fs)))
    if len(playlist) == 1:
        raise FileNotFoundError(f"No .wav files found in {tracks_dir}")
    return playlist


def select_material_index(digit, n_tracks):
    """Map a number-key digit (0-9) to a track index, or None if out of range."""
    if 0 <= digit < n_tracks:
        return digit
    return None


def constant_power_gains(icld_db):
    """Constant-power stereo gains. + icld_db = louder left; left^2 + right^2 = 1."""
    right_gain = np.sqrt(1.0 / (1.0 + 10.0 ** (icld_db / 10.0)))
    left_gain = right_gain * 10.0 ** (icld_db / 20.0)
    return left_gain, right_gain


def fractional_delay(signal, delay_samples, n_fft=None):
    """Delay a signal by (possibly fractional) samples. Positive delay = later."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[-1]
    if n_fft is None:
        n_fft = n
    spectrum = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft)
    phase = np.exp(-2j * np.pi * freqs * delay_samples)
    delayed = np.fft.irfft(spectrum * phase, n=n_fft)
    return delayed[:n]


def apply_cues(left, right, icld_db, ictd_s, fs):
    """Apply ICLD/ICTD cues on top of an existing stereo pair.

    icld_db: + = louder left channel.
    ictd_s:  + = left leads (delay applied to the right channel).
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_gain, right_gain = constant_power_gains(icld_db)
    left = left * left_gain
    right = right * right_gain
    delay_samples = ictd_s * fs
    if delay_samples >= 0:
        right = fractional_delay(right, delay_samples)
    else:
        left = fractional_delay(left, -delay_samples)
    return np.stack([left, right], axis=1)


def render_stereo(mono, icld_db, ictd_s, fs):
    """Render a mono track to a stereo pair with the given ICLD/ICTD cues."""
    mono = np.asarray(mono, dtype=np.float64)
    return apply_cues(mono, mono, icld_db, ictd_s, fs)


def load_stereo_filters(filters_path):
    """Load a 2-channel filter set from a run folder or .npy file; returns (2, n_taps)."""
    path = Path(filters_path)
    if path.is_dir():
        for candidate in (path / "npy" / "spk_filters.npy", path / "spk_filters.npy"):
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not find spk_filters.npy under {filters_path} (looked in ./ and ./npy/)."
            )
    filters = np.load(path)
    if filters.ndim == 3:
        filters = filters[0]
    if filters.ndim != 2:
        raise ValueError(f"Expected filters shaped (n_sources, n_taps) or (batch, n_sources, n_taps); got {filters.shape}.")
    if filters.shape[0] != 2:
        raise ValueError(f"Expected 2 stereo filters, got {filters.shape[0]}.")
    return filters.astype(np.float64)


def apply_stereo_filters(mono, filters):
    """Convolve a mono track with the left/right filters; returns (left, right)."""
    stim = torch.tensor(np.asarray(mono), dtype=torch.float32).view(1, 1, -1)
    filt = torch.tensor(filters, dtype=torch.float32).unsqueeze(0)
    channels = []
    for source_idx in range(2):
        source_filter = filt[:, source_idx:source_idx + 1, :]
        filtered = TF.convolve(stim, source_filter, mode="full")
        channels.append(filtered.squeeze().numpy())
    min_len = min(len(ch) for ch in channels)
    return channels[0][:min_len], channels[1][:min_len]


def render_material(mono, filters, icld_db, ictd_s, fs):
    """Render a mono track through optional stereo filters, then apply ICLD/ICTD on top."""
    if filters is None:
        left = right = np.asarray(mono, dtype=np.float64)
    else:
        left, right = apply_stereo_filters(mono, filters)
    return apply_cues(left, right, icld_db, ictd_s, fs)


def normalize_for_playback(audio, target_peak=0.12):
    """Peak-normalize the whole stereo array jointly to preserve the ICLD/ICTD ratio."""
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


def run_playback_loop(playlist, fs, device_index, target_peak, filters=None, filters_label=None):
    devices = sd.query_devices()
    device_name = devices[device_index]["name"]

    material_idx = 0
    icld_db = 0.0
    ictd_ms = 0.0
    # Arrows edit the pending cues; they are only applied to playback on Enter.
    pending_icld_db = 0.0
    pending_ictd_ms = 0.0

    def rendered_audio():
        mono = playlist[material_idx][1]
        stereo = render_material(mono, filters, icld_db, ictd_ms / 1000.0, fs)
        return normalize_for_playback(stereo, target_peak=target_peak)

    player = LoopPlayer(rendered_audio(), fs, device_index)
    player.start()
    player.playing = False

    def refresh_audio():
        player.set_audio(rendered_audio())

    def select_material(new_idx):
        nonlocal material_idx
        material_idx = new_idx % len(playlist)
        refresh_audio()

    def set_pending_icld(value):
        nonlocal pending_icld_db
        pending_icld_db = float(np.clip(value, -ICLD_LIMIT_DB, ICLD_LIMIT_DB))

    def set_pending_ictd(value):
        nonlocal pending_ictd_ms
        pending_ictd_ms = round(float(np.clip(value, -ICTD_LIMIT_MS, ICTD_LIMIT_MS)), 1)

    def apply_pending():
        nonlocal icld_db, ictd_ms
        icld_db = pending_icld_db
        ictd_ms = pending_ictd_ms
        refresh_audio()

    filter_line = f"Filters: {filters_label}" if filters is not None else "Filters: none (direct stereo)"

    pygame.init()
    screen = pygame.display.set_mode((760, 460))
    pygame.display.set_caption("ICLD / ICTD listening")
    font = pygame.font.SysFont(None, 28)

    running = True
    clock = pygame.time.Clock()
    while running:
        material_name = playlist[material_idx][0]
        icld_sign = "left louder" if pending_icld_db > 0 else ("right louder" if pending_icld_db < 0 else "centred")
        ictd_sign = "left leads" if pending_ictd_ms > 0 else ("right leads" if pending_ictd_ms < 0 else "centred")
        dirty = pending_icld_db != icld_db or pending_ictd_ms != ictd_ms
        apply_status = "Enter to apply pending change" if dirty else "Applied"
        status = "Playing - Space to pause" if player.playing else "Paused - Space to play"
        screen.fill((25, 25, 25))
        for i, line in enumerate([
            status,
            f"Material: {material_name} (key {material_idx})",
            "0-9 to change program material",
            f"ICLD: {pending_icld_db:+.0f} dB ({icld_sign})   [applied {icld_db:+.0f} dB]",
            "Up/Down to change ICLD (+/- means left/right louder)",
            f"ICTD: {pending_ictd_ms:+.1f} ms ({ictd_sign})   [applied {ictd_ms:+.1f} ms]",
            "Right/Left to change ICTD (+/- means left/right leads)",
            apply_status,
            filter_line,
            f"Device {device_index}: {device_name}   ESC to quit",
        ]):
            screen.blit(font.render(line, True, (220, 220, 220)), (20, 20 + 40 * i))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not getattr(event, "repeat", 0):
                    player.toggle()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and not getattr(event, "repeat", 0):
                    apply_pending()
                elif event.key == pygame.K_UP:
                    set_pending_icld(pending_icld_db + ICLD_STEP_DB)
                elif event.key == pygame.K_DOWN:
                    set_pending_icld(pending_icld_db - ICLD_STEP_DB)
                elif event.key == pygame.K_RIGHT:
                    set_pending_ictd(pending_ictd_ms + ICTD_STEP_MS)
                elif event.key == pygame.K_LEFT:
                    set_pending_ictd(pending_ictd_ms - ICTD_STEP_MS)
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    idx = select_material_index(event.key - pygame.K_0, len(playlist))
                    if idx is not None:
                        select_material(idx)

        clock.tick(30)

    player.stop()
    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively pan a stereo pair by ICLD (Up/Down) and ICTD (Left/Right); "
        "number keys switch program material."
    )
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz.")
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument(
        "--audio-tracks-dir",
        type=Path,
        default=ROOT / "data" / "audio_tracks",
        help="Directory of mono .wav program material (keys 0-9 select tracks alphabetically).",
    )
    parser.add_argument("--target-peak", type=float, default=0.12, help="Playback peak level.")
    parser.add_argument(
        "--filters",
        type=Path,
        default=None,
        help="Path to a run folder (with npy/spk_filters.npy) or a 2-channel spk_filters .npy. "
        "The material is convolved with these stereo filters, then ICLD/ICTD is applied on top.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    playlist = build_material_playlist(args.audio_tracks_dir, args.fs)
    print(f"Loaded {len(playlist)} program materials from {args.audio_tracks_dir}:")
    for idx, (name, _) in enumerate(playlist):
        print(f"  key {idx}: {name}")

    filters = None
    filters_label = None
    if args.filters is not None:
        filters = load_stereo_filters(args.filters)
        filters_label = str(args.filters)
        print(f"Loaded stereo filters from {args.filters}: shape {filters.shape} (2, n_taps)")

    device_index = select_output_device(args.device)
    run_playback_loop(playlist, args.fs, device_index, args.target_peak, filters, filters_label)


if __name__ == "__main__":
    main()
