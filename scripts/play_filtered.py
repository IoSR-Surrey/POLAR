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


def build_stimulus_playlist(fs, burst_s, silence_s, seed, audio_tracks_dir):
    playlist = [("pink noise", make_pink_noise_burst(fs, burst_s, silence_s, seed))]
    tracks_dir = Path(audio_tracks_dir)
    if tracks_dir.is_dir():
        for path in sorted(tracks_dir.glob("*.wav")):
            playlist.append((path.stem, load_mono_wav(path, fs)))
    return playlist


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


def validate_n_sources(n_sources):
    if n_sources not in (2, 4):
        raise ValueError(f"Expected 2 or 4 loudspeakers, got {n_sources}.")


def load_spk_pos(output_dir, n_sources):
    pos_path = output_dir / "npy" / "spk_pos.npy"
    if not pos_path.exists():
        return None
    spk_pos = np.load(pos_path)
    if spk_pos.ndim == 1:
        spk_pos = spk_pos.reshape(-1, 2)
    if spk_pos.shape[0] != n_sources:
        raise ValueError(
            f"spk_pos has {spk_pos.shape[0]} speakers but spk_filters has {n_sources}."
        )
    return spk_pos


def setup_description(n_sources, spk_pos=None):
    if n_sources == 2:
        return "2 speakers (stereo)"
    if spk_pos is not None:
        upper = [i for i, pos in enumerate(spk_pos) if pos[1] >= 0.0]
        lower = [i for i, pos in enumerate(spk_pos) if pos[1] < 0.0]
        if upper and lower:
            return f"4 speakers (upper {upper}, lower {lower})"
    return "4 speakers (2x2 arrays)"


def format_channel_map(channel_map):
    return ", ".join(f"spk{i}->{ch}" for i, ch in enumerate(channel_map))


def parse_channel_map(channel_map_str):
    if channel_map_str is None:
        return None
    parts = [int(x.strip()) for x in channel_map_str.split(",")]
    if not parts:
        raise ValueError("--channel-map must contain at least one channel.")
    if any(ch < 1 for ch in parts):
        raise ValueError("--channel-map channels must be 1-based positive integers.")
    return parts


def effective_channel_map(channel_map_arg, n_sources):
    if channel_map_arg is None:
        return list(range(1, n_sources + 1))
    if len(channel_map_arg) == n_sources:
        return list(channel_map_arg)
    print(
        f"Warning: channel map length {len(channel_map_arg)} != {n_sources} speakers; "
        f"using 1..{n_sources}."
    )
    return list(range(1, n_sources + 1))


def max_required_channels(channel_map_arg, filter_sets):
    return max(
        max(effective_channel_map(channel_map_arg, filter_set["n_sources"]))
        for filter_set in filter_sets
    )


def load_filter_sets(output_dirs):
    filter_sets = []
    for output_dir in output_dirs:
        spk_filters = load_spk_filters(output_dir)
        n_sources, n_taps = spk_filters.shape
        validate_n_sources(n_sources)
        spk_pos = load_spk_pos(output_dir, n_sources)
        try:
            label = str(output_dir.relative_to(ROOT))
        except ValueError:
            label = str(output_dir)
        filter_sets.append({
            "label": label,
            "spk_filters": spk_filters,
            "spk_pos": spk_pos,
            "n_sources": n_sources,
        })
        setup = setup_description(n_sources, spk_pos)
        print(f"Loaded spk_filters from {output_dir}: {n_sources} speakers, {n_taps} taps ({setup})")
        if spk_pos is not None:
            for i, pos in enumerate(spk_pos):
                print(f"  spk {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
    return filter_sets


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


def route_channels(audio, channel_map):
    audio = np.asarray(audio, dtype=np.float64)
    n_samples = audio.shape[0]
    n_physical = max(channel_map)
    routed = np.zeros((n_samples, n_physical), dtype=np.float64)
    for source_idx, physical_ch in enumerate(channel_map):
        routed[:, physical_ch - 1] = audio[:, source_idx]
    return routed


def normalize_for_playback(audio, target_peak=0.12):
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def prepare_playback_audio(logical_audio, channel_map):
    return normalize_for_playback(route_channels(logical_audio, channel_map))


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
    def __init__(self, audio, fs, device_index, channel_map):
        self.audio = audio
        self.fs = fs
        self.device_index = device_index
        self.channel_map = list(channel_map)
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

    def restart(self, audio, channel_map):
        new_map = list(channel_map)
        layout_changed = (
            tuple(new_map) != tuple(self.channel_map)
            or audio.shape[1] != self.audio.shape[1]
        )
        was_playing = self.playing
        self.audio = audio
        self.channel_map = new_map
        self.position = 0
        if layout_changed and self.stream is not None:
            self.stop()
            self.start()
            self.playing = was_playing


def run_playback_loop(playlist, filter_sets, fs, device_index, channel_map_arg):
    devices = sd.query_devices()
    device_name = devices[device_index]["name"]

    pygame.init()
    screen = pygame.display.set_mode((760, 420))
    pygame.display.set_caption("Filtered playback")
    font = pygame.font.SysFont(None, 28)

    folder_idx = 0
    source_idx = 0
    channel_map = effective_channel_map(channel_map_arg, filter_sets[folder_idx]["n_sources"])

    def filtered_audio():
        return apply_spk_filters(playlist[source_idx][1], filter_sets[folder_idx]["spk_filters"])

    player = LoopPlayer(prepare_playback_audio(filtered_audio(), channel_map), fs, device_index, channel_map)
    player.start()

    def select_source(new_idx):
        nonlocal source_idx
        source_idx = new_idx % len(playlist)
        player.set_audio(prepare_playback_audio(filtered_audio(), channel_map))

    def select_folder(new_idx):
        nonlocal folder_idx, channel_map
        folder_idx = new_idx % len(filter_sets)
        channel_map = effective_channel_map(channel_map_arg, filter_sets[folder_idx]["n_sources"])
        player.restart(prepare_playback_audio(filtered_audio(), channel_map), channel_map)

    running = True
    clock = pygame.time.Clock()
    while running:
        filter_set = filter_sets[folder_idx]
        setup = setup_description(filter_set["n_sources"], filter_set["spk_pos"])
        screen.fill((25, 25, 25))
        status = "Playing — Space to pause" if player.playing else "Paused — Space to play"
        source_name = playlist[source_idx][0]
        folder_name = filter_set["label"]
        for i, line in enumerate([
            status,
            f"Source: {source_name} ({source_idx + 1}/{len(playlist)})",
            "Up/Down to change source",
            f"Filters: {folder_name} ({folder_idx + 1}/{len(filter_sets)})",
            "Left/Right to change filters",
            f"Setup: {setup}",
            f"Channel map: {format_channel_map(channel_map)}",
            f"Device {device_index}: {device_name}",
            f"{max(channel_map)} physical channels ({len(channel_map)} speakers)   ESC to quit",
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
                elif event.key == pygame.K_UP and not getattr(event, "repeat", 0):
                    select_source(source_idx - 1)
                elif event.key == pygame.K_DOWN and not getattr(event, "repeat", 0):
                    select_source(source_idx + 1)
                elif event.key == pygame.K_LEFT and not getattr(event, "repeat", 0):
                    select_folder(folder_idx - 1)
                elif event.key == pygame.K_RIGHT and not getattr(event, "repeat", 0):
                    select_folder(folder_idx + 1)

        clock.tick(30)

    player.stop()
    pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play pink noise or audio tracks through loudspeakers filtered by saved spk_filters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "play_filtered_config.json",
        help="JSON config with a list of output folders to cycle through.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to a single optimisation output run folder.")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate in Hz.")
    parser.add_argument("--burst-s", type=float, default=0.25, help="Pink-noise burst duration in seconds.")
    parser.add_argument("--silence-s", type=float, default=0.25, help="Silence before and after the burst in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pink-noise generation.")
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument(
        "--channel-map",
        default=None,
        help="Comma-separated 1-based physical output channels per logical speaker "
        "(e.g. 1,2 for stereo or 3,4,5,6 for a 4-speaker beamforming setup). "
        "Defaults to 1..N.",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--save-wav", type=Path, default=None, help="Optional path to save the filtered multi-channel WAV.")
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

    channel_map_arg = parse_channel_map(args.channel_map)
    filter_sets = load_filter_sets(output_dirs)
    first_channel_map = effective_channel_map(channel_map_arg, filter_sets[0]["n_sources"])
    if channel_map_arg is not None and len(channel_map_arg) != filter_sets[0]["n_sources"]:
        print(
            f"Note: --channel-map has {len(channel_map_arg)} entries for the first folder's "
            f"{filter_sets[0]['n_sources']} speakers; per-folder maps may fall back to 1..N."
        )

    playlist = build_stimulus_playlist(
        args.fs, args.burst_s, args.silence_s, args.seed, ROOT / "data" / "audio_tracks"
    )
    n_tracks = len(playlist) - 1
    print(f"Loaded {len(playlist)} stimuli (pink noise + {n_tracks} tracks)")

    filtered_audio = apply_spk_filters(playlist[0][1], filter_sets[0]["spk_filters"])
    routed_audio = route_channels(filtered_audio, first_channel_map)
    print(
        f"Filtered audio shape: {filtered_audio.shape} (samples, logical channels); "
        f"routed shape: {routed_audio.shape} (samples, physical channels)"
    )

    if args.save_wav is not None:
        save_wav(args.save_wav, routed_audio, args.fs)
        print(f"Saved filtered burst to {args.save_wav}")

    if args.no_play:
        return

    required_channels = max_required_channels(channel_map_arg, filter_sets)
    output_device = select_output_device(required_channels, args.device)
    run_playback_loop(playlist, filter_sets, args.fs, output_device, channel_map_arg)


if __name__ == "__main__":
    main()
