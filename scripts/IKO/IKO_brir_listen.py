import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
import sounddevice as sd
import torch
import torchaudio.functional as TF
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location(
    "iko_brir_optimisation", ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py",
)
iko_brir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iko_brir)

play_binaural_spec = importlib.util.spec_from_file_location(
    "play_binaural", ROOT / "scripts" / "listening" / "play_binaural.py",
)
play_binaural = importlib.util.module_from_spec(play_binaural_spec)
play_binaural_spec.loader.exec_module(play_binaural)
LoopPlayer = play_binaural.LoopPlayer
list_devices = play_binaural.list_devices
normalize_for_playback = play_binaural.normalize_for_playback
select_output_device = play_binaural.select_output_device

from utils.IKO_functions import num_hoa_channels


N_IKO_LOUDSPEAKERS = 20
IKO_DATA_DIR = Path("data/IKO")
SPH_TO_DRIVER_NPY = IKO_DATA_DIR / "hoa_to_driver" / "hoa_to_driver.npy"
BRIR_NPY = IKO_DATA_DIR / "driver_to_vl" / "brir_centre.npy"


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


def one_hot_hoa_weight(order, channel_idx):
    """Return a one-hot HOA weight vector for the given channel index."""
    n_channels = num_hoa_channels(order)
    if channel_idx < 0 or channel_idx >= n_channels:
        raise ValueError(f"channel_idx must be in [0, {n_channels - 1}], got {channel_idx}.")
    weights = np.zeros(n_channels, dtype=np.float64)
    weights[channel_idx] = 1.0
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


def build_mono_stimulus(args, fs, seed):
    if args.input_wav is not None:
        return load_mono_wav(args.input_wav, fs)
    if args.burst:
        return iko_brir.make_pink_noise_burst(fs, args.burst_s, args.silence_s, seed)
    return iko_brir.pink_noise(int(round(args.duration_s * fs)), random_state=seed)


def binaural_to_playback_audio(binaural_signal):
    """Convert (1, 2, T) torch tensor to (T, 2) float array for playback."""
    audio = binaural_signal.detach().cpu().numpy()
    if audio.ndim == 3 and audio.shape[0] == 1 and audio.shape[1] == 2:
        return np.asarray(audio[0].T, dtype=np.float64)
    raise ValueError(f"Expected binaural shape (1, 2, T), got {tuple(audio.shape)}.")


def render_binaural(weights, mono_signal, device, dtype, sph_to_driver_path, brir_path):
    n_eigenbeams = weights.shape[0]
    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        sph_to_driver_path, n_eigenbeams, N_IKO_LOUDSPEAKERS, device, dtype,
    )
    brirs = iko_brir.load_iko_brirs(brir_path, N_IKO_LOUDSPEAKERS, device, dtype)
    weights_t = torch.tensor(weights, device=device, dtype=dtype)
    signal_t = torch.tensor(mono_signal, device=device, dtype=dtype).view(1, -1)
    binaural_signal, speaker_ir = iko_brir.render_binaural_brir(
        signal_t, weights_t, eigenbeam_to_loudspeaker, brirs,
    )
    return binaural_signal, speaker_ir


def save_outputs(output_dir, weights, mono_signal, speaker_ir, binaural_signal, fs, peak):
    output_dir = Path(output_dir)
    npy_dir = output_dir / "npy"
    wav_dir = output_dir / "wav"
    npy_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    np.save(npy_dir / "hoa_weights.npy", np.asarray(weights, dtype=np.float64).reshape(-1))
    np.save(npy_dir / "mono_signal.npy", np.asarray(mono_signal, dtype=np.float64).reshape(-1))
    np.save(npy_dir / "speaker_ir.npy", speaker_ir.detach().cpu().numpy())
    np.save(npy_dir / "binaural_signal.npy", binaural_signal.detach().cpu().numpy())

    playback_audio = normalize_for_playback(binaural_to_playback_audio(binaural_signal), target_peak=peak)
    wav_path = wav_dir / "binaural.wav"
    wavfile.write(wav_path, fs, playback_audio)
    return npy_dir, wav_path


def sh_azimuth_to_screen(cx, cy, radius, azimuth_rad):
    """Map SH/matplotlib polar azimuth (0=east, CCW) to pygame screen coordinates."""
    return (
        int(cx + radius * np.cos(azimuth_rad)),
        int(cy - radius * np.sin(azimuth_rad)),
    )


def polar_pattern_polyline(order, weights, plot_radius, cx, cy):
    """Return pygame polyline segment(s) for the horizontal beam pattern.

    Uses |w·Y| as radius (signed values fold in polar coords) and appends a 2π
    wrap sample so the curve closes along the perimeter without a cross-circle chord.
    """
    weights_t = torch.tensor(np.asarray(weights, dtype=np.float64).reshape(-1))
    azimuths, pattern = iko_brir.beam_pattern_over_circle(order, weights_t)
    plot_azimuths = iko_brir._beam_pattern_plot_azimuths(azimuths)
    pattern = np.abs(pattern)

    sort_idx = np.argsort(plot_azimuths)
    plot_azimuths = plot_azimuths[sort_idx]
    pattern = pattern[sort_idx]
    plot_azimuths = np.concatenate([plot_azimuths, [plot_azimuths[0] + 2 * np.pi]])
    pattern = np.concatenate([pattern, [pattern[0]]])

    peak = max(float(np.max(pattern)), 1e-6)
    points = [
        sh_azimuth_to_screen(cx, cy, (val / peak) * plot_radius, az)
        for az, val in zip(plot_azimuths, pattern)
    ]
    return [points]


def make_polar_pattern_surface(order, weights, size, fs):
    """Render horizontal beam pattern as a pygame surface."""
    surface = pygame.Surface((size, size))
    surface.fill((35, 35, 35))
    cx, cy = size // 2, size // 2
    plot_radius = size // 2 - 16

    for frac in (0.25, 0.5, 0.75, 1.0):
        pygame.draw.circle(surface, (60, 60, 60), (cx, cy), int(plot_radius * frac), 1)
    pygame.draw.line(surface, (50, 50, 50), (cx, cy - plot_radius), (cx, cy + plot_radius), 1)
    pygame.draw.line(surface, (50, 50, 50), (cx - plot_radius, cy), (cx + plot_radius, cy), 1)

    for segment in polar_pattern_polyline(order, weights, plot_radius, cx, cy):
        if len(segment) > 1:
            pygame.draw.lines(surface, (100, 180, 255), False, segment, 2)

    listener_positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    geometry = iko_brir.setup_geometry(2.0, 30.0, -30.0, listener_positions, fs)
    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])
    for az, color in [(listener_az, (80, 140, 255)), (reflector_az, (140, 140, 140))]:
        plot_az = float(iko_brir._beam_pattern_plot_azimuths(az))
        x, y = sh_azimuth_to_screen(cx, cy, plot_radius, plot_az)
        pygame.draw.line(surface, color, (cx, cy), (x, y), 2)

    return surface


def run_playback_loop(audio, fs, device_index, peak, order, weights):
    player = LoopPlayer(normalize_for_playback(audio, target_peak=peak), fs, device_index)
    player.start()

    pygame.init()
    screen = pygame.display.set_mode((760, 420))
    pygame.display.set_caption("IKO BRIR binaural playback")
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 22)
    pattern_surface = make_polar_pattern_surface(order, weights, 320, fs)
    devices = sd.query_devices()
    device_name = devices[device_index]["name"]
    duration_s = audio.shape[0] / fs
    pattern_rect = pattern_surface.get_rect(topleft=(400, 50))

    running = True
    clock = pygame.time.Clock()
    while running:
        screen.fill((25, 25, 25))
        status = "Playing — Space to pause" if player.playing else "Paused — Space to play"
        for i, line in enumerate([
            status,
            f"Duration: {duration_s:.3f} s   Device {device_index}: {device_name}",
            "ESC to quit",
        ]):
            screen.blit(font.render(line, True, (220, 220, 220)), (20, 20 + 32 * i))

        screen.blit(font.render("Beam pattern", True, (220, 220, 220)), (pattern_rect.centerx - 70, 16))
        screen.blit(pattern_surface, pattern_rect)
        screen.blit(small_font.render("Listener", True, (80, 140, 255)), (pattern_rect.right + 12, pattern_rect.top + 8))
        screen.blit(small_font.render("Reflector", True, (140, 140, 140)), (pattern_rect.right + 12, pattern_rect.top + 32))
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
        description="Render custom HOA weights through IKO BRIR propagation and listen.",
    )
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument(
        "--weights",
        help="HOA weights as a .npy/.json file path or comma-separated values (16 floats for order 3).",
    )
    weight_group.add_argument(
        "--one-hot",
        type=int,
        metavar="CHANNEL",
        help="Use a one-hot HOA weight vector (channel index 0..15 for order 3).",
    )
    parser.add_argument("--order", type=int, default=3, help="HOA order (default 3 -> 16 channels).")
    parser.add_argument("--input-wav", type=str, default=None, help="Optional mono WAV stimulus.")
    parser.add_argument("--burst", action="store_true", help="Use a pink-noise burst (default stimulus).")
    parser.add_argument("--continuous", action="store_true", help="Use continuous pink noise instead of a burst.")
    parser.add_argument("--duration-s", type=float, default=3.0, help="Duration for continuous pink noise.")
    parser.add_argument("--burst-s", type=float, default=0.25, help="Burst length when using burst stimulus.")
    parser.add_argument("--silence-s", type=float, default=0.25, help="Leading/trailing silence for burst stimulus.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to config value).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for npy/wav outputs.")
    parser.add_argument("--sph-to-driver", type=Path, default=SPH_TO_DRIVER_NPY, help="Path to hoa_to_driver.npy.")
    parser.add_argument("--brir", type=Path, default=BRIR_NPY, help="Path to measured loudspeaker BRIRs (.npy).")
    parser.add_argument("--peak", type=float, default=0.12, help="Target peak after normalisation before playback.")
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--no-play", action="store_true", help="Skip playback (still saves outputs).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    with open("config/hyperpameters.json", "r") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    device = config["env"]["device"]
    dtype = torch.float32
    seed = config["env"].get("seed", 0) if args.seed is None else args.seed

    if args.one_hot is not None:
        weights = one_hot_hoa_weight(args.order, args.one_hot)
    else:
        weights = load_weights(args.weights, args.order)

    if args.continuous:
        args.burst = False
    elif not args.burst and args.input_wav is None:
        args.burst = True

    mono_signal = build_mono_stimulus(args, fs, seed)
    binaural_signal, speaker_ir = render_binaural(
        weights, mono_signal, device, dtype, args.sph_to_driver.resolve(), args.brir.resolve(),
    )
    playback_audio = binaural_to_playback_audio(binaural_signal)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = Path("outputs") / "iko_brir_listen" / timestamp
    else:
        output_dir = args.output_dir.resolve()
    npy_dir, wav_path = save_outputs(
        output_dir, weights, mono_signal, speaker_ir, binaural_signal, fs, args.peak,
    )

    print(f"HOA weights ({len(weights)} ch): {weights}")
    print(
        f"Binaural shape: {tuple(binaural_signal.shape)} "
        f"({playback_audio.shape[0]} samples @ {fs} Hz)"
    )
    print(f"Saved npy to {npy_dir.resolve()}")
    print(f"Saved WAV to {wav_path.resolve()}")

    if args.no_play:
        return

    output_device = select_output_device(args.device)
    run_playback_loop(playback_audio, fs, output_device, args.peak, args.order, weights)


if __name__ == "__main__":
    main()
