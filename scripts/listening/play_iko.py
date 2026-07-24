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
import pygame
import sounddevice as sd
import torch
import torchaudio.functional as TF


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


play_filtered = _load_module("play_filtered", ROOT / "scripts" / "listening" / "play_filtered.py")
iko_brir = _load_module("iko_brir_optimisation", ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py")

build_stimulus_playlist = play_filtered.build_stimulus_playlist
load_config_folders = play_filtered.load_config_folders
format_channel_map = play_filtered.format_channel_map
parse_channel_map = play_filtered.parse_channel_map
effective_channel_map = play_filtered.effective_channel_map
max_required_channels = play_filtered.max_required_channels
route_channels = play_filtered.route_channels
prepare_playback_audio = play_filtered.prepare_playback_audio
normalize_for_playback = play_filtered.normalize_for_playback
LoopPlayer = play_filtered.LoopPlayer
select_output_device = play_filtered.select_output_device
list_devices = play_filtered.list_devices
save_wav = play_filtered.save_wav

N_IKO_LOUDSPEAKERS = iko_brir.N_IKO_LOUDSPEAKERS
SPH_TO_DRIVER_NPY = iko_brir.SPH_TO_DRIVER_NPY


def load_hoa_weights(output_dir, stage):
    weights_path = output_dir / "npy" / f"{stage}_hoa_weights.npy"
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find HOA weights at {weights_path}")
    weights = np.load(weights_path)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        key = f"{stage}_hoa_weights"
        if key in manifest and manifest[key].get("skipped"):
            raise ValueError(f"{key} was skipped in manifest: {manifest[key].get('reason')}")
    return np.asarray(weights, dtype=np.float64).reshape(-1)


def load_weight_sets(output_dirs, stage):
    weight_sets = []
    for output_dir in output_dirs:
        weights = load_hoa_weights(output_dir, stage)
        try:
            label = str(output_dir.relative_to(ROOT))
        except ValueError:
            label = str(output_dir)
        weight_sets.append({
            "label": label,
            "weights": weights,
            "stage": stage,
            "n_sources": N_IKO_LOUDSPEAKERS,
        })
        print(
            f"Loaded {stage} HOA weights from {output_dir}: "
            f"{weights.shape[0]} eigenbeams -> {N_IKO_LOUDSPEAKERS} drivers"
        )
    return weight_sets


def speaker_ir_from_weights(weights, eigenbeam_to_loudspeaker, device, dtype):
    weights_t = torch.tensor(weights, device=device, dtype=dtype)
    return iko_brir.combine_speaker_impulse_responses(weights_t, eigenbeam_to_loudspeaker)


def apply_speaker_ir(stimulus, speaker_ir):
    stim = torch.tensor(stimulus, dtype=torch.float32).view(1, 1, -1)
    filters = speaker_ir.detach().cpu().to(dtype=torch.float32).unsqueeze(0)
    filtered = TF.convolve(stim, filters, mode="full")
    return np.asarray(filtered.squeeze(0).T, dtype=np.float64)


def render_driver_signals(stimulus, weights, eigenbeam_to_loudspeaker, device, dtype):
    speaker_ir = speaker_ir_from_weights(weights, eigenbeam_to_loudspeaker, device, dtype)
    return apply_speaker_ir(stimulus, speaker_ir)


def run_playback_loop(playlist, weight_sets, fs, device_index, channel_map_arg, eigenbeam_to_loudspeaker, torch_device, dtype):
    devices = sd.query_devices()
    device_name = devices[device_index]["name"]

    folder_idx = 0
    source_idx = 0
    channel_map = effective_channel_map(channel_map_arg, N_IKO_LOUDSPEAKERS)

    def driver_audio():
        return render_driver_signals(
            playlist[source_idx][1],
            weight_sets[folder_idx]["weights"],
            eigenbeam_to_loudspeaker,
            torch_device,
            dtype,
        )

    player = LoopPlayer(prepare_playback_audio(driver_audio(), channel_map), fs, device_index, channel_map)
    player.start()

    def select_source(new_idx):
        nonlocal source_idx
        source_idx = new_idx % len(playlist)
        player.set_audio(prepare_playback_audio(driver_audio(), channel_map))

    def select_folder(new_idx):
        nonlocal folder_idx, channel_map
        folder_idx = new_idx % len(weight_sets)
        channel_map = effective_channel_map(channel_map_arg, N_IKO_LOUDSPEAKERS)
        player.restart(prepare_playback_audio(driver_audio(), channel_map), channel_map)

    pygame.init()
    screen = pygame.display.set_mode((760, 420))
    pygame.display.set_caption("IKO driver playback")
    font = pygame.font.SysFont(None, 28)

    running = True
    clock = pygame.time.Clock()
    while running:
        weight_set = weight_sets[folder_idx]
        screen.fill((25, 25, 25))
        status = "Playing — Space to pause" if player.playing else "Paused — Space to play"
        source_name = playlist[source_idx][0]
        folder_name = weight_set["label"]
        for i, line in enumerate([
            status,
            f"Source: {source_name} ({source_idx + 1}/{len(playlist)})",
            "Up/Down to change source",
            f"Weights: {folder_name} ({folder_idx + 1}/{len(weight_sets)})",
            f"Stage: {weight_set['stage']}   {N_IKO_LOUDSPEAKERS} IKO drivers",
            "Left/Right to change weights",
            f"Channel map: {format_channel_map(channel_map)}",
            f"Device {device_index}: {device_name}",
            f"{max(channel_map)} physical channels   ESC to quit",
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
        description="Play pink noise or audio tracks through the physical IKO drivers "
        "via saved HOA weights and sph_to_driver."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "play_iko_config.json",
        help="JSON config with a list of IKO optimisation output folders to cycle through.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to a single IKO optimisation output run folder.",
    )
    parser.add_argument(
        "--stage",
        choices=("initial", "final"),
        default="final",
        help="Which saved HOA weights to use from each run folder.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="Sample rate in Hz. Defaults to config/hyperpameters.json env.fs.",
    )
    parser.add_argument("--burst-s", type=float, default=0.25, help="Pink-noise burst duration in seconds.")
    parser.add_argument("--silence-s", type=float, default=0.25, help="Silence before and after the burst in seconds.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pink-noise generation.")
    parser.add_argument(
        "--sph-to-driver",
        type=Path,
        default=SPH_TO_DRIVER_NPY,
        help="Path to hoa_to_driver.npy.",
    )
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument(
        "--channel-map",
        default=None,
        help="Comma-separated 1-based physical output channels per IKO driver "
        f"(default 1..{N_IKO_LOUDSPEAKERS}).",
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument(
        "--save-wav",
        type=Path,
        default=None,
        help="Optional path to save the routed multi-channel WAV.",
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

    with open(ROOT / "config" / "hyperpameters.json", encoding="utf-8") as f:
        config = json.load(f)
    fs = args.fs if args.fs is not None else int(config["env"]["fs"])
    torch_device = config["env"]["device"]
    dtype = torch.float32

    weight_sets = load_weight_sets(output_dirs, args.stage)
    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        args.sph_to_driver.resolve(),
        weight_sets[0]["weights"].shape[0],
        N_IKO_LOUDSPEAKERS,
        torch_device,
        dtype,
    )

    channel_map_arg = parse_channel_map(args.channel_map)
    first_channel_map = effective_channel_map(channel_map_arg, N_IKO_LOUDSPEAKERS)
    if channel_map_arg is not None and len(channel_map_arg) != N_IKO_LOUDSPEAKERS:
        print(
            f"Note: --channel-map has {len(channel_map_arg)} entries for "
            f"{N_IKO_LOUDSPEAKERS} drivers; may fall back to 1..N."
        )

    playlist = build_stimulus_playlist(
        fs, args.burst_s, args.silence_s, args.seed, ROOT / "data" / "audio_tracks",
    )
    n_tracks = len(playlist) - 1
    print(f"Loaded {len(playlist)} stimuli (pink noise + {n_tracks} tracks)")

    driver_audio = render_driver_signals(
        playlist[0][1], weight_sets[0]["weights"], eigenbeam_to_loudspeaker, torch_device, dtype,
    )
    routed_audio = route_channels(driver_audio, first_channel_map)
    print(
        f"Driver signal shape: {driver_audio.shape} (samples, logical channels); "
        f"routed shape: {routed_audio.shape} (samples, physical channels)"
    )

    if args.save_wav is not None:
        save_wav(args.save_wav, routed_audio, fs)
        print(f"Saved routed burst to {args.save_wav}")

    if args.no_play:
        return

    required_channels = max_required_channels(channel_map_arg, weight_sets)
    output_device = select_output_device(required_channels, args.device)
    run_playback_loop(
        playlist,
        weight_sets,
        fs,
        output_device,
        channel_map_arg,
        eigenbeam_to_loudspeaker,
        torch_device,
        dtype,
    )


if __name__ == "__main__":
    main()
