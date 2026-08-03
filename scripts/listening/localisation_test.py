import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pygame
import sounddevice as sd
from scipy.io import wavfile


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
# Stimulus filenames follow:
#   {optimisation_type}_{target_angle}[_{program_material}].wav
# e.g. 2026-08-02-135216_+30.wav
#      2026-08-02-135216_+30_cello_1.wav
#      vbap_+15_cello_1.wav
# Fields before program_material must not contain underscores. program_material
# is optional and may itself contain underscores (joined from remaining parts).
STIMULUS_FIELDS = ("optimisation_type", "target_angle")


def _parse_angle_field(value, name):
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Could not parse target_angle '{value}' as a number in '{name}'."
        )


def parse_stimulus_filename(name):
    """Parse a stimulus filename into its fields.

    Returns a dict with keys from STIMULUS_FIELDS, 'target_angle' as float
    (degrees, + = left), and 'program_material' (str, possibly empty).
    Raises ValueError if the name does not match.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Expected at least 2 underscore-separated fields "
            f"(optimisation_type_{{target_angle}}[_{{program_material}}]), "
            f"got {len(parts)} in '{name}'."
        )

    return {
        "optimisation_type": parts[0],
        "target_angle": _parse_angle_field(parts[1], name),
        "program_material": "_".join(parts[2:]),
    }


def discover_stimuli(folder):
    """Return a list of stimulus dicts for all valid WAVs in `folder`.

    Each dict has the parsed fields plus 'path' (Path) and 'filename' (str).
    Files that do not match the naming convention are skipped with a warning.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Stimulus folder does not exist: {folder}")
    stimuli = []
    for path in sorted(folder.glob("*.wav")):
        try:
            fields = parse_stimulus_filename(path.name)
        except ValueError as exc:
            print(f"Skipping {path.name}: {exc}", file=sys.stderr)
            continue
        fields["path"] = path
        fields["filename"] = path.name
        stimuli.append(fields)
    return stimuli


def expand_stimuli_with_repetitions(stimuli, n_repetitions):
    """Repeat each stimulus `n_repetitions` times (for randomised multi-pass trials)."""
    n = int(n_repetitions)
    if n < 1:
        raise ValueError(f"n_repetitions must be >= 1, got {n_repetitions}.")
    return list(stimuli) * n


# --------------------------------------------------------------------------- #
# Audio loading / playback
# --------------------------------------------------------------------------- #
def load_config_fs(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return int(config["env"]["fs"])


def load_multichannel_wav(path, target_fs):
    """Load a (possibly) multi-channel WAV as a float64 (T, n_ch) array.

    Preserves the channel count (one column per loudspeaker) and resamples to
    target_fs if needed. Integer PCM is scaled to [-1, 1].
    """
    fs, data = wavfile.read(path)
    audio = np.asarray(data, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if np.issubdtype(data.dtype, np.integer):
        audio = audio / np.iinfo(data.dtype).max
    if fs != target_fs:
        import torch
        import torchaudio.functional as TF

        resampled = TF.resample(
            torch.tensor(audio.T, dtype=torch.float32),
            fs,
            target_fs,
        ).numpy()
        audio = resampled.T
    return audio


# Keep in sync with create_localisation_stimuli.DEFAULT_TARGET_RMS.
DEFAULT_TARGET_RMS = 0.02484308


def normalize_for_playback(audio, target_rms=DEFAULT_TARGET_RMS):
    """Scale so the global RMS equals target_rms, preserving inter-channel level."""
    audio = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio.astype(np.float32)


def select_output_device(device_index, min_channels):
    """Return an output device index with at least min_channels output channels."""
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


def list_devices():
    print(sd.query_devices())


class LoopPlayer:
    """Continuously loops a multi-channel signal on a sounddevice OutputStream."""

    def __init__(self, audio, fs, device_index):
        self.audio = np.ascontiguousarray(audio, dtype=np.float32)
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
        n = len(self.audio)
        pos = self.position
        end = pos + frames
        if end <= n:
            outdata[:] = self.audio[pos:end]
            self.position = end % n
        else:
            first = n - pos
            outdata[:first] = self.audio[pos:]
            rem = frames - first
            outdata[first:] = self.audio[:rem]
            self.position = rem % n

    def start(self):
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

    def restart(self):
        """Restart playback from the beginning of the signal."""
        self.position = 0
        self.playing = True


# --------------------------------------------------------------------------- #
# Angle geometry (screen: front at top, + = left, - = right)
# --------------------------------------------------------------------------- #
def wrap180(angle_deg):
    """Wrap an angle in degrees to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def signed_error(reported_deg, target_deg):
    """Signed circular error (reported - target) wrapped to [-180, 180)."""
    return wrap180(reported_deg - target_deg)


def azimuth_to_screen(angle_deg, cx, cy, radius):
    """Map an azimuth (0 front, + left, - right) to screen coordinates."""
    a = np.deg2rad(angle_deg)
    x = cx - radius * np.sin(a)
    y = cy - radius * np.cos(a)
    return int(round(x)), int(round(y))


def click_to_azimuth(mx, my, cx, cy):
    """Map a screen click to an azimuth (0 front, + left, - right), or None at centre."""
    dx = mx - cx
    dy = my - cy
    if dx == 0 and dy == 0:
        return None
    return wrap180(np.degrees(np.arctan2(-dx, -dy)))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def compute_group_stats(errors_deg):
    """Summary stats for a set of signed errors (degrees)."""
    errors = np.asarray(errors_deg, dtype=np.float64)
    n = int(errors.size)
    if n == 0:
        return {"n": 0, "mean_signed_error": float("nan"), "mean_abs_error": float("nan"),
                "rmse": float("nan"), "circular_sd_deg": float("nan")}
    ang = np.deg2rad(errors)
    resultant = np.abs(np.mean(np.exp(1j * ang)))
    circ_sd = float(np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))) if resultant > 0 else float("inf")
    return {
        "n": n,
        "mean_signed_error": float(np.mean(errors)),
        "mean_abs_error": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "circular_sd_deg": circ_sd,
    }


def aggregate_stats(rows):
    """Aggregate trial rows into overall, per-optimisation-type and per-seat stats.

    `rows` is a list of dicts with at least 'signed_error', 'optimisation_type',
    'seat_id' keys. Includes a one-sample bias t-test per optimisation type and a
    one-way ANOVA on absolute error across optimisation types when applicable.
    """
    from scipy import stats

    errors = [float(r["signed_error"]) for r in rows]
    result = {"overall": compute_group_stats(errors)}

    by_type = {}
    for r in rows:
        by_type.setdefault(str(r["optimisation_type"]), []).append(float(r["signed_error"]))
    result["by_optimisation_type"] = {}
    result["bias_ttest_by_type"] = {}
    for opt_type, errs in sorted(by_type.items()):
        result["by_optimisation_type"][opt_type] = compute_group_stats(errs)
        if len(errs) >= 2:
            t_stat, p_val = stats.ttest_1samp(errs, 0.0)
            result["bias_ttest_by_type"][opt_type] = {"t": float(t_stat), "p": float(p_val)}

    seats = {str(r["seat_id"]) for r in rows}
    if len(seats) > 1:
        by_seat = {}
        for r in rows:
            by_seat.setdefault(str(r["seat_id"]), []).append(float(r["signed_error"]))
        result["by_seat"] = {seat: compute_group_stats(errs) for seat, errs in sorted(by_seat.items())}

    abs_groups = [np.abs(errs) for errs in by_type.values() if len(errs) >= 2]
    if len(abs_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*abs_groups)
        result["anova_abs_error_by_type"] = {"F": float(f_stat), "p": float(p_val)}

    return result


# --------------------------------------------------------------------------- #
# CSV logging
# --------------------------------------------------------------------------- #
CSV_COLUMNS = [
    "timestamp", "seat_id", "participant_id", "filename", "optimisation_type",
    "program_material", "target_angle", "reported_angle", "signed_error", "abs_error",
    "n_replays", "trial_index", "n_trials",
]


def append_trial_row(results_path, row):
    """Append one trial row to the CSV, writing the header if the file is new."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_trial_rows(results_path):
    with open(results_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Analysis + plots
# --------------------------------------------------------------------------- #
def analyse_csv(results_path):
    """Compute stats and write a <stem>_stats.json plus summary plots next to the CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path = Path(results_path)
    rows = read_trial_rows(results_path)
    if not rows:
        raise ValueError(f"No trial rows found in {results_path}.")

    stats_summary = aggregate_stats(rows)
    stats_path = results_path.with_name(results_path.stem + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    targets = np.array([float(r["target_angle"]) for r in rows])
    reported = np.array([float(r["reported_angle"]) for r in rows])
    errors = np.array([float(r["signed_error"]) for r in rows])
    opt_types = [str(r["optimisation_type"]) for r in rows]
    unique_types = sorted(set(opt_types))

    # Scatter: target vs reported, coloured by optimisation type.
    scatter_path = results_path.with_name(results_path.stem + "_scatter.png")
    fig, ax = plt.subplots()
    for opt_type in unique_types:
        mask = np.array([t == opt_type for t in opt_types])
        ax.scatter(targets[mask], reported[mask], label=opt_type, alpha=0.7)
    lo = float(min(targets.min(), reported.min())) - 10.0
    hi = float(max(targets.max(), reported.max())) + 10.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal")
    ax.set_xlabel("Target angle (deg, + = left)")
    ax.set_ylabel("Reported angle (deg, + = left)")
    ax.set_title("Reported vs target azimuth")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    # Signed error grouped by optimisation type.
    error_path = results_path.with_name(results_path.stem + "_error_by_type.png")
    fig, ax = plt.subplots()
    grouped = [errors[np.array([t == opt_type for t in opt_types])] for opt_type in unique_types]
    ax.boxplot(grouped, showmeans=True)
    ax.set_xticks(range(1, len(unique_types) + 1))
    ax.set_xticklabels(unique_types)
    for i, group in enumerate(grouped, start=1):
        jitter = (np.random.default_rng(0).random(group.size) - 0.5) * 0.2
        ax.scatter(np.full(group.size, i) + jitter, group, alpha=0.5, color="tab:blue")
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_ylabel("Signed error (deg)")
    ax.set_xlabel("Optimisation type")
    ax.set_title("Localisation signed error by optimisation type")
    fig.tight_layout()
    fig.savefig(error_path, dpi=150)
    plt.close(fig)

    return {
        "stats_path": stats_path,
        "scatter_path": scatter_path,
        "error_path": error_path,
        "stats": stats_summary,
    }


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
BG_COLOR = (25, 25, 25)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (100, 180, 255)
MARKER_COLOR = (255, 120, 120)
CIRCLE_COLOR = (90, 90, 90)

WINDOW_SIZE = (820, 820)


def browse_directory(initial=None):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(
        initialdir=str(initial) if initial else os.getcwd(),
        title="Select stimulus folder",
    )
    root.destroy()
    return path or None


DEFAULT_RESULTS_DIR = ROOT / "data" / "localisation_responses"


def sanitize_results_field(value):
    """Make a setup field safe for use in a results filename."""
    value = str(value).strip()
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '):
        value = value.replace(ch, "-")
    return value or "unknown"


def build_results_filename(seat, participant, timestamp=None):
    """Build results CSV filename from setup config: seat, participant, timestamp."""
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"localisation_results_{sanitize_results_field(seat)}_"
        f"{sanitize_results_field(participant)}_{ts}.csv"
    )


def default_results_path(seat, participant, results_dir=None, timestamp=None):
    """Default results CSV path under data/localisation_responses/."""
    results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    return results_dir / build_results_filename(seat, participant, timestamp=timestamp)


def resolve_results_path(state, results_dir=None, timestamp=None):
    """Return the results path to use: manual override, or auto from seat/participant."""
    if state.get("results_overridden") and state.get("results_path"):
        return Path(state["results_path"])
    ts = timestamp if timestamp is not None else state.get("results_timestamp")
    return default_results_path(
        state["seat"], state["participant"], results_dir=results_dir, timestamp=ts
    )


def browse_save_csv(initial=None):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_path = Path(initial) if initial else default_results_path("seat", "participant")
    path = filedialog.asksaveasfilename(
        initialdir=str(initial_path.parent),
        initialfile=initial_path.name,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Choose where to save results",
    )
    root.destroy()
    return path or None


def draw_text_lines(screen, font, lines, x, y, spacing=32, color=TEXT_COLOR):
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, color), (x, y + spacing * i))


def run_setup_screen(screen, font, small_font, defaults):
    """Collect seat, participant, repetitions, stimulus folder and results path.

    Results path defaults to localisation_results_{seat}_{participant}_{timestamp}.csv
    under data/localisation_responses/ unless overridden (Ctrl+R or --results).

    Returns a dict with keys seat, participant, repetitions, stimulus_folder,
    results_path, or None if the user quit.
    """
    state = {
        "seat": defaults.get("seat") or "",
        "participant": defaults.get("participant") or "",
        "repetitions": str(defaults.get("repetitions") if defaults.get("repetitions") is not None else 1),
        "stimulus_folder": defaults.get("stimulus_folder"),
        "results_path": defaults.get("results_path"),
        "results_overridden": bool(defaults.get("results_path")),
        "results_timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    fields = ("seat", "participant", "repetitions")
    active_idx = 0
    message = ""
    clock = pygame.time.Clock()

    while True:
        screen.fill(BG_COLOR)
        folder_text = str(state["stimulus_folder"]) if state["stimulus_folder"] else "(Ctrl+F to browse)"
        if state["seat"].strip() and state["participant"].strip():
            preview_path = resolve_results_path(state)
            results_text = str(preview_path)
        else:
            results_text = "(auto: localisation_results_{seat}_{participant}_{timestamp}.csv)"
        active = fields[active_idx]
        seat_prefix = "> " if active == "seat" else "  "
        part_prefix = "> " if active == "participant" else "  "
        reps_prefix = "> " if active == "repetitions" else "  "
        draw_text_lines(screen, font, [
            "Localisation listening test - setup",
            "",
            f"{seat_prefix}Seat ID:        {state['seat']}",
            f"{part_prefix}Participant ID: {state['participant']}",
            f"{reps_prefix}Repetitions:    {state['repetitions']}",
            "",
            f"  Stimulus folder: {folder_text}",
            f"  Results CSV:     {results_text}",
        ], 30, 30)
        draw_text_lines(screen, small_font, [
            "TAB: switch seat/participant/repetitions    type to edit    Backspace: delete",
            "Ctrl+F: browse stimulus folder    Ctrl+R: override results CSV path",
            "ENTER: begin test    ESC: quit",
            "",
            "Results filename is built from seat + participant (+ timestamp) by default.",
            "Repetitions = how many times each stimulus file is presented (then shuffled).",
        ], 30, 30 + 32 * 9, spacing=26)
        if message:
            screen.blit(font.render(message, True, MARKER_COLOR), (30, WINDOW_SIZE[1] - 60))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                return None
            elif event.key == pygame.K_TAB:
                active_idx = (active_idx + 1) % len(fields)
            elif event.key == pygame.K_BACKSPACE:
                state[active] = state[active][:-1]
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                error = validate_setup(state)
                if error:
                    message = error
                else:
                    state["results_path"] = str(resolve_results_path(state))
                    return state
            else:
                handled = handle_setup_hotkey(event, state)
                if not handled and event.unicode and event.unicode.isprintable():
                    if active == "repetitions" and not event.unicode.isdigit():
                        continue
                    state[active] += event.unicode

        clock.tick(30)


def handle_setup_hotkey(event, state):
    """Handle browse hotkeys (Ctrl+F / Ctrl+R). Returns True if handled."""
    ctrl = event.mod & pygame.KMOD_CTRL
    if ctrl and event.key == pygame.K_f:
        folder = browse_directory(state.get("stimulus_folder"))
        if folder:
            state["stimulus_folder"] = folder
        return True
    if ctrl and event.key == pygame.K_r:
        suggested = None
        if state["seat"].strip() and state["participant"].strip():
            suggested = resolve_results_path(state)
        elif state.get("results_path"):
            suggested = state["results_path"]
        path = browse_save_csv(suggested)
        if path:
            state["results_path"] = path
            state["results_overridden"] = True
        return True
    return False


def validate_setup(state):
    if not state["seat"].strip():
        return "Seat ID is required."
    if not state["participant"].strip():
        return "Participant ID is required."
    reps_text = str(state.get("repetitions", "")).strip()
    if not reps_text:
        return "Repetitions is required (integer >= 1)."
    try:
        reps = int(reps_text)
    except ValueError:
        return "Repetitions must be an integer >= 1."
    if reps < 1:
        return "Repetitions must be an integer >= 1."
    if not state["stimulus_folder"]:
        return "Stimulus folder is required (Ctrl+F)."
    if not Path(state["stimulus_folder"]).is_dir():
        return "Stimulus folder does not exist."
    return None


def draw_response_circle(screen, font, small_font, center, radius, reported_angle, info_lines):
    cx, cy = center
    pygame.draw.circle(screen, CIRCLE_COLOR, center, radius, 2)
    pygame.draw.circle(screen, (70, 70, 70), center, 4)

    # Cardinal direction labels.
    labels = [(0.0, "FRONT (0)"), (90.0, "LEFT (+90)"), (-90.0, "RIGHT (-90)"), (180.0, "BACK (180)")]
    for angle, label in labels:
        lx, ly = azimuth_to_screen(angle, cx, cy, radius + 28)
        text = small_font.render(label, True, TEXT_COLOR)
        rect = text.get_rect(center=(lx, ly))
        screen.blit(text, rect)

    # Tick marks every 30 degrees.
    for angle in range(-150, 210, 30):
        inner = azimuth_to_screen(angle, cx, cy, radius - 10)
        outer = azimuth_to_screen(angle, cx, cy, radius)
        pygame.draw.line(screen, CIRCLE_COLOR, inner, outer, 1)

    if reported_angle is not None:
        mx, my = azimuth_to_screen(reported_angle, cx, cy, radius)
        pygame.draw.line(screen, MARKER_COLOR, center, (mx, my), 2)
        pygame.draw.circle(screen, MARKER_COLOR, (mx, my), 9)

    draw_text_lines(screen, font, info_lines, 20, 20)


def run_trial_loop(screen, font, small_font, stimuli, session, fs, device_index):
    """Run all trials, appending each confirmed response to the CSV.

    Returns True if the session completed, False if the user quit early.
    """
    center = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 + 20)
    radius = 300
    n_trials = len(stimuli)
    clock = pygame.time.Clock()
    player = None
    completed = True

    for trial_index, stimulus in enumerate(stimuli):
        audio = normalize_for_playback(stimulus["audio"], target_rms=session["rms"])
        if player is not None:
            player.stop()
        player = LoopPlayer(audio, fs, device_index)
        player.start()
        player.playing = True
        player.position = 0
        n_replays = 0
        reported_angle = None
        trial_running = True

        while trial_running:
            screen.fill(BG_COLOR)
            info_lines = [
                f"Trial {trial_index + 1} / {n_trials}",
                "Click where the sound appears to come from.",
                "SPACE: replay    ENTER: confirm    ESC: quit",
                f"Reported: {reported_angle:.1f} deg" if reported_angle is not None else "Reported: --",
                f"Replays: {n_replays}",
            ]
            draw_response_circle(screen, font, small_font, center, radius, reported_angle, info_lines)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    player.stop()
                    return False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    angle = click_to_azimuth(event.pos[0], event.pos[1], center[0], center[1])
                    if angle is not None:
                        reported_angle = angle
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        player.stop()
                        return False
                    elif event.key == pygame.K_SPACE and not getattr(event, "repeat", 0):
                        player.restart()
                        n_replays += 1
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and reported_angle is not None:
                        err = signed_error(reported_angle, stimulus["target_angle"])
                        row = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "seat_id": session["seat"],
                            "participant_id": session["participant"],
                            "filename": stimulus["filename"],
                            "optimisation_type": stimulus["optimisation_type"],
                            "program_material": stimulus.get("program_material", ""),
                            "target_angle": stimulus["target_angle"],
                            "reported_angle": round(reported_angle, 2),
                            "signed_error": round(err, 2),
                            "abs_error": round(abs(err), 2),
                            "n_replays": n_replays,
                            "trial_index": trial_index,
                            "n_trials": n_trials,
                        }
                        append_trial_row(session["results_path"], row)
                        trial_running = False

            clock.tick(30)

    if player is not None:
        player.stop()
    return completed


def run_done_screen(screen, font, small_font, results_path, analysis):
    clock = pygame.time.Clock()
    lines = [
        "Session complete. Results saved to:",
        str(results_path),
        "",
        "Analysis written:",
        str(analysis["stats_path"].name),
        str(analysis["scatter_path"].name),
        str(analysis["error_path"].name),
        "",
        f"Overall mean |error|: {analysis['stats']['overall']['mean_abs_error']:.2f} deg",
        f"Overall RMSE: {analysis['stats']['overall']['rmse']:.2f} deg",
        "",
        "Press ESC or close the window to exit.",
    ]
    while True:
        screen.fill(BG_COLOR)
        draw_text_lines(screen, small_font, lines, 30, 40, spacing=30)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        clock.tick(30)


def run_gui(args, fs):
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Localisation listening test")
    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 24)

    defaults = {
        "seat": args.seat,
        "participant": args.participant,
        "repetitions": args.repetitions,
        "stimulus_folder": str(args.stimuli) if args.stimuli else None,
        "results_path": str(args.results) if args.results else None,
    }
    setup = run_setup_screen(screen, font, small_font, defaults)
    if setup is None:
        pygame.quit()
        return

    stimuli = discover_stimuli(setup["stimulus_folder"])
    if not stimuli:
        show_error_screen(screen, font, [
            "No valid stimulus WAVs found in",
            f"{setup['stimulus_folder']}",
            "",
            "Expected names like: {optimisation_type}_{target_angle}[_{program_material}].wav",
            "Press ESC to exit.",
        ])
        pygame.quit()
        return

    for stimulus in stimuli:
        stimulus["audio"] = load_multichannel_wav(stimulus["path"], fs)
    max_channels = max(s["audio"].shape[1] for s in stimuli)

    try:
        device_index = select_output_device(args.device, max_channels)
    except (RuntimeError, ValueError) as exc:
        show_error_screen(screen, font, [str(exc), "", "Press ESC to exit."])
        pygame.quit()
        return

    n_repetitions = int(str(setup["repetitions"]).strip())
    stimuli = expand_stimuli_with_repetitions(stimuli, n_repetitions)
    random.Random(args.seed).shuffle(stimuli)

    session = {
        "seat": setup["seat"].strip(),
        "participant": setup["participant"].strip(),
        "results_path": Path(setup["results_path"]),
        "rms": args.rms,
    }

    completed = run_trial_loop(screen, font, small_font, stimuli, session, fs, device_index)

    if completed:
        try:
            analysis = analyse_csv(session["results_path"])
            run_done_screen(screen, font, small_font, session["results_path"], analysis)
        except Exception as exc:  # noqa: BLE001 - surface any analysis error to the operator
            show_error_screen(screen, font, [
                "Session complete but analysis failed:",
                str(exc),
                "",
                "Raw results are still saved. Press ESC to exit.",
            ])

    pygame.quit()


def show_error_screen(screen, font, lines):
    clock = pygame.time.Clock()
    while True:
        screen.fill(BG_COLOR)
        draw_text_lines(screen, font, lines, 30, 40)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return
        clock.tick(30)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Localisation listening-test GUI (azimuth response on a circle)."
    )
    parser.add_argument("--stimuli", type=Path, default=None, help="Folder of labeled stimulus WAVs.")
    parser.add_argument(
        "--results", type=Path, default=None,
        help="Override results CSV path. Default: data/localisation_responses/"
        "localisation_results_{seat}_{participant}_{timestamp}.csv",
    )
    parser.add_argument("--seat", type=str, default=None, help="Seat / listening-position ID.")
    parser.add_argument("--participant", type=str, default=None, help="Participant ID.")
    parser.add_argument("--device", type=int, default=None, help="Output device index for sounddevice.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument(
        "--fs", type=int, default=None,
        help="Playback sample rate in Hz. Defaults to config/hyperpameters.json env.fs.",
    )
    parser.add_argument(
        "--rms", type=float, default=DEFAULT_TARGET_RMS,
        help="Target RMS after normalisation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for trial-order shuffling.")
    parser.add_argument(
        "--repetitions", type=int, default=1, metavar="N",
        help="How many times each stimulus file is presented (shuffled together). Default: 1.",
    )
    parser.add_argument(
        "--analyse", type=Path, default=None,
        help="Analyse an existing results CSV (stats + plots) and exit; no GUI.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.repetitions < 1:
        raise SystemExit(f"--repetitions must be >= 1, got {args.repetitions}.")

    if args.list_devices:
        list_devices()
        return

    if args.analyse is not None:
        analysis = analyse_csv(args.analyse)
        print(f"Stats:   {analysis['stats_path']}")
        print(f"Scatter: {analysis['scatter_path']}")
        print(f"Errors:  {analysis['error_path']}")
        print(json.dumps(analysis["stats"], indent=2))
        return

    if args.fs is not None:
        fs = args.fs
    else:
        config_path = ROOT / "config" / "hyperpameters.json"
        fs = load_config_fs(config_path) if config_path.exists() else 48000

    run_gui(args, fs)


if __name__ == "__main__":
    main()
