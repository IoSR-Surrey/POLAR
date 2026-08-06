"""
Overnight plan until local 08:00:
  1) Re-anchor R3 geometry with distance atten + no TB
  2) Lower lr / more epochs on that geometry
  3) Capacity (taps/bands) under careful schedule
  4) Convex lambda_loc weight sweep on best practical schedule

Start the next job only if local time is still before the start deadline.
Do not gate on estimated runtime; do not kill jobs that overrun the deadline.
"""
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "hyperpameters.json"
PLAY_CONFIG_PATH = ROOT / "config" / "play_filtered_config.json"
LOG_DIR = ROOT / "outputs" / "cagrad"
SWEEP_LOG_PATH = LOG_DIR / "overnight_anchor_sweep_log.json"

# Hard stop: no new job that cannot finish before this local time tomorrow (or today if still before).
DEADLINE_HOUR = 8
DEADLINE_MINUTE = 0

# Rough runtime budgets (minutes) from recent machine timings, with slack.
# 6-band 128taps ~15–25 min / 100 ep; 12-band can be ~2–4 h / 100 ep.
SWEEP_RUNS = [
    # Phase 1: schedule
    {
        "id": "S0_r3_reanchor",
        "phase": "schedule",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.01,
        "num_epochs": 100,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "est_minutes": 40,
    },
    {
        "id": "S1_r3_lr005_ep200",
        "phase": "schedule",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 200,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "est_minutes": 70,
    },
    # Phase 2: capacity under careful schedule
    {
        "id": "C1_256t_6b_care",
        "phase": "capacity",
        "n_samples": 256,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "est_minutes": 70,
    },
    # Phase 3: lambda trade-off early (cheap on 6-band) before long 12-band capacity
    {
        "id": "L_lam025",
        "phase": "lambda",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.25,
        "est_minutes": 55,
    },
    {
        "id": "L_lam050",
        "phase": "lambda",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "est_minutes": 55,
    },
    {
        "id": "L_lam075",
        "phase": "lambda",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.75,
        "est_minutes": 55,
    },
    {
        "id": "L_lam090",
        "phase": "lambda",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.9,
        "est_minutes": 55,
    },
    # Phase 2b: heavier capacity if time remains
    {
        "id": "C2_128t_12b_care",
        "phase": "capacity",
        "n_samples": 128,
        "n_bands": 12,
        "erb_factor": 0.25,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "est_minutes": 360,
    },
    {
        "id": "C3_256t_12b_care",
        "phase": "capacity",
        "n_samples": 256,
        "n_bands": 12,
        "erb_factor": 0.25,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "est_minutes": 480,
    },
]

SAVED_RE = re.compile(r"Saved outputs to (.+)")
ANCHOR_PATH = "outputs/cagrad/2026-08-04_005407"


def deadline_datetime(now=None):
    now = now or datetime.now()
    # Next 08:00 local (if already past 08:00 today, use tomorrow).
    candidate = now.replace(hour=DEADLINE_HOUR, minute=DEADLINE_MINUTE, second=0, microsecond=0)
    if now >= candidate:
        candidate = candidate + timedelta(days=1)
    return candidate


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def apply_run(config, run):
    cfg = deepcopy(config)
    cfg.setdefault("env", {})
    cfg["env"]["apply_distance_attenuation"] = True
    cfg["filter_estimator"]["n_samples"] = run["n_samples"]
    cfg["stimulus_dataset"]["n_bands"] = run["n_bands"]
    cfg["stimulus_dataset"]["n_samples_per_band"] = 32
    cfg["stimulus_dataset"]["erb_factor"] = run["erb_factor"]
    cfg["stimulus_dataset"]["band_mode"] = "linear_erb"
    cfg["stimulus_dataset"]["dataset_type"] = "bandpass"
    cfg["stimulus_dataset"]["stimulus_type"] = "noise"
    cfg["training"]["num_epochs"] = run["num_epochs"]
    cfg["training"]["lr"] = run["lr"]
    cfg["training"]["batch_size"] = 64
    cfg["training"]["cagrad_c"] = run["cagrad_c"]
    cfg["training"]["mtl_optimizer"] = "cagrad"
    cfg["training"]["lambda_loc"] = run["lambda_loc"]
    cfg["saving"]["use_tensorboard"] = False
    return cfg


def relative_output_path(saved_path):
    path = Path(saved_path.strip())
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def can_start(_est_minutes, deadline):
    # Only refuse to *start* after the deadline; ignore duration estimates.
    return datetime.now() < deadline


def run_one(run, base_config):
    print(
        f"\n===== Starting {run['id']} phase={run['phase']} "
        f"taps={run['n_samples']} bands={run['n_bands']} "
        f"lr={run['lr']} ep={run['num_epochs']} "
        f"lambda_loc={run['lambda_loc']} c={run['cagrad_c']} =====",
        flush=True,
    )
    save_config(apply_run(base_config, run))
    log_path = LOG_DIR / f"overnight_{run['id']}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "conda",
        "run",
        "-n",
        "polar-env",
        "--no-capture-output",
        "python",
        str(ROOT / "scripts" / "scratch.py"),
    ]
    started = datetime.now()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)
    elapsed = (datetime.now() - started).total_seconds() / 60.0
    combined = log_path.read_text(encoding="utf-8", errors="replace")
    print(combined[-2500:], flush=True)
    match = None
    for match in SAVED_RE.finditer(combined):
        pass
    oom = "OutOfMemoryError" in combined or "CUDA out of memory" in combined
    result = {
        "id": run["id"],
        "phase": run["phase"],
        "config": {k: run[k] for k in run if k != "est_minutes"},
        "est_minutes": run["est_minutes"],
        "elapsed_minutes": round(elapsed, 1),
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        result["status"] = "oom" if oom else "failed"
        result["path"] = None
        print(f"{run['id']} {result['status']} (exit {proc.returncode}) after {elapsed:.1f} min", flush=True)
        return result
    if match is None:
        result["status"] = "no_path"
        result["path"] = None
        print(f"{run['id']} finished without output path after {elapsed:.1f} min", flush=True)
        return result
    path = relative_output_path(match.group(1))
    result["status"] = "success"
    result["path"] = path
    print(f"{run['id']} saved to {path} ({elapsed:.1f} min)", flush=True)
    return result


def write_play_config(old_folders, new_folders):
    """Write playlist JSON with a blank line between existing and new folders."""
    folders_old = []
    seen = set()
    for path in old_folders:
        if path and path not in seen:
            folders_old.append(path)
            seen.add(path)
    folders_new = []
    for path in new_folders:
        if path and path not in seen:
            folders_new.append(path)
            seen.add(path)

    lines = ['{', '    "folders": [']
    for i, path in enumerate(folders_old):
        trailing = "," if (i < len(folders_old) - 1 or folders_new) else ""
        lines.append(f'        "{path}"{trailing}')
    if folders_new:
        if folders_old:
            lines.append("")  # blank line separator
        for i, path in enumerate(folders_new):
            trailing = "," if i < len(folders_new) - 1 else ""
            lines.append(f'        "{path}"{trailing}')
    lines.append("    ]")
    lines.append("}")
    lines.append("")
    PLAY_CONFIG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Updated {PLAY_CONFIG_PATH} "
        f"({len(folders_old)} existing + {len(folders_new)} new, blank line separator)",
        flush=True,
    )


def append_play_config(paths):
    with open(PLAY_CONFIG_PATH, "r", encoding="utf-8") as f:
        play = json.load(f)
    old = list(play.get("folders", []))
    write_play_config(old, paths)


def main():
    deadline = deadline_datetime()
    print(f"Overnight sweep start {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(
        f"Start deadline (no new starts after this; in-progress not killed): "
        f"{deadline.isoformat(timespec='seconds')}",
        flush=True,
    )

    base_config = load_config()
    original = deepcopy(base_config)
    results = []
    skipped = []

    try:
        for run in SWEEP_RUNS:
            if not can_start(run.get("est_minutes"), deadline):
                msg = (
                    f"SKIP {run['id']}: past start deadline "
                    f"{deadline.isoformat(timespec='seconds')}"
                )
                print(msg, flush=True)
                skipped.append({"id": run["id"], "reason": "past_start_deadline"})
                # No more starts after 08:00.
                for later in SWEEP_RUNS:
                    if later["id"] == run["id"]:
                        continue
                    if SWEEP_RUNS.index(later) > SWEEP_RUNS.index(run):
                        skipped.append({"id": later["id"], "reason": "past_start_deadline"})
                break
            result = run_one(run, base_config)
            results.append(result)
            SWEEP_LOG_PATH.write_text(
                json.dumps({"results": results, "skipped": skipped}, indent=2),
                encoding="utf-8",
            )
            if result["status"] == "oom" and run["n_samples"] >= 256 and run["n_bands"] >= 12:
                print("OOM on heavy capacity run; skipping remaining larger capacity configs.", flush=True)
                for later in SWEEP_RUNS:
                    if later["id"] == run["id"]:
                        continue
                    if later["phase"] == "capacity" and later["n_samples"] * later["n_bands"] > run["n_samples"] * run["n_bands"]:
                        skipped.append({"id": later["id"], "reason": "skipped_after_oom"})
    finally:
        restore = deepcopy(original)
        restore.setdefault("env", {})
        restore["env"]["apply_distance_attenuation"] = True
        restore["filter_estimator"]["n_samples"] = 128
        restore["stimulus_dataset"]["n_bands"] = 6
        restore["stimulus_dataset"]["erb_factor"] = 0.1
        restore["stimulus_dataset"]["n_samples_per_band"] = 32
        restore["training"]["num_epochs"] = 100
        restore["training"]["lr"] = 0.01
        restore["training"]["cagrad_c"] = 1.0
        restore["training"]["lambda_loc"] = 0.5
        restore["saving"]["use_tensorboard"] = False
        save_config(restore)

    new_paths = [r["path"] for r in results if r.get("status") == "success" and r.get("path")]
    # Keep prior anchor in playlist and append new successes.
    anchor_and_new = []
    if ANCHOR_PATH not in (json.loads(PLAY_CONFIG_PATH.read_text(encoding="utf-8")).get("folders") or []):
        anchor_and_new.append(ANCHOR_PATH)
    anchor_and_new.extend(new_paths)
    append_play_config(anchor_and_new)

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "deadline": deadline.isoformat(timespec="seconds"),
        "results": results,
        "skipped": skipped,
    }
    SWEEP_LOG_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== Overnight summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
