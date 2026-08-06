"""
Continue overnight tests until local 08:00 for *starts only*.
Never skip based on runtime estimates; never kill a running job.

Queue (order):
  C3 capacity, then λ×capacity interactions, λ fine grid, 64-tap care, long schedule.
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
SWEEP_LOG_PATH = LOG_DIR / "morning_extension_sweep_log.json"

DEADLINE_HOUR = 8
DEADLINE_MINUTE = 0

# Start-anything-before-deadline; no finish-time filtering.
SWEEP_RUNS = [
    # Remaining capacity from main overnight plan
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
    },
    # Metrics: L025 best ang MAE + slightly best LSD among λ grid; test interaction
    {
        "id": "X_lam025_256t6b",
        "phase": "lambda_x_capacity",
        "n_samples": 256,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.25,
    },
    {
        "id": "X_lam025_128t12b",
        "phase": "lambda_x_capacity",
        "n_samples": 128,
        "n_bands": 12,
        "erb_factor": 0.25,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.25,
    },
    # Push colouration weight further / fill gap 0.25–0.5
    {
        "id": "X_lam010",
        "phase": "lambda_fine",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.1,
    },
    {
        "id": "X_lam035",
        "phase": "lambda_fine",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.35,
    },
    # 64-tap multi-obj with atten + care schedule (liked taps length)
    {
        "id": "X_64t6b_lam050",
        "phase": "capacity_low",
        "n_samples": 64,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
    },
    {
        "id": "X_64t6b_lam025",
        "phase": "capacity_low",
        "n_samples": 64,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.005,
        "num_epochs": 150,
        "cagrad_c": 1.0,
        "lambda_loc": 0.25,
    },
    # Longer schedule at R3 geometry (S1 may not be saturated)
    {
        "id": "X_S2_lr0025_ep300",
        "phase": "schedule",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.0025,
        "num_epochs": 300,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
    },
    # Diagnostic: atten OFF reanchor (R3-like), for attributing S0 LSD rise vs original R3
    {
        "id": "CTL_R3_no_atten",
        "phase": "control",
        "n_samples": 128,
        "n_bands": 6,
        "erb_factor": 0.1,
        "lr": 0.01,
        "num_epochs": 100,
        "cagrad_c": 1.0,
        "lambda_loc": 0.5,
        "apply_distance_attenuation": False,
    },
]

SAVED_RE = re.compile(r"Saved outputs to (.+)")

BASELINE_FOLDERS = [
    "outputs/cagrad/2026-08-03_193016",
    "outputs/cagrad/2026-08-03_194802",
    "outputs/cagrad/2026-08-03_201820",
    "outputs/cagrad/2026-08-03_224647",
    "outputs/cagrad/2026-08-03_235020",
    "outputs/cagrad/2026-08-04_005407",
    "outputs/cagrad/2026-08-04_011408",
    "outputs/cagrad/2026-08-04_012143",
    "outputs/cagrad/2026-08-04_092544",
    "outputs/cagrad/2026-08-04_094909",
    "outputs/cagrad/2026-08-04_105222",
    "outputs/cagrad/2026-08-04_111530",
    "outputs/cagrad/2026-08-04_151631",
]


def deadline_datetime(now=None):
    now = now or datetime.now()
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
    cfg["env"]["apply_distance_attenuation"] = bool(run.get("apply_distance_attenuation", True))
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


def run_one(run, base_config):
    atten = run.get("apply_distance_attenuation", True)
    print(
        f"\n===== Starting {run['id']} phase={run['phase']} "
        f"taps={run['n_samples']} bands={run['n_bands']} "
        f"lr={run['lr']} ep={run['num_epochs']} "
        f"lambda_loc={run['lambda_loc']} c={run['cagrad_c']} atten={atten} =====",
        flush=True,
    )
    save_config(apply_run(base_config, run))
    log_path = LOG_DIR / f"morning_ext_{run['id']}.log"
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
        "config": {k: run[k] for k in run},
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

    lines = ["{", '    "folders": [']
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


def overnight_new_paths():
    """All new-folder paths currently after the baseline block in playlist, or from logs."""
    paths = []
    log_paths = []
    if SWEEP_LOG_PATH.exists():
        data = json.loads(SWEEP_LOG_PATH.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            if r.get("status") == "success" and r.get("path"):
                log_paths.append(r["path"])
    anchor_log = LOG_DIR / "overnight_anchor_sweep_log.json"
    if anchor_log.exists():
        data = json.loads(anchor_log.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            if r.get("status") == "success" and r.get("path"):
                paths.append(r["path"])
    # playlist may already have paths
    if PLAY_CONFIG_PATH.exists():
        folders = json.loads(PLAY_CONFIG_PATH.read_text(encoding="utf-8")).get("folders") or []
        base = set(BASELINE_FOLDERS)
        for p in folders:
            if p not in base:
                paths.append(p)
    paths.extend(log_paths)
    out = []
    seen = set()
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out


def main():
    deadline = deadline_datetime()
    print(f"Morning extension start {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(
        f"Policy: start next job only if now < {deadline.isoformat(timespec='seconds')}; "
        f"never kill in-progress jobs; no runtime estimate gating.",
        flush=True,
    )

    base_config = load_config()
    original = deepcopy(base_config)
    results = []
    skipped = []
    new_paths = overnight_new_paths()

    try:
        for run in SWEEP_RUNS:
            now = datetime.now()
            if now >= deadline:
                print(
                    f"SKIP {run['id']}: past start deadline "
                    f"{deadline.isoformat(timespec='seconds')} "
                    f"(job would not be started; ongoing jobs would not be killed).",
                    flush=True,
                )
                skipped.append({"id": run["id"], "reason": "past_start_deadline"})
                # Stop queue entirely after 08:00 — do not start later jobs either.
                for later in SWEEP_RUNS:
                    if later["id"] == run["id"]:
                        continue
                    if any(s.get("id") == later["id"] for s in skipped):
                        continue
                    if any(r.get("id") == later["id"] for r in results):
                        continue
                    # only skip remaining not-yet-reached
                    if SWEEP_RUNS.index(later) > SWEEP_RUNS.index(run):
                        skipped.append({"id": later["id"], "reason": "past_start_deadline"})
                break

            result = run_one(run, base_config)
            results.append(result)
            if result.get("status") == "success" and result.get("path"):
                new_paths.append(result["path"])
                write_play_config(BASELINE_FOLDERS, new_paths)

            SWEEP_LOG_PATH.write_text(
                json.dumps(
                    {
                        "results": results,
                        "skipped": skipped,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if result["status"] == "oom" and run["n_samples"] >= 256 and run["n_bands"] >= 12:
                print("OOM on C3; continuing with lighter remaining queue.", flush=True)
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

    write_play_config(BASELINE_FOLDERS, new_paths)
    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "start_deadline": deadline.isoformat(timespec="seconds"),
        "results": results,
        "skipped": skipped,
    }
    SWEEP_LOG_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== Morning extension summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
