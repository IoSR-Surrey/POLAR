"""CAGrad ball-size (cagrad_c) sweep, holding other knobs fixed."""
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "config" / "hyperpameters.json"
PLAY_CONFIG_PATH = ROOT / "config" / "play_filtered_config.json"
SWEEP_LOG_PATH = ROOT / "outputs" / "cagrad" / "ball_size_sweep_log.json"

from utils.miscellaneous import write_run_config

# Skip c=1.0 on both recipes (already have current / R3) and c=2.0 on 12-band (R1).
SWEEP_RUNS = [
    # Best colouration recipe (R3): full c grid minus existing c=1.0
    {"id": "B1_r3_c0.5", "n_samples": 128, "n_bands": 6, "cagrad_c": 0.5, "erb_factor": 0.1},
    {"id": "B2_r3_c1.5", "n_samples": 128, "n_bands": 6, "cagrad_c": 1.5, "erb_factor": 0.1},
    {"id": "B3_r3_c2.0", "n_samples": 128, "n_bands": 6, "cagrad_c": 2.0, "erb_factor": 0.1},
    # 12-band recipe: fill missing c values (have c=1.0 current, c=2.0 R1)
    {"id": "B4_12_c0.5", "n_samples": 128, "n_bands": 12, "cagrad_c": 0.5, "erb_factor": 0.25},
    {"id": "B5_12_c1.5", "n_samples": 128, "n_bands": 12, "cagrad_c": 1.5, "erb_factor": 0.25},
]

SAVED_RE = re.compile(r"Saved outputs to (.+)")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def apply_run(config, run):
    cfg = deepcopy(config)
    cfg["filter_estimator"]["n_samples"] = run["n_samples"]
    cfg["stimulus_dataset"]["n_bands"] = run["n_bands"]
    cfg["stimulus_dataset"]["n_samples_per_band"] = 32
    cfg["stimulus_dataset"]["erb_factor"] = run["erb_factor"]
    cfg["training"]["num_epochs"] = 100
    cfg["training"]["batch_size"] = 64
    cfg["training"]["cagrad_c"] = run["cagrad_c"]
    cfg["training"]["mtl_optimizer"] = "cagrad"
    cfg["saving"]["use_tensorboard"] = True
    return cfg


def relative_output_path(saved_path):
    path = Path(saved_path.strip())
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def run_one(run, base_config):
    print(
        f"\n===== Starting {run['id']}: taps={run['n_samples']}, "
        f"bands={run['n_bands']}, cagrad_c={run['cagrad_c']}, "
        f"erb_factor={run['erb_factor']} =====",
        flush=True,
    )
    save_config(apply_run(base_config, run))
    log_path = ROOT / "outputs" / "cagrad" / f"ball_size_{run['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    combined = log_path.read_text(encoding="utf-8", errors="replace")
    print(combined[-2000:], flush=True)
    match = None
    for match in SAVED_RE.finditer(combined):
        pass
    oom = "OutOfMemoryError" in combined or "CUDA out of memory" in combined
    if proc.returncode != 0:
        status = "oom" if oom else "failed"
        print(f"{run['id']} {status} (exit {proc.returncode})", flush=True)
        return {"id": run["id"], "status": status, "path": None, "returncode": proc.returncode, **run}
    if match is None:
        print(f"{run['id']} finished but no output path found", flush=True)
        return {"id": run["id"], "status": "no_path", "path": None, "returncode": proc.returncode, **run}
    path = relative_output_path(match.group(1))
    print(f"{run['id']} saved to {path}", flush=True)
    return {"id": run["id"], "status": "success", "path": path, "returncode": 0, **run}


def append_play_config(new_paths):
    with open(PLAY_CONFIG_PATH, "r", encoding="utf-8") as f:
        play = json.load(f)
    folders = list(play.get("folders", []))
    seen = set(folders)
    for path in new_paths:
        if path and path not in seen:
            folders.append(path)
            seen.add(path)
    with open(PLAY_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"folders": folders}, f, indent=4)
        f.write("\n")
    print(f"Updated {PLAY_CONFIG_PATH} with {len(folders)} folders", flush=True)


def main():
    base_config = load_config()
    original_config = deepcopy(base_config)
    results = []
    SWEEP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(SWEEP_LOG_PATH.parent, {
        "script": "scripts/ball_size_sweep.py",
        "sweep_runs": SWEEP_RUNS,
        "base_config": base_config,
        "child_script": "scripts/scratch.py",
    }, filename="ball_size_sweep_run_config.json")
    try:
        for run in SWEEP_RUNS:
            result = run_one(run, base_config)
            results.append(result)
            SWEEP_LOG_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        restore = deepcopy(original_config)
        restore["filter_estimator"]["n_samples"] = 128
        restore["stimulus_dataset"]["n_bands"] = 12
        restore["stimulus_dataset"]["n_samples_per_band"] = 32
        restore["stimulus_dataset"]["erb_factor"] = 0.25
        restore["training"]["num_epochs"] = 100
        restore["training"]["cagrad_c"] = 1.0
        save_config(restore)

    new_paths = [r["path"] for r in results if r.get("status") == "success" and r.get("path")]
    append_play_config(new_paths)

    print("\n===== Ball-size sweep summary =====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
