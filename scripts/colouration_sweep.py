"""Drive the colouration overnight CAGrad sweep via scripts/scratch.py."""
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
SWEEP_LOG_PATH = ROOT / "outputs" / "cagrad" / "colouration_sweep_log.json"

from utils.miscellaneous import write_run_config

LIKED_BASELINES = [
    "outputs/cagrad/2026-08-03_193016",
    "outputs/cagrad/2026-08-03_194802",
]

# Pre-sweep llado+LSD run (128 taps / 12 bands / c=1.0 / 150 epochs).
CURRENT_BASELINE = "outputs/cagrad/2026-08-03_201820"

# n_bands=6 needs erb_factor<=~0.12 for linear_erb coverage of [20, 20000] Hz.
SWEEP_RUNS = [
    {"id": "R1", "n_samples": 128, "n_bands": 12, "cagrad_c": 2.0, "erb_factor": 0.25},
    {"id": "R2", "n_samples": 64, "n_bands": 12, "cagrad_c": 1.0, "erb_factor": 0.25},
    {"id": "R3", "n_samples": 128, "n_bands": 6, "cagrad_c": 1.0, "erb_factor": 0.1},
    {"id": "R4", "n_samples": 64, "n_bands": 6, "cagrad_c": 1.0, "erb_factor": 0.1},
    {"id": "R5", "n_samples": 32, "n_bands": 12, "cagrad_c": 1.0, "erb_factor": 0.25},
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
    print(f"\n===== Starting {run['id']}: taps={run['n_samples']}, "
          f"bands={run['n_bands']}, cagrad_c={run['cagrad_c']}, "
          f"erb_factor={run['erb_factor']} =====", flush=True)
    save_config(apply_run(base_config, run))
    log_path = ROOT / "outputs" / "cagrad" / f"colouration_sweep_{run['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    # conda run ensures CUDA/nvrtc DLLs are on PATH (plain env python.exe does not).
    cmd = ["conda", "run", "-n", "polar-env", "--no-capture-output",
           "python", str(ROOT / "scripts" / "scratch.py")]
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    combined = log_path.read_text(encoding="utf-8", errors="replace")
    print(combined[-4000:], flush=True)
    match = None
    for match in SAVED_RE.finditer(combined):
        pass
    oom = "OutOfMemoryError" in combined or "CUDA out of memory" in combined
    if proc.returncode != 0:
        status = "oom" if oom else "failed"
        print(f"{run['id']} {status} (exit {proc.returncode})", flush=True)
        return {"id": run["id"], "status": status, "path": None, "returncode": proc.returncode}
    if match is None:
        print(f"{run['id']} finished but no output path found", flush=True)
        return {"id": run["id"], "status": "no_path", "path": None, "returncode": proc.returncode}
    path = relative_output_path(match.group(1))
    print(f"{run['id']} saved to {path}", flush=True)
    return {"id": run["id"], "status": "success", "path": path, "returncode": 0}


def update_play_config(paths):
    folders = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        folders.append(path)
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
        "script": "scripts/colouration_sweep.py",
        "sweep_runs": SWEEP_RUNS,
        "liked_baselines": LIKED_BASELINES,
        "current_baseline": CURRENT_BASELINE,
        "base_config": base_config,
        "child_script": "scripts/scratch.py",
    }, filename="colouration_sweep_run_config.json")
    try:
        for run in SWEEP_RUNS:
            result = run_one(run, base_config)
            results.append(result)
            SWEEP_LOG_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        # Restore memory-safe defaults close to the pre-sweep current job.
        restore = deepcopy(original_config)
        restore["filter_estimator"]["n_samples"] = 128
        restore["stimulus_dataset"]["n_bands"] = 12
        restore["stimulus_dataset"]["n_samples_per_band"] = 32
        restore["training"]["num_epochs"] = 100
        restore["training"]["cagrad_c"] = 1.0
        restore["stimulus_dataset"]["erb_factor"] = 0.25
        save_config(restore)

    sweep_paths = [r["path"] for r in results if r.get("status") == "success" and r.get("path")]
    update_play_config(LIKED_BASELINES + [CURRENT_BASELINE] + sweep_paths)

    print("\n===== Sweep summary =====", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"Current baseline included: {CURRENT_BASELINE}", flush=True)
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
