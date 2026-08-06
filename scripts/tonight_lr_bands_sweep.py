"""
Tonight: 12-band atten-on care spine with slightly calmer schedule,
then the lambda optimisations discussed after listening.

Schedule: lr=0.0035, 180 epochs (mild drop vs care 0.005/150; not the half-step
0.0025 that regressed as S2-style on 6b).

Runs (all: 128t, 12b, erb=0.25, c=1, atten on):
  A) schedule control at lambda 0.5 (fair compare to #36 under new lr)
  B) low lambda where colour can actually move: 0.05, 0.10, 0.15
  C) optional high-loc twin of #20 on 12b: lambda 0.9

Start any job only before 08:00; never kill in progress.
"""
import json
import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "hyperpameters.json"
PLAY = ROOT / "config" / "play_filtered_config.json"
LOG_DIR = ROOT / "outputs" / "cagrad"
SWEEP_LOG = LOG_DIR / "tonight_lr_bands_sweep_log.json"
SAVED_RE = re.compile(r"Saved outputs to (.+)")

DEADLINE_HOUR = 8
DEADLINE_MINUTE = 0

# Shared geometry: preferred 12-band atten-on care path
BASE = {
    "n_samples": 128,
    "n_bands": 12,
    "erb_factor": 0.25,
    "cagrad_c": 1.0,
    "lr": 0.0035,
    "num_epochs": 180,
    "apply_distance_attenuation": True,
    "n_samples_per_band": 32,
}

SWEEP_RUNS = [
    {
        "id": "A_12b_lam050_lr0035",
        "lambda_loc": 0.50,
        **BASE,
    },
    {
        "id": "B_12b_lam005_lr0035",
        "lambda_loc": 0.05,
        **BASE,
    },
    {
        "id": "B_12b_lam010_lr0035",
        "lambda_loc": 0.10,
        **BASE,
    },
    {
        "id": "B_12b_lam015_lr0035",
        "lambda_loc": 0.15,
        **BASE,
    },
    {
        "id": "C_12b_lam090_lr0035",
        "lambda_loc": 0.90,
        **BASE,
    },
]


def deadline_datetime(now=None):
    now = now or datetime.now()
    candidate = now.replace(hour=DEADLINE_HOUR, minute=DEADLINE_MINUTE, second=0, microsecond=0)
    if now >= candidate:
        candidate = candidate + timedelta(days=1)
    return candidate


def load_config():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def save_config(cfg):
    CFG_PATH.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")


def apply_run(config, run):
    cfg = deepcopy(config)
    cfg.setdefault("env", {})
    cfg["env"]["apply_distance_attenuation"] = bool(run.get("apply_distance_attenuation", True))
    cfg["filter_estimator"]["n_samples"] = run["n_samples"]
    cfg["stimulus_dataset"]["n_bands"] = run["n_bands"]
    cfg["stimulus_dataset"]["n_samples_per_band"] = run["n_samples_per_band"]
    cfg["stimulus_dataset"]["erb_factor"] = run["erb_factor"]
    cfg["stimulus_dataset"]["band_mode"] = "linear_erb"
    cfg["stimulus_dataset"]["dataset_type"] = "bandpass"
    cfg["stimulus_dataset"]["stimulus_type"] = "noise"
    cfg["training"]["num_epochs"] = run["num_epochs"]
    cfg["training"]["lr"] = run["lr"]
    cfg["training"]["batch_size"] = 64
    cfg["training"]["cagrad_c"] = run["cagrad_c"]
    cfg["training"]["lambda_loc"] = run["lambda_loc"]
    cfg["training"]["mtl_optimizer"] = "cagrad"
    cfg["saving"]["use_tensorboard"] = False
    return cfg


def restore_care_defaults(base_config):
    restore = deepcopy(base_config)
    restore.setdefault("env", {})
    restore["env"]["apply_distance_attenuation"] = True
    restore["filter_estimator"]["n_samples"] = 128
    restore["stimulus_dataset"]["n_bands"] = 12
    restore["stimulus_dataset"]["erb_factor"] = 0.25
    restore["stimulus_dataset"]["n_samples_per_band"] = 32
    restore["training"]["num_epochs"] = 180
    restore["training"]["lr"] = 0.0035
    restore["training"]["cagrad_c"] = 1.0
    restore["training"]["lambda_loc"] = 0.5
    restore["saving"]["use_tensorboard"] = False
    return restore


def relative_output_path(saved_path):
    path = Path(saved_path.strip())
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def append_playlist(rel):
    play = json.loads(PLAY.read_text(encoding="utf-8"))
    folders = list(play.get("folders", []))
    if rel not in folders:
        folders.append(rel)
    lines = ["{", '    "folders": [']
    for i, p in enumerate(folders):
        trailing = "," if i < len(folders) - 1 else ""
        lines.append(f'        "{p}"{trailing}')
        if i == 12 and len(folders) > 13:
            lines.append("")
    lines.append("    ]")
    lines.append("}")
    lines.append("")
    PLAY.write_text("\n".join(lines), encoding="utf-8")
    print(f"APPENDED {rel} (#{len(folders)})", flush=True)


def run_one(run, base_config):
    print(
        f"\n===== {run['id']}: taps={run['n_samples']} bands={run['n_bands']} "
        f"erb={run['erb_factor']} c={run['cagrad_c']} lambda={run['lambda_loc']} "
        f"lr={run['lr']} ep={run['num_epochs']} atten={run['apply_distance_attenuation']} =====",
        flush=True,
    )
    save_config(apply_run(base_config, run))
    log_path = LOG_DIR / f"tonight_{run['id']}.log"
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
    text = log_path.read_text(encoding="utf-8", errors="replace")
    print(text[-2000:], flush=True)
    match = None
    for match in SAVED_RE.finditer(text):
        pass
    result = {
        "id": run["id"],
        "config": {k: run[k] for k in run},
        "elapsed_minutes": round(elapsed, 1),
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        result["status"] = "failed"
        result["path"] = None
        return result
    if match is None:
        result["status"] = "no_path"
        result["path"] = None
        return result
    path = relative_output_path(match.group(1))
    result["status"] = "success"
    result["path"] = path
    append_playlist(path)
    print(f"{run['id']} -> {path} ({elapsed:.1f} min)", flush=True)
    return result


def main():
    deadline = deadline_datetime()
    print(f"Tonight sweep start {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(
        f"Start deadline {deadline.isoformat(timespec='seconds')} "
        f"(no new starts after; never kill in progress)",
        flush=True,
    )
    print(
        "Recipe: lr=0.0035 / 180 ep · 12b · atten on · "
        "lambda in {0.50, 0.05, 0.10, 0.15, 0.90}",
        flush=True,
    )

    base_config = load_config()
    original = deepcopy(base_config)
    results = []
    skipped = []

    try:
        for i, run in enumerate(SWEEP_RUNS):
            if datetime.now() >= deadline:
                print(f"SKIP {run['id']}: past start deadline", flush=True)
                for later in SWEEP_RUNS[i:]:
                    skipped.append({"id": later["id"], "reason": "past_start_deadline"})
                break
            result = run_one(run, base_config)
            results.append(result)
            SWEEP_LOG.write_text(
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
    finally:
        save_config(restore_care_defaults(original))

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "start_deadline": deadline.isoformat(timespec="seconds"),
        "results": results,
        "skipped": skipped,
    }
    SWEEP_LOG.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== Tonight summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
