"""
Multi-day 4-param grid search (ntaps, n_bands, lr, lambda_loc).

Centre: 128 taps · 12 bands · lr=0.0035 · (lambda around low range).
Lambda emphasis on small values (0.05–0.25); high-lambda above ~0.25 was
observed near-identical, so factorial only covers low λ, plus the in-progress
centre high-control cell (lambda=0.5 @ centre geometry/schedule).

Fixed: cagrad_c=1, atten ON, 180 ep, samples_per_band=32, batch=64.

Resume-safe: skips cells already present under outputs/cagrad/*/json/run_config.json
with matching knobs. Appends successes to play_filtered_config.json.

No start deadline (multi-day). Never kills an in-progress run.
"""
import json
import os
import re
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "hyperpameters.json"
PLAY = ROOT / "config" / "play_filtered_config.json"
LOG_DIR = ROOT / "outputs" / "cagrad"
SWEEP_LOG = LOG_DIR / "grid_4param_log.json"
SAVED_RE = re.compile(r"Saved outputs to (.+)")

# --- grid definition (centred on current care spine) ---
N_TAPS = [64, 128, 256]
N_BANDS = [6, 12, 24]
LRS = [0.0025, 0.0035, 0.005]
# Low λ only for factorial (higher ≈ same). 0.5 is a single high-control extra below.
LAMBDAS = [0.05, 0.10, 0.15, 0.25]

CENTER_TAPS = 128
CENTER_BANDS = 12
CENTER_LR = 0.0035
CENTER_LAM = 0.15  # middle of low-λ set for prioritisation

NUM_EPOCHS = 180
CAGRAD_C = 1.0
N_SAMPLES_PER_BAND = 32
ATTEN = True

# erb_factor must cover [20, 20k] for linear_erb
ERB_BY_BANDS = {
    6: 0.1,
    12: 0.25,
    24: 0.25,
}

# Extra high-λ controls at centre only (not full factorial — high λ looked similar).
# Includes the currently-running / queued tonight jobs so they count as grid hits.
EXTRA_CELLS = [
    {
        "n_samples": CENTER_TAPS,
        "n_bands": CENTER_BANDS,
        "lr": CENTER_LR,
        "lambda_loc": 0.50,  # in progress tonight A
    },
    {
        "n_samples": CENTER_TAPS,
        "n_bands": CENTER_BANDS,
        "lr": CENTER_LR,
        "lambda_loc": 0.90,  # still on old tonight queue after A
    },
]


def cell_id(cell):
    return (
        f"t{cell['n_samples']}_b{cell['n_bands']}_"
        f"lr{cell['lr']:.4f}_lam{cell['lambda_loc']:.2f}"
    ).replace(".", "p")


def make_cell(n_samples, n_bands, lr, lambda_loc):
    return {
        "id": cell_id(
            {
                "n_samples": n_samples,
                "n_bands": n_bands,
                "lr": lr,
                "lambda_loc": lambda_loc,
            }
        ),
        "n_samples": n_samples,
        "n_bands": n_bands,
        "erb_factor": ERB_BY_BANDS[n_bands],
        "lr": float(lr),
        "lambda_loc": float(lambda_loc),
        "num_epochs": NUM_EPOCHS,
        "cagrad_c": CAGRAD_C,
        "n_samples_per_band": N_SAMPLES_PER_BAND,
        "apply_distance_attenuation": ATTEN,
    }


def priority(cell):
    """Lower = run sooner: near centre first, then cheaper (fewer bands/taps)."""
    dt = abs(cell["n_samples"] - CENTER_TAPS) / CENTER_TAPS
    db = abs(cell["n_bands"] - CENTER_BANDS) / CENTER_BANDS
    dlr = abs(cell["lr"] - CENTER_LR) / CENTER_LR
    dlam = abs(cell["lambda_loc"] - CENTER_LAM) / max(CENTER_LAM, 1e-6)
    # rough relative cost
    cost = (cell["n_samples"] / 128.0) * (cell["n_bands"] / 12.0)
    return (dt + db + dlr + dlam, cost)


def build_grid():
    cells = []
    for nt, nb, lr, lam in product(N_TAPS, N_BANDS, LRS, LAMBDAS):
        cells.append(make_cell(nt, nb, lr, lam))
    for extra in EXTRA_CELLS:
        cells.append(
            make_cell(
                extra["n_samples"],
                extra["n_bands"],
                extra["lr"],
                extra["lambda_loc"],
            )
        )
    # de-dupe by id
    by_id = {c["id"]: c for c in cells}
    ordered = sorted(by_id.values(), key=priority)
    return ordered


def float_close(a, b, rtol=1e-5, atol=1e-9):
    return abs(float(a) - float(b)) <= max(atol, rtol * max(abs(float(a)), abs(float(b))))


def cell_key(cell):
    return (
        int(cell["n_samples"]),
        int(cell["n_bands"]),
        round(float(cell["lr"]), 6),
        round(float(cell["lambda_loc"]), 4),
        int(cell["num_epochs"]),
        round(float(cell["cagrad_c"]), 3),
        bool(cell["apply_distance_attenuation"]),
    )


def key_from_run_config(cfg):
    try:
        fe = cfg["filter_estimator"]
        sd = cfg["stimulus_dataset"]
        tr = cfg["training"]
        env = cfg.get("env", {})
        return (
            int(fe["n_samples"]),
            int(sd["n_bands"]),
            round(float(tr["lr"]), 6),
            round(float(tr.get("lambda_loc", 0.5)), 4),
            int(tr["num_epochs"]),
            round(float(tr.get("cagrad_c", 1.0)), 3),
            bool(env.get("apply_distance_attenuation", False)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def scan_completed():
    """Map cell_key -> relative output path for finished matching runs."""
    found = {}
    root = LOG_DIR
    if not root.exists():
        return found
    for run_cfg_path in root.glob("*/json/run_config.json"):
        try:
            data = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cfg = data.get("config", data)
        key = key_from_run_config(cfg)
        if key is None:
            continue
        # require completed training artifact
        out_dir = run_cfg_path.parent.parent
        if not (out_dir / "npy" / "spk_filters.npy").exists():
            continue
        rel = out_dir.resolve().relative_to(ROOT.resolve()).as_posix()
        found[key] = rel
    return found


def load_config():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def save_config(cfg):
    CFG_PATH.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")


def apply_run(config, run):
    cfg = deepcopy(config)
    cfg.setdefault("env", {})
    cfg["env"]["apply_distance_attenuation"] = bool(run["apply_distance_attenuation"])
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


def restore_centre_defaults(base_config):
    restore = deepcopy(base_config)
    restore.setdefault("env", {})
    restore["env"]["apply_distance_attenuation"] = True
    restore["filter_estimator"]["n_samples"] = CENTER_TAPS
    restore["stimulus_dataset"]["n_bands"] = CENTER_BANDS
    restore["stimulus_dataset"]["erb_factor"] = ERB_BY_BANDS[CENTER_BANDS]
    restore["stimulus_dataset"]["n_samples_per_band"] = N_SAMPLES_PER_BAND
    restore["training"]["num_epochs"] = NUM_EPOCHS
    restore["training"]["lr"] = CENTER_LR
    restore["training"]["cagrad_c"] = CAGRAD_C
    restore["training"]["lambda_loc"] = CENTER_LAM
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


def write_status(results, skipped, remaining):
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "n_done": sum(1 for r in results if r.get("status") == "success"),
        "n_failed": sum(1 for r in results if r.get("status") == "failed"),
        "n_skipped": len(skipped),
        "n_remaining": remaining,
        "results": results,
        "skipped": skipped,
    }
    SWEEP_LOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one(run, base_config):
    print(
        f"\n===== {run['id']}: taps={run['n_samples']} bands={run['n_bands']} "
        f"erb={run['erb_factor']} c={run['cagrad_c']} lambda={run['lambda_loc']} "
        f"lr={run['lr']} ep={run['num_epochs']} atten={run['apply_distance_attenuation']} =====",
        flush=True,
    )
    save_config(apply_run(base_config, run))
    log_path = LOG_DIR / f"grid_{run['id']}.log"
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
    print(text[-1500:], flush=True)
    match = None
    for match in SAVED_RE.finditer(text):
        pass
    result = {
        "id": run["id"],
        "config": {
            "n_samples": run["n_samples"],
            "n_bands": run["n_bands"],
            "lr": run["lr"],
            "lambda_loc": run["lambda_loc"],
            "num_epochs": run["num_epochs"],
            "cagrad_c": run["cagrad_c"],
        },
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


def _python_cmd_match(pattern):
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
                "Measure-Object | Select-Object -ExpandProperty Count",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip() or "0") > 0
    except (subprocess.CalledProcessError, ValueError):
        return False


def scratch_running():
    return _python_cmd_match(r"scratch\\.py")


def tonight_driver_running():
    """Old 5-run queue whose cells we absorb into this grid."""
    return _python_cmd_match(r"tonight_lr_bands_sweep")


def wait_for_gpu_free(poll_s=45):
    """Let the currently-running / queued centre jobs finish first (they map into the grid).

    Do not kill them — next cells after A (lambda 0.05/0.10/0.15 + 0.90 centre)
    are part of EXTRA or the low-lambda factorial at centre.
    """
    if not scratch_running() and not tonight_driver_running():
        return
    print(
        "Waiting for current tonight queue / scratch.py to finish "
        "(those centre cells integrate into the grid)...",
        flush=True,
    )
    while scratch_running() or tonight_driver_running():
        time.sleep(poll_s)
        bits = []
        if scratch_running():
            bits.append("scratch")
        if tonight_driver_running():
            bits.append("tonight_driver")
        print(
            f"  still busy ({', '.join(bits) if bits else '?'}) "
            f"@ {datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
    print("GPU free — starting remaining grid cells.", flush=True)
    time.sleep(3)


def main():
    grid = build_grid()
    print(f"Grid 4-param start {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(
        f"Levels: taps={N_TAPS} bands={N_BANDS} lr={LRS} lambda={LAMBDAS} "
        f"+ high-control lambda=0.5 at centre  -> {len(grid)} cells",
        flush=True,
    )
    print(f"Fixed: ep={NUM_EPOCHS} c={CAGRAD_C} atten={ATTEN}", flush=True)
    print("Policy: multi-day, resume skips done cells, never kill mid-run", flush=True)

    wait_for_gpu_free()

    # Re-scan after wait so finishing job is picked up
    completed = scan_completed()
    base_config = load_config()
    original = deepcopy(base_config)
    results = []
    skipped = []

    try:
        pending = []
        for cell in grid:
            key = cell_key(cell)
            if key in completed:
                skipped.append(
                    {
                        "id": cell["id"],
                        "reason": "already_completed",
                        "path": completed[key],
                    }
                )
                print(f"SKIP {cell['id']}: already {completed[key]}", flush=True)
            else:
                pending.append(cell)

        print(
            f"Queue: {len(pending)} to run, {len(skipped)} skipped of {len(grid)} total",
            flush=True,
        )
        write_status(results, skipped, len(pending))

        for i, cell in enumerate(pending):
            # re-check in case another process wrote a matching run
            completed = scan_completed()
            key = cell_key(cell)
            if key in completed:
                skipped.append(
                    {
                        "id": cell["id"],
                        "reason": "already_completed",
                        "path": completed[key],
                    }
                )
                print(f"SKIP {cell['id']}: already {completed[key]}", flush=True)
                write_status(results, skipped, len(pending) - i - 1)
                continue

            result = run_one(cell, base_config)
            results.append(result)
            if result.get("path"):
                completed[key] = result["path"]
            write_status(results, skipped, max(0, len(pending) - i - 1))
    finally:
        save_config(restore_centre_defaults(original))

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "n_grid": len(grid),
        "results": results,
        "skipped": skipped,
    }
    SWEEP_LOG.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== Grid 4-param summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
