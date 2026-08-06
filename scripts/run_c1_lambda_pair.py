"""Sequential: same as C2/#21 (128t 12b c=1 care atten) with lambda_loc variants.
Skips ones that already exist with the same key knobs.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "hyperpameters.json"
BACKUP = ROOT / "config" / "hyperpameters_lam_care_backup.json"
PLAY = ROOT / "config" / "play_filtered_config.json"
LOG_DIR = ROOT / "outputs" / "cagrad"
SAVED_RE = re.compile(r"Saved outputs to (.+)")

RUNS = [
    {"id": "R1care_c1_lam025", "lambda_loc": 0.25, "skip_if": "outputs/cagrad/2026-08-05_024108"},
    {"id": "R1care_c1_lam075", "lambda_loc": 0.75, "skip_if": None},
]


def apply_base(cfg, lambda_loc):
    cfg = json.loads(json.dumps(cfg))
    cfg["env"]["apply_distance_attenuation"] = True
    cfg["filter_estimator"]["n_samples"] = 128
    cfg["stimulus_dataset"]["n_bands"] = 12
    cfg["stimulus_dataset"]["erb_factor"] = 0.25
    cfg["stimulus_dataset"]["n_samples_per_band"] = 32
    cfg["stimulus_dataset"]["band_mode"] = "linear_erb"
    cfg["stimulus_dataset"]["dataset_type"] = "bandpass"
    cfg["stimulus_dataset"]["stimulus_type"] = "noise"
    cfg["training"]["num_epochs"] = 150
    cfg["training"]["lr"] = 0.005
    cfg["training"]["cagrad_c"] = 1.0
    cfg["training"]["lambda_loc"] = lambda_loc
    cfg["training"]["mtl_optimizer"] = "cagrad"
    cfg["training"]["batch_size"] = 64
    cfg["saving"]["use_tensorboard"] = False
    return cfg


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


def main():
    original = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    BACKUP.write_text(json.dumps(original, indent=4) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for run in RUNS:
            if run.get("skip_if") and (ROOT / run["skip_if"]).exists():
                print(
                    f"SKIP {run['id']}: already have {run['skip_if']}",
                    flush=True,
                )
                continue

            print(
                f"===== Starting {run['id']} lambda_loc={run['lambda_loc']} "
                f"(#21 recipe: 128t 12b c=1 lr=0.005 ep150 atten) =====",
                flush=True,
            )
            CFG_PATH.write_text(
                json.dumps(apply_base(original, run["lambda_loc"]), indent=4) + "\n",
                encoding="utf-8",
            )
            log_path = LOG_DIR / f"{run['id']}.log"
            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    [
                        "conda",
                        "run",
                        "-n",
                        "polar-env",
                        "--no-capture-output",
                        "python",
                        str(ROOT / "scripts" / "scratch.py"),
                    ],
                    cwd=str(ROOT),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            text = log_path.read_text(encoding="utf-8", errors="replace")
            print(text[-1500:], flush=True)
            if proc.returncode != 0:
                print(f"{run['id']} FAILED exit={proc.returncode}", flush=True)
                continue
            match = None
            for match in SAVED_RE.finditer(text):
                pass
            if match is None:
                print(f"{run['id']} no output path", flush=True)
                continue
            rel = relative_output_path(match.group(1))
            append_playlist(rel)
            print(f"{run['id']} -> {rel}", flush=True)
    finally:
        if BACKUP.exists():
            shutil.copy(BACKUP, CFG_PATH)
            BACKUP.unlink()
            print("Restored hyperpameters.json", flush=True)


if __name__ == "__main__":
    main()
