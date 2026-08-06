"""One-shot: R1 (128t, 12b, c=2, 100 ep) with distance attenuation ON, then restore config + append playlist."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config" / "hyperpameters.json"
BACKUP = ROOT / "config" / "hyperpameters_R1_atten_backup.json"
PLAY = ROOT / "config" / "play_filtered_config.json"
LOG = ROOT / "outputs" / "cagrad" / "R1_atten_on.log"
SAVED_RE = re.compile(r"Saved outputs to (.+)")


def main():
    original = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    BACKUP.write_text(json.dumps(original, indent=4) + "\n", encoding="utf-8")

    cfg = json.loads(json.dumps(original))
    cfg["env"]["apply_distance_attenuation"] = True
    cfg["filter_estimator"]["n_samples"] = 128
    cfg["stimulus_dataset"]["n_bands"] = 12
    cfg["stimulus_dataset"]["erb_factor"] = 0.25
    cfg["stimulus_dataset"]["n_samples_per_band"] = 32
    cfg["stimulus_dataset"]["band_mode"] = "linear_erb"
    cfg["stimulus_dataset"]["dataset_type"] = "bandpass"
    cfg["stimulus_dataset"]["stimulus_type"] = "noise"
    cfg["training"]["num_epochs"] = 100
    cfg["training"]["lr"] = 0.01
    cfg["training"]["cagrad_c"] = 2.0
    cfg["training"]["lambda_loc"] = 0.5
    cfg["training"]["mtl_optimizer"] = "cagrad"
    cfg["training"]["batch_size"] = 64
    cfg["saving"]["use_tensorboard"] = False
    CFG_PATH.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    print("Config: R1 + atten on (128t / 12b / c=2 / 100 ep)", flush=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    LOG.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(LOG, "w", encoding="utf-8") as log_f:
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
    finally:
        if BACKUP.exists():
            shutil.copy(BACKUP, CFG_PATH)
            BACKUP.unlink()
            print("Restored hyperpameters.json", flush=True)

    text = LOG.read_text(encoding="utf-8", errors="replace")
    match = None
    for match in SAVED_RE.finditer(text):
        pass
    print(text[-2000:], flush=True)

    if proc.returncode != 0:
        print(f"FAILED exit={proc.returncode}", flush=True)
        raise SystemExit(proc.returncode)

    if match is None:
        print("Finished but no Saved outputs path found", flush=True)
        raise SystemExit(2)

    path = Path(match.group(1).strip())
    try:
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix().replace("\\", "/")

    play = json.loads(PLAY.read_text(encoding="utf-8"))
    folders = list(play.get("folders", []))
    if rel not in folders:
        folders.append(rel)
    # Keep blank line after baseline 13 (indices 0-12) if both blocks exist
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
    print(f"APPENDED playlist as last entry: {rel} (#{len(folders)})", flush=True)
    print(f"Saved outputs to {rel}", flush=True)


if __name__ == "__main__":
    main()
