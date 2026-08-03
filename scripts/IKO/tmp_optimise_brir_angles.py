"""Temporary batch script: optimise IKO BRIR for multiple target angles and export ramped driver white noise."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.seed import set_seed

ANGLES = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20]
HYPERPARAMETERS_PATH = ROOT / "config" / "hyperpameters.json"
OUTPUTS_ROOT = ROOT / "outputs" / "iko_brir"
MANIFEST_PATH = OUTPUTS_ROOT / "angle_sweep_manifest.json"
DURATION_S = 2.0
RAMP_S = 0.05
TARGET_PEAK = 0.95
STAGE = "final"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iko_brir = _load_module("iko_brir_optimisation", ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py")
export_iko = _load_module("export_iko_drivers", ROOT / "scripts" / "listening" / "export_iko_drivers.py")


def load_config():
    with open(HYPERPARAMETERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    with open(HYPERPARAMETERS_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


def list_run_dirs():
    if not OUTPUTS_ROOT.exists():
        return set()
    return {p for p in OUTPUTS_ROOT.iterdir() if p.is_dir()}


def angle_suffix(angle):
    return f"ang{angle:+d}".replace("+", "")


def rename_run_dir(run_dir, angle):
    new_path = run_dir.parent / f"{run_dir.name}_{angle_suffix(angle)}"
    run_dir.rename(new_path)
    return new_path


def main():
    config = load_config()
    set_seed(config["env"]["seed"])
    original_target_angle = config["training"]["target_angle"]
    fs = int(config["env"]["fs"])
    device = config["env"]["device"]
    dtype = torch.float32
    n_eigenbeams = 16  # 3rd-order HOA

    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        iko_brir.SPH_TO_DRIVER_NPY, n_eigenbeams, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype,
    )
    mono = export_iko.white_noise(int(round(DURATION_S * fs)), random_state=config["env"]["seed"])
    mono = export_iko.apply_ramp(mono, fs, RAMP_S)

    manifest = {}
    try:
        for angle in ANGLES:
            print(f"\n=== Optimising for target angle {angle}° ===")
            config["training"]["target_angle"] = float(angle)
            save_config(config)

            before = list_run_dirs()
            iko_brir.main()
            after = list_run_dirs()
            new_dirs = after - before
            if len(new_dirs) != 1:
                raise RuntimeError(f"Expected one new output folder, found {len(new_dirs)}: {new_dirs}")

            run_dir = rename_run_dir(new_dirs.pop(), angle)
            print(f"Renamed run folder to {run_dir.name}")

            export_iko.export_run(
                run_dir, STAGE, fs, device, dtype, eigenbeam_to_loudspeaker, mono, target_peak=TARGET_PEAK,
            )
            manifest[str(angle)] = str(run_dir.relative_to(ROOT)).replace("\\", "/")
    finally:
        config["training"]["target_angle"] = original_target_angle
        save_config(config)
        print(f"\nRestored target_angle to {original_target_angle}° in {HYPERPARAMETERS_PATH}")

    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        f.write("\n")
    print(f"\nWrote manifest to {MANIFEST_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
