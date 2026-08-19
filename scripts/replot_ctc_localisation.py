"""Replot CTC localisation arrows using 8 x 100 ms white-noise averaging."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

from propagation.model import BinauralFreeFieldPropagation
from utils.sofa_functions import load_sofa

CTC_ROOT = ROOT / "outputs/ctc"
PLOT_NAMES = {
    "Proposed (Col. + IPD)": "localisation_proposed",
    "Proposed (Col. only)": "localisation_col_only",
    "CTC": "localisation_ctc",
}


def _load_ctc_module():
    spec = importlib.util.spec_from_file_location(
        "ctc_optimisation", ROOT / "scripts" / "ctc_optimisation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replot_run(run_dir, ctc_mod, device, dtype):
    npy = run_dir / "npy"
    spk_path = npy / "spk_filters.npy"
    if not spk_path.exists():
        print(f"skip {run_dir.name}: no spk_filters.npy")
        return

    with open(run_dir / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)

    fs = int(run_config["fs"])
    target_angle = float(run_config["args"]["target_angle"])
    sofa_path = run_config["args"]["sofa_path"]
    apply_distance_attenuation = bool(run_config.get("apply_distance_attenuation", True))
    speaker_distance = float(run_config.get("speaker_distance", 2.0))

    sofaile = load_sofa(sofa_path)
    lstn_pos, spk_pos, _, target_angles, _ = ctc_mod._build_geometry(
        dtype, device, target_angle, speaker_distance=speaker_distance,
    )
    prop = BinauralFreeFieldPropagation(
        fs, lstn_pos, spk_pos, sofaile,
        apply_distance_attenuation=apply_distance_attenuation,
        device=device, dtype=dtype,
    )

    proposed = np.load(spk_path)
    if proposed.ndim == 3:
        proposed = proposed[0]
    col_path = npy / "spk_filters_col_only.npy"
    if not col_path.exists():
        print(f"skip {run_dir.name}: no spk_filters_col_only.npy")
        return
    proposed_col_only = np.load(col_path)
    if proposed_col_only.ndim == 3:
        proposed_col_only = proposed_col_only[0]
    ctc_filters = np.load(npy / "ctc_filters.npy")

    old_proposed = None
    old_path = npy / "localisation_results.Proposed (Col. + IPD).localisation_angles.npy"
    if old_path.exists():
        old_proposed = float(np.load(old_path).reshape(-1)[0])

    figs, results = ctc_mod.plot_ctc_localisation(
        proposed=proposed,
        proposed_col_only=proposed_col_only,
        ctc=ctc_filters,
        prop=prop,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        target_angles=target_angles,
        target_angle_deg=target_angle,
        sofaile=sofaile,
        fs=fs,
        device=device,
        dtype=dtype,
    )

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for fig, (name, res) in zip(figs, results.items()):
        out = plot_dir / f"{PLOT_NAMES[name]}.png"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        plt.close(fig)
        angle = float(np.atleast_1d(np.asarray(res["localisation_angles"]).reshape(-1))[0])
        extra = ""
        if name == "Proposed (Col. + IPD)" and old_proposed is not None:
            extra = f" (old {old_proposed:.1f})"
        print(f"  {out.name}: {angle:.1f} deg{extra}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Replotting CTC localisation on {device}")
    ctc_mod = _load_ctc_module()
    for run_dir in sorted(p for p in CTC_ROOT.iterdir() if p.is_dir()):
        print(run_dir.name)
        replot_run(run_dir, ctc_mod, device, dtype)


if __name__ == "__main__":
    main()
