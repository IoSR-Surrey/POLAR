"""Replot localisation_arrows.png from a run folder using the 8 x 100 ms probe."""
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

from plotting.evaluation import compute_white_noise_plot_outputs
from plotting.miscellaneous import listener_labels_from_lstn_pos
from plotting.spatial_plots import localisation_arrows_plot
from utils.miscellaneous import target_angle_for_listeners
from utils.sofa_functions import load_sofa

RUNS = [
    ROOT / "outputs/cagrad/2026-08-05_045408",
    ROOT / "outputs/cagrad/2026-08-05_050357",
]


class _SavingManager:
    def __init__(self, saved_data):
        self.saved_data = saved_data


def _json_value(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_saved_data(run_dir):
    npy = run_dir / "npy"
    js = run_dir / "json"
    sofa_path = _json_value(js / "sofa_path.json")
    return {
        "lstn_pos": [torch.as_tensor(np.load(npy / "lstn_pos.npy"))],
        "spk_pos": [torch.as_tensor(np.load(npy / "spk_pos.npy"))],
        "spk_filters": [torch.as_tensor(np.load(npy / "spk_filters.npy"))],
        "gtfb": [torch.as_tensor(np.load(npy / "gtfb.npy"))],
        "target_position": [torch.as_tensor(np.load(npy / "target_position.npy"))],
        "fs": [_json_value(js / "fs.json")],
        "apply_distance_attenuation": [_json_value(js / "apply_distance_attenuation.json")],
        "sofaile": [load_sofa(sofa_path)],
        "localisation_angles": [np.load(npy / "localisation_angles.npy")],
        "lsd": [np.load(npy / "lsd.npy")],
    }


def replot_run(run_dir):
    saved_data = load_saved_data(run_dir)
    old_loc = np.asarray(saved_data["localisation_angles"][0]).reshape(-1)
    old_lsd = np.asarray(saved_data["lsd"][0]).reshape(-1)
    outputs = compute_white_noise_plot_outputs(_SavingManager(saved_data))
    if outputs is None:
        raise RuntimeError(f"Could not evaluate {run_dir}")

    lstn_pos = saved_data["lstn_pos"][0]
    target_angle = target_angle_for_listeners(lstn_pos, saved_data["target_position"][0])
    loc = np.asarray(outputs["localisation_angles"]).reshape(-1)
    lsd = np.asarray(outputs["lsd"]).reshape(-1)
    print(run_dir.name)
    for i, pos in enumerate(np.asarray(lstn_pos)):
        print(
            f"  ({pos[0]:.2f}, {pos[1]:.2f}): "
            f"az {loc[i]:.1f} deg (old {old_loc[i]:.1f}), "
            f"LSD {lsd[i]:.2f} dB (old {old_lsd[i]:.2f})"
        )

    localisation_arrows_plot(
        lstn_pos,
        saved_data["spk_pos"][0],
        outputs["localisation_angles"],
        outputs["localisation_uncertainty"],
        lsd=outputs["lsd"],
        target_angle=target_angle,
        listener_labels=listener_labels_from_lstn_pos(lstn_pos),
    )
    out = run_dir / "plots" / "localisation_arrows.png"
    fig = plt.gcf()
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    for run_dir in RUNS:
        replot_run(run_dir)


if __name__ == "__main__":
    main()
