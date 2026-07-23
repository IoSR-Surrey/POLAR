import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.IKO_functions import num_hoa_channels

IKO_ORDER = 3
N_CHANNELS = num_hoa_channels(IKO_ORDER)
N_SPEAKERS = 20
SPH_TO_DRIVER_NPY = ROOT / "data" / "IKO" / "sph_to_driver.npy"


def load_iko_brir_optimisation_module():
    module_path = ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py"
    spec = importlib.util.spec_from_file_location("iko_brir_optimisation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hoa_flat_index_to_nm(order, channel_idx):
    idx = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if idx == channel_idx:
                return n, m
            idx += 1
    raise IndexError(f"channel_idx {channel_idx} out of range for order {order}")


def driver_energy_coefficient_of_variation(energies):
    energies = np.asarray(energies, dtype=np.float64)
    return float(energies.std() / (energies.mean() + 1e-12))


def main():
    iko_brir = load_iko_brir_optimisation_module()
    if not SPH_TO_DRIVER_NPY.is_file():
        raise FileNotFoundError(f"Missing measured data: {SPH_TO_DRIVER_NPY}")

    sph_to_driver = iko_brir.load_sph_to_driver(SPH_TO_DRIVER_NPY, N_CHANNELS, N_SPEAKERS, "cpu", torch.float32)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path("outputs") / "iko_brir_fingerprint" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharey=False)
    summary_rows = []
    for channel_idx, ax in enumerate(axes.ravel()):
        n, m = hoa_flat_index_to_nm(IKO_ORDER, channel_idx)
        energies = iko_brir.fingerprint_driver_energies(sph_to_driver, channel_idx).detach().cpu().numpy()
        cv = driver_energy_coefficient_of_variation(energies)
        summary_rows.append((channel_idx, n, m, cv, float(np.max(energies) / np.min(energies))))
        ax.bar(np.arange(N_SPEAKERS), energies, color="steelblue")
        ax.set_title(f"ACN{channel_idx} n={n} m={m:+d}\nCV={cv:.2f}", fontsize=8)
        ax.set_xlabel("Driver", fontsize=7)
        ax.set_xticks(np.arange(0, N_SPEAKERS, 5))

    fig.suptitle("One-hot driver excitation fingerprints (pre-BRIR)")
    fig.tight_layout()
    summary_plot = output_dir / "fingerprint_summary.png"
    fig.savefig(summary_plot, dpi=150)
    plt.close(fig)

    summary_path = output_dir / "fingerprint_summary.txt"
    with open(summary_path, "w") as f:
        f.write("channel_idx\tn\tm\tcv\tmax_over_min\n")
        for channel_idx, n, m, cv, dynamic_range in summary_rows:
            f.write(f"{channel_idx}\t{n}\t{m:+d}\t{cv:.6f}\t{dynamic_range:.6f}\n")

    print(f"Saved fingerprint plot to {summary_plot}")
    print(f"Saved fingerprint table to {summary_path}")


if __name__ == "__main__":
    main()
