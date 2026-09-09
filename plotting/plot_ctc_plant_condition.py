"""Plot condition numbers of the CTC plant inverse H(ω)^{-1}.

Uses the same ±30° HRTF plant and DFT length as ``scripts/ctc_optimisation.py``.

Example:
  python plotting/plot_ctc_plant_condition.py
  python plotting/plot_ctc_plant_condition.py --run-dir outputs/ctc/2026-08-19_002700
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.sofa_functions import load_sofa


def _load_ctc_module():
    spec = importlib.util.spec_from_file_location(
        "ctc_optimisation", ROOT / "scripts" / "ctc_optimisation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _latest_ctc_run(ctc_root):
    runs = sorted(
        p for p in Path(ctc_root).iterdir()
        if p.is_dir() and p.name[:4].isdigit()
    )
    return runs[-1] if runs else None


def _peak_table(stats, key="cond_inv", f_min=80.0, f_max=20000.0, n_peaks=8):
    freqs = stats["freqs"]
    y = stats[key]
    band = (freqs >= f_min) & (freqs <= f_max)
    f = freqs[band]
    c = y[band]
    peaks = []
    for i in range(c.size):
        left = c[i - 1] if i > 0 else -np.inf
        right = c[i + 1] if i < c.size - 1 else -np.inf
        if c[i] >= left and c[i] >= right:
            peaks.append((float(c[i]), float(f[i])))
    peaks.sort(reverse=True)
    return peaks[:n_peaks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sofa-path", type=str, default=None)
    parser.add_argument("--fs", type=int, default=None)
    parser.add_argument("--ctc-n-fft", type=int, default=None)
    parser.add_argument("--n-fft-dense", type=int, default=4096)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_ctc_run(ROOT / "outputs" / "ctc")
    cfg = {}
    if run_dir is not None and (run_dir / "run_config.json").exists():
        with open(run_dir / "run_config.json", encoding="utf-8") as f:
            cfg = json.load(f)

    sofa_path = args.sofa_path or cfg.get("args", {}).get(
        "sofa_path", "data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa",
    )
    fs = int(args.fs or cfg.get("fs", 48000))
    ctc_n_fft = int(
        args.ctc_n_fft
        or cfg.get("ctc_n_fft")
        or cfg.get("args", {}).get("n_taps", 256)
    )
    n_fft_dense = max(int(args.n_fft_dense), ctc_n_fft)

    ctc_mod = _load_ctc_module()
    sofaile = load_sofa(sofa_path)
    stats = ctc_mod.analyse_plant_inverse_condition(
        ctc_mod.compute_plant_matrix(sofaile, n_fft=ctc_n_fft), fs,
    )
    stats_dense = ctc_mod.analyse_plant_inverse_condition(
        ctc_mod.compute_plant_matrix(sofaile, n_fft=n_fft_dense), fs,
    )

    print(f"SOFA: {sofa_path}")
    print(f"Plant: 2x2 HRTFs at ±30°, fs={fs} Hz")
    ctc_mod.print_plant_condition_summary(stats)
    if n_fft_dense != ctc_n_fft:
        ctc_mod.print_plant_condition_summary(stats_dense)
    print("Largest local peaks of κ(H^{-1}) (dense grid, 80–20 kHz):")
    for kappa, freq in _peak_table(stats_dense):
        print(f"  {freq:8.1f} Hz   κ = {kappa:.3g}")
    print("Largest local peaks of ||H^{-1}||_2 (dense grid, 80–20 kHz):")
    for gain, freq in _peak_table(stats_dense, key="inv_spectral_norm"):
        print(f"  {freq:8.1f} Hz   ||H^{{-1}}||_2 = {gain:.3g}")

    if args.output:
        out_path = Path(args.output)
    elif run_dir is not None:
        out_path = run_dir / "plots" / "ctc_plant_condition.png"
    else:
        out_path = ROOT / "outputs" / "ctc" / "ctc_plant_condition.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gain_path = out_path.with_name("ctc_plant_inverse_gain.png")

    fig = ctc_mod.plot_ctc_plant_condition(stats, stats_dense)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.pdf')}")

    fig_gain = ctc_mod.plot_ctc_plant_inverse_gain(stats, stats_dense)
    fig_gain.savefig(gain_path, bbox_inches="tight", dpi=150)
    fig_gain.savefig(gain_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig_gain)
    print(f"Saved {gain_path}")
    print(f"Saved {gain_path.with_suffix('.pdf')}")

    if run_dir is not None:
        npy_dir = run_dir / "npy"
        npy_dir.mkdir(exist_ok=True)
        np.save(npy_dir / "plant_condition.freqs.npy", stats["freqs"])
        np.save(npy_dir / "plant_condition.cond_inv.npy", stats["cond_inv"])
        np.save(npy_dir / "plant_condition.cond_H.npy", stats["cond_H"])
        np.save(npy_dir / "plant_condition.sigma_max.npy", stats["sigma_max"])
        np.save(npy_dir / "plant_condition.sigma_min.npy", stats["sigma_min"])
        np.save(npy_dir / "plant_condition.inv_spectral_norm.npy", stats["inv_spectral_norm"])
        np.save(npy_dir / "plant_condition.dense.freqs.npy", stats_dense["freqs"])
        np.save(npy_dir / "plant_condition.dense.cond_inv.npy", stats_dense["cond_inv"])
        np.save(npy_dir / "plant_condition.dense.sigma_max.npy", stats_dense["sigma_max"])
        np.save(npy_dir / "plant_condition.dense.sigma_min.npy", stats_dense["sigma_min"])
        np.save(npy_dir / "plant_condition.dense.inv_spectral_norm.npy", stats_dense["inv_spectral_norm"])
        print(f"Saved arrays under {npy_dir}")


if __name__ == "__main__":
    main()
