"""Sweep stimuli at omni-only HOA init and report estimated azimuths.

Mirrors IKO_brir_optimisation initialisation (weights = [1, 0, ..., 0]) and can use either
random white noise (to quantify stimulus-driven variance) or a unit impulse (deterministic).
"""

import argparse
import importlib.util
import json
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
from utils.miscellaneous import load_npy_to_torch, write_run_config
from utils.sofa_functions import load_sofa, resample_sofa_ir, get_template_for_llado2025_desena_2020
from models.listener_based.llado2025.model import llado2025_grad_surgery


def load_iko_brir_optimisation_module():
    module_path = ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py"
    spec = importlib.util.spec_from_file_location("iko_brir_optimisation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate localisation azimuth for many noise instances at omni HOA init.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of independent trials.")
    parser.add_argument("--signal-length", type=int, default=256, help="Mono stimulus length (samples).")
    parser.add_argument(
        "--stimulus",
        choices=("noise", "impulse"),
        default="noise",
        help="Stimulus type: unseeded-style white noise draws, or a unit impulse.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for noise reproducibility. If omitted, uses config env.seed (or 0).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "hyperpameters.json",
        help="Path to hyperparameters JSON.",
    )
    return parser.parse_args()


def make_stimulus(stimulus, length, device, dtype, seed=None):
    if stimulus == "impulse":
        signal = torch.zeros(1, length, device=device, dtype=dtype)
        signal[0, 0] = 1.0
        return signal
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(1, length, device=device, dtype=dtype)


def main():
    args = parse_args()
    iko_brir = load_iko_brir_optimisation_module()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    device = config["env"]["device"]
    dtype = torch.float32
    target_angle = float(config["training"]["target_angle"])
    base_seed = args.seed if args.seed is not None else int(config["env"].get("seed", 0))

    iko_order = 3
    n_eigenbeams = num_hoa_channels(iko_order)

    sofaile = resample_sofa_ir(load_sofa("data/SOFA/mit_kemar_large_pinna.sofa"), fs)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    llado_model.eval()

    brirs = iko_brir.load_iko_brirs(
        iko_brir.BRIR_NPY, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype,
    )
    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        iko_brir.SPH_TO_DRIVER_NPY, n_eigenbeams, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype,
    )

    # Same omni-only init as IKO_brir_optimisation.main
    weights = torch.zeros(n_eigenbeams, device=device, dtype=dtype)
    weights[0] = 1.0
    target = torch.tensor(target_angle, device=device, dtype=dtype)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path("outputs") / "iko_brir_init_noise_sweep" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(output_dir, {
        "script": "scripts/IKO/IKO_brir_init_noise_sweep.py",
        "config": config,
        "args": {
            "stimulus": args.stimulus,
            "n_trials": args.n_trials,
            "signal_length": args.signal_length,
            "seed": base_seed,
        },
        "target_angle": target_angle,
    })

    azimuths = np.empty(args.n_trials, dtype=np.float64)
    loc_losses = np.empty(args.n_trials, dtype=np.float64)
    seeds = np.empty(args.n_trials, dtype=np.int64)

    print(f"Omni HOA weights: {weights.detach().cpu().numpy()}")
    print(
        f"stimulus={args.stimulus}, n_trials={args.n_trials}, "
        f"signal_length={args.signal_length}, base_seed={base_seed}"
    )
    print(f"target_angle={target_angle}° (used only for loc loss)")
    print("-" * 60)

    with torch.no_grad():
        for i in range(args.n_trials):
            trial_seed = base_seed + i
            seeds[i] = trial_seed
            signal = make_stimulus(
                args.stimulus, args.signal_length, device, dtype,
                seed=None if args.stimulus == "impulse" else trial_seed,
            )
            binaural, _ = iko_brir.render_binaural_brir(
                signal, weights, eigenbeam_to_loudspeaker, brirs,
            )
            azimuth, loss = iko_brir.evaluate_localisation(llado_model, binaural, target)
            azimuths[i] = float(azimuth.item())
            loc_losses[i] = float(loss.item())
            print(
                f"trial {i + 1:3d}/{args.n_trials}: "
                f"{'impulse' if args.stimulus == 'impulse' else f'seed={trial_seed}'}, "
                f"azimuth={azimuths[i]:+7.2f}°, loc_loss={loc_losses[i]:.4f}"
            )

    azimuth_std = float(azimuths.std(ddof=1)) if args.n_trials > 1 else 0.0
    loc_loss_std = float(loc_losses.std(ddof=1)) if args.n_trials > 1 else 0.0

    print("-" * 60)
    print(
        f"azimuth: mean={azimuths.mean():+.2f}°, std={azimuth_std:.2f}°, "
        f"min={azimuths.min():+.2f}°, max={azimuths.max():+.2f}°, "
        f"median={np.median(azimuths):+.2f}°"
    )
    print(
        f"loc_loss: mean={loc_losses.mean():.4f}, std={loc_loss_std:.4f}, "
        f"min={loc_losses.min():.4f}, max={loc_losses.max():.4f}"
    )
    if args.stimulus == "impulse":
        print(f"all_identical={bool(np.allclose(azimuths, azimuths[0]))}")

    results = {
        "stimulus": args.stimulus,
        "base_seed": base_seed,
        "n_trials": args.n_trials,
        "signal_length": args.signal_length,
        "target_angle": target_angle,
        "weights": weights.detach().cpu().numpy().tolist(),
        "seeds": seeds.tolist(),
        "azimuths_deg": azimuths.tolist(),
        "loc_losses": loc_losses.tolist(),
        "summary": {
            "azimuth_mean": float(azimuths.mean()),
            "azimuth_std": azimuth_std,
            "azimuth_min": float(azimuths.min()),
            "azimuth_max": float(azimuths.max()),
            "azimuth_median": float(np.median(azimuths)),
            "loc_loss_mean": float(loc_losses.mean()),
            "loc_loss_std": loc_loss_std,
            "all_identical": bool(np.allclose(azimuths, azimuths[0])),
        },
    }
    np.savez(
        output_dir / "init_noise_sweep.npz",
        seeds=seeds,
        azimuths_deg=azimuths,
        loc_losses=loc_losses,
        weights=weights.detach().cpu().numpy(),
    )
    with open(output_dir / "init_noise_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(azimuths, bins=min(20, max(5, args.n_trials // 3)), edgecolor="black", alpha=0.8)
    axes[0].axvline(azimuths.mean(), color="C1", linestyle="--", label=f"mean={azimuths.mean():+.2f}°")
    axes[0].axvline(0.0, color="gray", linestyle=":", label="0°")
    axes[0].set_xlabel("Estimated azimuth (°)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Omni init: azimuth vs {args.stimulus}")
    axes[0].legend()

    axes[1].scatter(np.arange(1, args.n_trials + 1), azimuths, s=18, alpha=0.8)
    axes[1].axhline(azimuths.mean(), color="C1", linestyle="--")
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Estimated azimuth (°)")
    axes[1].set_title("Per-trial azimuth")

    fig.tight_layout()
    plot_path = output_dir / "azimuth_vs_noise.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Results saved to {output_dir}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
