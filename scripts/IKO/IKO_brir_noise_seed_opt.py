"""Optimise IKO BRIR from omni init with different noise stimulus draws; compare final beam patterns."""

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
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.IKO_functions import compute_real_sh_matrix, num_hoa_channels
from utils.miscellaneous import load_npy_to_torch
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from models.listener_based.llado2025.model import llado2025_grad_surgery


def load_iko_brir_optimisation_module():
    module_path = ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py"
    spec = importlib.util.spec_from_file_location("iko_brir_optimisation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare final beam patterns after optimising with different noise draws.",
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of independent noise seeds.")
    parser.add_argument("--n-epochs", type=int, default=100, help="Optimisation epochs per seed.")
    parser.add_argument("--signal-length", type=int, default=256, help="Mono stimulus length.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (trials use seed, seed+1, ...).")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "hyperpameters.json",
    )
    return parser.parse_args()


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_optimisation(iko_brir, signal, shared, n_epochs, lr, target, lambdas):
    device = shared["device"]
    dtype = shared["dtype"]
    n_eigenbeams = shared["n_eigenbeams"]

    weight_model = nn.Linear(1, n_eigenbeams, bias=False, device=device, dtype=dtype)
    weight_model.weight.data.zero_()
    weight_model.weight.data[0, 0] = 1.0
    optimizer = torch.optim.Adam(weight_model.parameters(), lr=lr)

    with torch.no_grad():
        initial_weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
        initial_binaural, _ = iko_brir.render_binaural_brir(
            signal, initial_weights, shared["eigenbeam_to_loudspeaker"], shared["brirs"],
        )
        initial_azimuth, _ = iko_brir.evaluate_localisation(shared["llado_model"], initial_binaural, target)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
        binaural_signal, _ = iko_brir.render_binaural_brir(
            signal, weights, shared["eigenbeam_to_loudspeaker"], shared["brirs"],
        )
        _, loss = iko_brir.evaluate_localisation(shared["llado_model"], binaural_signal, target)
        l1_loss = torch.nn.functional.l1_loss(weights, torch.zeros_like(weights))
        (
            sidelobe_loss,
            listener_directivity_loss,
            reflector_directivity_loss,
            _,
            _,
            _,
        ) = iko_brir.compute_directivity_loss(
            weights,
            shared["sh_matrix_full"],
            shared["listener_mask"],
            shared["reflector_mask"],
            shared["sh_matrix"],
        )
        vertical_loss = iko_brir.compute_vertical_radiation_loss(weights, shared["sh_matrix_vertical"])
        total_loss = (
            lambdas["loc"] * loss
            + lambdas["l1"] * l1_loss
            + lambdas["sidelobe"] * sidelobe_loss
            + lambdas["directivity"] * (listener_directivity_loss + reflector_directivity_loss)
            + lambdas["vertical"] * vertical_loss
        )
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
        final_binaural, _ = iko_brir.render_binaural_brir(
            signal, final_weights, shared["eigenbeam_to_loudspeaker"], shared["brirs"],
        )
        final_azimuth, final_loss = iko_brir.evaluate_localisation(
            shared["llado_model"], final_binaural, target,
        )
        azimuths, pattern = iko_brir.beam_pattern_over_circle(shared["iko_order"], final_weights)

    return {
        "initial_azimuth": float(initial_azimuth.item()),
        "final_azimuth": float(final_azimuth.item()),
        "final_loc_loss": float(final_loss.item()),
        "final_weights": final_weights.detach().cpu().numpy(),
        "pattern_azimuths": azimuths,
        "pattern": pattern,
    }


def main():
    args = parse_args()
    iko_brir = load_iko_brir_optimisation_module()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    device = config["env"]["device"]
    dtype = torch.float32
    lr = float(config["training"].get("lr", 0.01))
    target_angle = float(config["training"]["target_angle"])
    iko_config = config["iko"]
    lambdas = {
        "loc": iko_config["lambda_loc"],
        "l1": iko_config["lambda_l1"],
        "sidelobe": iko_config["lambda_sidelobe"],
        "directivity": iko_config["lambda_directivity"],
        "vertical": iko_config.get("lambda_vertical", 0.0),
    }

    iko_order = 3
    n_eigenbeams = num_hoa_channels(iko_order)
    listener_positions = torch.tensor([[0.0, 0.0]], dtype=dtype)

    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    llado_model.eval()

    brirs = iko_brir.load_iko_brirs(iko_brir.BRIR_NPY, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype)
    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        iko_brir.SPH_TO_DRIVER_NPY, n_eigenbeams, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype,
    )
    geometry = iko_brir.setup_geometry(2.0, 30.0, -30.0, listener_positions, fs)
    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            iko_order,
            iko_brir.physical_azimuths_to_math_sh(geometry["sh_azimuths_rad"].numpy()),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(device=device, dtype=dtype)
    sh_matrix_full, listener_mask, reflector_mask = iko_brir.build_pattern_sampling(
        iko_order,
        float(geometry["sh_azimuths_rad"][0]),
        float(geometry["sh_azimuths_rad"][1]),
        iko_config["lobe_half_width_deg"],
        iko_config["pattern_num_points"],
        device,
        dtype,
    )
    sh_matrix_vertical = iko_brir.build_vertical_sampling(
        iko_order,
        iko_brir.expand_vertical_elevations_deg(
            iko_config.get("vertical_elevation_deg", [30.0, 60.0, 90.0]),
        ),
        iko_config.get("vertical_num_azimuths", iko_config["pattern_num_points"]),
        device,
        dtype,
    )

    shared = {
        "device": device,
        "dtype": dtype,
        "n_eigenbeams": n_eigenbeams,
        "iko_order": iko_order,
        "llado_model": llado_model,
        "brirs": brirs,
        "eigenbeam_to_loudspeaker": eigenbeam_to_loudspeaker,
        "sh_matrix": sh_matrix,
        "sh_matrix_full": sh_matrix_full,
        "listener_mask": listener_mask,
        "reflector_mask": reflector_mask,
        "sh_matrix_vertical": sh_matrix_vertical,
    }
    target = torch.tensor(target_angle, device=device, dtype=dtype)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path("outputs") / "iko_brir_noise_seed_opt" / timestamp
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.n_seeds)]
    results = []
    print(
        f"Optimising {args.n_seeds} noise draws × {args.n_epochs} epochs, "
        f"signal_length={args.signal_length}, target={target_angle}°"
    )
    print("-" * 70)

    for seed in seeds:
        torch.manual_seed(seed)
        signal = torch.randn(1, args.signal_length, device=device, dtype=dtype)
        out = run_optimisation(iko_brir, signal, shared, args.n_epochs, lr, target, lambdas)
        out["seed"] = seed
        results.append(out)
        print(
            f"seed={seed}: init_az={out['initial_azimuth']:+.2f}°, "
            f"final_az={out['final_azimuth']:+.2f}°, "
            f"final_loc_loss={out['final_loc_loss']:.4f}, "
            f"||w||={np.linalg.norm(out['final_weights']):.4f}"
        )

    weight_matrix = np.stack([r["final_weights"] for r in results], axis=0)
    pattern_matrix = np.stack([r["pattern"] for r in results], axis=0)
    cos_w = np.zeros((args.n_seeds, args.n_seeds))
    cos_p = np.zeros((args.n_seeds, args.n_seeds))
    for i in range(args.n_seeds):
        for j in range(args.n_seeds):
            cos_w[i, j] = cosine_similarity(weight_matrix[i], weight_matrix[j])
            cos_p[i, j] = cosine_similarity(pattern_matrix[i], pattern_matrix[j])

    # Exclude diagonal for summary stats
    offdiag = ~np.eye(args.n_seeds, dtype=bool)
    print("-" * 70)
    print(
        f"final weight cosine (off-diag): mean={cos_w[offdiag].mean():.4f}, "
        f"min={cos_w[offdiag].min():.4f}"
    )
    print(
        f"beam pattern cosine (off-diag): mean={cos_p[offdiag].mean():.4f}, "
        f"min={cos_p[offdiag].min():.4f}"
    )
    print(
        f"final azimuth: mean={np.mean([r['final_azimuth'] for r in results]):+.2f}°, "
        f"std={np.std([r['final_azimuth'] for r in results], ddof=1):.2f}°"
    )

    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])
    plot_listener_az = float(iko_brir._beam_pattern_plot_azimuths(listener_az))
    plot_reflector_az = float(iko_brir._beam_pattern_plot_azimuths(reflector_az))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    for r in results:
        plot_az = iko_brir._beam_pattern_plot_azimuths(r["pattern_azimuths"])
        ax.plot(plot_az, np.abs(r["pattern"]), alpha=0.8, label=f"seed {r['seed']}")
    r_lim = float(np.max(np.abs(pattern_matrix))) * 1.1
    ax.plot([plot_listener_az, plot_listener_az], [0, r_lim], color="blue", linestyle="--", linewidth=1.5, label="Listener")
    ax.plot([plot_reflector_az, plot_reflector_az], [0, r_lim], color="gray", linestyle="--", linewidth=1.5, label="Reflector")
    ax.set_title("Final beam patterns across noise seeds")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "beam_pattern_overlay_seeds.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(cos_w, vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("Final weight cosine")
    axes[0].set_xticks(range(args.n_seeds))
    axes[0].set_yticks(range(args.n_seeds))
    axes[0].set_xticklabels(seeds)
    axes[0].set_yticklabels(seeds)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(cos_p, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("Beam pattern cosine")
    axes[1].set_xticks(range(args.n_seeds))
    axes[1].set_yticks(range(args.n_seeds))
    axes[1].set_xticklabels(seeds)
    axes[1].set_yticklabels(seeds)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots_dir / "similarity_matrices.png", dpi=150)
    plt.close(fig)

    np.savez(
        output_dir / "seed_opt_results.npz",
        seeds=np.array(seeds),
        final_weights=weight_matrix,
        patterns=pattern_matrix,
        pattern_azimuths=results[0]["pattern_azimuths"],
        initial_azimuths=np.array([r["initial_azimuth"] for r in results]),
        final_azimuths=np.array([r["final_azimuth"] for r in results]),
        weight_cosine=cos_w,
        pattern_cosine=cos_p,
    )
    summary = {
        "n_seeds": args.n_seeds,
        "n_epochs": args.n_epochs,
        "signal_length": args.signal_length,
        "base_seed": args.seed,
        "target_angle": target_angle,
        "seeds": seeds,
        "initial_azimuths": [r["initial_azimuth"] for r in results],
        "final_azimuths": [r["final_azimuth"] for r in results],
        "final_weights": [r["final_weights"].tolist() for r in results],
        "weight_cosine_offdiag_mean": float(cos_w[offdiag].mean()),
        "weight_cosine_offdiag_min": float(cos_w[offdiag].min()),
        "pattern_cosine_offdiag_mean": float(cos_p[offdiag].mean()),
        "pattern_cosine_offdiag_min": float(cos_p[offdiag].min()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
