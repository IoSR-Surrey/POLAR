import os
import sys
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from propagation.helpers import _calculate_source_angle_2D
from utils.IKO_functions import compute_real_sh_matrix
from utils.miscellaneous import load_npy_to_torch
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020, get_IRs, sofa_find
from models.listener_based.llado2025.model import llado2025_grad_surgery


def setup_geometry(d, iko_angle_deg, reflector_angle_deg, listener_positions, fs, c=343.0):
    """Build scene geometry, SH directions (from IKO), SOFA azimuths (from listener), and reflection delay."""
    iko_position = torch.tensor(
        [[d * torch.cos(torch.deg2rad(torch.tensor(iko_angle_deg))), d * torch.sin(torch.deg2rad(torch.tensor(iko_angle_deg)))]],
        dtype=torch.float32,
    )
    reflector_position = torch.tensor(
        [[d * torch.cos(torch.deg2rad(torch.tensor(reflector_angle_deg))), d * torch.sin(torch.deg2rad(torch.tensor(reflector_angle_deg)))]],
        dtype=torch.float32,
    )

    iko_to_listener = listener_positions - iko_position
    iko_to_reflector = reflector_position - iko_position
    sh_azimuths_rad = torch.cat([
        torch.atan2(iko_to_listener[:, 1], iko_to_listener[:, 0]),
        torch.atan2(iko_to_reflector[:, 1], iko_to_reflector[:, 0]),
    ], dim=0)
    sh_elevations_rad = torch.zeros(2, dtype=torch.float32)

    iko_azimuth_deg = _calculate_source_angle_2D(listener_positions, iko_position)
    reflector_azimuth_deg = _calculate_source_angle_2D(listener_positions, reflector_position)

    direct_dist = torch.norm(iko_position - listener_positions, dim=-1)
    reflect_dist = torch.norm(reflector_position - iko_position, dim=-1) + torch.norm(reflector_position - listener_positions, dim=-1)
    delay_samples = int(torch.round((reflect_dist - direct_dist) / c * fs).item())

    return {
        "iko_position": iko_position,
        "reflector_position": reflector_position,
        "sh_azimuths_rad": sh_azimuths_rad,
        "sh_elevations_rad": sh_elevations_rad,
        "iko_azimuth_deg": iko_azimuth_deg,
        "reflector_azimuth_deg": reflector_azimuth_deg,
        "delay_samples": delay_samples,
    }


def get_binaural_hrir(listener_pos, source_pos, sofaile, device, dtype):
    """HRIR at listener for a source at source_pos: (1, 2, ir_len)."""
    source_azimuth_deg = _calculate_source_angle_2D(listener_pos, source_pos)
    idx = sofa_find(source_azimuth_deg.T, sofaile)
    return get_IRs(idx, sofaile, device=device, dtype=dtype)


def render_binaural_path(mono, hrir, weight, pre_delay_samples, post_delay_samples, device, dtype, mode="full"):
    """Convolve padded mono with HRIR and apply directional weight. Returns (1, 2, T)."""
    mono = mono.view(1, -1)
    if pre_delay_samples > 0:
        mono = torch.cat([torch.zeros(1, pre_delay_samples, device=device, dtype=dtype), mono], dim=1)
    if post_delay_samples > 0:
        mono = torch.cat([mono, torch.zeros(1, post_delay_samples, device=device, dtype=dtype)], dim=1)

    mono = mono.view(1, 1, -1)
    hrir = hrir.reshape(1, 2, -1)
    ear_signals = TF.convolve(mono, hrir, mode=mode).reshape(1, 1, 2, -1).squeeze(1)
    return ear_signals * weight


def render_binaural_sum(signal, iko_hrir, reflector_hrir, weights, sh_matrix, delay_samples, reflector_attenuation, device, dtype):
    """Render direct + reflected paths and sum to (1, 2, T)."""
    directional_weights = sh_matrix @ weights
    iko_weight = directional_weights[0]
    reflector_weight = directional_weights[1]

    # Pad both monos to the same length so convolved outputs align:
    # direct:   [signal, zeros(delay)]  — arrives first
    # reflect:  [zeros(delay), signal]  — arrives delay_samples later
    direct_signal = render_binaural_path(signal, iko_hrir, iko_weight, 0, delay_samples, device, dtype)
    reflector_signal = render_binaural_path(signal, reflector_hrir, reflector_weight * reflector_attenuation, delay_samples, 0, device, dtype)
    return direct_signal + reflector_signal, directional_weights


def evaluate_localisation(llado_model, binaural_signal, target_angle):
    """Forward pass through localisation model; return azimuth and MSE loss."""
    estimated_azimuth = llado_model(binaural_signal, None)["output"]
    loss = torch.mean((estimated_azimuth - target_angle) ** 2)
    return estimated_azimuth, loss


def beam_pattern_over_circle(order, weights, num_points=360):
    """Full horizontal beam pattern for plotting."""
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    elevations = np.zeros(num_points)
    Y = compute_real_sh_matrix(order, azimuths, elevations)
    w = weights.detach().cpu().numpy().reshape(-1)
    return azimuths, Y @ w


def sh_weights_to_nm(order, weights):
    """Map flat HOA weight vector to (n, m, value) using the same ordering as compute_real_sh_matrix."""
    w = weights.detach().cpu().numpy().reshape(-1)
    nm_values = []
    idx = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            nm_values.append((n, m, w[idx]))
            idx += 1
    return nm_values


def plot_weight_decomposition(order, initial_weights, final_weights, plots_dir):
    """Plot HOA weights on an (m, n) grid with the weight value at each valid (n, m) cell."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    m_ticks = list(range(-order, order + 1))
    n_ticks = list(range(order + 1))
    all_weights = np.concatenate([
        initial_weights.detach().cpu().numpy().reshape(-1),
        final_weights.detach().cpu().numpy().reshape(-1),
    ])
    vmax = max(float(np.max(np.abs(all_weights))), 1e-6)

    im = None
    for ax, weights, label in zip(axes, [initial_weights, final_weights], ["Initial", "Optimised"]):
        grid = np.full((order + 1, 2 * order + 1), np.nan)
        for n, m, w in sh_weights_to_nm(order, weights):
            grid[n, m + order] = w

        im = ax.imshow(
            grid, origin="lower", aspect="equal", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            extent=[-order - 0.5, order + 0.5, -0.5, order + 0.5],
        )
        for n, m, w in sh_weights_to_nm(order, weights):
            ax.text(m, n, f"{w:.3f}", ha="center", va="center", fontsize=8, color="black")

        ax.set_xticks(m_ticks)
        ax.set_yticks(n_ticks)
        ax.set_xlabel("m")
        ax.set_ylabel("n (order)")
        ax.set_title(f"{label} HOA weights")
        ax.grid(True, color="white", linewidth=0.5, alpha=0.5)

    fig.subplots_adjust(left=0.06, right=0.86, wspace=0.35)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Weight")
    fig.savefig(plots_dir / "hoa_weights.png", dpi=150)
    plt.close(fig)


def plot_results(output_dir, order, initial_weights, final_weights, geometry, initial_azimuth, final_azimuth, target_angle):
    """Save polar beam-pattern and scene localisation plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 5))
    for ax, weights, label in zip(axes, [initial_weights, final_weights], ["Initial", "Optimised"]):
        azimuths, pattern = beam_pattern_over_circle(order, weights)
        ax.plot(azimuths, pattern, label="Pattern")
        r_lim = max(float(np.max(np.abs(pattern))), 0.1) * 1.1
        ax.plot([listener_az, listener_az], [0, r_lim], color="blue", linestyle="--", linewidth=1.5, label="Listener")
        ax.plot([reflector_az, reflector_az], [0, r_lim], color="gray", linestyle="--", linewidth=1.5, label="Reflector")
        ax.set_title(f"{label} beam pattern")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "beam_pattern.png", dpi=150)
    plt.close(fig)

    listener_pos = geometry["listener_positions"].numpy()
    iko_pos = geometry["iko_position"].numpy()
    reflector_pos = geometry["reflector_position"].numpy()
    arrow_len = 1.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, azimuth, label in zip(axes, [initial_azimuth, final_azimuth], ["Initial", "Optimised"]):
        ax.scatter(listener_pos[:, 0], listener_pos[:, 1], c="blue", marker="o", label="Listener", zorder=3)
        ax.scatter(iko_pos[:, 0], iko_pos[:, 1], c="black", marker="x", label="IKO", zorder=3)
        ax.scatter(reflector_pos[:, 0], reflector_pos[:, 1], c="gray", marker="s", label="Reflector", zorder=3)
        az_rad = np.deg2rad(float(azimuth))
        ax.arrow(
            listener_pos[0, 0], listener_pos[0, 1],
            arrow_len * np.cos(az_rad), arrow_len * np.sin(az_rad),
            head_width=0.15, head_length=0.1, fc="red", ec="red", label="Estimated",
        )
        target_rad = np.deg2rad(target_angle)
        ax.arrow(
            listener_pos[0, 0], listener_pos[0, 1],
            arrow_len * np.cos(target_rad), arrow_len * np.sin(target_rad),
            head_width=0.15, head_length=0.1, fc="green", ec="green", linestyle="--", label="Target",
        )
        ax.set_aspect("equal")
        ax.set_title(f"{label} localisation ({float(azimuth):.1f}°)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "localisation_scene.png", dpi=150)
    plt.close(fig)

    plot_weight_decomposition(order, initial_weights, final_weights, plots_dir)


def main():
    with open("config/hyperpameters.json", "r") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    device = config["env"]["device"]
    dtype = torch.float32
    n_epochs = config["training"]["num_epochs"]
    lr = 0.01
    target_angle = config["training"]["target_angle"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path("outputs") / "iko" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)

    d = 2.0
    iko_order = 3
    listener_positions = torch.tensor([[0.0, 0.0]], dtype=dtype)
    reflector_attenuation = 0.8

    geometry = setup_geometry(d, 30.0, -30.0, listener_positions, fs)
    geometry["listener_positions"] = listener_positions

    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            iko_order,
            geometry["sh_azimuths_rad"].numpy(),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(device=device, dtype=dtype)

    iko_hrir = get_binaural_hrir(listener_positions, geometry["iko_position"], sofaile, device, dtype)
    reflector_hrir = get_binaural_hrir(listener_positions, geometry["reflector_position"], sofaile, device, dtype)

    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    llado_model.eval()

    weight_model = nn.Linear(1, (iko_order + 1) ** 2, bias=False, device=device, dtype=dtype)
    weight_model.weight.data.zero_()
    weight_model.weight.data[0, 0] = 1.0  # omnidirectional (0th-order) SH only
    optimizer = torch.optim.Adam(weight_model.parameters(), lr=lr)

    signal = torch.randn(1, 256, device=device, dtype=dtype)
    target = torch.tensor(target_angle, device=device, dtype=dtype)

    lambda_l1 = 1.5

    with torch.no_grad():
        initial_weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
        initial_binaural, _ = render_binaural_sum(
            signal, iko_hrir, reflector_hrir, initial_weights, sh_matrix,
            geometry["delay_samples"], reflector_attenuation, device, dtype,
        )
        initial_azimuth, initial_loss = evaluate_localisation(llado_model, initial_binaural, target)
        initial_l1_loss = torch.nn.functional.l1_loss(initial_weights, torch.zeros_like(initial_weights))

    print(f"Initial azimuth: {initial_azimuth.item():.2f}°, loc loss: {initial_loss.item():.4f}, L1 loss: {initial_l1_loss.item():.4f}")

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
        binaural_signal, _ = render_binaural_sum(
            signal, iko_hrir, reflector_hrir, weights, sh_matrix,
            geometry["delay_samples"], reflector_attenuation, device, dtype,
        )
        _, loss = evaluate_localisation(llado_model, binaural_signal, target)

        l1_loss = torch.nn.functional.l1_loss(weights, torch.zeros_like(weights))
        total_loss = loss + lambda_l1 * l1_loss
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                azimuth, _ = evaluate_localisation(llado_model, binaural_signal, target)
            print(
                f"Epoch {epoch + 1}/{n_epochs}, loc loss: {loss.item():.4f}, "
                f"L1 loss: {l1_loss.item():.4f}, total loss: {total_loss.item():.4f}, "
                f"azimuth: {azimuth.item():.2f}°"
            )

    with torch.no_grad():
        final_weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
        final_binaural, _ = render_binaural_sum(
            signal, iko_hrir, reflector_hrir, final_weights, sh_matrix,
            geometry["delay_samples"], reflector_attenuation, device, dtype,
        )
        final_azimuth, final_loss = evaluate_localisation(llado_model, final_binaural, target)
        final_l1_loss = torch.nn.functional.l1_loss(final_weights, torch.zeros_like(final_weights))

    print(
        f"Final azimuth: {final_azimuth.item():.2f}°, loc loss: {final_loss.item():.4f}, "
        f"L1 loss: {final_l1_loss.item():.4f} (target: {target_angle}°)"
    )

    plot_results(
        output_dir, iko_order, initial_weights, final_weights, geometry,
        initial_azimuth.item(), final_azimuth.item(), target_angle,
    )
    print(f"Plots saved to {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
