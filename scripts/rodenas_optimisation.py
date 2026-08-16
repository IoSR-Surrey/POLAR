import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from utils.seed import set_seed
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import (load_npy_to_torch,
                                target_angle_for_listeners,
                                create_mtl_optimizer,
                                run_mtl_optimizer_step,
                                git_commit_info,
                                write_run_config)
from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.LSD.model import LSD
from models.loudspeaker_based.second_moment.model import SecondMoment
from models.train import ModelTrainer
from logger.saving import SavingManager
from logger.printing import print_loss_and_estimates
from plotting.XY_plots import (
    time_domain_filters_plot,
    frequency_domain_filters_plot,
    loss_over_epochs_plot,
    plot_loss_and_estimates,
    loss_grad_norms_over_epochs_plot,
    cagrad_dual_weights_over_epochs_plot,
    cagrad_task_coeffs_over_epochs_plot,
    raw_grad_cosine_to_reference_plot,
)
from plotting.spatial_plots import localisation_arrows_plot
from plotting.batch_utils import (
    average_estimates_dicts,
    average_interaural_cues_dicts,
    average_loss_dicts,
    average_mtl_saved_values,
    get_white_noise_batch,
)
from datasets.stimulus_dataset import get_stim_loader


def plot_rodenas_mtl_diagnostics(saving_manager):
    """Loss curves and CAGrad cosine / weight diagnostics (same as scratch CAGrad branch)."""
    if "loss_dict" in saving_manager.saved_data:
        loss_over_epochs_plot(saving_manager.saved_data["loss_dict"], "Loss over epochs")
        saving_manager.save_new_plots("loss_over_epochs")

    if "loss_dict" in saving_manager.saved_data and "modelTrainer.estimates" in saving_manager.saved_data:
        n_listeners = None
        if "lstn_pos" in saving_manager.saved_data:
            lstn = saving_manager.saved_data["lstn_pos"][0]
            n_listeners = int(np.asarray(lstn.detach().cpu() if torch.is_tensor(lstn) else lstn).shape[0])
        plot_loss_and_estimates(
            saving_manager.saved_data["loss_dict"],
            saving_manager.saved_data["modelTrainer.estimates"],
            "Loss and Estimates",
            n_listeners=n_listeners,
        )
        loss_names = [
            f"loss_and_estimates_{key}"
            for key in saving_manager.saved_data["loss_dict"][0].keys()
            if key != "total"
        ]
        if "total" in saving_manager.saved_data["loss_dict"][0]:
            loss_names.append("loss_and_estimates_total")
        saving_manager.save_new_plots("loss_and_estimates", names=loss_names)

    if "raw_grads" not in saving_manager.saved_data or "effective_grad" not in saving_manager.saved_data:
        return

    task_names = list(saving_manager.saved_data["raw_grads"][0].keys())
    raw_grad_norms = [
        {key: torch.norm(epoch_grads[key]) for key in epoch_grads}
        for epoch_grads in saving_manager.saved_data["raw_grads"]
    ]
    loss_grad_norms_over_epochs_plot(raw_grad_norms, "Per-task raw gradient norm", norm_kind="Raw")
    saving_manager.save_new_plots("raw_grad_norms")

    if "cagrad_dual_weights" in saving_manager.saved_data:
        cagrad_dual_weights_over_epochs_plot(
            saving_manager.saved_data["cagrad_dual_weights"], task_names,
            "CAGrad task mix (weights sum to 1)",
        )
        saving_manager.save_new_plots("cagrad_dual_weights")

    if "cagrad_task_coeffs" in saving_manager.saved_data:
        cagrad_task_coeffs_over_epochs_plot(
            saving_manager.saved_data["cagrad_task_coeffs"], task_names,
            "CAGrad gradient multiplier per task",
        )
        saving_manager.save_new_plots("cagrad_task_coeffs")

    summed_raw_grad = [
        torch.stack([epoch_grads[key] for key in epoch_grads]).sum(dim=0)
        for epoch_grads in saving_manager.saved_data["raw_grads"]
    ]
    raw_grad_cosine_to_reference_plot(
        saving_manager.saved_data["raw_grads"], summed_raw_grad, task_names,
        "Task gradient vs plain (summed) direction",
    )
    saving_manager.save_new_plots("raw_vs_plain_cosine_similarity")
    raw_grad_cosine_to_reference_plot(
        saving_manager.saved_data["raw_grads"], saving_manager.saved_data["effective_grad"], task_names,
        "Task gradient vs CAGrad update direction",
    )
    saving_manager.save_new_plots("raw_vs_cagrad_cosine_similarity")


def build_array_positions(array_centre, offsets, invert_offsets=False):
    signed_offsets = -offsets if invert_offsets else offsets
    return [(array_centre[0], array_centre[1] + off) for off in signed_offsets]


def build_listener_position_weights(lstn_pos, y_pos_lo, y_pos_hi, y_neg_lo, y_neg_hi, w_high, w_low):
    """
    Weight each listener by its global y coordinate only: high in [y_pos_lo, y_pos_hi] and
    [y_neg_lo, y_neg_hi], low elsewhere (e.g. centre strip and far lateral positions).
    A small tolerance keeps linspace endpoints like +/-0.20 m symmetric about the origin.
    """
    eps = 1e-8
    listener_weights = []
    for listener_pos in lstn_pos:
        y = float(listener_pos[1])
        in_pos_band = (y_pos_lo - eps) <= y <= (y_pos_hi + eps)
        in_neg_band = (y_neg_lo - eps) <= y <= (y_neg_hi + eps)
        listener_weights.append(w_high if (in_pos_band or in_neg_band) else w_low)
    return np.asarray(listener_weights, dtype=np.float32)


def make_listener_weighted_loss(listener_weights, device, dtype, pointwise="squared"):
    """
    Per-listener error reduced over the listener dimension, weighted by per-listener weights.
    Estimates and targets are shaped (N_signals, N_listeners); the weighted mean matches the
    old ListenerBasedLoss. ``pointwise`` selects the element-wise error: "squared" (MSE, used
    for localisation) or "abs" (L1, used for colouration).
    """
    weights = torch.as_tensor(listener_weights, device=device, dtype=dtype).view(1, -1)
    weight_sum = weights.sum().clamp(min=1e-12)

    def criterion(estimates, targets):
        if pointwise == "abs":
            error = (estimates - targets).abs()
        else:
            error = (estimates - targets) ** 2
        # Mean over signals, weighted mean over listeners (weights broadcast on dim 0).
        return torch.sum(error * weights, dim=-1).mean() / weight_sum

    return criterion


def localisation_error_metrics(target_angles, loc_before, loc_after, listener_weights):
    """
    Per-listener localisation error (estimated minus target azimuth, in degrees) before and
    after optimisation, reported both unweighted and weighted by the listener-position weights.
    """
    target = np.asarray(target_angles, dtype=float).ravel()
    before = np.asarray(loc_before, dtype=float).ravel()
    after = np.asarray(loc_after, dtype=float).ravel()
    weights = np.asarray(listener_weights, dtype=float).ravel()

    err_before = before - target
    err_after = after - target
    sum_w = float(np.sum(weights))

    def _wmean(values):
        return float(np.sum(weights * values) / sum_w) if sum_w > 0.0 else float(np.mean(values))

    return {
        "target_angle_deg": target.tolist(),
        "loc_before_deg": before.tolist(),
        "loc_after_deg": after.tolist(),
        "mae_before_deg": float(np.mean(np.abs(err_before))),
        "mae_after_deg": float(np.mean(np.abs(err_after))),
        "rmse_before_deg": float(np.sqrt(np.mean(err_before ** 2))),
        "rmse_after_deg": float(np.sqrt(np.mean(err_after ** 2))),
        "weighted_mae_before_deg": _wmean(np.abs(err_before)),
        "weighted_mae_after_deg": _wmean(np.abs(err_after)),
        "weighted_rmse_before_deg": float(np.sqrt(_wmean(err_before ** 2))),
        "weighted_rmse_after_deg": float(np.sqrt(_wmean(err_after ** 2))),
    }


def compute_group_directivity_db(
    filters, spk_pos_np, group_indices, fs, freqs_eval_hz, radius, n_angles=360, c=343.0
):
    """
    Frequency-averaged far-field directivity (in dB) of a loudspeaker group as a function of
    observation angle around the group centroid. Mirrors ``plot_arrows_with_directivity`` with
    ``directivity_mode='avg'`` so the returned curve matches the plotted blue curve.
    """
    group_indices = np.asarray(group_indices, dtype=int).reshape(-1)
    h_group = np.atleast_2d(np.asarray(filters, dtype=float))[group_indices]
    p_group = np.asarray(spk_pos_np, dtype=float)[group_indices]
    group_mid = p_group.mean(axis=0)

    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    n_taps = h_group.shape[1]
    n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omega = 2 * np.pi * freqs
    H_group = np.fft.rfft(h_group, n=n_fft, axis=1)
    freq_bin_idx = [int(np.argmin(np.abs(freqs - f))) for f in freqs_eval_hz]

    mag2 = np.zeros((len(freqs_eval_hz), n_angles))
    for k, ang in enumerate(theta):
        obs = group_mid + np.array([radius * np.cos(ang), radius * np.sin(ang)])
        tau = np.linalg.norm(obs - p_group, axis=1) / c
        phase = np.exp(-1j * np.outer(tau, omega))
        H_sum = np.sum(H_group * phase, axis=0)
        for fi, bi in enumerate(freq_bin_idx):
            mag2[fi, k] = np.abs(H_sum[bi]) ** 2

    dir_avg_db = 10.0 * np.log10(np.mean(mag2, axis=0) + 1e-12)
    return theta, dir_avg_db, group_mid


def _db_at_nearest_angle_deg(sample_angles_deg, levels_db, target_angle_deg):
    sample_angles_deg = np.asarray(sample_angles_deg, dtype=np.float64).ravel()
    levels_db = np.asarray(levels_db, dtype=np.float64).ravel()
    target_wrapped = float(np.mod(target_angle_deg, 360.0))
    samples_wrapped = np.mod(sample_angles_deg, 360.0)
    separation_deg = np.abs((samples_wrapped - target_wrapped + 180.0) % 360.0 - 180.0)
    return float(levels_db[int(np.argmin(separation_deg))])


def directivity_match_metrics(theta, dir_avg_db, group_mid, overlay_curves, overlay_labels=None):
    """
    Compare the estimated group directivity against each Rodenas overlay curve after
    normalisation. Both curves are aligned to 0 dB at the array steering direction (the wedge
    mid-angle) exactly as in the figure, then the overlay is interpolated onto the estimated
    angle grid and dB error statistics are computed over the overlapping angular range.
    """
    cy = float(group_mid[1])
    if cy >= 0.0:
        theta_wedge0 = np.pi
        theta_wedge1 = np.pi + np.pi / 3.0
    else:
        theta_wedge0 = np.pi - np.pi / 3.0
        theta_wedge1 = np.pi
    mid_rad = 0.5 * (theta_wedge0 + theta_wedge1)
    mid_angle_deg = float(np.rad2deg(mid_rad) % 360.0)

    delta_theta = np.abs((theta - mid_rad + np.pi) % (2 * np.pi) - np.pi)
    k_mid = int(np.argmin(delta_theta))
    dir_est_aligned = dir_avg_db - dir_avg_db[k_mid]

    theta_deg = np.rad2deg(theta)
    wedge0_deg = float(np.rad2deg(theta_wedge0))
    wedge1_deg = float(np.rad2deg(theta_wedge1))
    theta_wrapped = np.mod(theta, 2 * np.pi)
    mask = (theta_wrapped >= theta_wedge0) & (theta_wrapped <= theta_wedge1)
    theta_sel_deg = np.mod(theta_deg[mask], 360.0)
    est_sel = dir_est_aligned[mask]

    overlay_curves = [np.asarray(curve, dtype=float) for curve in overlay_curves]
    overlay_angle_offset_deg = 180.0 - float(np.min(np.mod(overlay_curves[0][:, 0], 360.0)))

    metrics = []
    for ci, curve in enumerate(overlay_curves):
        label = overlay_labels[ci] if (overlay_labels and ci < len(overlay_labels)) else f"overlay_{ci}"

        angles_deg_plot = np.mod(curve[:, 0] + overlay_angle_offset_deg, 360.0)
        db_sel = np.flip(curve[:, 1])
        if cy < 0.0:
            angles_deg_plot = np.mod(360.0 - angles_deg_plot, 360.0)
        db_at_mid = _db_at_nearest_angle_deg(angles_deg_plot, db_sel, mid_angle_deg)
        db_aligned = db_sel - db_at_mid

        order = np.argsort(angles_deg_plot)
        ov_ang = angles_deg_plot[order]
        ov_db = db_aligned[order]

        lo = max(wedge0_deg, float(ov_ang.min()))
        hi = min(wedge1_deg, float(ov_ang.max()))
        in_range = (theta_sel_deg >= lo) & (theta_sel_deg <= hi)
        if not np.any(in_range):
            metrics.append({"label": label, "rmse_db": float("nan"), "mae_db": float("nan"),
                            "max_abs_db": float("nan"), "n_points": 0, "angle_range_deg": [lo, hi]})
            continue

        ov_interp = np.interp(theta_sel_deg[in_range], ov_ang, ov_db)
        diff = est_sel[in_range] - ov_interp
        metrics.append({
            "label": label,
            "rmse_db": float(np.sqrt(np.mean(diff ** 2))),
            "mae_db": float(np.mean(np.abs(diff))),
            "max_abs_db": float(np.max(np.abs(diff))),
            "n_points": int(np.sum(in_range)),
            "angle_range_deg": [lo, hi],
        })
    return metrics


def _to_numpy_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(float, copy=False)
    return np.asarray(x, dtype=float)


def plot_pair_directivity_polar(
    est_filters,
    spk_pos,
    fs,
    group_indices,
    radius=None,
    n_angles=720,
    freqs_eval_hz=(250, 500, 1000, 2000, 4000),
    c=343.0,
    overlay_curve=None,
    overlay_curves=None,
    overlay_labels=None,
    show=False,
    db_floor=None,
    overlay_avg_directivity=True,
):
    """
    Polar plot of the frequency-dependent directivity for a selected loudspeaker group.

    For each azimuth on a circle, this sums the geometric-delay-compensated FIRs and plots
    10*log10(|H|^2) in dB (shared scale, no per-curve peak normalisation). An optional dashed
    curve overlays the log of the linear average of the per-band powers, and any provided
    ``overlay_curves`` (shape (N, 2) as [angle_deg, level_db]) are scattered for reference.
    """
    filters = np.atleast_2d(np.asarray(est_filters, dtype=float))
    spk_pos = _to_numpy_float(spk_pos)

    group_indices = np.asarray(group_indices, dtype=int).reshape(-1)
    if group_indices.size < 1:
        raise ValueError("group_indices must contain at least one loudspeaker index.")

    group_mid = np.mean(spk_pos[group_indices], axis=0)

    if radius is None:
        radius = float(np.mean(np.linalg.norm(spk_pos[group_indices], axis=1)))

    n_taps = filters.shape[1]
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    freqs_eval_hz = [float(f) for f in freqs_eval_hz]
    mag2_by_freq = np.zeros((len(freqs_eval_hz), len(theta)), dtype=float)

    h_group = filters[group_indices]
    p_group = spk_pos[group_indices]

    n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omega = 2.0 * np.pi * freqs
    H_group = np.fft.rfft(h_group, n=n_fft, axis=1)
    freq_bin_idx = [int(np.argmin(np.abs(freqs - f_hz))) for f_hz in freqs_eval_hz]
    eps = 1e-12

    for k, ang in enumerate(theta):
        obs = group_mid + np.array([radius * np.cos(ang), radius * np.sin(ang)], dtype=float)
        tau_group = np.linalg.norm(obs - p_group, axis=1) / c
        phase_group = np.exp(-1j * np.outer(tau_group, omega))
        H_sum = np.sum(H_group * phase_group, axis=0)
        for fi, bi in enumerate(freq_bin_idx):
            mag2_by_freq[fi, k] = np.abs(H_sum[bi]) ** 2

    log_mean_mag2 = 10.0 * np.log10(np.mean(mag2_by_freq, axis=0) + eps)

    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(111, projection="polar")
    min_db = float("inf")
    max_db = float("-inf")
    for fi, f_hz in enumerate(freqs_eval_hz):
        dir_db = 10.0 * np.log10(mag2_by_freq[fi] + eps)
        min_db = min(min_db, float(np.min(dir_db)))
        max_db = max(max_db, float(np.max(dir_db)))
        ax.plot(theta, dir_db, linewidth=1.6, label=f"{int(round(f_hz))} Hz")

    if overlay_avg_directivity:
        min_db = min(min_db, float(np.min(log_mean_mag2)))
        max_db = max(max_db, float(np.max(log_mean_mag2)))
        ax.plot(
            theta,
            log_mean_mag2,
            linewidth=2.0,
            linestyle="--",
            color="tab:red",
            label=r"Average over eval. freqs",
            zorder=6,
        )

    overlay_curve_list = []
    if overlay_curve is not None:
        overlay_curve_list.append(overlay_curve)
    if overlay_curves is not None:
        overlay_curve_list.extend(overlay_curves)
    if overlay_labels is not None and len(overlay_labels) != len(overlay_curve_list):
        raise ValueError("overlay_labels must contain one label per overlay curve.")

    overlay_colors = ["k", "tab:purple", "tab:brown", "tab:gray", "tab:olive"]
    for curve_idx, curve in enumerate(overlay_curve_list):
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError("Each overlay curve must have shape (N, 2) as [angle_deg, level_db].")
        overlay_theta = np.deg2rad(np.mod(curve[:, 0], 360.0))
        overlay_db = curve[:, 1]
        min_db = min(min_db, float(np.min(overlay_db)))
        max_db = max(max_db, float(np.max(overlay_db)))
        ax.scatter(
            overlay_theta,
            overlay_db,
            color=overlay_colors[curve_idx % len(overlay_colors)],
            s=20,
            marker="o",
            label=overlay_labels[curve_idx] if overlay_labels is not None else ("Data overlay" if curve_idx == 0 else f"Data overlay {curve_idx + 1}"),
            zorder=5,
        )

    span = max(max_db - min_db, 1e-9)
    pad = 0.05 * span
    y_floor = max(-40.0, min_db - pad) if db_floor is None else float(db_floor)
    y_ceiling = max_db + pad
    ax.set_ylim(y_floor, y_ceiling)
    idx_str = ", ".join(str(int(i)) for i in group_indices.tolist())
    ax.set_title(f"Group directivity (sources {idx_str}), r={radius:.1f} m")
    fig.subplots_adjust(right=0.72)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    ax.grid(True)

    if show:
        plt.show()

    return fig


def plot_arrows_with_directivity(
    est_filters,
    spk_pos,
    lstn_pos,
    loc,
    fs,
    group_idx,
    unc=None,
    col=None,
    target_pos=None,
    show_listener_target_lines=False,
    vbap_values=None,
    show=True,
    n_angles=360,
    freqs_eval_hz=(250, 500, 1000, 2000, 4000),
    directivity_observation_radius_m=None,
    c=343.0,
    directivity_mode="avg",
    directivity_freq_idx=2,
    directivity_scale=1.0,
    directivity_floor_db=-20,
    directivity_label="Proposed Method",
    overlay_curve=None,
    overlay_curves=None,
    overlay_labels=None,
    legend_ncol=None,
    legend_bbox_to_anchor=(0.5, 1.02),
    figsize=(6, 6),
):
    use_lsd_colorbar = col is not None
    if unc is None:
        unc = torch.zeros_like(loc)
    if col is None:
        col = torch.zeros_like(loc)

    lstn_pos = lstn_pos.detach().cpu().numpy().squeeze().reshape(-1, 2)
    spk_pos = spk_pos.detach().cpu().numpy().squeeze()
    loc = loc.detach().cpu().numpy().squeeze().reshape(-1)
    unc = unc.detach().cpu().numpy().squeeze().reshape(-1)
    col = col.detach().cpu().numpy().squeeze().reshape(-1)

    overlay_curve_list = []
    if overlay_curve is not None:
        overlay_curve_list.append(overlay_curve)
    if overlay_curves is not None:
        overlay_curve_list.extend(overlay_curves)

    if overlay_labels is not None and len(overlay_labels) != len(overlay_curve_list):
        raise ValueError("overlay_labels must contain one label per overlay curve.")

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x', label='Loudspeakers')
    ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', label='Virtual Listeners')

    target_pos_np = None
    if target_pos is not None:
        target_pos_np = _to_numpy_float(target_pos).reshape(-1)
        if target_pos_np.size != 2:
            raise ValueError("target_pos must be a 2D position.")
        ax.scatter(
            target_pos_np[0],
            target_pos_np[1],
            marker="^",
            c="red",
            s=60,
            label="Target",
            zorder=6,
        )

    arrow_len = 0.2
    cmap = cm.viridis
    norm = colors.Normalize(vmin=0, vmax=18.5)
    arrow_mono_color = "#2E0854"

    for x, y, ang, leng, col_val in zip(lstn_pos[:, 0], lstn_pos[:, 1], loc, unc, col):
        dx = (1 - leng) * arrow_len * np.cos(np.deg2rad(ang))
        dy = (1 - leng) * arrow_len * np.sin(np.deg2rad(ang))
        if use_lsd_colorbar:
            color = cmap(norm(max(col_val, 1e-3)))
        else:
            color = arrow_mono_color

        ax.arrow(x, y, dx, dy,
                 head_width=0.05, head_length=0.08,
                 fc=color, ec=color)

    if use_lsd_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label="LSD")

    if len(overlay_curve_list) > 0:
        curve_ref_global = np.asarray(overlay_curve_list[0])
        overlay_angle_offset_deg = 180.0 - float(np.min(np.mod(curve_ref_global[:, 0], 360.0)))
    else:
        overlay_angle_offset_deg = 0.0

    for i, g in enumerate(group_idx):

        group_indices = np.asarray(g, dtype=int).reshape(-1)

        filters = np.atleast_2d(_to_numpy_float(est_filters))
        h_group = filters[group_indices]
        p_group = spk_pos[group_indices]

        group_mid = np.mean(p_group, axis=0)
        cy = float(group_mid[1])
        if cy > 0.0:
            theta_wedge0 = np.pi
            theta_wedge1 = np.pi + np.pi / 3.0
        elif cy < 0.0:
            theta_wedge0 = np.pi - np.pi / 3.0
            theta_wedge1 = np.pi
        else:
            theta_wedge0 = np.pi
            theta_wedge1 = np.pi + np.pi / 3.0

        theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        n_taps = h_group.shape[1]
        n_fft = max(4096, int(2 ** np.ceil(np.log2(max(64, n_taps)))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
        omega = 2*np.pi*freqs
        H_group = np.fft.rfft(h_group, n=n_fft, axis=1)

        freq_bin_idx = [np.argmin(np.abs(freqs - f)) for f in freqs_eval_hz]

        mag2 = np.zeros((len(freqs_eval_hz), len(theta)))

        if directivity_observation_radius_m is None:
            radius = float(np.mean(np.linalg.norm(p_group - group_mid, axis=1)))
        else:
            radius = float(directivity_observation_radius_m)

        for k, ang in enumerate(theta):
            obs = group_mid + np.array([radius*np.cos(ang), radius*np.sin(ang)])
            tau = np.linalg.norm(obs - p_group, axis=1) / c
            phase = np.exp(-1j * np.outer(tau, omega))
            H_sum = np.sum(H_group * phase, axis=0)

            for fi, bi in enumerate(freq_bin_idx):
                mag2[fi, k] = np.abs(H_sum[bi])**2

        dir_db = 10*np.log10(mag2 + 1e-12)

        if directivity_mode == "avg":
            dir_use = 10*np.log10(np.mean(mag2, axis=0) + 1e-12)
        else:
            dir_use = dir_db[directivity_freq_idx]

        theta_wrapped = np.mod(theta, 2 * np.pi)
        mask = (theta_wrapped >= theta_wedge0) & (theta_wrapped <= theta_wedge1)
        mid_rad = 0.5 * (theta_wedge0 + theta_wedge1)

        delta_theta = np.abs((theta - mid_rad + np.pi) % (2 * np.pi) - np.pi)
        k_mid = int(np.argmin(delta_theta))
        dir_at_mid = float(dir_use[k_mid])

        dir_aligned = dir_use - dir_at_mid
        dir_clipped = np.maximum(dir_aligned, directivity_floor_db)
        r = directivity_scale * (dir_clipped - directivity_floor_db) / (0 - directivity_floor_db)

        theta_sel = theta[mask]
        r_sel = r[mask]

        cx, cy = group_mid

        x_dir = cx + r_sel*np.cos(theta_sel)
        y_dir = cy + r_sel*np.sin(theta_sel)

        ax.plot(x_dir, y_dir, color="tab:blue", linewidth=2,
                label=directivity_label if i == 0 else None, zorder=50)

        line_extent = 0.0
        for db_ref in (-10.0, 0.0, 10.0):
            line_extent = max(
                line_extent,
                directivity_scale * (db_ref - directivity_floor_db) / (0.0 - directivity_floor_db),
            )
        th_oblique_ray = theta_wedge1 if cy >= 0.0 else theta_wedge0
        n_arc = 72
        th_arc = np.linspace(theta_wedge0, theta_wedge1, n_arc)
        theta_label = 0.5 * (theta_wedge0 + theta_wedge1)
        cos_tl, sin_tl = np.cos(theta_label), np.sin(theta_label)
        label_radial_nudge = 0.04 * line_extent

        ref_line_kw = dict(color="0.45", linewidth=0.9, linestyle="-", zorder=35, alpha=0.75)
        ref_arc_kw = dict(color="0.45", linewidth=0.8, linestyle="--", zorder=34, alpha=0.75)

        ax.plot(
            [cx, cx + line_extent * np.cos(np.pi)],
            [cy, cy + line_extent * np.sin(np.pi)],
            **ref_line_kw,
        )
        ax.plot(
            [cx, cx + line_extent * np.cos(th_oblique_ray)],
            [cy, cy + line_extent * np.sin(th_oblique_ray)],
            **ref_line_kw,
        )
        for db_ref in (-10.0, 0.0, 10.0):
            circle_r = directivity_scale * (db_ref - directivity_floor_db) / (0.0 - directivity_floor_db)
            if circle_r > 0.0:
                ax.plot(
                    cx + circle_r * np.cos(th_arc),
                    cy + circle_r * np.sin(th_arc),
                    **ref_arc_kw,
                )
                if db_ref > 0.0:
                    db_lbl = f"+{int(round(db_ref))} dB"
                elif db_ref < 0.0:
                    db_lbl = f"{int(round(db_ref))} dB"
                else:
                    db_lbl = "0 dB"
                r_lbl = circle_r + label_radial_nudge
                ax.text(
                    cx + r_lbl * cos_tl,
                    cy + r_lbl * sin_tl,
                    db_lbl,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.35",
                    zorder=36,
                )

        mid_angle_deg = float(np.rad2deg(mid_rad) % 360.0)

        overlay_colors = ["k", "tab:purple", "tab:brown", "tab:gray"]

        for curve_idx, curve in enumerate(overlay_curve_list):
            curve = np.asarray(curve, dtype=float)

            angles_deg_plot = np.mod(curve[:, 0] + overlay_angle_offset_deg, 360.0)
            db_sel = np.flip(curve[:, 1])
            if cy < 0.0:
                angles_deg_plot = np.mod(360.0 - angles_deg_plot, 360.0)
            theta_overlay = np.deg2rad(angles_deg_plot)

            db_at_mid = _db_at_nearest_angle_deg(angles_deg_plot, db_sel, mid_angle_deg)
            db_aligned = db_sel - db_at_mid
            db_clipped = np.maximum(db_aligned, directivity_floor_db)

            r_overlay = directivity_scale * (db_clipped - directivity_floor_db) / (0 - directivity_floor_db)

            cx, cy = group_mid
            x_overlay = cx + r_overlay * np.cos(theta_overlay)
            y_overlay = cy + r_overlay * np.sin(theta_overlay)
            order = np.argsort(angles_deg_plot)
            oc = overlay_colors[curve_idx % len(overlay_colors)]
            ax.plot(
                x_overlay[order],
                y_overlay[order],
                linestyle="-",
                linewidth=1.4,
                color=oc,
                label=overlay_labels[curve_idx] if (overlay_labels and i == 0) else None,
                zorder=5,
            )

    if show_listener_target_lines and target_pos_np is not None:
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()
        tx, ty = float(target_pos_np[0]), float(target_pos_np[1])
        for x, y in zip(lstn_pos[:, 0], lstn_pos[:, 1]):
            p_lstn = np.array([float(x), float(y)], dtype=float)
            p_tgt = np.array([tx, ty], dtype=float)
            d = p_tgt - p_lstn
            dist = float(np.linalg.norm(d))
            if dist < 1e-12:
                continue
            u = d / dist
            extend = 0.2 * dist
            p0 = p_lstn - extend * u
            p1 = p_tgt + extend * u
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linestyle="--",
                linewidth=0.8,
                color="0.65",
                alpha=0.6,
                zorder=1,
            )
        ax.set_xlim(xlim_before)
        ax.set_ylim(ylim_before)

    ax.set_aspect('equal')
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if legend_ncol is None:
        legend_ncol = 1 if len(unique) <= 2 else 2
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(pad=0.35, rect=[0.02, 0.02, 0.98, 0.91])

    if show:
        plt.show()

    return fig


def main(model_weights=None):
    with open('config/hyperpameters.json', 'r') as f:
        config = json.load(f)
    set_seed(config["env"]["seed"])
    # 51-listener Rodenas geometry is too large for an 8GB GPU with Llado/CAGrad;
    # keep training on CPU like the original script.
    device = torch.device("cpu")
    dtype = getattr(torch, config["env"]["dtype"])
    mtl_optimizer = "cagrad"  # config["training"]["mtl_optimizer"]
    skip_keys = config["saving"]["skip_keys"]
    use_tensorboard = False  # heavy run (51 listeners × many bandpass batches)
    fs = config["env"]["fs"]
    sofa_path = "data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
    gtfb_path = "data/banks/gtfb_48k.npy"
    apply_distance_attenuation = config["env"]["apply_distance_attenuation"]
    if model_weights is None:
        model_weights = [1.0, 1.0]  # loc (llado), colouration (LSD)
    else:
        model_weights = [float(model_weights[0]), float(model_weights[1])]
    filter_initialisation = {"init_weight": "middle", "fs": fs, "n_bands": 4, "band_mode": "log"}

    sofaile = load_sofa(sofa_path)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch(gtfb_path, device=device, dtype=dtype)
    lag_values = torch.arange(-0.001 * fs, 0.001 * fs + 1, 1, dtype=dtype, device=device) / fs

    speakers_per_array = 2
    speaker_distance = 2.0
    speaker_spacing_m = 0.18
    inner_angle_deg = 30.0
    # Place all loudspeakers on a circle of radius speaker_distance. Inner pair at
    # +/- inner_angle_deg; outer pair at the same radius, chord-spaced by speaker_spacing_m.
    pair_angle_delta_deg = float(np.rad2deg(2.0 * np.arcsin(speaker_spacing_m / (2.0 * speaker_distance))))
    speaker_angles_deg = np.array([
        inner_angle_deg,                              # 0: +30 inner
        -inner_angle_deg,                             # 1: -30 inner
        inner_angle_deg + pair_angle_delta_deg,       # 2: +outer
        -inner_angle_deg - pair_angle_delta_deg,      # 3: -outer
    ], dtype=float)
    spk_pos = torch.tensor([
        [speaker_distance * np.cos(np.deg2rad(a)), speaker_distance * np.sin(np.deg2rad(a))]
        for a in speaker_angles_deg
    ], dtype=dtype, device=device)
    # Upper (+y) = indices 0,2; lower (-y) = indices 1,3
    upper_indices = (0, 2)
    lower_indices = (1, 3)
    upper_centre = spk_pos[list(upper_indices)].detach().cpu().numpy().mean(axis=0)
    lower_centre = spk_pos[list(lower_indices)].detach().cpu().numpy().mean(axis=0)

    num_lstn_pos = 51
    lstn_pos = torch.tensor(
        [[0.0, float(y)] for y in np.linspace(-1.25, 1.25, num_lstn_pos)],
        dtype=dtype, device=device,
    )

    d = speaker_distance * np.cos(np.deg2rad(inner_angle_deg))
    target_angle = config["training"]["target_angle"]
    target_position = torch.tensor(
        [d, float(np.tan(np.deg2rad(target_angle)) * d)],
        dtype=dtype, device=device,
    )
    target_angles = target_angle_for_listeners(lstn_pos, target_position)

    listener_weights = build_listener_position_weights(lstn_pos, 0.20, 1.25, -1.25, -0.20, 1.0, 0.0)
    if float(np.sum(listener_weights)) <= 0.0:
        listener_weights = np.ones(len(lstn_pos), dtype=np.float32)

    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    lsd_model = LSD(fs, sofaile, target_angles.unsqueeze(-1), device=device, dtype=dtype)

    prop = BinauralFreeFieldPropagation(
        fs, lstn_pos, spk_pos, sofaile,
        apply_distance_attenuation=apply_distance_attenuation,
        device=device, dtype=dtype,
    )

    n_sources = len(spk_pos)
    n_taps = config["filter_estimator"]["n_samples"]
    nn = MultiSourceFIR(n_taps, n_sources)
    nn.set_filter_initialisation(**filter_initialisation)
    # Tie symmetric pairs: +30 with -30 (0↔1), +outer with -outer (2↔3).
    nn.branches[1] = nn.branches[0]
    nn.branches[3] = nn.branches[2]
    nn = nn.to(device=device, dtype=dtype)
    second_moment = SecondMoment(n_taps, n_sources, device=device, dtype=dtype)

    modelTrainer = ModelTrainer(
        [llado_model, lsd_model],
        criterions=[
            make_listener_weighted_loss(listener_weights, device, dtype, pointwise="squared"),
            make_listener_weighted_loss(listener_weights, device, dtype, pointwise="squared"),
        ],
        weights=model_weights,
    )

    targets = {
        "llado2025_grad_surgery": target_angles.unsqueeze(0),
        "LSD": torch.zeros(1, len(lstn_pos), dtype=dtype, device=device),
    }

    # Keep bandpass multi-band coverage, but fewer draws per band for CPU wall-time
    # (51 listeners forces batch_size=1).
    rodenas_batch_size = 1
    n_bands = config["stimulus_dataset"]["n_bands"]
    n_samples_per_band = 4
    stimulus_dataset_run = {
        **config["stimulus_dataset"],
        "n_samples_per_band": n_samples_per_band,
    }

    stim_loader, stim_dataset = get_stim_loader(
        fs=fs,
        n_bands=n_bands,
        n_samples_per_band=n_samples_per_band,
        batch_size=rodenas_batch_size,
        stimulus_length_ms=config["stimulus_dataset"]["stimulus_length_ms"],
        ramp_time_ms=config["stimulus_dataset"]["ramp_time_ms"],
        f_min=config["stimulus_dataset"]["f_min"],
        f_max=config["stimulus_dataset"]["f_max"],
        band_mode=config["stimulus_dataset"]["band_mode"],
        erb_factor=config["stimulus_dataset"].get("erb_factor", 1.0),
        dataset_type=config["stimulus_dataset"]["dataset_type"],
        stimulus_type=config["stimulus_dataset"]["stimulus_type"],
        seed=config["env"]["seed"],
        shuffle=config["training"]["shuffle"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        drop_last=config["training"]["drop_last"],
    )
    eval_config = {
        **config,
        "stimulus_dataset": stimulus_dataset_run,
        "training": {**config["training"], "batch_size": 1},
    }
    eval_stim = get_white_noise_batch(eval_config, device=device, dtype=dtype, shuffle=False)

    optimizer = torch.optim.Adam(nn.parameters(), lr=config["training"]["lr"])
    mtl = create_mtl_optimizer(modelTrainer, mtl_optimizer, training_config=config["training"])

    saving_manager = SavingManager(
        f"outputs/rodenas/{mtl_optimizer}", skip_keys=skip_keys, timestamp=True, use_tensorboard=use_tensorboard,
        task_names=list(modelTrainer.model_dict.keys()),
    )

    git_info = git_commit_info()
    run_config = {
        "script": "scripts/rodenas_optimisation.py",
        "git": git_info,
        "config": config,
        "env": {
            "fs": fs,
            "device": str(device),
            "dtype": config["env"]["dtype"],
            "seed": config["env"]["seed"],
            "sofa_path": sofa_path,
            "gtfb_path": gtfb_path,
            "apply_distance_attenuation": apply_distance_attenuation,
        },
        "mtl_optimizer": mtl_optimizer,
        "n_taps": n_taps,
        "n_sources": n_sources,
        "speakers_per_array": speakers_per_array,
        "speaker_distance": speaker_distance,
        "speaker_spacing_m": speaker_spacing_m,
        "inner_angle_deg": inner_angle_deg,
        "pair_angle_delta_deg": pair_angle_delta_deg,
        "speaker_angles_deg": speaker_angles_deg.tolist(),
        "upper_indices": list(upper_indices),
        "lower_indices": list(lower_indices),
        "num_lstn_pos": num_lstn_pos,
        "target_angle": target_angle,
        "model_weights": model_weights,
        "optimizer": "Adam",
        "lr": config["training"]["lr"],
        "num_epochs": config["training"]["num_epochs"],
        "batch_size": rodenas_batch_size,
        "cagrad_c": config["training"].get("cagrad_c"),
        "filter_initialisation": filter_initialisation,
        "stimulus_dataset": stimulus_dataset_run,
    }
    write_run_config(saving_manager.output_dir, run_config)
    saving_manager.save("run_config", run_config)
    saving_manager.save("git_commit", git_info["commit"])
    saving_manager.save("git_dirty", git_info["dirty"])
    saving_manager.save("gtfb", gtfb)
    saving_manager.save("lag_values", lag_values)
    saving_manager.save("tau_iacc", 1e-2)
    saving_manager.save("target_angle", target_angle)
    saving_manager.save("target_position", target_position)
    saving_manager.save("target_angles", target_angles)
    saving_manager.save("lstn_pos", lstn_pos)
    saving_manager.save("spk_pos", spk_pos)
    saving_manager.save("sofaile", sofaile)
    saving_manager.save("listener_weights", listener_weights)
    saving_manager.save("stimulus_dataset", stimulus_dataset_run)
    saving_manager.save("filter_initialisation", filter_initialisation)
    saving_manager.save("mtl_optimizer", mtl_optimizer)
    saving_manager.save("model_weights", model_weights)
    saving_manager.save("lr", config["training"]["lr"])
    saving_manager.save("num_epochs", config["training"]["num_epochs"])
    saving_manager.save("batch_size", rodenas_batch_size)
    saving_manager.save("n_taps", n_taps)
    saving_manager.save("n_sources", n_sources)
    saving_manager.save("fs", fs)
    saving_manager.save("device", str(device))
    saving_manager.save("dtype", config["env"]["dtype"])
    saving_manager.save("seed", config["env"]["seed"])

    dummy_input = torch.ones(1, 1, dtype=dtype, device=device)
    n_listeners = len(lstn_pos)

    with torch.no_grad():
        spk_filters = nn(dummy_input)
        binaural_signals = prop(eval_stim, spk_filters)
        modelTrainer.compute_estimates(binaural_signals, spk_filters, eval_stim)
        initial_localisation = modelTrainer.estimates["llado2025_grad_surgery"].mean(dim=0).detach().cpu().numpy()
        binaural_signals_flat = binaural_signals.view(
            binaural_signals.shape[0] * binaural_signals.shape[1], 2, -1
        )
        initial_lsd = (
            lsd_model(binaural_signals_flat, eval_stim)["output"]
            .view(eval_stim.shape[0], n_listeners)
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )

    print(
        f"Starting Rodenas CAGrad training on {device}: "
        f"{config['training']['num_epochs']} epochs, batch_size={rodenas_batch_size}, "
        f"n_listeners={n_listeners}, n_taps={n_taps}, weights={model_weights}, "
        f"bands={n_bands}x{n_samples_per_band}, "
        f"spk_angles_deg={np.round(speaker_angles_deg, 2).tolist()}",
        flush=True,
    )
    for epoch in range(config["training"]["num_epochs"]):
        batch_loss_dicts = []
        batch_estimates = []
        batch_cues = {}
        batch_mtl_saved = []

        for batch_idx, stim in enumerate(stim_loader):
            if batch_idx % 32 == 0:
                print(f"  epoch {epoch} batch {batch_idx}/{len(stim_loader)}", flush=True)
            stim = stim.to(device=device, dtype=dtype)
            spk_filters = nn(dummy_input)
            binaural_signals = prop(stim, spk_filters)

            modelTrainer.compute_estimates(binaural_signals, spk_filters, stim)
            loss_dict = modelTrainer.compute_loss(targets)
            batch_loss_dicts.append({key: value.detach() for key, value in loss_dict.items()})
            batch_estimates.append(
                {key: value.detach() for key, value in modelTrainer.estimates.items()}
            )

            for model_name, kwargs in modelTrainer.kwargs_output.items():
                if "interauralcues" in kwargs:
                    cues = kwargs["interauralcues"]
                    batch_cues.setdefault(model_name, []).append(
                        {"itd": cues["itd"].detach(), "ild": cues["ild"].detach()}
                    )

            optimizer.zero_grad()
            batch_mtl_saved.append(
                run_mtl_optimizer_step(
                    mtl_optimizer, mtl, loss_dict, nn, optimizer, saving_manager, modelTrainer, save=False
                )
            )

        saving_manager.set_step(epoch)
        epoch_loss_dict = average_loss_dicts(batch_loss_dicts)
        epoch_estimates = average_estimates_dicts(batch_estimates)
        saving_manager.save("loss_dict", epoch_loss_dict)
        saving_manager.save("modelTrainer.estimates", epoch_estimates)
        for model_name, cues_dicts in batch_cues.items():
            saving_manager.save(
                f"{model_name}_cues", average_interaural_cues_dicts(cues_dicts, n_listeners)
            )
        for key, value in average_mtl_saved_values(batch_mtl_saved).items():
            saving_manager.save(key, value)

        saving_manager.flush_tensorboard()
        print_loss_and_estimates(epoch_loss_dict, epoch_estimates, epoch)

    with torch.no_grad():
        spk_filters = nn(dummy_input)
        binaural_signals = prop(eval_stim, spk_filters)
        binaural_signals = binaural_signals.view(
            binaural_signals.shape[0] * binaural_signals.shape[1], 2, -1
        )
        localisation_angles = (
            llado_model(binaural_signals, eval_stim)["output"]
            .view(eval_stim.shape[0], n_listeners)
            .mean(dim=0, keepdim=True)
        )
        lsd = (
            lsd_model(binaural_signals, eval_stim)["output"]
            .view(eval_stim.shape[0], n_listeners)
            .mean(dim=0, keepdim=True)
        )

    saving_manager.save("binaural_signals", binaural_signals)
    saving_manager.save("localisation_angles", localisation_angles)
    saving_manager.save("lsd", lsd)
    saving_manager.save("spk_filters", spk_filters)

    saving_manager.export_to_disk()
    print(f"Saved outputs to {saving_manager.output_dir}")

    est_filters = spk_filters.squeeze(0).detach().cpu().numpy()
    est_filters /= np.max(np.abs(est_filters)) / 0.9
    est_filters = est_filters * 1.41
    loc_np = localisation_angles.squeeze(0).detach().cpu().numpy()
    lsd_np = lsd.squeeze(0).detach().cpu().numpy()
    lstn_pos_np = lstn_pos.detach().cpu().numpy()
    spk_pos_np = spk_pos.detach().cpu().numpy()

    time_domain_filters_plot(est_filters, fs=fs)
    saving_manager.save_new_plots("time_domain_filters")
    frequency_domain_filters_plot(est_filters, fs=fs)
    saving_manager.save_new_plots("frequency_domain_filters")
    localisation_arrows_plot(
        lstn_pos, spk_pos, loc_np,
        np.zeros_like(loc_np), lsd_np,
        target_angle=target_angles.detach().cpu().numpy(),
    )
    saving_manager.save_new_plots("localisation_arrows")

    fig_listener_weights, ax_listener_weights = plt.subplots(figsize=(6, 3))
    ax_listener_weights.plot(lstn_pos_np[:, 1], listener_weights, linewidth=2)
    ax_listener_weights.set_xlabel("Listener position y (m)")
    ax_listener_weights.set_ylabel("Listener weight")
    ax_listener_weights.set_title("Listener-position weights")
    ax_listener_weights.grid(True, alpha=0.25)
    saving_manager.save_new_plots("listener_weights")

    rodenas2003_fig12_opt = np.loadtxt(ROOT / "data" / "miscellaneous" / "rodenas2003_fig12_opt.csv", delimiter=",", skiprows=1)
    rodenas2003_fig12_opt = [[-((angle_deg + 180) % 360.0), level_db] for angle_deg, level_db in rodenas2003_fig12_opt]
    rodenas2003_fig12_1000Hz = np.loadtxt(ROOT / "data" / "miscellaneous" / "rodenas2003_fig12_1000Hz.csv", delimiter=",", skiprows=1)
    rodenas2003_fig12_1000Hz = [[-((angle_deg + 180) % 360.0), level_db] for angle_deg, level_db in rodenas2003_fig12_1000Hz]

    right_array_indices = lower_indices
    left_array_indices = upper_indices

    loc_metrics = localisation_error_metrics(
        target_angles=target_angles.detach().cpu().numpy(),
        loc_before=initial_localisation,
        loc_after=loc_np,
        listener_weights=listener_weights,
    )
    col_w = np.asarray(listener_weights, dtype=float).ravel()
    col_sum_w = float(np.sum(col_w))
    col_metrics = {
        "mean_before_db": float(np.mean(initial_lsd)),
        "mean_after_db": float(np.mean(lsd_np)),
        "weighted_mean_before_db": float(np.sum(col_w * initial_lsd) / col_sum_w) if col_sum_w > 0 else float(np.mean(initial_lsd)),
        "weighted_mean_after_db": float(np.sum(col_w * lsd_np) / col_sum_w) if col_sum_w > 0 else float(np.mean(lsd_np)),
    }
    dir_theta, dir_avg_db, dir_group_mid = compute_group_directivity_db(
        filters=est_filters, spk_pos_np=spk_pos_np, group_indices=right_array_indices,
        fs=fs, freqs_eval_hz=[250, 500, 1000, 2000, 4000], radius=10.0,
    )
    directivity_metrics = directivity_match_metrics(
        theta=dir_theta, dir_avg_db=dir_avg_db, group_mid=dir_group_mid,
        overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
        overlay_labels=["rodenas2003 (optimum)", "rodenas2003 (1000 Hz)"],
    )
    analysis = {"localisation": loc_metrics, "colouration": col_metrics, "directivity_match": directivity_metrics}

    print("\n================ Results analysis ================")
    print(f"  Loc MAE  : {loc_metrics['mae_before_deg']:.2f} -> {loc_metrics['mae_after_deg']:.2f}")
    print(f"  Loc RMSE : {loc_metrics['rmse_before_deg']:.2f} -> {loc_metrics['rmse_after_deg']:.2f}")
    print(f"  LSD mean : {col_metrics['mean_before_db']:.2f} -> {col_metrics['mean_after_db']:.2f}")
    for m in directivity_metrics:
        print(f"  {m['label']}: RMSE {m['rmse_db']:.2f} dB, MAE {m['mae_db']:.2f} dB")
    print("==================================================\n")

    with open(saving_manager.output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=4)

    plot_pair_directivity_polar(
        est_filters=est_filters, spk_pos=spk_pos_np, fs=fs,
        group_indices=right_array_indices, radius=10.0,
        freqs_eval_hz=[250, 500, 1000, 2000, 4000],
        overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
        overlay_labels=["rodenas2003 (opt.)", "rodenas2003 (1 kHz)"],
        db_floor=-40, show=False,
    )
    saving_manager.save_new_plots("directivity_polar")

    plot_arrows_with_directivity(
        est_filters=est_filters, spk_pos=spk_pos, lstn_pos=lstn_pos,
        loc=torch.as_tensor(loc_np), fs=fs,
        group_idx=[left_array_indices, right_array_indices],
        target_pos=target_position, show_listener_target_lines=True, show=False,
        freqs_eval_hz=[250, 500, 1000, 2000, 4000],
        directivity_observation_radius_m=10.0, directivity_mode="avg",
        directivity_scale=0.8, directivity_floor_db=-20,
        overlay_curves=[rodenas2003_fig12_opt, rodenas2003_fig12_1000Hz],
        overlay_labels=["rodenas2003 (optimum)", "rodenas2003 (1000 Hz)"],
        legend_bbox_to_anchor=(0.28, 1.02), figsize=(5, 5),
    )
    saving_manager.save_new_plots("arrows_and_directivity")

    est_filters_centre = np.stack([
        np.sum(est_filters[list(upper_indices)], axis=0),
        np.sum(est_filters[list(lower_indices)], axis=0),
    ], axis=0)
    spk_pos_centre = np.stack([upper_centre, lower_centre], axis=0)
    prop_centre = BinauralFreeFieldPropagation(
        fs, lstn_pos, torch.as_tensor(spk_pos_centre, dtype=dtype, device=device), sofaile,
        apply_distance_attenuation=apply_distance_attenuation, device=device, dtype=dtype,
    )
    with torch.no_grad():
        h_centre = torch.as_tensor(est_filters_centre, dtype=dtype, device=device).unsqueeze(0)
        bin_stim_centre = prop_centre(eval_stim, h_centre)
        bin_stim_centre = bin_stim_centre.view(bin_stim_centre.shape[0] * bin_stim_centre.shape[1], 2, -1)
        loc_centre = (
            llado_model(bin_stim_centre, eval_stim)["output"]
            .view(eval_stim.shape[0], n_listeners)
            .mean(dim=0)
        )

    plot_arrows_with_directivity(
        est_filters=est_filters_centre,
        spk_pos=torch.as_tensor(spk_pos_centre, dtype=dtype),
        lstn_pos=lstn_pos.detach().cpu(), loc=loc_centre.detach().cpu(), fs=fs,
        group_idx=[(0,), (1,)], show=False,
        freqs_eval_hz=[250, 500, 1000, 2000, 4000],
        directivity_observation_radius_m=10.0, directivity_mode="avg",
        directivity_scale=0.8, directivity_floor_db=-20,
        directivity_label="Omni source", legend_ncol=1, figsize=(5, 5),
    )
    saving_manager.save_new_plots("arrows_and_directivity_centre")

    plot_rodenas_mtl_diagnostics(saving_manager)

    saving_manager.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rodenas bandpass CAGrad optimisation")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=2,
        metavar=("LOC", "COL"),
        default=None,
        help="Task weights for localisation (llado) and colouration (LSD). Default: 1 1",
    )
    args = parser.parse_args()
    main(model_weights=args.weights)
