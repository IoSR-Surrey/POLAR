"""Optimise stereo CTC-like filters with colouration (LSD) + IPD losses.

Ports the old DAFx2026_fig7 colouration+IPD optimisation onto the current
filter / propagation / ModelTrainer pipeline, and compares against classical
crosstalk-cancellation (CTC) inversion.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from filter_estimation.model import MultiSourceFIR
from logger.printing import print_loss_and_estimates
from logger.saving import SavingManager
from models.listener_based.ipd.model import IPD
from models.listener_based.LSD.model import LSD
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.desena2020.model import desena2020
from models.train import ModelTrainer
from plotting.spatial_plots import localisation_arrows_plot
from plotting.batch_utils import mean_over_stimulus_batch
from plotting.evaluation import _broadband_white_noise_batch
from propagation.model import BinauralFreeFieldPropagation
from datasets.stimulus_dataset import get_stim_loader
from utils.miscellaneous import (
    create_mtl_optimizer,
    load_npy_to_torch,
    run_mtl_optimizer_step,
    target_angle_for_listeners,
    write_run_config,
)
from utils.seed import set_seed
from utils.sofa_functions import get_IRs, get_template_for_llado2025_desena_2020, load_sofa, sofa_find


def compute_xtc_filters(H, reg=0.0):
    n_fft = H.shape[-1]
    G = np.zeros_like(H, dtype=np.complex64)
    eye = np.eye(2)
    for k in range(n_fft):
        G[:, :, k] = np.linalg.inv(H[:, :, k] + reg * eye)
    return G


def compute_plant_matrix(sofaile, left_spk_angle=30.0, right_spk_angle=-30.0, n_fft=256):
    """2x2 plant H(ω): rows = ears, columns = loudspeakers (±30° HRIRs)."""
    left_spk_hrirs = get_IRs(sofa_find(left_spk_angle, sofaile), sofaile).numpy()[0]
    right_spk_hrirs = get_IRs(sofa_find(right_spk_angle, sofaile), sofaile).numpy()[0]
    if n_fft is None:
        n_fft = left_spk_hrirs.shape[-1]
    irs = np.stack([left_spk_hrirs, right_spk_hrirs], axis=1)  # (2 ears, 2 spk, T)
    return np.fft.fft(irs, n=n_fft, axis=-1)


def analyse_plant_inverse_condition(H, fs, reg=0.0):
    """Per-bin condition number of H^{-1} (and of H) on the positive-frequency DFT grid.

    CTC inverts G[k] = (H[k] + reg I)^{-1}. For an invertible matrix,
    κ(H^{-1}) = κ(H) = σ_max / σ_min. ``cond_inv`` is measured on the
    realised inverse so numerical error at near-singular bins is visible.
    """
    n_fft = H.shape[-1]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    n_r = freqs.shape[0]
    eye = np.eye(2, dtype=np.complex128)
    cond_H = np.empty(n_r)
    cond_inv = np.empty(n_r)
    sigma_max = np.empty(n_r)
    sigma_min = np.empty(n_r)
    inv_norm = np.empty(n_r)
    det_abs = np.empty(n_r)
    for k in range(n_r):
        Hk = np.asarray(H[:, :, k], dtype=np.complex128) + reg * eye
        s = np.linalg.svd(Hk, compute_uv=False)
        sigma_max[k] = float(s[0])
        sigma_min[k] = float(s[-1])
        cond_H[k] = float(s[0] / max(s[-1], np.finfo(float).tiny))
        Gk = np.linalg.inv(Hk)
        cond_inv[k] = float(np.linalg.cond(Gk))
        inv_norm[k] = float(np.linalg.norm(Gk, 2))
        det_abs[k] = float(np.abs(np.linalg.det(Hk)))
    return {
        "freqs": freqs,
        "cond_H": cond_H,
        "cond_inv": cond_inv,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "inv_spectral_norm": inv_norm,
        "det_abs": det_abs,
        "n_fft": n_fft,
        "fs": fs,
        "reg": reg,
    }


def plant_condition_band_summary(stats, f_min=80.0, f_max=20000.0):
    """Min / median / max of κ(H^{-1}) in a frequency band."""
    freqs = stats["freqs"]
    cond = stats["cond_inv"]
    band = (freqs >= f_min) & (freqs <= f_max) & np.isfinite(cond)
    cond_b = cond[band]
    freqs_b = freqs[band]
    i_min = int(np.argmin(cond_b))
    i_max = int(np.argmax(cond_b))
    return {
        "f_min": f_min,
        "f_max": f_max,
        "n_bins": int(band.sum()),
        "min": float(cond_b[i_min]),
        "min_hz": float(freqs_b[i_min]),
        "median": float(np.median(cond_b)),
        "mean": float(np.mean(cond_b)),
        "max": float(cond_b[i_max]),
        "max_hz": float(freqs_b[i_max]),
        "rel_cond_H_error": float(
            np.nanmax(np.abs(stats["cond_inv"][band] - stats["cond_H"][band]) / np.maximum(stats["cond_H"][band], 1e-12))
        ),
    }


def print_plant_condition_summary(stats, f_min=80.0, f_max=20000.0):
    s = plant_condition_band_summary(stats, f_min=f_min, f_max=f_max)
    print(
        f"Plant inverse condition number (n_fft={stats['n_fft']}, "
        f"reg={stats['reg']:g}, {f_min:g}–{f_max:g} Hz, {s['n_bins']} bins):"
    )
    print(f"  min    κ(H^{{-1}}) = {s['min']:.3g} at {s['min_hz']:.1f} Hz")
    print(f"  median κ(H^{{-1}}) = {s['median']:.3g}")
    print(f"  mean   κ(H^{{-1}}) = {s['mean']:.3g}")
    print(f"  max    κ(H^{{-1}}) = {s['max']:.3g} at {s['max_hz']:.1f} Hz")
    print(f"  max |κ(H^{{-1}})-κ(H)| / κ(H) = {s['rel_cond_H_error']:.2e}")


def plot_ctc_plant_condition(stats, stats_dense=None, f_min=80.0, f_max=20000.0):
    """κ(H^{-1}) and singular values of the 2x2 plant vs frequency."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    fig, (ax_cond, ax_svd) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)

    cond_color = "#0072B2"
    smax_color = "#009E73"
    smin_color = "#D55E00"

    def _band(st):
        m = (st["freqs"] >= f_min) & (st["freqs"] <= f_max)
        return st["freqs"][m], st["cond_inv"][m], st["sigma_max"][m], st["sigma_min"][m], st["inv_spectral_norm"][m]

    if stats_dense is not None:
        f_d, cond_d, smax_d, smin_d, _ = _band(stats_dense)
        ax_cond.semilogx(f_d, cond_d, color=cond_color, lw=2.2, label=r"$\kappa(H^{-1})$")
        ax_svd.semilogx(f_d, smax_d, color=smax_color, lw=2.2, label=r"$\sigma_{\mathrm{max}}(H)$")
        ax_svd.semilogx(f_d, smin_d, color=smin_color, lw=2.2, label=r"$\sigma_{\mathrm{min}}(H)$")
        f_b, cond_b, _, _, _ = _band(stats)
        ax_cond.semilogx(
            f_b, cond_b, color="0.25", marker="o", ms=3.5, ls="none",
            label=rf"CTC bins ($N={stats['n_fft']}$)", zorder=3,
        )
    else:
        f_b, cond_b, smax_b, smin_b, _ = _band(stats)
        ax_cond.semilogx(f_b, cond_b, color=cond_color, lw=2.2, label=r"$\kappa(H^{-1})$")
        ax_svd.semilogx(f_b, smax_b, color=smax_color, lw=2.2, label=r"$\sigma_{\mathrm{max}}(H)$")
        ax_svd.semilogx(f_b, smin_b, color=smin_color, lw=2.2, label=r"$\sigma_{\mathrm{min}}(H)$")

    ax_cond.set_yscale("log")
    ax_cond.set_ylabel(r"$\kappa(H^{-1})$")
    ax_cond.set_title("Plant inverse conditioning")
    ax_cond.set_xlim(f_min, f_max)
    ax_cond.grid(True, which="both", alpha=0.35)
    ax_cond.legend(loc="best", frameon=True)
    ax_cond.tick_params(axis="both", labelsize=12)

    ax_svd.set_yscale("log")
    ax_svd.set_xlabel("Frequency [Hz]")
    ax_svd.set_ylabel("Singular value")
    ax_svd.set_xlim(f_min, f_max)
    ax_svd.grid(True, which="both", alpha=0.35)
    ax_svd.legend(loc="best", frameon=True)
    ax_svd.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    return fig


def plot_ctc_plant_inverse_gain(stats, stats_dense=None, f_min=80.0, f_max=20000.0):
    """||H^{-1}||_2 vs frequency, with the CTC DFT bins overlaid."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    gain_color = "#0072B2"

    def _band(st):
        m = (st["freqs"] >= f_min) & (st["freqs"] <= f_max)
        return st["freqs"][m], st["inv_spectral_norm"][m]

    if stats_dense is not None:
        f_d, g_d = _band(stats_dense)
        ax.semilogx(f_d, g_d, color=gain_color, lw=2.2, label=r"$\|H^{-1}\|_2$")
        f_b, g_b = _band(stats)
        ax.semilogx(
            f_b, g_b, color="0.25", marker="o", ms=3.5, ls="none",
            label=rf"CTC bins ($N={stats['n_fft']}$)", zorder=3,
        )
    else:
        f_b, g_b = _band(stats)
        ax.semilogx(f_b, g_b, color=gain_color, lw=2.2, label=r"$\|H^{-1}\|_2$")

    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\|H^{-1}\|_2$")
    ax.set_title("Plant inverse gain")
    ax.set_xlim(f_min, f_max)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", frameon=True)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    return fig


def compute_ctc_filters(sofaile, target_angle, left_spk_angle=30.0, right_spk_angle=-30.0, n_fft=256):
    """Classical CTC filters that render ``target_angle`` at the ears.

    256-point circular inverse, fftshifted for linear convolution in the plant.
    """
    ref_hrirs = get_IRs(sofa_find(target_angle, sofaile), sofaile).numpy()[0]
    H = compute_plant_matrix(sofaile, left_spk_angle, right_spk_angle, n_fft)
    n_fft = H.shape[-1]
    G = compute_xtc_filters(H)
    ref_hrtf = np.fft.fft(ref_hrirs, n=n_fft, axis=-1)

    left_f = G[0, 0] * ref_hrtf[0] + G[0, 1] * ref_hrtf[1]
    right_f = G[1, 0] * ref_hrtf[0] + G[1, 1] * ref_hrtf[1]

    left = np.real(np.fft.ifft(left_f)).reshape(1, -1)
    right = np.real(np.fft.ifft(right_f)).reshape(1, -1)
    filts = np.concatenate((left, right), axis=0)
    return np.fft.fftshift(filts, axes=-1)


class CircularMixingPropagation:
    """256-point circular mix used to plot classical CTC: ears = ifft(H (S G))."""

    def __init__(
        self,
        sofaile,
        n_fft,
        lstn_pos,
        spk_pos,
        apply_distance_attenuation=True,
        left_spk_angle=30.0,
        right_spk_angle=-30.0,
        device="cpu",
        dtype=torch.float32,
    ):
        self.n_fft = int(n_fft)
        left_h = get_IRs(sofa_find(left_spk_angle, sofaile), sofaile, device=device, dtype=dtype)
        right_h = get_IRs(sofa_find(right_spk_angle, sofaile), sofaile, device=device, dtype=dtype)
        irs = torch.stack([left_h, right_h], dim=2)
        self.H = torch.fft.fft(irs, n=self.n_fft, dim=-1)
        dist = torch.linalg.vector_norm(spk_pos - lstn_pos, dim=-1).reshape(-1)
        if apply_distance_attenuation:
            self.src_gain = (1.0 / dist).to(device=device, dtype=dtype)
        else:
            self.src_gain = torch.ones(irs.shape[2], device=device, dtype=dtype)

    def __call__(self, stim, spk_filters, listener_indices=None):
        return self.forward(stim, spk_filters, listener_indices)

    def forward(self, stim, spk_filters, listener_indices=None):
        n_fft = self.n_fft
        S = torch.fft.fft(stim, n=n_fft, dim=-1)
        G = torch.fft.fft(spk_filters, n=n_fft, dim=-1)
        SG = S * G * self.src_gain.view(1, -1, 1)
        H = self.H if listener_indices is None else self.H[listener_indices]
        ears_f = torch.einsum("lesk,bsk->blek", H, SG)
        return torch.fft.ifft(ears_f, n=n_fft, dim=-1).real


def smooth_third_octave(freq, mag_db):
    smoothed = np.zeros_like(mag_db)
    k = 2 ** (1 / 6)
    for i, fc in enumerate(freq):
        idx = np.where((freq >= fc / k) & (freq <= fc * k))[0]
        smoothed[i] = np.mean(mag_db[idx])
    return smoothed


def _pad_time(x, n):
    pad = n - x.shape[-1]
    return x if pad == 0 else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])


def _centre_pad(x, n):
    pad = n - x.shape[-1]
    if pad <= 0:
        return x
    left = pad // 2
    right = pad - left
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)])


def _wrap_phase(x, bound=np.pi):
    return (x + bound) % (2.0 * bound) - bound


def _ipd_from_spec(spec):
    """Wrapped IPD; positive = left leads."""
    return _wrap_phase(np.angle(spec[0]) - np.angle(spec[1]))


def _ipd_from_time(x, fs, n_fft_min=4096):
    n_fft = max(n_fft_min, x.shape[-1])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    spec = np.fft.rfft(x, n=n_fft, axis=-1)
    return freqs, _ipd_from_spec(spec)


def _table_wrapped_phase_error(
    x, y, fs, remove_linear=False, n_fft_min=4096, f_min=20.0, f_max=20000.0,
):
    """Per-bin wrap(phi_x - phi_y) used by the Uniform [rad] table column."""
    n_fft = max(n_fft_min, x.shape[-1], y.shape[-1])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    px = np.unwrap(np.angle(np.fft.rfft(x, n=n_fft, axis=-1)), axis=-1)
    py = np.unwrap(np.angle(np.fft.rfft(y, n=n_fft, axis=-1)), axis=-1)
    if remove_linear:
        band = (freqs >= f_min) & (freqs <= f_max)
        for p in (px, py):
            for ch in range(p.shape[0]):
                coef = np.polyfit(freqs[band], p[ch, band], 1)
                p[ch] = p[ch] - np.polyval(coef, freqs)
    return freqs, _wrap_phase(px - py)


def _prepare_ctc_plot_signals(
    proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim, ctc_prop=None,
):
    """Render impulse responses. Learnt filters use ``prop`` (linear full conv);
    CTC uses ``ctc_prop`` (circular DFT mix) when given.
    """
    ctc_render = ctc_prop if ctc_prop is not None else prop
    with torch.no_grad():
        proposed_t = torch.as_tensor(proposed, device=stim.device, dtype=stim.dtype).unsqueeze(0)
        col_only_t = torch.as_tensor(proposed_col_only, device=stim.device, dtype=stim.dtype).unsqueeze(0)
        ctc_t = torch.as_tensor(ctc, device=stim.device, dtype=stim.dtype).unsqueeze(0)

        bin_proposed = prop(stim, proposed_t).detach().cpu().numpy()[0, 0]
        bin_col_only = prop(stim, col_only_t).detach().cpu().numpy()[0, 0]
        bin_ctc = ctc_render(stim, ctc_t).detach().cpu().numpy()[0, 0]

    ref = reference_bin.detach().cpu().numpy()
    if ref.ndim == 3:
        ref = ref[0]

    # Propagation uses full convolution, so rendered ears are longer than the
    # HRIR reference. Pad to a common length before FFT so np.fft.rfft does
    # not truncate longer signals.
    n_ear = max(ref.shape[-1], bin_proposed.shape[-1], bin_col_only.shape[-1], bin_ctc.shape[-1])
    ref = _pad_time(ref, n_ear)
    bin_proposed = _pad_time(bin_proposed, n_ear)
    bin_col_only = _pad_time(bin_col_only, n_ear)
    bin_ctc = _pad_time(bin_ctc, n_ear)

    n_filts = max(proposed.shape[-1], proposed_col_only.shape[-1], ctc.shape[-1])
    return {
        "proposed": proposed,
        "proposed_col_only": proposed_col_only,
        "ctc": ctc,
        "ref": ref,
        "bin_proposed": bin_proposed,
        "bin_col_only": bin_col_only,
        "bin_ctc": bin_ctc,
        "n_filts": n_filts,
        "n_ear": n_ear,
        "freqs_filt": np.fft.rfftfreq(n_filts, d=1 / fs),
        "freqs_ref": np.fft.rfftfreq(n_ear, d=1 / fs),
    }


def _plot_ctc_2x2(
    freqs_filt,
    freqs_ref,
    filt_curves,
    ear_curves,
    ylabel,
    ylim,
    titles=("Left Loudspeaker Filter", "Right Loudspeaker Filter", "Left Ear Signal", "Right Ear Signal"),
):
    fontsize = 18
    fig = plt.figure(figsize=(8, 6))

    ax1 = plt.subplot(2, 2, 1)
    l1, = ax1.semilogx(freqs_filt, filt_curves["proposed"][0], linewidth=3)
    l1b, = ax1.semilogx(freqs_filt, filt_curves["col_only"][0], linewidth=3)
    l2, = ax1.semilogx(freqs_filt, filt_curves["ctc"][0], linewidth=3)
    ax1.grid()
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_title(titles[0], fontsize=fontsize)
    ax1.set_xlim([80, 20000])
    ax1.tick_params(axis="both", labelsize=12)
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title(titles[1], fontsize=fontsize)
    ax2.semilogx(freqs_filt, filt_curves["proposed"][1], linewidth=3)
    ax2.semilogx(freqs_filt, filt_curves["col_only"][1], linewidth=3)
    ax2.semilogx(freqs_filt, filt_curves["ctc"][1], linewidth=3)
    ax2.grid()
    ax2.set_xlim([80, 20000])
    ax2.tick_params(axis="both", labelsize=12)
    if ylim is not None:
        ax2.set_ylim(ylim)

    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogx(freqs_ref, ear_curves["proposed"][0], linewidth=3)
    ax3.semilogx(freqs_ref, ear_curves["col_only"][0], linewidth=3)
    ax3.semilogx(freqs_ref, ear_curves["ctc"][0], linewidth=3)
    l5, = ax3.semilogx(
        freqs_ref, ear_curves["ref"][0],
        linewidth=4, color="black", linestyle="--",
    )
    ax3.grid()
    ax3.set_xlim([80, 20000])
    ax3.tick_params(axis="both", labelsize=12)
    if ylim is not None:
        ax3.set_ylim(ylim)
    ax3.set_title(titles[2], fontsize=fontsize)
    ax3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax3.set_ylabel(ylabel, fontsize=fontsize)

    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogx(freqs_ref, ear_curves["proposed"][1], linewidth=3)
    ax4.semilogx(freqs_ref, ear_curves["col_only"][1], linewidth=3)
    ax4.semilogx(freqs_ref, ear_curves["ctc"][1], linewidth=3)
    ax4.semilogx(
        freqs_ref, ear_curves["ref"][1],
        linewidth=4, color="black", linestyle="--",
    )
    ax4.grid()
    ax4.set_xlim([80, 20000])
    ax4.tick_params(axis="both", labelsize=12)
    if ylim is not None:
        ax4.set_ylim(ylim)
    ax4.set_title(titles[3], fontsize=fontsize)
    ax4.set_xlabel("Frequency [Hz]", fontsize=fontsize)

    fig.legend(
        [l1, l1b, l2, l5],
        ["Proposed Method (Col. + IPD)", "Proposed Method (Col. only)", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        fontsize=fontsize,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def _set_phase_ylim(ax, *arrays, min_lim=2.0 * np.pi, max_ticks=7):
    ymax = float(min_lim)
    for a in arrays:
        ymax = max(ymax, float(np.nanmax(np.abs(np.asarray(a)))))
    n = max(1, int(np.ceil(ymax / np.pi - 1e-9)))
    lim = n * np.pi
    ax.set_ylim([-lim, lim])
    step_k = 1
    while (2 * n) / step_k + 1 > max_ticks:
        step_k *= 2
    ks = np.arange(-n, n + 1, step_k)
    if ks[0] != -n:
        ks = np.insert(ks, 0, -n)
    if ks[-1] != n:
        ks = np.append(ks, n)
    ticks = ks * np.pi
    labels = []
    for k in ks:
        if k == 0:
            labels.append("0")
        elif k == 1:
            labels.append(r"$\pi$")
        elif k == -1:
            labels.append(r"$-\pi$")
        else:
            labels.append(rf"${k}\pi$")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def _plot_ctc_phase(freqs_filt, freqs_ear, filt_err, ipd, ipd_err):
    """Delay-stripped filter wrap vs CTC, unwrapped IPD, wrapped IPD error."""
    fontsize = 18
    fig = plt.figure(figsize=(8, 6))
    titles = (
        "Left Loudspeaker vs CTC",
        "Right Loudspeaker vs CTC",
        "IPD",
        "IPD Error",
    )

    ax1 = plt.subplot(2, 2, 1)
    l1, = ax1.semilogx(freqs_filt, filt_err["proposed"][0], linewidth=3)
    l1b, = ax1.semilogx(freqs_filt, filt_err["col_only"][0], linewidth=3)
    l2, = ax1.semilogx(freqs_filt, filt_err["ctc"][0], linewidth=3)
    ax1.grid()
    ax1.set_ylabel("Phase error [rad]", fontsize=fontsize)
    ax1.set_title(titles[0], fontsize=fontsize)
    ax1.set_xlim([80, 20000])
    ax1.tick_params(axis="both", labelsize=12)
    _set_phase_ylim(
        ax1, filt_err["proposed"][0], filt_err["col_only"][0], filt_err["ctc"][0],
        min_lim=np.pi,
    )

    ax2 = plt.subplot(2, 2, 2)
    ax2.semilogx(freqs_filt, filt_err["proposed"][1], linewidth=3)
    ax2.semilogx(freqs_filt, filt_err["col_only"][1], linewidth=3)
    ax2.semilogx(freqs_filt, filt_err["ctc"][1], linewidth=3)
    ax2.grid()
    ax2.set_title(titles[1], fontsize=fontsize)
    ax2.set_xlim([80, 20000])
    ax2.tick_params(axis="both", labelsize=12)
    _set_phase_ylim(
        ax2, filt_err["proposed"][1], filt_err["col_only"][1], filt_err["ctc"][1],
        min_lim=np.pi,
    )

    ax3 = plt.subplot(2, 2, 3)
    ax3.semilogx(freqs_ear, ipd["proposed"], linewidth=3)
    ax3.semilogx(freqs_ear, ipd["col_only"], linewidth=3)
    ax3.semilogx(freqs_ear, ipd["ctc"], linewidth=3)
    l5, = ax3.semilogx(freqs_ear, ipd["ref"], linewidth=4, color="black", linestyle="--")
    ax3.grid()
    ax3.set_xlim([80, 20000])
    ax3.tick_params(axis="both", labelsize=12)
    _set_phase_ylim(ax3, ipd["proposed"], ipd["col_only"], ipd["ctc"], ipd["ref"])
    ax3.set_title(titles[2], fontsize=fontsize)
    ax3.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax3.set_ylabel("IPD [rad]", fontsize=fontsize)

    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogx(freqs_ear, ipd_err["proposed"], linewidth=3)
    ax4.semilogx(freqs_ear, ipd_err["col_only"], linewidth=3)
    ax4.semilogx(freqs_ear, ipd_err["ctc"], linewidth=3)
    ax4.semilogx(freqs_ear, np.zeros_like(freqs_ear), linewidth=4, color="black", linestyle="--")
    ax4.grid()
    ax4.set_xlim([80, 20000])
    ax4.tick_params(axis="both", labelsize=12)
    _set_phase_ylim(
        ax4, ipd_err["proposed"], ipd_err["col_only"], ipd_err["ctc"],
        min_lim=np.pi,
    )
    ax4.set_title(titles[3], fontsize=fontsize)
    ax4.set_xlabel("Frequency [Hz]", fontsize=fontsize)
    ax4.set_ylabel("IPD error [rad]", fontsize=fontsize)

    fig.legend(
        [l1, l1b, l2, l5],
        ["Proposed Method (Col. + IPD)", "Proposed Method (Col. only)", "CTC", "Reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        fontsize=fontsize,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def plot_ctc_time_domain(proposed, proposed_col_only, ctc, fs):
    """Time-domain taps of left/right loudspeaker filters."""
    fontsize = 18
    n_taps = max(proposed.shape[-1], proposed_col_only.shape[-1], ctc.shape[-1])
    time_ms = np.arange(n_taps) / fs * 1000
    proposed = _centre_pad(proposed, n_taps)
    proposed_col_only = _centre_pad(proposed_col_only, n_taps)
    ctc = _centre_pad(ctc, n_taps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    titles = ("Left Loudspeaker Filter", "Right Loudspeaker Filter")
    labels = ("Proposed Method (Col. + IPD)", "Proposed Method (Col. only)", "CTC")
    series = (proposed, proposed_col_only, ctc)

    for ch, ax in enumerate(axes):
        lines = []
        for filt in series:
            lines.append(ax.plot(time_ms, filt[ch], linewidth=2)[0])
        ax.set_title(titles[ch], fontsize=fontsize)
        ax.set_xlabel("Time [ms]", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True)
        if ch == 0:
            ax.set_ylabel("Amplitude", fontsize=fontsize)

    fig.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=fontsize,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def plot_ctc_comparison(proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim, ctc_prop=None):
    """1/3-octave magnitude, wrapped filter phase vs CTC, and ear IPD."""
    fig_mag = plot_ctc_magnitude(
        proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim,
        ctc_prop=ctc_prop, smooth=True,
    )
    sigs = _prepare_ctc_plot_signals(
        proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim, ctc_prop=ctc_prop,
    )

    proposed_spec = np.fft.rfft(sigs["proposed"], n=sigs["n_filts"], axis=-1)
    ctc_spec = np.fft.rfft(sigs["ctc"], n=sigs["n_filts"], axis=-1)
    proposed_f = 20 * np.log10(np.abs(proposed_spec) + 1e-12)
    ctc_f = 20 * np.log10(np.abs(ctc_spec) + 1e-12)
    lsd = np.sqrt(np.mean((ctc_f - proposed_f) ** 2, axis=-1)).mean()
    print(f"Log Spectral Distance between CTC and proposed filters: {lsd:.2f} dB")

    freqs_filt, filt_proposed = _table_wrapped_phase_error(
        sigs["proposed"], sigs["ctc"], fs, remove_linear=True,
    )
    _, filt_col_only = _table_wrapped_phase_error(
        sigs["proposed_col_only"], sigs["ctc"], fs, remove_linear=True,
    )

    freqs_ear, ipd_w_proposed = _ipd_from_time(sigs["bin_proposed"], fs)
    _, ipd_w_col_only = _ipd_from_time(sigs["bin_col_only"], fs)
    _, ipd_w_ctc = _ipd_from_time(sigs["bin_ctc"], fs)
    _, ipd_w_ref = _ipd_from_time(sigs["ref"], fs)
    ipd_w = {
        "proposed": ipd_w_proposed,
        "col_only": ipd_w_col_only,
        "ctc": ipd_w_ctc,
        "ref": ipd_w_ref,
    }
    ipd = {key: np.unwrap(val) for key, val in ipd_w.items()}
    ipd_err = {
        "proposed": _wrap_phase(ipd_w["proposed"] - ipd_w["ref"]),
        "col_only": _wrap_phase(ipd_w["col_only"] - ipd_w["ref"]),
        "ctc": _wrap_phase(ipd_w["ctc"] - ipd_w["ref"]),
    }

    fig_phase = _plot_ctc_phase(
        freqs_filt,
        freqs_ear,
        filt_err={
            "proposed": filt_proposed,
            "col_only": filt_col_only,
            "ctc": np.zeros_like(filt_proposed),
        },
        ipd=ipd,
        ipd_err=ipd_err,
    )
    return fig_mag, fig_phase


def plot_ctc_magnitude(
    proposed,
    proposed_col_only,
    ctc,
    reference_bin,
    fs,
    prop,
    stim,
    ctc_prop=None,
    smooth=False,
):
    """Magnitude comparison from the raw optimiser filters (unnormalized rfft)."""
    sigs = _prepare_ctc_plot_signals(
        proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim, ctc_prop=ctc_prop,
    )
    n_filts = sigs["n_filts"]
    n_ear = sigs["n_ear"]
    freqs_filt = sigs["freqs_filt"]
    freqs_ref = sigs["freqs_ref"]
    proposed, proposed_col_only, ctc = sigs["proposed"], sigs["proposed_col_only"], sigs["ctc"]
    bin_proposed, bin_col_only, bin_ctc = sigs["bin_proposed"], sigs["bin_col_only"], sigs["bin_ctc"]
    ref = sigs["ref"]

    def _mag(x, n):
        return 20.0 * np.log10(np.abs(np.fft.rfft(x, n=n, axis=-1)) + 1e-12)

    proposed_f = _mag(proposed, n_filts)
    col_only_f = _mag(proposed_col_only, n_filts)
    ctc_f = _mag(ctc, n_filts)
    bin_proposed_f = _mag(bin_proposed, n_ear)
    bin_col_only_f = _mag(bin_col_only, n_ear)
    bin_ctc_f = _mag(bin_ctc, n_ear)
    ref_f = _mag(ref, n_ear)

    if smooth:
        proposed_f = np.stack([smooth_third_octave(freqs_filt, proposed_f[0]),
                               smooth_third_octave(freqs_filt, proposed_f[1])])
        col_only_f = np.stack([smooth_third_octave(freqs_filt, col_only_f[0]),
                               smooth_third_octave(freqs_filt, col_only_f[1])])
        ctc_f = np.stack([smooth_third_octave(freqs_filt, ctc_f[0]),
                          smooth_third_octave(freqs_filt, ctc_f[1])])
        bin_proposed_f = np.stack([smooth_third_octave(freqs_ref, bin_proposed_f[0]),
                                   smooth_third_octave(freqs_ref, bin_proposed_f[1])])
        bin_col_only_f = np.stack([smooth_third_octave(freqs_ref, bin_col_only_f[0]),
                                   smooth_third_octave(freqs_ref, bin_col_only_f[1])])
        bin_ctc_f = np.stack([smooth_third_octave(freqs_ref, bin_ctc_f[0]),
                              smooth_third_octave(freqs_ref, bin_ctc_f[1])])
        ref_f = np.stack([smooth_third_octave(freqs_ref, ref_f[0]),
                          smooth_third_octave(freqs_ref, ref_f[1])])

    return _plot_ctc_2x2(
        freqs_filt,
        freqs_ref,
        filt_curves={
            "proposed": proposed_f,
            "col_only": col_only_f,
            "ctc": ctc_f,
        },
        ear_curves={
            "proposed": bin_proposed_f,
            "col_only": bin_col_only_f,
            "ctc": bin_ctc_f,
            "ref": ref_f,
        },
        ylabel="Magnitude [dB]",
        ylim=[-45, 15],
    )


def _evaluate_llado_localisation(filters, prop, stim, llado_model, desena_model, lsd_model, n_listeners):
    """Run llado/desena/LSD on filters rendered through ``prop`` with ``stim``."""
    filters_t = torch.as_tensor(filters, device=stim.device, dtype=stim.dtype)
    if filters_t.ndim == 2:
        filters_t = filters_t.unsqueeze(0)

    with torch.no_grad():
        binaural = prop(stim, filters_t)
        binaural_flat = binaural.reshape(-1, 2, binaural.shape[-1])
        loc = llado_model(binaural_flat, stim)["output"]
        unc = desena_model(binaural_flat, stim)["output"]
        lsd = lsd_model(binaural_flat, stim)["output"]

    return {
        "localisation_angles": mean_over_stimulus_batch(loc, n_listeners),
        "localisation_uncertainty": mean_over_stimulus_batch(unc, n_listeners),
        "lsd": mean_over_stimulus_batch(lsd, n_listeners),
    }


def plot_ctc_localisation(
    proposed,
    proposed_col_only,
    ctc,
    prop,
    lstn_pos,
    spk_pos,
    target_angles,
    target_angle_deg,
    sofaile,
    fs,
    device,
    dtype,
    stim=None,
    ctc_prop=None,
    ctc_stim=None,
    gtfb_path="data/banks/gtfb_48k.npy",
):
    """Localisation arrows for proposed / col-only / CTC via the llado model."""
    gtfb = load_npy_to_torch(gtfb_path, device=device, dtype=dtype)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    desena_model = desena2020(fs, hrirs, gtfb, device=device, dtype=dtype)
    lsd_model = LSD(fs, sofaile, target_angles, device=device, dtype=dtype)
    n_listeners = len(lstn_pos)

    if stim is None:
        stim = _broadband_white_noise_batch(fs, device, dtype)
    ctc_render = ctc_prop if ctc_prop is not None else prop
    if ctc_stim is None:
        ctc_stim = stim

    results = {
        "Proposed (Col. + IPD)": _evaluate_llado_localisation(
            proposed, prop, stim, llado_model, desena_model, lsd_model, n_listeners
        ),
        "Proposed (Col. only)": _evaluate_llado_localisation(
            proposed_col_only, prop, stim, llado_model, desena_model, lsd_model, n_listeners
        ),
        "CTC": _evaluate_llado_localisation(
            ctc, ctc_render, ctc_stim, llado_model, desena_model, lsd_model, n_listeners
        ),
    }

    for name, res in results.items():
        angle = float(np.atleast_1d(np.asarray(res["localisation_angles"]).reshape(-1))[0])
        unc = float(np.atleast_1d(np.asarray(res["localisation_uncertainty"]).reshape(-1))[0])
        lsd = float(np.atleast_1d(np.asarray(res["lsd"]).reshape(-1))[0])
        print(f"Llado localisation [{name}]: angle={angle:.2f} deg, uncertainty={unc:.3f}, LSD={lsd:.2f} dB")

    figs = []
    for name, res in results.items():
        angle = float(np.atleast_1d(np.asarray(res["localisation_angles"]).reshape(-1))[0])
        localisation_arrows_plot(
            lstn_pos,
            spk_pos,
            res["localisation_angles"],
            res["localisation_uncertainty"],
            lsd=res["lsd"],
            target_angle=target_angle_deg,
        )
        fig_i = plt.gcf()
        fig_i.axes[0].set_title(f"{name}\n(estimated {angle:.1f}°)", fontsize=12)
        figs.append(fig_i)
    return figs, results


def _build_geometry(dtype, device, target_angle, speaker_distance=2.0, speaker_base_angle=30.0):
    lstn_pos = torch.tensor([[0.0, 0.0]], dtype=dtype, device=device)
    base_angle_rad = np.deg2rad(speaker_base_angle)
    spk_pos = torch.tensor([
        [speaker_distance * np.cos(base_angle_rad), speaker_distance * np.sin(base_angle_rad)],
        [speaker_distance * np.cos(base_angle_rad), -speaker_distance * np.sin(base_angle_rad)],
    ], dtype=dtype, device=device)

    target_distance = 2.0
    target_pos = torch.tensor(
        [target_distance, float(np.tan(np.deg2rad(target_angle)) * target_distance)],
        dtype=dtype,
        device=device,
    )
    target_angles = target_angle_for_listeners(lstn_pos, target_pos).unsqueeze(-1)
    return lstn_pos, spk_pos, target_pos, target_angles, speaker_distance


def _ctc_distance_attenuation_gain(lstn_pos, spk_pos):
    """Per-speaker gain that undoes prop 1/r so classical CTC matches design level."""
    spk_dist = torch.linalg.vector_norm(
        spk_pos - lstn_pos, dim=-1
    ).detach().cpu().numpy()
    return spk_dist[:, None]


def _optimise_filters(
    models,
    criterions,
    weights,
    targets,
    prop,
    filter_input,
    n_taps,
    n_sources,
    lr,
    num_epochs,
    device,
    dtype,
    stim=None,
    stim_loader=None,
    mtl_optimizer="cagrad",
    cagrad_c=1.0,
    saving_manager=None,
):
    if stim is None and stim_loader is None:
        raise ValueError("Provide stim or stim_loader.")

    filter_estimator = MultiSourceFIR(n_taps, n_sources)
    filter_estimator.set_filter_initialisation("middle")
    filter_estimator = filter_estimator.to(device=device, dtype=dtype)

    model_trainer = ModelTrainer(models, criterions=criterions, weights=weights)
    optimizer = torch.optim.Adam(filter_estimator.parameters(), lr=lr)
    mtl = create_mtl_optimizer(
        model_trainer,
        mtl_optimizer,
        training_config={"cagrad_c": cagrad_c},
    )

    stim_iter = iter(stim_loader) if stim_loader is not None else None

    for epoch in range(num_epochs):
        if saving_manager is not None:
            saving_manager.set_step(epoch)

        if stim_iter is not None:
            try:
                batch_stim = next(stim_iter)
            except StopIteration:
                stim_iter = iter(stim_loader)
                batch_stim = next(stim_iter)
            batch_stim = batch_stim.to(device=device, dtype=dtype)
        else:
            batch_stim = stim

        spk_filters = filter_estimator(filter_input)
        binaural_signals = prop(batch_stim, spk_filters)

        model_trainer.compute_estimates(binaural_signals, spk_filters, batch_stim)
        loss_dict = model_trainer.compute_loss(targets)

        optimizer.zero_grad()
        run_mtl_optimizer_step(
            mtl_optimizer,
            mtl,
            loss_dict,
            filter_estimator,
            optimizer,
            saving_manager,
            model_trainer,
            save=saving_manager is not None,
        )

        print_loss_and_estimates(loss_dict, model_trainer.estimates, epoch)

    with torch.no_grad():
        eval_stim = stim if stim is not None else batch_stim
        spk_filters = filter_estimator(filter_input)
        binaural_signals = prop(eval_stim, spk_filters)
        model_trainer.compute_estimates(binaural_signals, spk_filters, eval_stim)

    return spk_filters, binaural_signals, model_trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-angle", type=float, default=45.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-taps", type=int, default=256)
    parser.add_argument(
        "--ctc-n-fft",
        type=int,
        default=None,
        help="FFT length for the CTC inverse and circular plot mix. Default: --n-taps.",
    )
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--col-weight", type=float, default=1.0)
    parser.add_argument("--ipd-weight", type=float, default=0.1)
    parser.add_argument("--ipd-f-min", type=float, default=200.0)
    parser.add_argument("--ipd-f-max", type=float, default=20000.0)
    parser.add_argument("--ipd-mse", action="store_true", help="Use mean squared wrapped IPD error instead of MAE.")
    parser.add_argument("--cagrad-c", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--sofa-path", type=str, default="data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    parser.add_argument("--output-dir", type=str, default="outputs/ctc")
    parser.add_argument("--skip-col-only", action="store_true")
    parser.add_argument("--stimulus", type=str, choices=["impulse", "bandpass"], default="impulse")
    parser.add_argument("--n-bands", type=int, default=6)
    parser.add_argument("--n-samples-per-band", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stimulus-length-ms", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    fs = 48000
    dtype = getattr(torch, args.dtype)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    sofaile = load_sofa(args.sofa_path)
    apply_distance_attenuation = True
    lstn_pos, spk_pos, target_pos, target_angles, speaker_distance = _build_geometry(
        dtype, device, args.target_angle
    )
    n_sources = len(spk_pos)
    n_listeners = len(lstn_pos)

    prop = BinauralFreeFieldPropagation(
        fs, lstn_pos, spk_pos, sofaile,
        apply_distance_attenuation=apply_distance_attenuation,
        device=device, dtype=dtype,
    )

    lsd_model = LSD(fs, sofaile, target_angles, device=device, dtype=dtype)
    ipd_model = IPD(
        fs, sofaile, target_angles,
        f_min=args.ipd_f_min, f_max=args.ipd_f_max,
        use_mse=args.ipd_mse,
        device=device, dtype=dtype,
    )
    ipd_loss_name = "MSE" if args.ipd_mse else "MAE"
    print(f"LSD trainer: MSE vs 0 (squares the LSD RMS)")
    print(f"IPD band: {args.ipd_f_min:g}–{args.ipd_f_max:g} Hz ({ipd_loss_name} in the model)")
    # IPD(use_mse=True) already returns mean(e^2); L1 vs 0 keeps that as MSE.
    # mse_loss vs 0 would square it again.
    ipd_criterion = F.l1_loss if args.ipd_mse else F.mse_loss
    print(f"IPD trainer: {'L1 vs 0' if args.ipd_mse else 'MSE vs 0'}")

    # Impulse kept for plotting / final eval; training may use bandpass batches.
    plot_stim = torch.zeros(1, 1, 1, dtype=dtype, device=device)
    plot_stim[:, :, 0] = 1.0
    filter_input = torch.ones(1, 1, dtype=dtype, device=device)

    stim_loader = None
    if args.stimulus == "bandpass":
        stim_loader, _ = get_stim_loader(
            fs=fs,
            n_bands=args.n_bands,
            n_samples_per_band=args.n_samples_per_band,
            batch_size=args.batch_size,
            stimulus_length_ms=args.stimulus_length_ms,
            ramp_time_ms=1.0,
            f_min=20.0,
            f_max=20000.0,
            band_mode="linear_erb",
            erb_factor=0.25,
            dataset_type="bandpass",
            stimulus_type="noise",
            shuffle=True,
        )
        print(
            f"Using bandpass stimulus dataset: "
            f"{args.n_bands} bands x {args.n_samples_per_band} samples, "
            f"batch_size={args.batch_size}, length={args.stimulus_length_ms} ms"
        )
    else:
        print("Using impulse stimulus")

    ctc_n_fft = args.n_taps if args.ctc_n_fft is None else args.ctc_n_fft

    saving_manager = SavingManager(args.output_dir, timestamp=True)
    write_run_config(saving_manager.output_dir, {
        "script": "scripts/ctc_optimisation.py",
        "args": vars(args),
        "mtl_optimizer": "cagrad",
        "cagrad_c": args.cagrad_c,
        "fs": fs,
        "apply_distance_attenuation": apply_distance_attenuation,
        "speaker_distance": speaker_distance,
        "target_pos": target_pos.detach().cpu().tolist(),
        "lstn_pos": lstn_pos.detach().cpu().tolist(),
        "spk_pos": spk_pos.detach().cpu().tolist(),
        "lsd_criterion": "mse_loss",
        "ipd_criterion": "l1_loss" if args.ipd_mse else "mse_loss",
        "ipd_model": "mse" if args.ipd_mse else "mae",
        "ctc_n_fft": ctc_n_fft,
    })

    targets_joint = {
        "LSD": torch.zeros(1, n_listeners, dtype=dtype, device=device),
        "IPD": torch.zeros(1, n_listeners, dtype=dtype, device=device),
    }

    print(f"=== Optimising colouration + IPD with CAGrad (c={args.cagrad_c}) ===")
    t0 = time.time()
    spk_filters, binaural_signals, trainer = _optimise_filters(
        models=[lsd_model, ipd_model],
        criterions=[F.mse_loss, ipd_criterion],
        weights=[args.col_weight, args.ipd_weight],
        targets=targets_joint,
        prop=prop,
        stim=plot_stim,
        stim_loader=stim_loader,
        filter_input=filter_input,
        n_taps=args.n_taps,
        n_sources=n_sources,
        lr=args.lr,
        num_epochs=args.epochs,
        device=device,
        dtype=dtype,
        mtl_optimizer="cagrad",
        cagrad_c=args.cagrad_c,
        saving_manager=saving_manager,
    )
    print(f"Colouration + IPD optimisation time: {time.time() - t0:.2f}s")

    est_filters = spk_filters.squeeze(0).detach().cpu().numpy()
    est_filters /= np.max(np.abs(est_filters)) / 0.9

    if args.skip_col_only:
        est_filters_col_only = est_filters.copy()
        spk_filters_col_only_np = spk_filters.squeeze(0).detach().cpu().numpy()
    else:
        print("=== Optimising colouration only ===")
        t0 = time.time()
        spk_filters_col_only, _, _ = _optimise_filters(
            models=[lsd_model],
            criterions=[F.mse_loss],
            weights=[args.col_weight],
            targets={"LSD": torch.zeros(1, n_listeners, dtype=dtype, device=device)},
            prop=prop,
            stim=plot_stim,
            stim_loader=stim_loader,
            filter_input=filter_input,
            n_taps=args.n_taps,
            n_sources=n_sources,
            lr=args.lr,
            num_epochs=args.epochs,
            device=device,
            dtype=dtype,
            mtl_optimizer="plain",
            cagrad_c=args.cagrad_c,
            saving_manager=None,
        )
        print(f"Colouration-only optimisation time: {time.time() - t0:.2f}s")
        spk_filters_col_only_np = spk_filters_col_only.squeeze(0).detach().cpu().numpy()
        est_filters_col_only = spk_filters_col_only_np.copy()
        est_filters_col_only /= np.max(np.abs(est_filters_col_only)) / 0.9

    ctc_filters = compute_ctc_filters(sofaile, args.target_angle, n_fft=ctc_n_fft)
    print(f"CTC inverse n_fft={ctc_n_fft}; all methods plotted through linear full convolution")
    plant_H = compute_plant_matrix(sofaile, n_fft=ctc_n_fft)
    plant_stats = analyse_plant_inverse_condition(plant_H, fs)
    plant_n_fft_dense = max(4096, ctc_n_fft)
    plant_stats_dense = analyse_plant_inverse_condition(
        compute_plant_matrix(sofaile, n_fft=plant_n_fft_dense), fs,
    )
    print_plant_condition_summary(plant_stats)
    if plant_n_fft_dense != ctc_n_fft:
        print_plant_condition_summary(plant_stats_dense)
    # Classical CTC assumes no 1/r; when prop applies attenuation, boost filters by r
    # so CTC ear levels match the same plant. Safe because 1/r is frequency-independent.
    if apply_distance_attenuation:
        ctc_gain = _ctc_distance_attenuation_gain(lstn_pos, spk_pos)
        ctc_filters = ctc_filters * ctc_gain
        print(
            f"Scaled CTC filters by speaker distance "
            f"(gain={ctc_gain.ravel().tolist()}) "
            f"to compensate for distance attenuation"
        )

    reference_bin = lsd_model.compute_binaural_reference_signals(plot_stim)
    proposed_raw = spk_filters.squeeze(0).detach().cpu().numpy()
    fig_mag_raw = plot_ctc_magnitude(
        proposed=proposed_raw,
        proposed_col_only=spk_filters_col_only_np,
        ctc=ctc_filters,
        reference_bin=reference_bin,
        fs=fs,
        prop=prop,
        stim=plot_stim,
        smooth=False,
    )
    fig_mag, fig_phase = plot_ctc_comparison(
        proposed=proposed_raw,
        proposed_col_only=spk_filters_col_only_np,
        ctc=ctc_filters,
        reference_bin=reference_bin,
        fs=fs,
        prop=prop,
        stim=plot_stim,
    )
    fig_taps = plot_ctc_time_domain(
        proposed=proposed_raw,
        proposed_col_only=spk_filters_col_only_np,
        ctc=ctc_filters,
        fs=fs,
    )
    loc_figs, loc_results = plot_ctc_localisation(
        proposed=proposed_raw,
        proposed_col_only=spk_filters_col_only_np,
        ctc=ctc_filters,
        prop=prop,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        target_angles=target_angles,
        target_angle_deg=args.target_angle,
        sofaile=sofaile,
        fs=fs,
        device=device,
        dtype=dtype,
    )
    fig_cond = plot_ctc_plant_condition(plant_stats, plant_stats_dense)
    fig_gain = plot_ctc_plant_inverse_gain(plant_stats, plant_stats_dense)

    saving_manager.save("spk_filters", spk_filters)
    saving_manager.save("spk_filters_col_only", spk_filters_col_only_np)
    saving_manager.save("est_filters", est_filters)
    saving_manager.save("est_filters_col_only", est_filters_col_only)
    saving_manager.save("ctc_filters", ctc_filters)
    saving_manager.save("plant_condition", {
        "freqs": plant_stats["freqs"],
        "cond_inv": plant_stats["cond_inv"],
        "cond_H": plant_stats["cond_H"],
        "sigma_max": plant_stats["sigma_max"],
        "sigma_min": plant_stats["sigma_min"],
        "inv_spectral_norm": plant_stats["inv_spectral_norm"],
        "n_fft": plant_stats["n_fft"],
        "fs": plant_stats["fs"],
        "reg": plant_stats["reg"],
        "dense": {
            "freqs": plant_stats_dense["freqs"],
            "cond_inv": plant_stats_dense["cond_inv"],
            "cond_H": plant_stats_dense["cond_H"],
            "sigma_max": plant_stats_dense["sigma_max"],
            "sigma_min": plant_stats_dense["sigma_min"],
            "inv_spectral_norm": plant_stats_dense["inv_spectral_norm"],
            "n_fft": plant_stats_dense["n_fft"],
        },
        "summary_80_20k": plant_condition_band_summary(plant_stats_dense),
    })
    saving_manager.save("binaural_signals", binaural_signals)
    saving_manager.save("final_lsd", trainer.estimates["LSD"])
    saving_manager.save("final_ipd", trainer.estimates["IPD"])
    saving_manager.save("localisation_results", loc_results)
    saving_manager.export_to_disk()
    saving_manager.save_new_plots(
        "ctc_comparison",
        names=[
            "ctc_comparison_raw",
            "ctc_comparison",
            "ctc_comparison_phase",
            "ctc_comparison_taps",
            "localisation_proposed",
            "localisation_col_only",
            "localisation_ctc",
            "ctc_plant_condition",
            "ctc_plant_inverse_gain",
        ],
    )
    plt.close(fig_mag_raw)
    plt.close(fig_mag)
    plt.close(fig_phase)
    plt.close(fig_taps)
    plt.close(fig_cond)
    plt.close(fig_gain)
    for fig in loc_figs:
        plt.close(fig)

    print(f"Saved outputs to {saving_manager.output_dir}")
    print(f"Final LSD: {trainer.estimates['LSD'].mean().item():.4f}")
    print(f"Final IPD: {trainer.estimates['IPD'].mean().item():.4f}")


if __name__ == "__main__":
    main()
