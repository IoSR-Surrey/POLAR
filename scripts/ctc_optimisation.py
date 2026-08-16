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
from propagation.model import BinauralFreeFieldPropagation
from datasets.stimulus_dataset import get_stim_loader
from utils.dsp import bandpass_fir
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


def compute_ctc_filters(sofaile, target_angle, left_spk_angle=30.0, right_spk_angle=-30.0, n_fft=256):
    """Classical CTC filters that render ``target_angle`` at the ears."""
    left_spk_hrirs = get_IRs(sofa_find(left_spk_angle, sofaile), sofaile).numpy()[0]
    right_spk_hrirs = get_IRs(sofa_find(right_spk_angle, sofaile), sofaile).numpy()[0]
    ref_hrirs = get_IRs(sofa_find(target_angle, sofaile), sofaile).numpy()[0]

    irs = np.stack([left_spk_hrirs, right_spk_hrirs], axis=1)  # (2 ears, 2 spk, T)
    H = np.fft.fft(irs, n=n_fft, axis=-1)
    G = compute_xtc_filters(H)
    ref_hrtf = np.fft.fft(ref_hrirs, n=n_fft, axis=-1)

    left_f = G[0, 0] * ref_hrtf[0] + G[0, 1] * ref_hrtf[1]
    right_f = G[1, 0] * ref_hrtf[0] + G[1, 1] * ref_hrtf[1]

    left = np.real(np.fft.ifft(left_f)).reshape(1, -1)
    right = np.real(np.fft.ifft(right_f)).reshape(1, -1)
    filts = np.concatenate((left, right), axis=0)
    return np.fft.fftshift(filts, axes=-1)


def smooth_third_octave(freq, mag_db):
    smoothed = np.zeros_like(mag_db)
    k = 2 ** (1 / 6)
    for i, fc in enumerate(freq):
        idx = np.where((freq >= fc / k) & (freq <= fc * k))[0]
        smoothed[i] = np.mean(mag_db[idx])
    return smoothed


def _rfft_norm(x, n):
    return np.fft.rfft(x, n=n, axis=-1, norm="ortho")


def _pad_time(x, n):
    pad = n - x.shape[-1]
    return x if pad == 0 else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])


def _align_to_reference(x, ref, ref_channel=0):
    """Circularly shift ``x`` so channel ``ref_channel`` aligns with ``ref``."""
    corr = np.correlate(x[ref_channel], ref[ref_channel], mode="full")
    lag = int(np.argmax(corr) - (len(ref[ref_channel]) - 1))
    return np.stack([np.roll(x[ch], -lag) for ch in range(x.shape[0])], axis=0)


def _unwrapped_phase(spec, freqs=None, remove_linear=False, f_min=80.0, f_max=20000.0):
    """Unwrapped phase; optionally remove bulk linear delay estimated in-band."""
    phase = np.unwrap(np.angle(spec), axis=-1)
    if not remove_linear:
        return phase
    if freqs is None:
        raise ValueError("freqs is required when remove_linear=True")
    band = (freqs >= f_min) & (freqs <= f_max)
    out = np.empty_like(phase)
    for ch in range(phase.shape[0]):
        coef = np.polyfit(freqs[band], phase[ch, band], 1)
        out[ch] = phase[ch] - np.polyval(coef, freqs)
    return out


def _prepare_ctc_plot_signals(proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim):
    proposed = bandpass_fir(proposed, fs=fs)
    proposed_col_only = bandpass_fir(proposed_col_only, fs=fs)
    ctc = bandpass_fir(ctc, fs=fs)

    with torch.no_grad():
        proposed_t = torch.as_tensor(proposed, device=stim.device, dtype=stim.dtype).unsqueeze(0)
        col_only_t = torch.as_tensor(proposed_col_only, device=stim.device, dtype=stim.dtype).unsqueeze(0)
        ctc_t = torch.as_tensor(ctc, device=stim.device, dtype=stim.dtype).unsqueeze(0)

        bin_proposed = prop(stim, proposed_t).detach().cpu().numpy()[0, 0]
        bin_col_only = prop(stim, col_only_t).detach().cpu().numpy()[0, 0]
        bin_ctc = prop(stim, ctc_t).detach().cpu().numpy()[0, 0]

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

    n_filts = proposed.shape[-1]
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
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title(titles[1], fontsize=fontsize)
    ax2.semilogx(freqs_filt, filt_curves["proposed"][1], linewidth=3)
    ax2.semilogx(freqs_filt, filt_curves["col_only"][1], linewidth=3)
    ax2.semilogx(freqs_filt, filt_curves["ctc"][1], linewidth=3)
    ax2.grid()
    ax2.set_xlim([80, 20000])
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


def plot_ctc_time_domain(proposed, proposed_col_only, ctc, fs):
    """Time-domain taps of left/right loudspeaker filters."""
    fontsize = 18
    n_taps = proposed.shape[-1]
    time_ms = np.arange(n_taps) / fs * 1000

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


def plot_ctc_comparison(proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim):
    """Fig7-style magnitude and phase comparison of filters and ear signals."""
    sigs = _prepare_ctc_plot_signals(proposed, proposed_col_only, ctc, reference_bin, fs, prop, stim)

    proposed_spec = _rfft_norm(sigs["proposed"], sigs["n_filts"])
    col_only_spec = _rfft_norm(sigs["proposed_col_only"], sigs["n_filts"])
    ctc_spec = _rfft_norm(sigs["ctc"], sigs["n_filts"])
    ref_spec = _rfft_norm(sigs["ref"], sigs["n_ear"])
    bin_proposed_spec = _rfft_norm(sigs["bin_proposed"], sigs["n_ear"])
    bin_col_only_spec = _rfft_norm(sigs["bin_col_only"], sigs["n_ear"])
    bin_ctc_spec = _rfft_norm(sigs["bin_ctc"], sigs["n_ear"])

    proposed_f = 20 * np.log10(np.abs(proposed_spec) + 1e-12)
    col_only_f = 20 * np.log10(np.abs(col_only_spec) + 1e-12)
    ctc_f = 20 * np.log10(np.abs(ctc_spec) + 1e-12)

    lsd = np.sqrt(np.mean((ctc_f - proposed_f) ** 2, axis=-1)).mean()
    print(f"Log Spectral Distance between CTC and proposed filters: {lsd:.2f} dB")

    ref_f = 20 * np.log10(np.abs(ref_spec) + 1e-12)
    bin_proposed_f = 20 * np.log10(np.abs(bin_proposed_spec) + 1e-12)
    bin_col_only_f = 20 * np.log10(np.abs(bin_col_only_spec) + 1e-12)
    bin_ctc_f = 20 * np.log10(np.abs(bin_ctc_spec) + 1e-12)

    fig_mag = _plot_ctc_2x2(
        sigs["freqs_filt"],
        sigs["freqs_ref"],
        filt_curves={
            "proposed": [smooth_third_octave(sigs["freqs_filt"], proposed_f[0]),
                         smooth_third_octave(sigs["freqs_filt"], proposed_f[1])],
            "col_only": [smooth_third_octave(sigs["freqs_filt"], col_only_f[0]),
                         smooth_third_octave(sigs["freqs_filt"], col_only_f[1])],
            "ctc": [smooth_third_octave(sigs["freqs_filt"], ctc_f[0]),
                    smooth_third_octave(sigs["freqs_filt"], ctc_f[1])],
        },
        ear_curves={
            "proposed": [smooth_third_octave(sigs["freqs_ref"], bin_proposed_f[0]),
                         smooth_third_octave(sigs["freqs_ref"], bin_proposed_f[1])],
            "col_only": [smooth_third_octave(sigs["freqs_ref"], bin_col_only_f[0]),
                         smooth_third_octave(sigs["freqs_ref"], bin_col_only_f[1])],
            "ctc": [smooth_third_octave(sigs["freqs_ref"], bin_ctc_f[0]),
                    smooth_third_octave(sigs["freqs_ref"], bin_ctc_f[1])],
            "ref": [smooth_third_octave(sigs["freqs_ref"], ref_f[0]),
                    smooth_third_octave(sigs["freqs_ref"], ref_f[1])],
        },
        ylabel="Magnitude [dB]",
        ylim=[-60, 10],
    )

    # Absolute unwrapped phase is dominated by bulk delay: the HRIR reference is
    # early, while CTC/proposed go through centered FIRs (~n_taps/2). Align ears
    # to the reference before taking phase; remove linear phase from filters only.
    bin_proposed_aligned = _align_to_reference(sigs["bin_proposed"], sigs["ref"])
    bin_col_only_aligned = _align_to_reference(sigs["bin_col_only"], sigs["ref"])
    bin_ctc_aligned = _align_to_reference(sigs["bin_ctc"], sigs["ref"])

    proposed_p = _unwrapped_phase(proposed_spec, sigs["freqs_filt"], remove_linear=True)
    col_only_p = _unwrapped_phase(col_only_spec, sigs["freqs_filt"], remove_linear=True)
    ctc_p = _unwrapped_phase(ctc_spec, sigs["freqs_filt"], remove_linear=True)
    ref_p = _unwrapped_phase(ref_spec)
    bin_proposed_p = _unwrapped_phase(_rfft_norm(bin_proposed_aligned, sigs["n_ear"]))
    bin_col_only_p = _unwrapped_phase(_rfft_norm(bin_col_only_aligned, sigs["n_ear"]))
    bin_ctc_p = _unwrapped_phase(_rfft_norm(bin_ctc_aligned, sigs["n_ear"]))

    fig_phase = _plot_ctc_2x2(
        sigs["freqs_filt"],
        sigs["freqs_ref"],
        filt_curves={
            "proposed": proposed_p,
            "col_only": col_only_p,
            "ctc": ctc_p,
        },
        ear_curves={
            "proposed": bin_proposed_p,
            "col_only": bin_col_only_p,
            "ctc": bin_ctc_p,
            "ref": ref_p,
        },
        ylabel="Phase [rad]",
        ylim=None,
    )
    return fig_mag, fig_phase


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
    stim_length_ms=100.0,
    gtfb_path="data/banks/gtfb_48k.npy",
    seed=48,
):
    """Localisation arrows for proposed / col-only / CTC via the llado model."""
    gtfb = load_npy_to_torch(gtfb_path, device=device, dtype=dtype)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    desena_model = desena2020(fs, hrirs, gtfb, device=device, dtype=dtype)
    lsd_model = LSD(fs, sofaile, target_angles, device=device, dtype=dtype)
    n_listeners = len(lstn_pos)

    # White-noise probe (impulse is too short for gammatone / IACC cues).
    n_samples = int(round(stim_length_ms * fs / 1000.0))
    rng = np.random.default_rng(seed)
    stim = torch.as_tensor(
        rng.standard_normal((1, 1, n_samples), dtype=np.float32),
        device=device,
        dtype=dtype,
    )
    stim = stim / stim.std().clamp_min(1e-8)

    results = {
        "Proposed (Col. + IPD)": _evaluate_llado_localisation(
            proposed, prop, stim, llado_model, desena_model, lsd_model, n_listeners
        ),
        "Proposed (Col. only)": _evaluate_llado_localisation(
            proposed_col_only, prop, stim, llado_model, desena_model, lsd_model, n_listeners
        ),
        "CTC": _evaluate_llado_localisation(
            ctc, prop, stim, llado_model, desena_model, lsd_model, n_listeners
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
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--col-weight", type=float, default=1.0)
    parser.add_argument("--ipd-weight", type=float, default=0.1)
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
    ipd_model = IPD(fs, sofaile, target_angles, f_max=20000.0, device=device, dtype=dtype)

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
    })

    targets_joint = {
        "LSD": torch.zeros(1, n_listeners, dtype=dtype, device=device),
        "IPD": torch.zeros(1, n_listeners, dtype=dtype, device=device),
    }

    print(f"=== Optimising colouration + IPD with CAGrad (c={args.cagrad_c}) ===")
    t0 = time.time()
    spk_filters, binaural_signals, trainer = _optimise_filters(
        models=[lsd_model, ipd_model],
        criterions=[F.l1_loss, F.l1_loss],
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
    else:
        print("=== Optimising colouration only ===")
        t0 = time.time()
        spk_filters_col_only, _, _ = _optimise_filters(
            models=[lsd_model],
            criterions=[F.l1_loss],
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
        est_filters_col_only = spk_filters_col_only.squeeze(0).detach().cpu().numpy()
        est_filters_col_only /= np.max(np.abs(est_filters_col_only)) / 0.9

    ctc_filters = compute_ctc_filters(sofaile, args.target_angle, n_fft=args.n_taps)
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
    fig_mag, fig_phase = plot_ctc_comparison(
        proposed=est_filters,
        proposed_col_only=est_filters_col_only,
        ctc=ctc_filters,
        reference_bin=reference_bin,
        fs=fs,
        prop=prop,
        stim=plot_stim,
    )
    fig_taps = plot_ctc_time_domain(
        proposed=est_filters,
        proposed_col_only=est_filters_col_only,
        ctc=ctc_filters,
        fs=fs,
    )
    loc_figs, loc_results = plot_ctc_localisation(
        proposed=est_filters,
        proposed_col_only=est_filters_col_only,
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
        seed=args.seed,
    )

    saving_manager.save("spk_filters", spk_filters)
    saving_manager.save("est_filters", est_filters)
    saving_manager.save("est_filters_col_only", est_filters_col_only)
    saving_manager.save("ctc_filters", ctc_filters)
    saving_manager.save("binaural_signals", binaural_signals)
    saving_manager.save("final_lsd", trainer.estimates["LSD"])
    saving_manager.save("final_ipd", trainer.estimates["IPD"])
    saving_manager.save("localisation_results", loc_results)
    saving_manager.export_to_disk()
    saving_manager.save_new_plots(
        "ctc_comparison",
        names=[
            "ctc_comparison",
            "ctc_comparison_phase",
            "ctc_comparison_taps",
            "localisation_proposed",
            "localisation_col_only",
            "localisation_ctc",
        ],
    )
    plt.close(fig_mag)
    plt.close(fig_phase)
    plt.close(fig_taps)
    for fig in loc_figs:
        plt.close(fig)

    print(f"Saved outputs to {saving_manager.output_dir}")
    print(f"Final LSD: {trainer.estimates['LSD'].mean().item():.4f}")
    print(f"Final IPD: {trainer.estimates['IPD'].mean().item():.4f}")


if __name__ == "__main__":
    main()
