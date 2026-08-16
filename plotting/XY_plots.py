import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy import signal

from utils.miscellaneous import ready_for_plotting
from utils.dsp import cosine_similarity
from plotting.batch_utils import mean_over_stimulus_batch, reduce_batch_dimension
from datasets.stimulus_dataset.dataset import _erb_bandwidth, _erb_rate_to_hz

def time_domain_filters_plot(filters, fs=48000):

    filters = ready_for_plotting(filters)
    if filters.ndim == 3:
        filters = filters[0]  # (batch, n_sources, n_taps) -> (n_sources, n_taps)
    elif filters.ndim == 4:
        filters = filters[-1][0]  # (epoch, batch, n_sources, n_taps) -> (n_sources, n_taps)

    n_filters = filters.shape[0]
    n_samples = filters.shape[1]
    time_axis = np.arange(n_samples) / fs * 1000  # ms

    grid_size = int(np.ceil(np.sqrt(n_filters)))
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10), squeeze=False)
    ax = ax.ravel()

    for i in range(n_filters):
        ax[i].plot(time_axis, filters[i, :])
        ax[i].set_title(f"Filter {i+1}")
        ax[i].set_xlabel("Time (ms)")
        ax[i].set_ylabel("Amplitude")
        ax[i].grid(True)

    # Hide any unused subplots
    for j in range(n_filters, grid_size * grid_size):
        ax[j].axis('off')

    plt.tight_layout()

def frequency_domain_filters_plot(filters, fs=48000, nfft = 2048):

    filters = ready_for_plotting(filters)

    if filters.ndim == 3:
        filters = filters[0]  # (batch, n_sources, n_taps) -> (n_sources, n_taps)
    elif filters.ndim == 4:
        filters = filters[-1][0]  # (epoch, batch, n_sources, n_taps) -> (n_sources, n_taps)
    
    n_filters = filters.shape[0]
    
    filters_fft = np.fft.rfft(filters, axis=1, n=nfft)
    filters_db = 20*np.log10(np.abs(filters_fft) + 1e-10)
    filters_phase = np.unwrap(np.angle(filters_fft), axis=1)
    frequency_axis = np.fft.rfftfreq(nfft, 1/fs)

    # Relative phase of each filter w.r.t. filter 1 (via complex ratio, then unwrap)
    relative_phases = []
    relative_labels = []
    for i in range(1, n_filters):
        relative_phases.append(np.unwrap(np.angle(filters_fft[i] * np.conj(filters_fft[0]))))
        relative_labels.append(f"Filter {i+1} − Filter 1")
    
    vmin = np.min(filters_db)
    vmax = np.max(filters_db)

    # mag per filter + overlaid mag + phase per filter + overlaid phase + relative phase + relative mag
    n_plots = 2 * n_filters + 4
    grid_size = int(np.ceil(np.sqrt(n_plots)))
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(12, 12), squeeze=False)
    ax = ax.ravel()

    for i in range(n_filters):
        ax[i].semilogx(frequency_axis, filters_db[i, :])
        ax[i].set_title(f"Filter {i+1} magnitude")
        ax[i].set_xlabel("Frequency (Hz)")
        ax[i].set_ylabel("Magnitude (dB)")
        ax[i].grid(True)
        ax[i].set_ylim(vmin, vmax)

    overlay_mag_idx = n_filters
    for i in range(n_filters):
        ax[overlay_mag_idx].semilogx(frequency_axis, filters_db[i, :], label=f"Filter {i+1}")
    ax[overlay_mag_idx].set_title("Overlaid filter magnitudes")
    ax[overlay_mag_idx].set_xlabel("Frequency (Hz)")
    ax[overlay_mag_idx].set_ylabel("Magnitude (dB)")
    ax[overlay_mag_idx].grid(True)
    ax[overlay_mag_idx].set_ylim(vmin, vmax)
    ax[overlay_mag_idx].legend()

    phase_start = n_filters + 1
    for i in range(n_filters):
        ax[phase_start + i].semilogx(frequency_axis, filters_phase[i, :])
        ax[phase_start + i].set_title(f"Filter {i+1} phase")
        ax[phase_start + i].set_xlabel("Frequency (Hz)")
        ax[phase_start + i].set_ylabel("Phase (rad)")
        ax[phase_start + i].grid(True)

    overlay_phase_idx = phase_start + n_filters
    for i in range(n_filters):
        ax[overlay_phase_idx].semilogx(frequency_axis, filters_phase[i, :], label=f"Filter {i+1}")
    ax[overlay_phase_idx].set_title("Overlaid filter phases")
    ax[overlay_phase_idx].set_xlabel("Frequency (Hz)")
    ax[overlay_phase_idx].set_ylabel("Phase (rad)")
    ax[overlay_phase_idx].grid(True)
    ax[overlay_phase_idx].legend()

    rel_phase_idx = overlay_phase_idx + 1
    for rel, label in zip(relative_phases, relative_labels):
        ax[rel_phase_idx].semilogx(frequency_axis, rel, label=label)
    if n_filters == 1:
        ax[rel_phase_idx].set_title("Relative phase (N/A for single filter)")
    else:
        ax[rel_phase_idx].set_title("Relative phase")
        ax[rel_phase_idx].legend()
    ax[rel_phase_idx].set_xlabel("Frequency (Hz)")
    ax[rel_phase_idx].set_ylabel("Phase (rad)")
    ax[rel_phase_idx].grid(True)

    rel_mag_idx = rel_phase_idx + 1
    for i in range(1, n_filters):
        ax[rel_mag_idx].semilogx(frequency_axis, filters_db[i] - filters_db[0], label=f"Filter {i+1} − Filter 1")
    if n_filters == 1:
        ax[rel_mag_idx].set_title("Magnitude difference (N/A for single filter)")
    else:
        ax[rel_mag_idx].set_title("Magnitude difference")
        ax[rel_mag_idx].legend()
    ax[rel_mag_idx].set_xlabel("Frequency (Hz)")
    ax[rel_mag_idx].set_ylabel("Magnitude (dB)")
    ax[rel_mag_idx].grid(True)

    # Hide any unused subplots
    for j in range(n_plots, grid_size * grid_size):
        ax[j].axis('off')

    plt.tight_layout()

def raw_grad_cosine_to_reference_plot(raw_grads, reference_grads, task_names, title):
    """Per-task cosine similarity between each task's raw gradient and a reference
    gradient (e.g. the summed 'plain' gradient or the CAGrad combined gradient),
    one line per task over epochs."""
    n_epochs = len(raw_grads)

    plt.figure()
    for name in task_names:
        cosines = [cosine_similarity(raw_grads[epoch][name], reference_grads[epoch]) for epoch in range(n_epochs)]
        plt.plot(ready_for_plotting(cosines), label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Cosine similarity")
    plt.ylim(-1.05, 1.05)
    plt.grid(True)
    plt.tight_layout()


def plot_loss_and_estimates(loss_dict, estimates, title, n_listeners=None, listener_labels=None):

    n_epochs = len(loss_dict)
    keys = [key for key in loss_dict[0].keys() if key != "total"]

    for key in keys:
        losses = np.squeeze(ready_for_plotting([loss_dict[epoch][key] for epoch in range(n_epochs)]))
        model_estimates = np.asarray(ready_for_plotting([estimates[epoch][key] for epoch in range(n_epochs)]))

        if n_listeners is not None:
            model_estimates = reduce_batch_dimension(model_estimates, n_listeners)

        # Squeeze out singleton dimensions before checking shape
        model_estimates = np.squeeze(model_estimates)

        # Skip plotting estimates for high-dimensional outputs (e.g., CueMatchingModel)
        if model_estimates.ndim > 2:
            continue

        n_series = model_estimates.shape[-1]
        if n_listeners is not None and n_series == n_listeners:
            series_labels = listener_labels or [f"VL {i}" for i in range(n_listeners)]
        else:
            series_labels = [f"{key} {i}" for i in range(n_series)]

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(losses)
        plt.title(f"{key} loss over epochs for {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        if "llado" in key:
            estimate_ylabel = "Azimuth (deg)"
        elif "desena" in key:
            estimate_ylabel = "Uncertainty"
        else:
            estimate_ylabel = "Estimate"

        plt.subplot(2, 1, 2)
        for i in range(n_series):
            plt.plot(np.squeeze(model_estimates[..., i]), label=series_labels[i])
        plt.title(f"{key} estimate over epochs for {title}")
        plt.xlabel("Epoch")
        plt.ylabel(estimate_ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if "total" in loss_dict[0]:
        total_losses = np.squeeze(ready_for_plotting([loss_dict[epoch]["total"] for epoch in range(n_epochs)]))
        plt.figure()
        plt.plot(total_losses)
        plt.title(f"Total loss over epochs for {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()


def loss_over_epochs_plot(loss_dict, title="Loss over epochs"):
    """Simple loss-over-epochs plot for all tasks."""

    n_epochs = len(loss_dict)
    keys = list(loss_dict[0].keys())

    plt.figure()
    for key in keys:
        losses = np.squeeze(ready_for_plotting([loss_dict[epoch][key] for epoch in range(n_epochs)]))
        plt.plot(losses, label=key)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()


def loss_grad_norms_over_epochs_plot(loss_grad_norms, title, norm_kind="Effective"):

    n_epochs = len(loss_grad_norms)
    keys = [key for key in loss_grad_norms[0].keys() if key != "total"]

    plt.figure()
    for key in keys:
        grad_norms = np.squeeze(ready_for_plotting([loss_grad_norms[epoch][key] for epoch in range(n_epochs)]))
        plt.plot(grad_norms, label=key)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(f"{norm_kind} grad norm")
    plt.grid(True)
    plt.tight_layout()


def task_grad_pairwise_cosine_similarity_plot(task_names, mean_matrix, pairwise_labels, pairwise_over_epochs, ref_cosines_over_epochs, title, use_combined_reference=False, combined_norm_over_epochs=None, pairwise_only=False):

    mean_matrix = ready_for_plotting(mean_matrix)
    n_tasks = len(task_names)
    if pairwise_only:
        n_rows = 2
    elif combined_norm_over_epochs is not None:
        n_rows = 4
    else:
        n_rows = 3

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    im = axes[0].imshow(mean_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(n_tasks))
    axes[0].set_yticks(range(n_tasks))
    axes[0].set_xticklabels(task_names, rotation=45, ha="right")
    axes[0].set_yticklabels(task_names)
    axes[0].set_title(f"Pairwise task grad cosine similarity (mean over epochs) for {title}")
    for i in range(n_tasks):
        for j in range(n_tasks):
            axes[0].text(j, i, f"{mean_matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes[0])

    for label, cosines in zip(pairwise_labels, pairwise_over_epochs):
        axes[1].plot(ready_for_plotting(cosines), label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title(f"Pairwise task grad cosine similarity over epochs for {title}")
    if pairwise_labels:
        axes[1].legend()
    axes[1].grid(True)

    if pairwise_only:
        plt.tight_layout()
        return

    for name in task_names:
        axes[2].plot(ref_cosines_over_epochs[name], label=name)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Cosine Similarity")
    if use_combined_reference:
        axes[2].set_title(f"Task grad cosine similarity to combined gradient for {title}")
    else:
        axes[2].set_title(f"Task grad cosine similarity to fixed reference [1, 0, ...] for {title}")
    axes[2].legend()
    axes[2].grid(True)

    if combined_norm_over_epochs is not None:
        axes[3].plot(ready_for_plotting(combined_norm_over_epochs), label="combined")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Grad norm")
        axes[3].set_title(f"Combined gradient norm over epochs for {title}")
        axes[3].legend()
        axes[3].grid(True)
    plt.tight_layout()


def cagrad_dual_weights_over_epochs_plot(cagrad_dual_weights, task_names, title):

    weights = np.asarray(ready_for_plotting(cagrad_dual_weights))

    plt.figure()
    for i, name in enumerate(task_names):
        plt.plot(weights[..., i], label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Dual weight")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()


def cagrad_task_coeffs_over_epochs_plot(cagrad_task_coeffs, task_names, title):

    coeffs = np.asarray(ready_for_plotting(cagrad_task_coeffs))

    plt.figure()
    for i, name in enumerate(task_names):
        plt.plot(coeffs[..., i], label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient multiplier")
    plt.grid(True)
    plt.tight_layout()


def gradnorm_task_weights_over_epochs_plot(gradnorm_task_weights, task_names, title):

    weights = np.asarray(ready_for_plotting(gradnorm_task_weights))

    plt.figure()
    for i, name in enumerate(task_names):
        plt.plot(weights[..., i], label=name)
    plt.legend()
    plt.title(f"GradNorm task weights over epochs for {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Task weight")
    plt.grid(True)
    plt.tight_layout()


def iacc_over_epochs_plot(iacc, listener_labels=None, n_listeners=None):
    """Plot Interaural Cross-Correlation Coefficient (IACC) over epochs."""
    iacc = np.asarray(ready_for_plotting(iacc))

    if n_listeners is not None:
        iacc = reduce_batch_dimension(iacc, n_listeners)

    # Shape: (n_epochs, n_listeners) or (n_epochs,) if single listener
    if iacc.ndim == 1:
        iacc = iacc[:, np.newaxis]

    n_listeners = iacc.shape[-1]
    if listener_labels is None:
        listener_labels = [f"VL {i}" for i in range(n_listeners)]

    plt.figure()
    for i in range(n_listeners):
        plt.plot(iacc[:, i], label=listener_labels[i])
    plt.title("Interaural Cross-Correlation (IACC) Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("IACC")
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def ear_signals_magnitude_phase_plot(binaural_signals, fs=48000, nfft=2048, listener_labels=None, n_listeners=None):
    """Plot magnitude and phase of the ear signals for each virtual listener."""
    binaural_signals = ready_for_plotting(binaural_signals)

    # Expected shape: (n_listeners, 2, signal_length) where 2 = [left, right]
    if binaural_signals.ndim == 2:
        binaural_signals = binaural_signals[np.newaxis, ...]
    if n_listeners is not None and binaural_signals.shape[0] != n_listeners:
        binaural_signals = mean_over_stimulus_batch(binaural_signals, n_listeners)
    n_listeners = binaural_signals.shape[0]

    if listener_labels is None:
        listener_labels = [f"VL {i}" for i in range(n_listeners)]

    freq_axis = np.fft.rfftfreq(nfft, 1 / fs)

    fig, axes = plt.subplots(n_listeners, 2, figsize=(12, 3 * n_listeners), squeeze=False)

    for i in range(n_listeners):
        left_fft = np.fft.rfft(binaural_signals[i, 0, :], n=nfft)
        right_fft = np.fft.rfft(binaural_signals[i, 1, :], n=nfft)

        left_mag = 20 * np.log10(np.abs(left_fft) + 1e-10)
        right_mag = 20 * np.log10(np.abs(right_fft) + 1e-10)
        left_phase = np.unwrap(np.angle(left_fft))
        right_phase = np.unwrap(np.angle(right_fft))

        axes[i, 0].semilogx(freq_axis, left_mag, label="Left")
        axes[i, 0].semilogx(freq_axis, right_mag, label="Right")
        axes[i, 0].set_title(f"{listener_labels[i]} - Magnitude")
        axes[i, 0].set_xlabel("Frequency (Hz)")
        axes[i, 0].set_ylabel("Magnitude (dB)")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        axes[i, 1].semilogx(freq_axis, left_phase, label="Left")
        axes[i, 1].semilogx(freq_axis, right_phase, label="Right")
        axes[i, 1].set_title(f"{listener_labels[i]} - Phase")
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Phase (rad)")
        axes[i, 1].legend()
        axes[i, 1].grid(True)

    plt.tight_layout()


def _passband_edges(dataset, band_idx, dataset_idx):
    if dataset.band_mode == "linear_erb":
        centre_hz = float(dataset.centre_freqs[dataset_idx])
        bandwidth = _erb_bandwidth(centre_hz) / dataset.erb_factor
        low = max(centre_hz - bandwidth / 2, 1.0)
        high = min(centre_hz + bandwidth / 2, dataset.fs / 2 - 1)
        return low, high
    low_freqs, high_freqs = dataset._band_edges()
    return float(low_freqs[band_idx]), float(high_freqs[band_idx])


def bandpass_dataset_plot(dataset):
    """Illustrate band-pass tiling of a BandpassWhiteNoiseDataset.

    Top: 4th-order Butterworth magnitude of every sample (colour = band).
    Bottom: passbands, sampled centres, and (for linear_erb) sampling slices.
    """
    n_bands = dataset.n_bands
    n_per_band = dataset.n_samples_per_band
    cmap = plt.cm.viridis
    colours = cmap(np.linspace(0.08, 0.92, n_bands))
    freqs = np.geomspace(max(dataset.f_min * 0.5, 10.0), dataset.fs / 2 - 1, 4096)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    ax_mag, ax_cov = axes

    for band_idx in range(n_bands):
        colour = colours[band_idx]
        for sample_idx in range(n_per_band):
            dataset_idx = band_idx * n_per_band + sample_idx
            low, high = _passband_edges(dataset, band_idx, dataset_idx)
            sos = signal.butter(4, [low, high], btype="band", fs=dataset.fs, output="sos")
            _, h = signal.sosfreqz(sos, worN=freqs, fs=dataset.fs)
            mag_db = 20 * np.log10(np.abs(h) + 1e-12)
            ax_mag.semilogx(freqs, mag_db, color=colour, alpha=0.85, linewidth=1.1)

    ax_mag.axhline(-3.0, color="0.45", linestyle=":", linewidth=0.9, zorder=0)
    ax_mag.text(dataset.f_min, -3.0, " −3 dB", color="0.35", va="bottom", ha="left", fontsize=12)
    ax_mag.set_ylabel("Magnitude (dB)", fontsize=14)
    ax_mag.set_ylim(-55, 3)
    ax_mag.tick_params(axis="both", labelsize=12)
    ax_mag.grid(True, which="both", alpha=0.35)
    ax_mag.set_title("4th-order Butterworth responses", fontsize=16)

    slice_shown = False
    for band_idx in range(n_bands):
        colour = colours[band_idx]
        y_band = band_idx

        if dataset.band_mode == "linear_erb":
            erb_low, erb_high, erb_anchors, _ = dataset._linear_erb_anchor_centres()
            if n_bands == 1:
                lo, hi = erb_low, erb_high
            else:
                delta = erb_anchors[1] - erb_anchors[0]
                lo = max(erb_anchors[band_idx] - 0.5 * delta, erb_low)
                hi = min(erb_anchors[band_idx] + 0.5 * delta, erb_high)
            f_lo = _erb_rate_to_hz(lo)
            f_hi = _erb_rate_to_hz(hi)
            ax_cov.add_patch(Rectangle(
                (f_lo, y_band - 0.42), f_hi - f_lo, 0.84,
                facecolor=colour, edgecolor="none", alpha=0.28, zorder=0,
            ))
            slice_shown = True

        offsets = np.linspace(-0.22, 0.22, n_per_band) if n_per_band > 1 else np.array([0.0])
        for sample_idx in range(n_per_band):
            dataset_idx = band_idx * n_per_band + sample_idx
            fc = float(dataset.centre_freqs[dataset_idx])
            low, high = _passband_edges(dataset, band_idx, dataset_idx)
            y = y_band + offsets[sample_idx]
            ax_cov.plot([low, high], [y, y], color=colour, linewidth=2.4, solid_capstyle="round", zorder=2)
            ax_cov.plot(fc, y, "o", color=colour, markersize=4.6, markeredgecolor="0.15", markeredgewidth=0.4, zorder=3)

    if dataset.band_mode == "linear_erb":
        _, _, _, anchor_hz = dataset._linear_erb_anchor_centres()
        ax_cov.plot(
            anchor_hz, np.arange(n_bands),
            marker="^", color="k", linestyle="none", markersize=6.5,
            markerfacecolor="white", markeredgewidth=1.1, zorder=4, label="ERB anchor",
        )
        for ax in axes:
            for f_a in anchor_hz:
                ax.axvline(f_a, color="0.75", linestyle="--", linewidth=0.6, zorder=0)

    ax_cov.set_xscale("log")
    ax_cov.set_xlim(dataset.f_min * 0.7, min(dataset.f_max * 1.15, dataset.fs / 2 - 1))
    ax_cov.set_ylim(-0.7, n_bands - 0.3)
    ax_cov.set_yticks(np.arange(n_bands))
    ax_cov.set_yticklabels([f"{i + 1}" for i in range(n_bands)])
    ax_cov.set_ylabel("Band", fontsize=14)
    ax_cov.set_xlabel("Frequency (Hz)", fontsize=14)
    ax_cov.tick_params(axis="both", labelsize=12)
    ax_cov.grid(True, axis="x", which="both", alpha=0.35)
    ax_cov.set_title("Passbands, sampled centres, and ERB-rate coverage", fontsize=16)

    legend_handles = [
        Line2D([0], [0], color="0.3", linewidth=2.4, label="Passband ($f_c \\pm B/2$)"),
        Line2D([0], [0], marker="o", color="0.3", linestyle="none",
               markersize=5, markeredgecolor="k", markeredgewidth=0.4, label="Sampled $f_c$"),
    ]
    if dataset.band_mode == "linear_erb":
        legend_handles.append(
            Line2D([0], [0], marker="^", color="k", linestyle="none",
                   markersize=7, markerfacecolor="white", markeredgewidth=1.1, label="ERB-rate anchor")
        )
        if slice_shown:
            legend_handles.append(Patch(facecolor="0.5", alpha=0.25, edgecolor="none", label="Centre-frequency slice"))
    ax_cov.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=0.92)

    if dataset.band_mode == "linear_erb":
        bw_note = f"$B(f_c)=\\mathrm{{ERB}}(f_c)/{dataset.erb_factor:g}$"
    elif dataset.band_mode == "erb":
        bw_note = r"$B(f_c)=\mathrm{ERB}(f_c)$"
    else:
        bw_note = f"{dataset.band_mode} partitions of $[{dataset.f_min:g}, {dataset.f_max:g}]$ Hz"
    fig.suptitle(
        f"Band-pass dataset : "
        f"{n_bands} bands $\\times$ {n_per_band} samples, {bw_note}",
        fontsize=16,
    )
    fig.tight_layout()
    return fig, axes