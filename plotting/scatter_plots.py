import matplotlib.pyplot as plt
import numpy as np

from utils.sofa_functions import sofa_find, get_IRs

from utils.peripheral_processing import estimate_interaural_cues
from utils.miscellaneous import ready_for_plotting

def split_binaural_signals_by_listener(binaural_signals, n_listeners):

    if binaural_signals.ndim == 4:
        return [binaural_signals[0, listener_idx][None, ...] for listener_idx in range(n_listeners)]

    if binaural_signals.ndim != 3:
        raise ValueError(f"Expected binaural signals with 3 or 4 dimensions, got shape {tuple(binaural_signals.shape)}")

    n_flat = binaural_signals.shape[0]
    if n_flat == n_listeners:
        return [binaural_signals[listener_idx][None, ...] for listener_idx in range(n_listeners)]

    if n_flat % n_listeners != 0:
        raise ValueError(
            f"Cannot split {n_flat} binaural signals across {n_listeners} listeners"
        )

    binaural_signals = binaural_signals.reshape(n_flat // n_listeners, n_listeners, 2, -1)
    return [binaural_signals[0, listener_idx][None, ...] for listener_idx in range(n_listeners)]


def _itd_ild_scatter_on_axes(ax, binaural_signals, sofaile, target_angle, gtfb, lag_values, tau_iacc, title):

    idx = sofa_find(target_angle, sofaile)
    hrir = get_IRs(idx, sofaile)

    estimated_interaural_cues = estimate_interaural_cues(binaural_signals, gtfb, lag_values, tau_iacc)
    target_interaural_cues = estimate_interaural_cues(hrir, gtfb, lag_values, tau_iacc)

    estimated_itd = ready_for_plotting(estimated_interaural_cues['itd']).ravel()
    estimated_ild = ready_for_plotting(estimated_interaural_cues['ild']).ravel()
    target_itd = ready_for_plotting(target_interaural_cues['itd']).ravel()
    target_ild = ready_for_plotting(target_interaural_cues['ild']).ravel()

    pair_indices = np.arange(estimated_itd.size)

    estimated_scatter = ax.scatter(
        estimated_itd, estimated_ild,
        c=pair_indices, cmap='viridis', marker='o', label='Estimated',
    )
    ax.scatter(
        target_itd, target_ild,
        c=pair_indices, cmap='viridis', marker='x', label='Target',
    )
    ax.legend()
    ax.set_xlabel('ITD [ms]')
    ax.set_ylabel('ILD [dB]')
    ax.grid(True)
    ax.set_xlim(-0.001, 0.001)
    ax.set_ylim(-20, 20)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1000:g}'))
    ax.set_title(title)
    return estimated_scatter


def _add_critical_band_colorbar(fig, scatter, axes, label='Critical band index'):
    fig.canvas.draw()
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    bottom = min(ax.get_position().y0 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    cbar_ax = fig.add_axes([right + 0.02, bottom, 0.02, top - bottom])
    fig.colorbar(scatter, cax=cbar_ax, label=label)


def itd_ild_scatter_plot(binaural_signals, sofaile, target_angle, gtfb, lag_values, tau_iacc, n_listeners=1, listener_labels=None):

    listener_binaural_signals = split_binaural_signals_by_listener(binaural_signals, n_listeners)
    if listener_labels is None:
        listener_labels = [f"VL {i}" for i in range(n_listeners)]

    if n_listeners == 1:
        fig, ax = plt.subplots()
        estimated_scatter = _itd_ild_scatter_on_axes(
            ax, listener_binaural_signals[0], sofaile, target_angle, gtfb, lag_values, tau_iacc,
            f'ITD-ILD Scatter Plot for {target_angle}°',
        )
        fig.tight_layout()
        _add_critical_band_colorbar(fig, estimated_scatter, ax)
        return

    n_cols = int(np.ceil(np.sqrt(n_listeners)))
    n_rows = int(np.ceil(n_listeners / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.ravel()

    estimated_scatter = None
    for listener_idx, listener_binaural in enumerate(listener_binaural_signals):
        estimated_scatter = _itd_ild_scatter_on_axes(
            axes[listener_idx], listener_binaural, sofaile, target_angle, gtfb, lag_values, tau_iacc,
            f'{listener_labels[listener_idx]} — {target_angle}°',
        )

    for j in range(n_listeners, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'ITD-ILD Scatter Plot for {target_angle}°')
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    _add_critical_band_colorbar(fig, estimated_scatter, axes[:n_listeners])


