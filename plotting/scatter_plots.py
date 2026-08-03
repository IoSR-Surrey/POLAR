import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from utils.sofa_functions import sofa_find, get_IRs

from utils.peripheral_processing import estimate_interaural_cues
from utils.miscellaneous import ready_for_plotting, target_angle_for_listeners

def split_interaural_cues_by_listener(itd, ild, n_listeners):

    itd = ready_for_plotting(itd)
    ild = ready_for_plotting(ild)

    if itd.ndim == 1:
        return [(itd, ild)]

    n_flat = itd.shape[0]
    if n_flat == n_listeners:
        return [(itd[listener_idx], ild[listener_idx]) for listener_idx in range(n_listeners)]

    if n_flat % n_listeners != 0:
        raise ValueError(
            f"Cannot split {n_flat} interaural cue rows across {n_listeners} listeners"
        )

    itd = itd.reshape(n_flat // n_listeners, n_listeners, -1)
    ild = ild.reshape(n_flat // n_listeners, n_listeners, -1)
    return [
        (itd[:, listener_idx].mean(axis=0), ild[:, listener_idx].mean(axis=0))
        for listener_idx in range(n_listeners)
    ]


def _compute_target_itd_ild(sofaile, target_angle, gtfb, lag_values, tau_iacc):

    idx = sofa_find(target_angle, sofaile)
    hrir = get_IRs(idx, sofaile)
    gtfb = gtfb.to(device=hrir.device, dtype=hrir.dtype)
    lag_values = lag_values.to(device=hrir.device, dtype=hrir.dtype)
    target_interaural_cues = estimate_interaural_cues(hrir, gtfb, lag_values, tau_iacc)
    return (
        ready_for_plotting(target_interaural_cues['itd']).ravel(),
        ready_for_plotting(target_interaural_cues['ild']).ravel(),
    )


def _style_itd_ild_axes(ax):

    ax.set_xlabel('ITD [ms]')
    ax.set_ylabel('ILD [dB]')
    ax.grid(True)
    ax.set_xlim(-0.001, 0.001)
    ax.set_ylim(-20, 20)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1000:g}'))


def _scatter_itd_ild_from_arrays(ax, estimated_itd, estimated_ild, target_itd, target_ild, title, draw_target=True):

    pair_indices = np.arange(estimated_itd.size)

    estimated_scatter = ax.scatter(
        estimated_itd, estimated_ild,
        c=pair_indices, cmap='viridis', marker='o', label='Estimated',
    )
    if draw_target:
        ax.scatter(
            target_itd, target_ild,
            c=pair_indices, cmap='viridis', marker='x', label='Target',
        )
        ax.legend()
    _style_itd_ild_axes(ax)
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


def _as_listener_angles(angles, n_listeners):
    angles = np.atleast_1d(np.squeeze(ready_for_plotting(angles))).astype(float)
    if angles.size == 1:
        return np.repeat(angles, n_listeners)
    if angles.size != n_listeners:
        raise ValueError(f"Expected {n_listeners} listener angles, got {angles.size}")
    return angles


def _format_scatter_title(prefix, target_angle, perceived_angle=None):
    target_str = f'{float(target_angle):.1f}°'
    if perceived_angle is None:
        return f'{prefix}{target_str}'
    return f'{prefix}{target_str} (perceived {float(perceived_angle):.1f}°)'


def itd_ild_scatter_plot(estimated_itd, estimated_ild, sofaile, target_angle, gtfb, lag_values, tau_iacc, n_listeners=1, listener_labels=None, localisation_angles=None):

    # Estimated points come from the saved final-epoch interaural cues (the same
    # grad-surgery pipeline used for the evolution GIF) so the two views agree.
    listener_cues = split_interaural_cues_by_listener(estimated_itd, estimated_ild, n_listeners)
    if listener_labels is None:
        listener_labels = [f"VL {i}" for i in range(n_listeners)]

    target_angles = _as_listener_angles(target_angle, n_listeners)

    if localisation_angles is not None:
        localisation_angles = np.atleast_1d(np.squeeze(ready_for_plotting(localisation_angles)))
        if localisation_angles.size == 1 and n_listeners > 1:
            localisation_angles = np.repeat(localisation_angles, n_listeners)
        elif localisation_angles.size != n_listeners:
            localisation_angles = None

    if n_listeners == 1:
        fig, ax = plt.subplots()
        cues_itd, cues_ild = listener_cues[0]
        target_itd, target_ild = _compute_target_itd_ild(sofaile, target_angles[0], gtfb, lag_values, tau_iacc)
        perceived = None if localisation_angles is None else localisation_angles[0]
        estimated_scatter = _scatter_itd_ild_from_arrays(
            ax, np.ravel(cues_itd), np.ravel(cues_ild), target_itd, target_ild,
            _format_scatter_title('ITD-ILD Scatter Plot for ', target_angles[0], perceived),
        )
        fig.tight_layout()
        _add_critical_band_colorbar(fig, estimated_scatter, ax)
        return

    n_cols = int(np.ceil(np.sqrt(n_listeners)))
    n_rows = int(np.ceil(n_listeners / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.ravel()

    estimated_scatter = None
    for listener_idx, (cues_itd, cues_ild) in enumerate(listener_cues):
        target_itd, target_ild = _compute_target_itd_ild(
            sofaile, target_angles[listener_idx], gtfb, lag_values, tau_iacc,
        )
        perceived = None if localisation_angles is None else localisation_angles[listener_idx]
        estimated_scatter = _scatter_itd_ild_from_arrays(
            axes[listener_idx], np.ravel(cues_itd), np.ravel(cues_ild), target_itd, target_ild,
            _format_scatter_title(f'{listener_labels[listener_idx]} — ', target_angles[listener_idx], perceived),
        )

    for j in range(n_listeners, len(axes)):
        axes[j].axis('off')

    fig.suptitle('ITD-ILD Scatter Plot')
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    _add_critical_band_colorbar(fig, estimated_scatter, axes[:n_listeners])


def cue_matching_itd_ild_scatter_plot(estimated_itd, estimated_ild, target_cues, n_listeners=1, listener_labels=None):
    """ITD/ILD scatter plot for CueMatchingModel with per-listener target cues."""

    listener_cues = split_interaural_cues_by_listener(estimated_itd, estimated_ild, n_listeners)
    if listener_labels is None:
        listener_labels = [f"VL {i}" for i in range(n_listeners)]

    # target_cues shape: (N_listeners, 2, N_banks) where dim 1: 0=ITD, 1=ILD
    target_cues = ready_for_plotting(target_cues)

    if n_listeners == 1:
        fig, ax = plt.subplots()
        cues_itd, cues_ild = listener_cues[0]
        target_itd = target_cues[0, 0, :].ravel()
        target_ild = target_cues[0, 1, :].ravel()
        estimated_scatter = _scatter_itd_ild_from_arrays(
            ax, np.ravel(cues_itd), np.ravel(cues_ild), target_itd, target_ild,
            f'ITD-ILD Scatter Plot (Cue Matching)',
        )
        fig.tight_layout()
        _add_critical_band_colorbar(fig, estimated_scatter, ax)
        return

    n_cols = int(np.ceil(np.sqrt(n_listeners)))
    n_rows = int(np.ceil(n_listeners / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.ravel()

    estimated_scatter = None
    for listener_idx, (cues_itd, cues_ild) in enumerate(listener_cues):
        target_itd = target_cues[listener_idx, 0, :].ravel()
        target_ild = target_cues[listener_idx, 1, :].ravel()
        estimated_scatter = _scatter_itd_ild_from_arrays(
            axes[listener_idx], np.ravel(cues_itd), np.ravel(cues_ild), target_itd, target_ild,
            f'{listener_labels[listener_idx]}',
        )

    for j in range(n_listeners, len(axes)):
        axes[j].axis('off')

    fig.suptitle('ITD-ILD Scatter Plot (Cue Matching)')
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    _add_critical_band_colorbar(fig, estimated_scatter, axes[:n_listeners])


def itd_ild_scatter_evolution_gif(saving_manager, n_listeners, listener_labels=None, frame_stride=1, fps=16):

    interaural_cues_series = saving_manager.saved_data["llado2025_grad_surgery_cues"]
    if len(interaural_cues_series) <= 1:
        return

    sofaile = saving_manager.saved_data["sofaile"][0]
    gtfb = saving_manager.saved_data["gtfb"][0]
    lag_values = saving_manager.saved_data["lag_values"][0]
    tau_iacc = saving_manager.saved_data["tau_iacc"][0]
    if "target_position" in saving_manager.saved_data and "lstn_pos" in saving_manager.saved_data:
        target_angle = target_angle_for_listeners(
            saving_manager.saved_data["lstn_pos"][0],
            saving_manager.saved_data["target_position"][0],
        )
    else:
        target_angle = saving_manager.saved_data["target_angle"][0]
    target_angles = _as_listener_angles(target_angle, n_listeners)

    if listener_labels is None:
        listener_labels = [f"VL {i}" for i in range(n_listeners)]

    if n_listeners == 1:
        fig, axes = plt.subplots()
        axes = [axes]
    else:
        n_cols = int(np.ceil(np.sqrt(n_listeners)))
        n_rows = int(np.ceil(n_listeners / n_cols))
        fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes_grid.ravel()
        for j in range(n_listeners, len(axes)):
            axes[j].axis('off')

    estimated_scatters = []
    for listener_idx in range(n_listeners):
        ax = axes[listener_idx]
        target_itd, target_ild = _compute_target_itd_ild(
            sofaile, target_angles[listener_idx], gtfb, lag_values, tau_iacc,
        )
        pair_indices = np.arange(target_itd.size)
        ax.scatter(
            target_itd, target_ild,
            c=pair_indices, cmap='viridis', marker='x', label='Target',
        )
        cues_itd, cues_ild = split_interaural_cues_by_listener(
            interaural_cues_series[0]["itd"], interaural_cues_series[0]["ild"], n_listeners,
        )[listener_idx]
        estimated_scatter = ax.scatter(
            cues_itd, cues_ild,
            c=pair_indices, cmap='viridis', marker='o', label='Estimated',
        )
        ax.legend()
        _style_itd_ild_axes(ax)
        ax.set_title(f'{listener_labels[listener_idx]} — {target_angles[listener_idx]:.1f}°')
        estimated_scatters.append(estimated_scatter)

    title = fig.suptitle(f'ITD-ILD Evolution — Epoch 0')
    fig.tight_layout(rect=[0, 0, 0.95, 0.96])
    _add_critical_band_colorbar(fig, estimated_scatters[0], axes[:n_listeners])

    gif_path = saving_manager.output_dir / "plots" / "itd_ild_scatter_evolution.gif"
    gif_path.parent.mkdir(exist_ok=True)

    frame_indices = list(range(0, len(interaural_cues_series), frame_stride))
    writer = animation.PillowWriter(fps=fps)
    with writer.saving(fig, str(gif_path), dpi=120):
        for epoch in frame_indices:
            cues = interaural_cues_series[epoch]
            for listener_idx, estimated_scatter in enumerate(estimated_scatters):
                cues_itd, cues_ild = split_interaural_cues_by_listener(
                    cues["itd"], cues["ild"], n_listeners,
                )[listener_idx]
                estimated_scatter.set_offsets(np.c_[cues_itd, cues_ild])
            title.set_text(f'ITD-ILD Evolution — Epoch {epoch}')
            writer.grab_frame()

    plt.close(fig)
