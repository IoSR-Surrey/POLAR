import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

from utils.miscellaneous import ready_for_plotting
from plotting.batch_utils import mean_over_stimulus_batch

LSD_LOWPASS_REFERENCE_DB = 14.31


def localisation_arrows_plot(lstn_pos, spk_pos, localisation_angles, uncertainty, lsd, target_angle=None, target_pos=None, listener_labels=None, max_arrow_length=1.2, lsd_vmax=None, arrow_head_width=0.12, arrow_head_length=0.14, arrow_linewidth=2, trained_listener_mask=None):

    lstn_pos = ready_for_plotting(lstn_pos)
    spk_pos = ready_for_plotting(spk_pos)
    localisation_angles = np.squeeze(ready_for_plotting(localisation_angles))
    uncertainty = np.squeeze(ready_for_plotting(uncertainty))

    fig, ax = plt.subplots()

    n_listeners = lstn_pos.shape[0]
    if localisation_angles.size != n_listeners:
        localisation_angles = mean_over_stimulus_batch(localisation_angles, n_listeners)
    if uncertainty.size != n_listeners:
        uncertainty = mean_over_stimulus_batch(uncertainty, n_listeners)

    localisation_angles = np.atleast_1d(localisation_angles)
    uncertainty = np.atleast_1d(uncertainty)

    ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x', label='Speaker', zorder=3)
    if trained_listener_mask is None:
        ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', marker='o', label='Listener', zorder=3)
    else:
        trained = np.asarray(trained_listener_mask, dtype=bool).reshape(-1)
        other = ~trained
        if np.any(other):
            ax.scatter(
                lstn_pos[other, 0], lstn_pos[other, 1],
                c='#D55E00', marker='o', label='Unseen listener', zorder=3,
            )
        if np.any(trained):
            ax.scatter(
                lstn_pos[trained, 0], lstn_pos[trained, 1],
                c='#0072B2', marker='o', label='Optimised listener', zorder=4,
            )

    arrow_head_len = arrow_head_length
    arrow_width = arrow_linewidth

    lsd = np.squeeze(ready_for_plotting(lsd))
    if lsd.size != n_listeners:
        lsd = mean_over_stimulus_batch(lsd, n_listeners)
    lsd = np.atleast_1d(lsd)
    has_lsd = not np.all(np.isinf(lsd))

    if has_lsd:
        # Headroom above the 7 kHz LP marker so it does not sit on the top tick.
        lsd_vmax = float(lsd_vmax if lsd_vmax is not None else max(16.0, lsd.max(), 1e-3))
        lsd_norm = colors.Normalize(vmin=0.0, vmax=lsd_vmax)
        lsd_cmap = colors.LinearSegmentedColormap.from_list(
            "green_red", [(0, 1, 0), (0.8, 0.4, 0), (0.8, 0, 0)]
        )
        lsd_sm = cm.ScalarMappable(cmap=lsd_cmap, norm=lsd_norm)
        lsd_sm.set_array([])

    if target_pos is not None:
        target_pos = np.asarray(ready_for_plotting(target_pos), dtype=float).reshape(-1)
        if target_pos.size != 2:
            raise ValueError("target_pos must be a 2D position.")
        ax.scatter(
            target_pos[0], target_pos[1], c='red', marker='^', s=70,
            label='Target', zorder=4,
        )

    # Broadcast target_angle to one azimuth per listener (per-listener target
    # direction, drawn as a dashed ray emanating from each listener).
    if target_angle is not None:
        target_angles = np.broadcast_to(np.atleast_1d(np.asarray(target_angle, dtype=float)), (n_listeners,))
    else:
        target_angles = None

    arrow_tips_x = []
    arrow_tips_y = []
    for i, (x, y, angle, unc) in enumerate(zip(lstn_pos[:, 0], lstn_pos[:, 1], localisation_angles, uncertainty)):
        length = (1.0 - unc) * max_arrow_length
        dx = length * np.cos(np.deg2rad(angle))
        dy = length * np.sin(np.deg2rad(angle))
        if has_lsd:
            arrow_color = lsd_cmap(lsd_norm(max(lsd[i], 0.0)))
        else:
            arrow_color = 'blue'
        ax.arrow(
            float(x), float(y), float(dx), float(dy),
            head_width=arrow_head_width, head_length=arrow_head_len,
            fc=arrow_color, ec=arrow_color, linewidth=arrow_width, zorder=2,
        )
        arrow_tips_x.append(x + dx)
        arrow_tips_y.append(y + dy)

        if listener_labels is not None and i < len(listener_labels):
            ax.annotate(listener_labels[i], (float(x), float(y)), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color='blue', zorder=4)

        if target_pos is not None:
            ax.plot(
                [float(x), float(target_pos[0])], [float(y), float(target_pos[1])],
                color='black', linestyle='dashed', lw=1.2, zorder=5,
                label='Target direction' if i == 0 else None,
            )
        elif target_angles is not None:
            t_dx = max_arrow_length * np.cos(np.deg2rad(target_angles[i]))
            t_dy = max_arrow_length * np.sin(np.deg2rad(target_angles[i]))
            ax.plot([float(x), float(x + t_dx)], [float(y), float(y + t_dy)],
                    color='black', linestyle='dashed', lw=1, zorder=1,
                    label='Target angle' if i == 0 else None)

    all_x = np.concatenate([lstn_pos[:, 0], spk_pos[:, 0], arrow_tips_x])
    all_y = np.concatenate([lstn_pos[:, 1], spk_pos[:, 1], arrow_tips_y])
    if target_pos is not None:
        all_x = np.concatenate([all_x, target_pos[:1]])
        all_y = np.concatenate([all_y, target_pos[1:2]])
    margin = 0.25
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.set_title('Estimated localisation')
    ax.grid(True, zorder=0)
    ax.legend()
    if has_lsd:
        # Extra pad so the 7 kHz LP label can sit left of the bar, clear of the ticks.
        cbar = fig.colorbar(lsd_sm, ax=ax, label='LSD [dB]', pad=0.14)
        if LSD_LOWPASS_REFERENCE_DB <= lsd_vmax:
            cbar.ax.axhline(LSD_LOWPASS_REFERENCE_DB, color='black', linewidth=0.8, linestyle='--')
            cbar.ax.annotate(
                '7 kHz LP',
                xy=(0.0, LSD_LOWPASS_REFERENCE_DB),
                xycoords=('axes fraction', 'data'),
                xytext=(-6, 0),
                textcoords='offset points',
                ha='right',
                va='center',
                fontsize=9,
                annotation_clip=False,
                clip_on=False,
            )
    fig.tight_layout()
