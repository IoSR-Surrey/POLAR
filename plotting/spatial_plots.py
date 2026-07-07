import matplotlib.pyplot as plt
import numpy as np

from utils.miscellaneous import ready_for_plotting


def localisation_arrows_plot(lstn_pos, spk_pos, localisation_angles, uncertainty, target_angle=None, listener_labels=None, max_arrow_length=1.2):

    lstn_pos = ready_for_plotting(lstn_pos)
    spk_pos = ready_for_plotting(spk_pos)
    localisation_angles = np.squeeze(ready_for_plotting(localisation_angles))
    uncertainty = np.squeeze(ready_for_plotting(uncertainty))

    fig, ax = plt.subplots()

    localisation_angles = np.atleast_1d(localisation_angles)
    uncertainty = np.atleast_1d(uncertainty)
    n_listeners = lstn_pos.shape[0]

    ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x', label='Speaker', zorder=3)
    ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', marker='o', label='Listener', zorder=3)

    arrow_head_width = 0.12
    arrow_head_len = 0.14
    arrow_width = 2

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
        ax.arrow(
            float(x), float(y), float(dx), float(dy),
            head_width=arrow_head_width, head_length=arrow_head_len,
            fc='blue', ec='blue', linewidth=arrow_width, zorder=2,
        )
        arrow_tips_x.append(x + dx)
        arrow_tips_y.append(y + dy)

        if listener_labels is not None and i < len(listener_labels):
            ax.annotate(listener_labels[i], (float(x), float(y)), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color='blue', zorder=4)

        if target_angles is not None:
            t_dx = max_arrow_length * np.cos(np.deg2rad(target_angles[i]))
            t_dy = max_arrow_length * np.sin(np.deg2rad(target_angles[i]))
            ax.plot([float(x), float(x + t_dx)], [float(y), float(y + t_dy)],
                    color='black', linestyle='dashed', lw=1, zorder=1,
                    label='Target angle' if i == 0 else None)

    all_x = np.concatenate([lstn_pos[:, 0], spk_pos[:, 0], arrow_tips_x])
    all_y = np.concatenate([lstn_pos[:, 1], spk_pos[:, 1], arrow_tips_y])
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
    fig.tight_layout()
