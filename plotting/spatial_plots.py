import matplotlib.pyplot as plt
import numpy as np

from utils.miscellaneous import ready_for_plotting


def localisation_arrows_plot(lstn_pos, spk_pos, localisation_angles, uncertainty, target_angle=None, listener_labels=None, max_arrow_length=1.2):

    lstn_pos = ready_for_plotting(lstn_pos)
    spk_pos = ready_for_plotting(spk_pos)
    localisation_angles = np.squeeze(ready_for_plotting(localisation_angles))
    uncertainty = np.squeeze(ready_for_plotting(uncertainty))

    fig, ax = plt.subplots()

    ax.scatter(spk_pos[:, 0], spk_pos[:, 1], c='black', marker='x', label='Speaker', zorder=3)
    ax.scatter(lstn_pos[:, 0], lstn_pos[:, 1], c='blue', marker='o', label='Listener', zorder=3)

    arrow_head_width = 0.12
    arrow_head_len = 0.14
    arrow_width = 2

    arrow_tips_x = []
    arrow_tips_y = []
    for x, y, angle, unc in zip(lstn_pos[:, 0], lstn_pos[:, 1], localisation_angles, uncertainty):
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

    if target_angle is not None:
        x_lin = np.linspace(-1, 3, 200)
        y_lin = x_lin * np.tan(np.deg2rad(target_angle))
        ax.plot(x_lin, y_lin, color='black', linestyle='dashed', lw=1, zorder=1, label='Target angle')

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
