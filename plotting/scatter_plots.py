import matplotlib.pyplot as plt
import numpy as np

from utils.sofa_functions import sofa_find, get_IRs

from utils.peripheral_processing import estimate_interaural_cues
from utils.miscellaneous import ready_for_plotting

def itd_ild_scatter_plot(binaural_signals, sofaile, target_angle,gtfb, lag_values, tau_iacc):

    idx = sofa_find(target_angle, sofaile)
    hrir = get_IRs(idx, sofaile)

    estimated_interaural_cues = estimate_interaural_cues(binaural_signals, gtfb, lag_values, tau_iacc)
    target_interaural_cues = estimate_interaural_cues(hrir, gtfb, lag_values, tau_iacc)

    estimated_itd = ready_for_plotting(estimated_interaural_cues['itd']).ravel()    
    estimated_ild = ready_for_plotting(estimated_interaural_cues['ild']).ravel()
    target_itd = ready_for_plotting(target_interaural_cues['itd']).ravel()
    target_ild = ready_for_plotting(target_interaural_cues['ild']).ravel()

    pair_indices = np.arange(estimated_itd.size)

    fig, ax = plt.subplots()

    estimated_scatter = ax.scatter(
        estimated_itd, estimated_ild,
        c=pair_indices, cmap='viridis', marker='o', label='Estimated',
    )
    ax.scatter(
        target_itd, target_ild,
        c=pair_indices, cmap='viridis', marker='x', label='Target',
    )
    fig.colorbar(estimated_scatter, ax=ax, label='Critical band index')
    ax.legend()
    ax.set_xlabel('ITD [ms]')
    ax.set_ylabel('ILD [dB]')
    ax.grid(True)
    ax.set_xlim(-0.001, 0.001)
    ax.set_ylim(-20, 20)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 1000:g}'))
    ax.set_title(f'ITD-ILD Scatter Plot for {target_angle}°')
    fig.tight_layout()


