
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from utils.miscellaneous import ready_for_plotting
from plotting.XY_plots import (time_domain_filters_plot,
                     frequency_domain_filters_plot,
                     cosine_similarity_over_epochs_plot,
                     grad_norms_over_epochs_plot,
                     filter_tap_branch_grad_norms_over_epochs_plot,
                     filter_tap_branch_cosine_similarity_over_epochs_plot,
                     plot_loss_and_estimates)
from plotting.scatter_plots import itd_ild_scatter_plot


def listener_labels_from_lstn_pos(lstn_pos):
    lstn_pos = ready_for_plotting(lstn_pos)
    return [f"VL {i} ({pos[0]:.2g}, {pos[1]:.2g})" for i, pos in enumerate(lstn_pos)]


def get_listener_plot_info(saving_manager):
    if "lstn_pos" not in saving_manager.saved_data:
        return 1, None

    lstn_pos = saving_manager.saved_data["lstn_pos"][0]
    n_listeners = len(ready_for_plotting(lstn_pos))
    return n_listeners, listener_labels_from_lstn_pos(lstn_pos)


def ALL_PLOTTING_FUNCTIONS(saving_manager):

    spk_filters = saving_manager.saved_data["spk_filters"][0]
    binaural_signals = saving_manager.saved_data["binaural_signals"][0]
    sofaile = saving_manager.saved_data["sofaile"][0]
    target_angle = saving_manager.saved_data["target_angle"][0]
    gtfb = saving_manager.saved_data["gtfb"][0]
    lag_values = saving_manager.saved_data["lag_values"][0]
    tau_iacc = saving_manager.saved_data["tau_iacc"][0]
    n_listeners, listener_labels = get_listener_plot_info(saving_manager)

    time_domain_filters_plot(spk_filters)
    frequency_domain_filters_plot(spk_filters)
    itd_ild_scatter_plot(
        binaural_signals, sofaile, target_angle, gtfb, lag_values, tau_iacc,
        n_listeners=n_listeners, listener_labels=listener_labels,
    )
    grad_norms_over_epochs_plot(saving_manager.saved_data["rectified_signals_grad_itd"], saving_manager.saved_data["rectified_signals_grad_ild"], "Grad Norms Plot")
    filter_tap_branch_grad_norms_over_epochs_plot(saving_manager.saved_data["grad_itd_on_taps_norm"], saving_manager.saved_data["grad_ild_on_taps_norm"])
    filter_tap_branch_cosine_similarity_over_epochs_plot(saving_manager.saved_data["grad_itd_on_taps"], saving_manager.saved_data["grad_ild_on_taps"], "Filter Taps")
    plot_loss_and_estimates(
        saving_manager.saved_data["loss_dict"],
        saving_manager.saved_data["modelTrainer.estimates"],
        "Loss and Estimates",
        n_listeners=n_listeners,
        listener_labels=listener_labels,
    )
    _close_buttons = create_close_button()
    plt.show()



def close_all(_event):
    plt.close('all')


def create_close_button():
    buttons = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.subplots_adjust(bottom=0.12)
        ax_button = fig.add_axes([0.81, 0.02, 0.15, 0.07])
        ax_button.set_zorder(10)
        button = Button(ax_button, 'Close All Figures')
        button.on_clicked(close_all)
        buttons.append(button)
    return buttons
