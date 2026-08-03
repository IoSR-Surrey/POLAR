import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from utils.miscellaneous import ready_for_plotting, target_angle_for_listeners
from logger.saving import task_grad_pairwise_cosine_similarity_stats
from plotting.XY_plots import (time_domain_filters_plot,
                     frequency_domain_filters_plot,
                     raw_grad_cosine_to_reference_plot,
                     plot_loss_and_estimates,
                     loss_over_epochs_plot,
                     loss_grad_norms_over_epochs_plot,
                     gradnorm_task_weights_over_epochs_plot,
                     cagrad_dual_weights_over_epochs_plot,
                     cagrad_task_coeffs_over_epochs_plot,
                     task_grad_pairwise_cosine_similarity_plot,
                     ear_signals_magnitude_phase_plot,
                     iacc_over_epochs_plot)
from plotting.scatter_plots import itd_ild_scatter_plot, itd_ild_scatter_evolution_gif, cue_matching_itd_ild_scatter_plot
from plotting.spatial_plots import localisation_arrows_plot
from plotting.evaluation import compute_white_noise_plot_outputs


def ALL_PLOTTING_FUNCTIONS(saving_manager, show_plots=False):

    with open("config/hyperpameters.json", "r") as f:
        config = json.load(f)
        fs = config["env"]["fs"]

    n_listeners, listener_labels = get_listener_plot_info(saving_manager)

    white_noise_plot_outputs = compute_white_noise_plot_outputs(saving_manager)
    if white_noise_plot_outputs is not None:
        if "binaural_signals" in white_noise_plot_outputs:
            saving_manager.saved_data["binaural_signals"] = [white_noise_plot_outputs["binaural_signals"]]
        if "localisation_angles" in white_noise_plot_outputs:
            saving_manager.saved_data["localisation_angles"] = [white_noise_plot_outputs["localisation_angles"]]
        if "localisation_uncertainty" in white_noise_plot_outputs:
            saving_manager.saved_data["localisation_uncertainty"] = [white_noise_plot_outputs["localisation_uncertainty"]]
        if "lsd" in white_noise_plot_outputs:
            saving_manager.saved_data["lsd"] = [white_noise_plot_outputs["lsd"]]
        if "llado2025_grad_surgery_cues" in white_noise_plot_outputs:
            saving_manager.saved_data["llado2025_grad_surgery_cues"][-1] = white_noise_plot_outputs["llado2025_grad_surgery_cues"]

    if "spk_filters" in saving_manager.saved_data:
        spk_filters = saving_manager.saved_data["spk_filters"][0]
        time_domain_filters_plot(spk_filters, fs=fs)
        saving_manager.save_new_plots("time_domain_filters")
        frequency_domain_filters_plot(spk_filters, fs=fs)
        saving_manager.save_new_plots("frequency_domain_filters")
    
    if "llado2025_grad_surgery_cues" in saving_manager.saved_data:
        final_cues = saving_manager.saved_data["llado2025_grad_surgery_cues"][-1]
        localisation_angles = (
            saving_manager.saved_data["localisation_angles"][0]
            if "localisation_angles" in saving_manager.saved_data
            else None
        )
        if "target_position" in saving_manager.saved_data and "lstn_pos" in saving_manager.saved_data:
            target_angle = ready_for_plotting(
                target_angle_for_listeners(
                    saving_manager.saved_data["lstn_pos"][0],
                    saving_manager.saved_data["target_position"][0],
                )
            )
        else:
            target_angle = saving_manager.saved_data["target_angle"][0]
        itd_ild_scatter_plot(
            final_cues["itd"],
            final_cues["ild"],
            saving_manager.saved_data["sofaile"][0],
            target_angle,
            saving_manager.saved_data["gtfb"][0],
            saving_manager.saved_data["lag_values"][0],
            saving_manager.saved_data["tau_iacc"][0],
            n_listeners=n_listeners,
            listener_labels=listener_labels,
            localisation_angles=localisation_angles,
        )
        saving_manager.save_new_plots("itd_ild_scatter")
        if len(saving_manager.saved_data["llado2025_grad_surgery_cues"]) > 1:
            itd_ild_scatter_evolution_gif(saving_manager, n_listeners=n_listeners, listener_labels=listener_labels)
            saving_manager.save_new_plots("itd_ild_scatter_evolution")

    if "CueMatchingModel_cues" in saving_manager.saved_data and "target_cues" in saving_manager.saved_data:
        final_cues = saving_manager.saved_data["CueMatchingModel_cues"][-1]
        target_cues = saving_manager.saved_data["target_cues"][0]
        cue_matching_itd_ild_scatter_plot(
            final_cues["itd"],
            final_cues["ild"],
            target_cues,
            n_listeners=n_listeners,
            listener_labels=listener_labels,
        )
        saving_manager.save_new_plots("cue_matching_itd_ild_scatter")

    if "loss_dict" in saving_manager.saved_data:
        loss_over_epochs_plot(saving_manager.saved_data["loss_dict"], "Loss over epochs")
        saving_manager.save_new_plots("loss_over_epochs")

    if "loss_dict" in saving_manager.saved_data and "modelTrainer.estimates" in saving_manager.saved_data:
        plot_loss_and_estimates(
            saving_manager.saved_data["loss_dict"],
            saving_manager.saved_data["modelTrainer.estimates"],
            "Loss and Estimates",
            n_listeners=n_listeners,
            listener_labels=listener_labels,
        )
        saving_manager.save_new_plots("loss_and_estimates")
        loss_names = [f"loss_and_estimates_{key}" for key in saving_manager.saved_data["loss_dict"][0].keys() if key != "total"]
        if "total" in saving_manager.saved_data["loss_dict"][0]:
            loss_names.append("loss_and_estimates_total")
        saving_manager.save_new_plots("loss_and_estimates", names=loss_names)

    # Which MTL diagnostics to plot is inferred from the saved keys (the run's
    # optimizer), not re-read from config: GradNorm saves per-task "effective_grads"
    # (dict per epoch); CAGrad saves a single combined "effective_grad" per epoch.
    if "raw_grads" in saving_manager.saved_data and "effective_grads" in saving_manager.saved_data:
        task_names = list(saving_manager.saved_data["raw_grads"][0].keys())
        raw_grad_norms = [
            {key: torch.norm(epoch_grads[key]) for key in epoch_grads}
            for epoch_grads in saving_manager.saved_data["raw_grads"]
        ]
        effective_grad_norms = [
            {key: torch.norm(epoch_grads[key]) for key in epoch_grads}
            for epoch_grads in saving_manager.saved_data["effective_grads"]
        ]
        loss_grad_norms_over_epochs_plot(raw_grad_norms, "Per-task raw gradient norm", norm_kind="Raw")
        saving_manager.save_new_plots("raw_grad_norms")
        loss_grad_norms_over_epochs_plot(effective_grad_norms, "Per-task effective gradient norm", norm_kind="Effective")
        saving_manager.save_new_plots("effective_grad_norms")

        if "gradnorm_task_weights" in saving_manager.saved_data:
            gradnorm_task_weights_over_epochs_plot(
                saving_manager.saved_data["gradnorm_task_weights"], task_names, "GradNorm"
            )
            saving_manager.save_new_plots("gradnorm_task_weights")

        task_grad_pairwise_cosine_similarity_plot(
            **task_grad_pairwise_cosine_similarity_stats(saving_manager.saved_data["effective_grads"]),
            title="Task Gradients on Filter Taps",
            pairwise_only=True,
        )
        saving_manager.save_new_plots("task_grad_pairwise_cosine_similarity")

    # CAGrad and GradNorm+CAGrad both save a single combined "effective_grad" per epoch.
    if "raw_grads" in saving_manager.saved_data and "effective_grad" in saving_manager.saved_data:
        task_names = list(saving_manager.saved_data["raw_grads"][0].keys())
        raw_grad_norms = [
            {key: torch.norm(epoch_grads[key]) for key in epoch_grads}
            for epoch_grads in saving_manager.saved_data["raw_grads"]
        ]
        loss_grad_norms_over_epochs_plot(raw_grad_norms, "Per-task raw gradient norm", norm_kind="Raw")
        saving_manager.save_new_plots("raw_grad_norms")

        if "gradnorm_task_weights" in saving_manager.saved_data:
            gradnorm_task_weights_over_epochs_plot(
                saving_manager.saved_data["gradnorm_task_weights"], task_names, "GradNorm"
            )
            saving_manager.save_new_plots("gradnorm_task_weights")

        if "cagrad_dual_weights" in saving_manager.saved_data:
            cagrad_dual_weights_over_epochs_plot(
                saving_manager.saved_data["cagrad_dual_weights"], task_names,
                "CAGrad task mix (weights sum to 1)",
            )
            saving_manager.save_new_plots("cagrad_dual_weights")

        if "cagrad_task_coeffs" in saving_manager.saved_data:
            cagrad_task_coeffs_over_epochs_plot(
                saving_manager.saved_data["cagrad_task_coeffs"], task_names,
                "CAGrad gradient multiplier per task",
            )
            saving_manager.save_new_plots("cagrad_task_coeffs")

        # "Plain" optimisation direction: the summed raw gradient (grad of the total
        # loss), i.e. what a vanilla backward pass would step along.
        summed_raw_grad = [
            torch.stack([epoch_grads[key] for key in epoch_grads]).sum(dim=0)
            for epoch_grads in saving_manager.saved_data["raw_grads"]
        ]
        # How each task's raw gradient aligns with the plain (summed) direction.
        raw_grad_cosine_to_reference_plot(
            saving_manager.saved_data["raw_grads"], summed_raw_grad, task_names,
            "Task gradient vs plain (summed) direction",
        )
        saving_manager.save_new_plots("raw_vs_plain_cosine_similarity")
        # How each task's raw gradient aligns with the CAGrad combined direction.
        raw_grad_cosine_to_reference_plot(
            saving_manager.saved_data["raw_grads"], saving_manager.saved_data["effective_grad"], task_names,
            "Task gradient vs CAGrad update direction",
        )
        saving_manager.save_new_plots("raw_vs_cagrad_cosine_similarity")


    if all(key in saving_manager.saved_data for key in ("lstn_pos", "spk_pos", "localisation_angles")):
        lstn_pos = saving_manager.saved_data["lstn_pos"][0]
        if "target_position" in saving_manager.saved_data:
            target_angle = ready_for_plotting(
                target_angle_for_listeners(lstn_pos, saving_manager.saved_data["target_position"][0])
            )
        else:
            target_angle = saving_manager.saved_data["target_angle"][0]
        if "localisation_uncertainty" in saving_manager.saved_data:
            uncertainty = saving_manager.saved_data["localisation_uncertainty"][0]
        else:
            uncertainty = torch.zeros_like(saving_manager.saved_data["localisation_angles"][0])
        if "lsd" in saving_manager.saved_data:
            lsd = saving_manager.saved_data["lsd"][0]
        else:
            lsd = torch.full_like(saving_manager.saved_data["localisation_angles"][0], float("inf"))        
        localisation_arrows_plot(
            lstn_pos,
            saving_manager.saved_data["spk_pos"][0],
            saving_manager.saved_data["localisation_angles"][0],
            uncertainty,
            lsd=lsd,
            target_angle=target_angle,
            listener_labels=listener_labels,
            
        )
        saving_manager.save_new_plots("localisation_arrows")

    if "binaural_signals" in saving_manager.saved_data:
        ear_signals_magnitude_phase_plot(
            saving_manager.saved_data["binaural_signals"][0],
            fs=fs,
            listener_labels=listener_labels,
            n_listeners=n_listeners,
        )
        saving_manager.save_new_plots("ear_signals_magnitude_phase")

    if "iacc" in saving_manager.saved_data:
        iacc_over_epochs_plot(
            saving_manager.saved_data["iacc"],
            listener_labels=listener_labels,
            n_listeners=n_listeners,
        )
        saving_manager.save_new_plots("iacc_over_epochs")

    if show_plots:
        create_close_button()
        plt.show()
    else:
        plt.close("all")


def listener_labels_from_lstn_pos(lstn_pos):
    lstn_pos = ready_for_plotting(lstn_pos)
    return [f"VL {i} ({pos[0]:.2g}, {pos[1]:.2g})" for i, pos in enumerate(lstn_pos)]


def get_listener_plot_info(saving_manager):
    if "lstn_pos" not in saving_manager.saved_data:
        return 1, None

    lstn_pos = saving_manager.saved_data["lstn_pos"][0]
    n_listeners = len(ready_for_plotting(lstn_pos))
    return n_listeners, listener_labels_from_lstn_pos(lstn_pos)


def close_all(_event):
    plt.close("all")


def create_close_button():
    buttons = []
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.subplots_adjust(bottom=0.12)
        ax_button = fig.add_axes([0.81, 0.02, 0.15, 0.07])
        ax_button.set_zorder(10)
        button = Button(ax_button, "Close All Figures")
        button.on_clicked(close_all)
        fig._close_all_button = button
        buttons.append(button)
    return buttons
