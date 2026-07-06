import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.miscellaneous import ready_for_plotting, _flatten_tap_grads
from utils.dsp import cosine_similarity

def time_domain_filters_plot(filters):

    filters = ready_for_plotting(filters)
    if filters.ndim == 3:
        filters = filters[0]  # (batch, n_sources, n_samples) -> (n_sources, n_samples)
    elif filters.ndim == 4:
        filters = filters[-1][0]  # (epoch, batch, n_samples, n_channels) -> (n_samples, n_channels)

    n_filters = filters.shape[0]
    n_samples = filters.shape[1]

    grid_size = int(np.ceil(np.sqrt(n_filters)))
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10), squeeze=False)
    ax = ax.ravel()

    for i in range(n_filters):
        ax[i].plot(filters[i, :])
        ax[i].set_title(f"Filter {i+1}")
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Amplitude")
        ax[i].grid(True)

    # Hide any unused subplots
    for j in range(n_filters, grid_size * grid_size):
        ax[j].axis('off')

    plt.tight_layout()

def frequency_domain_filters_plot(filters, fs=48000, nfft = 2048):

    filters = ready_for_plotting(filters)

    if filters.ndim == 3:
        filters = filters[0]  # (batch, n_sources, n_samples) -> (n_sources, n_samples)
    elif filters.ndim == 4:
        filters = filters[-1][0]  # (epoch, batch, n_samples, n_channels) -> (n_samples, n_channels)
    
    n_filters = filters.shape[0]
    n_samples = filters.shape[1]
    
    filters = np.fft.rfft(filters, axis=1, n=nfft)
    filters = 20*np.log10(np.abs(filters) + 1e-10)
    frequency_axis = np.fft.rfftfreq(nfft, 1/fs)

    grid_size = int(np.ceil(np.sqrt(n_filters)))
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10), squeeze=False)
    ax = ax.ravel()

    for i in range(n_filters):
        ax[i].semilogx(frequency_axis, filters[i, :])
        ax[i].set_title(f"Filter {i+1}")
        ax[i].set_xlabel("Frequency (Hz)")
        ax[i].set_ylabel("Magnitude (dB)")
        ax[i].grid(True)

    # Hide any unused subplots
    for j in range(n_filters, grid_size * grid_size):
        ax[j].axis('off')

    plt.tight_layout()

def cosine_similarity_over_epochs_plot(a, b, title):

    cosine_similarity_over_epochs = []
    for i in range(len(a)):
        cosine_similarity_over_epochs.append(cosine_similarity(a[i], b[i]))
        
    cosine_similarity_over_epochs = ready_for_plotting(cosine_similarity_over_epochs)

    plt.figure()
    plt.plot(cosine_similarity_over_epochs)
    plt.title(f"Cosine Similarity Over Epochs Plot for {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()


def filter_tap_branch_cosine_similarity_over_epochs_plot(grad_itd_on_taps, grad_ild_on_taps, title):

    cosine_similarity_over_epochs = []
    for i in range(len(grad_itd_on_taps)):
        a = _flatten_tap_grads(grad_itd_on_taps[i])
        b = _flatten_tap_grads(grad_ild_on_taps[i])
        if a is None or b is None:
            cosine_similarity_over_epochs.append(float("nan"))
        else:
            cosine_similarity_over_epochs.append(cosine_similarity(a, b))

    cosine_similarity_over_epochs = ready_for_plotting(cosine_similarity_over_epochs)

    plt.figure()
    plt.plot(cosine_similarity_over_epochs)
    plt.title(f"Filter Tap Branch Cosine Similarity Over Epochs for {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()


def grad_norms_over_epochs_plot(grad_itd, grad_ild, title):


    grad_norm_itd = [torch.norm(grad_itd[i]) for i in range(len(grad_itd))]
    grad_norm_ild = [torch.norm(grad_ild[i]) for i in range(len(grad_ild))]

    grad_norm_itd = ready_for_plotting(grad_norm_itd)
    grad_norm_ild = ready_for_plotting(grad_norm_ild)

    plt.figure()
    plt.plot(grad_norm_itd, label="Grad Norm ITD")
    plt.plot(grad_norm_ild, label="Grad Norm ILD")
    plt.legend()
    plt.title(f"Grad Norms Over Epochs Plot for {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Grad Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def filter_tap_branch_grad_norms_over_epochs_plot(grad_norm_itd_on_taps, grad_norm_ild_on_taps):

    grad_norm_itd = ready_for_plotting(grad_norm_itd_on_taps)
    grad_norm_ild = ready_for_plotting(grad_norm_ild_on_taps)

    plt.figure()
    plt.plot(grad_norm_itd, label="Filter tap grad norm ITD")
    plt.plot(grad_norm_ild, label="Filter tap grad norm ILD")
    plt.legend()
    plt.title(f"Filter Tap Branch Grad Norms Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Grad norm")
    plt.grid(True)
    plt.tight_layout()


def plot_loss_and_estimates(loss_dict, estimates, title, n_listeners=None, listener_labels=None):

    n_epochs = len(loss_dict)
    keys = [key for key in loss_dict[0].keys() if key != "total"]

    for key in keys:
        losses = np.squeeze(ready_for_plotting([loss_dict[epoch][key] for epoch in range(n_epochs)]))
        model_estimates = np.asarray(ready_for_plotting([estimates[epoch][key] for epoch in range(n_epochs)]))

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
        plt.subplot(2, 1, 2)
        for i in range(n_series):
            plt.plot(np.squeeze(model_estimates[..., i]), label=series_labels[i])
        plt.title(f"{key} estimate over epochs for {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Estimate")
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


def loss_grad_norms_over_epochs_plot(loss_grad_norms, title):

    n_epochs = len(loss_grad_norms)
    keys = [key for key in loss_grad_norms[0].keys() if key != "total"]

    plt.figure()
    for key in keys:
        grad_norms = np.squeeze(ready_for_plotting([loss_grad_norms[epoch][key] for epoch in range(n_epochs)]))
        plt.plot(grad_norms, label=key)
    plt.legend()
    plt.title(f"Effective loss grad norms on filter taps for {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Effective grad norm")
    plt.grid(True)
    plt.tight_layout()


def task_grad_pairwise_cosine_similarity_plot(task_names, mean_matrix, pairwise_labels, pairwise_over_epochs, ref_cosines_over_epochs, title):

    mean_matrix = ready_for_plotting(mean_matrix)
    n_tasks = len(task_names)

    fig, axes = plt.subplots(3, 1, figsize=(8, 14))

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

    for name in task_names:
        axes[2].plot(ref_cosines_over_epochs[name], label=name)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].set_title(f"Task grad cosine similarity to fixed reference [1, 0, ...] for {title}")
    axes[2].legend()
    axes[2].grid(True)
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