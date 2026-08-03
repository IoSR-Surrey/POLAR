import json

import numpy as np
import torch

from utils.miscellaneous import ready_for_plotting


def load_plot_config(config_path="config/hyperpameters.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_white_noise_batch(config, device=None, dtype=None, shuffle=False):
    from datasets.stimulus_dataset import get_stim_loader

    env_config = config["env"]
    stimulus_config = config["stimulus_dataset"]
    training_config = config["training"]

    if device is None:
        device = torch.device(env_config["device"])
    if dtype is None:
        dtype = getattr(torch, env_config["dtype"])

    stim_loader, _ = get_stim_loader(
        fs=env_config["fs"],
        n_bands=stimulus_config["n_bands"],
        n_samples_per_band=stimulus_config["n_samples_per_band"],
        batch_size=training_config["batch_size"],
        stimulus_length_ms=stimulus_config["stimulus_length_ms"],
        ramp_time_ms=stimulus_config["ramp_time_ms"],
        f_min=stimulus_config["f_min"],
        f_max=stimulus_config["f_max"],
        band_mode=stimulus_config.get("band_mode", "erb"),
        erb_factor=stimulus_config.get("erb_factor", 1.0),
        dataset_type=stimulus_config.get("dataset_type", "bandpass"),
        stimulus_type=stimulus_config.get("stimulus_type", "impulse"),
        shuffle=shuffle,
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        drop_last=training_config["drop_last"],
    )
    return next(iter(stim_loader)).to(device=device, dtype=dtype)


def mean_over_stimulus_batch(values, n_listeners):
    """Average flat (batch*n_listeners, ...) rows into (n_listeners, ...)."""
    values = ready_for_plotting(values)
    n_flat = values.shape[0]
    if n_flat == n_listeners:
        return values
    if n_flat % n_listeners != 0:
        raise ValueError(
            f"Cannot split {n_flat} rows across {n_listeners} listeners"
        )
    batch_size = n_flat // n_listeners
    return values.reshape(batch_size, n_listeners, *values.shape[1:]).mean(axis=0)


def mean_interaural_cues_over_batch(cues, n_listeners):
    return {
        "itd": mean_over_stimulus_batch(cues["itd"], n_listeners),
        "ild": mean_over_stimulus_batch(cues["ild"], n_listeners),
    }


def reduce_batch_dimension(values, n_listeners):
    """Mean over a stimulus-batch dimension when present in epoch series."""
    values = np.asarray(ready_for_plotting(values))
    if values.ndim < 3 or values.shape[-1] != n_listeners:
        return values

    batch_axis = values.ndim - 2
    if values.shape[batch_axis] == 1:
        return np.squeeze(values, axis=batch_axis)
    return values.mean(axis=batch_axis)


def average_loss_dicts(loss_dicts):
    return {
        key: torch.stack([loss_dict[key].detach() for loss_dict in loss_dicts]).mean()
        for key in loss_dicts[0]
    }


def average_estimates_dicts(estimates_dicts):
    # cat then mean so a smaller final batch (drop_last=False) is handled correctly
    return {
        key: torch.cat([estimates[key].detach() for estimates in estimates_dicts], dim=0).mean(dim=0)
        for key in estimates_dicts[0]
    }


def _mean_flat_rows_per_listener(values, n_listeners):
    values = values.detach().reshape(-1, n_listeners, *values.shape[1:]).mean(dim=0)
    return values


def average_interaural_cues_dicts(cues_dicts, n_listeners):
    return {
        "itd": torch.stack([
            _mean_flat_rows_per_listener(cues["itd"], n_listeners) for cues in cues_dicts
        ]).mean(dim=0),
        "ild": torch.stack([
            _mean_flat_rows_per_listener(cues["ild"], n_listeners) for cues in cues_dicts
        ]).mean(dim=0),
    }


def average_coherence_values(coherence_values, n_listeners):
    return torch.stack([
        _mean_flat_rows_per_listener(coherence, n_listeners) for coherence in coherence_values
    ]).mean(dim=0)


def average_task_vector_dicts(vector_dicts):
    return {
        key: torch.stack([item[key].detach() for item in vector_dicts]).mean(dim=0)
        if all(item[key] is not None for item in vector_dicts)
        else None
        for key in vector_dicts[0]
    }


def average_mtl_saved_values(saved_values_list):
    averaged = {}
    for key in saved_values_list[0]:
        values = [saved[key] for saved in saved_values_list]
        first = values[0]
        if isinstance(first, dict):
            if all(isinstance(v, torch.Tensor) and v.ndim == 0 for v in first.values()):
                averaged[key] = {
                    subkey: torch.stack([value[subkey].detach() for value in values]).mean()
                    for subkey in first
                }
            else:
                averaged[key] = average_task_vector_dicts(values)
        elif isinstance(first, torch.Tensor) and first.ndim == 0:
            averaged[key] = torch.stack([value.detach() for value in values]).mean()
        elif isinstance(first, torch.Tensor):
            averaged[key] = torch.stack([value.detach() for value in values]).mean(dim=0)
        else:
            averaged[key] = first
    return averaged
