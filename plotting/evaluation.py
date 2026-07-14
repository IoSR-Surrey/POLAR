import torch

from utils.sofa_functions import get_template_for_llado2025_desena_2020
from utils.miscellaneous import load_npy_to_torch, target_angle_for_listeners
from propagation.model import BinauralFreeFieldPropagation
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.desena2020.model import desena2020
from models.listener_based.LSD.model import LSD
from plotting.batch_utils import get_white_noise_batch, load_plot_config, mean_over_stimulus_batch


def _final_spk_filters(saved_data):
    spk_filters = saved_data["spk_filters"][-1]
    if spk_filters.ndim == 4:
        spk_filters = spk_filters[-1]
    if spk_filters.ndim == 3 and spk_filters.shape[0] > 1:
        spk_filters = spk_filters[:1]
    return spk_filters


def compute_white_noise_plot_outputs(saving_manager, config_path="config/hyperpameters.json"):
    """Evaluate final speaker filters on a white-noise batch for plotting."""
    saved_data = saving_manager.saved_data
    required_keys = ("lstn_pos", "spk_pos", "spk_filters", "sofaile")
    if not all(key in saved_data for key in required_keys):
        return None

    config = load_plot_config(config_path)
    env_config = config["env"]
    fs = env_config["fs"]
    dtype = getattr(torch, env_config["dtype"])
    device = torch.device(env_config["device"])

    lstn_pos = saved_data["lstn_pos"][0].to(device=device, dtype=dtype)
    spk_pos = saved_data["spk_pos"][0].to(device=device, dtype=dtype)
    sofaile = saved_data["sofaile"][0]
    spk_filters = _final_spk_filters(saved_data).to(device=device, dtype=dtype)
    stim = get_white_noise_batch(config, device=device, dtype=dtype, shuffle=False)

    if "gtfb" in saved_data:
        gtfb = saved_data["gtfb"][0].to(device=device, dtype=dtype)
    else:
        gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)

    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    prop = BinauralFreeFieldPropagation(fs, lstn_pos, spk_pos, sofaile, device=device, dtype=dtype)
    binaural_signals = prop(stim, spk_filters)
    binaural_flat = binaural_signals.reshape(-1, 2, binaural_signals.shape[-1])

    outputs = {
        "binaural_signals": binaural_signals.mean(dim=0),
        "stim": stim,
    }

    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    llado_output = llado_model(binaural_flat, stim)
    outputs["localisation_angles"] = mean_over_stimulus_batch(llado_output["output"], len(lstn_pos))
    cues = llado_output["kwargs"]["interauralcues"]
    outputs["llado2025_grad_surgery_cues"] = {
        "itd": mean_over_stimulus_batch(cues["itd"], len(lstn_pos)),
        "ild": mean_over_stimulus_batch(cues["ild"], len(lstn_pos)),
    }

    desena_model = desena2020(fs, hrirs, gtfb, device=device, dtype=dtype)
    outputs["localisation_uncertainty"] = mean_over_stimulus_batch(
        desena_model(binaural_flat, stim)["output"], len(lstn_pos)
    )

    if "target_position" in saved_data:
        target_position = saved_data["target_position"][0].to(device=device, dtype=dtype)
    else:
        target_angle = saved_data["target_angle"][0]
        target_distance = spk_pos[0].norm()
        target_position = torch.tensor(
            [
                target_distance * torch.cos(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))),
                target_distance * torch.sin(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))),
            ],
            dtype=dtype,
            device=device,
        )
    target_angles = target_angle_for_listeners(lstn_pos, target_position)
    lsd_model = LSD(fs, sofaile, target_angles.unsqueeze(-1), device=device, dtype=dtype)
    outputs["lsd"] = mean_over_stimulus_batch(
        lsd_model(binaural_flat, stim)["output"], len(lstn_pos)
    )

    return outputs
