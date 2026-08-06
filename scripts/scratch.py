import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn as nn

from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.seed import set_seed
from utils.miscellaneous import (load_npy_to_torch, 
                                target_angle_for_listeners, 
                                create_mtl_optimizer, 
                                run_mtl_optimizer_step,
                                git_commit_info,
                                write_run_config)
from utils.dsp import magnitude_squared_coherence

from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.desena2020.model import desena2020
from models.listener_based.cue_matching.model import CueMatchingModel
from models.listener_based.cue_matching.helpers import get_target_cues
from models.listener_based.LSD.model import LSD
from models.loudspeaker_based.second_moment.model import SecondMoment
from models.train import ModelTrainer
from logger.saving import SavingManager
from plotting.miscellaneous import ALL_PLOTTING_FUNCTIONS
from plotting.evaluation import compute_white_noise_plot_outputs
from plotting.batch_utils import (
    average_coherence_values,
    average_estimates_dicts,
    average_interaural_cues_dicts,
    average_loss_dicts,
    average_mtl_saved_values,
)
from logger.printing import print_loss_and_estimates
from datasets.stimulus_dataset import get_stim_loader

with open('config/hyperpameters.json', 'r') as f:
    config = json.load(f)
set_seed(config["env"]["seed"])
device = torch.device(config["env"]["device"])
dtype = getattr(torch, config["env"]["dtype"])
mtl_optimizer = "cagrad" #config["training"]["mtl_optimizer"]
skip_keys = config["saving"]["skip_keys"]
use_tensorboard = config["saving"]["use_tensorboard"]
fs = config["env"]["fs"]
sofa_path = "data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
gtfb_path = "data/banks/gtfb_48k.npy"
apply_distance_attenuation = config["env"]["apply_distance_attenuation"]
lag_window_s = 0.001
tau_iacc = 1e-2
d = 2
speaker_angles_deg = [30.0, -30.0]
listeners = [[0.0, 0.0], [0.0, 0.5], [0.0, -0.5]]
# Convex task balance: lambda_loc on llado, (1-lambda_loc) on LSD.
# Scaled to sum=2 so lambda_loc=0.5 matches historical [1.0, 1.0] weights.
lambda_loc = float(config["training"].get("lambda_loc", 0.5))
lambda_loc = min(max(lambda_loc, 0.0), 1.0)
model_weights = [2.0 * lambda_loc, 2.0 * (1.0 - lambda_loc)]
filter_initialisation = {"init_weight": "middle", "fs": fs, "n_bands": 4, "band_mode": "log"}
    
sofaile = load_sofa(sofa_path)
hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
gtfb = load_npy_to_torch(gtfb_path, device=device, dtype=dtype)
lag_values = torch.arange(-lag_window_s * fs, lag_window_s * fs + 1, 1, dtype=dtype, device=device) / fs

lstn_pos = torch.tensor(listeners, dtype=dtype, device=device)
speaker_angles = torch.tensor(speaker_angles_deg, dtype=dtype, device=device)
spk_pos = torch.stack([
    d * torch.cos(torch.deg2rad(speaker_angles)),
    d * torch.sin(torch.deg2rad(speaker_angles)),
], dim=1)

target_angle = config["training"]["target_angle"]
target_position = torch.tensor([d * torch.cos(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))), d * torch.sin(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device)))], dtype=dtype, device=device)
target_angles = target_angle_for_listeners(lstn_pos, target_position)
target_cues = get_target_cues(target_angles.unsqueeze(-1), sofaile, gtfb, device=device, dtype=dtype)

llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
desena_model = desena2020(fs, hrirs, gtfb, device=device, dtype=dtype)
lsd_model = LSD(fs, sofaile, target_angles.unsqueeze(-1), device=device, dtype=dtype)
cue_model = CueMatchingModel(fs, hrirs, gtfb, device=device, dtype=dtype)

prop = BinauralFreeFieldPropagation(fs, lstn_pos, spk_pos, sofaile, apply_distance_attenuation=apply_distance_attenuation, device=device, dtype=dtype)

n_sources = len(spk_pos)
n_taps = config["filter_estimator"]["n_samples"]
nn = MultiSourceFIR(n_taps, n_sources)
nn.set_filter_initialisation(**filter_initialisation)
nn = nn.to(device=device, dtype=dtype)
second_moment = SecondMoment(n_taps, n_sources, device=device, dtype=dtype)

modelTrainer = ModelTrainer(
    [llado_model, lsd_model],
    weights=model_weights,
)

targets = {
    "llado2025_grad_surgery": target_angles.unsqueeze(0),
    "LSD": torch.zeros_like(target_angles.unsqueeze(0)),
}

stim_loader, stim_dataset = get_stim_loader(
    fs=fs,
    n_bands=config["stimulus_dataset"]["n_bands"],
    n_samples_per_band=config["stimulus_dataset"]["n_samples_per_band"],
    batch_size=config["training"]["batch_size"],
    stimulus_length_ms=config["stimulus_dataset"]["stimulus_length_ms"],
    ramp_time_ms=config["stimulus_dataset"]["ramp_time_ms"],
    f_min=config["stimulus_dataset"]["f_min"],
    f_max=config["stimulus_dataset"]["f_max"],
    band_mode=config["stimulus_dataset"]["band_mode"],
    erb_factor=config["stimulus_dataset"].get("erb_factor", 1.0),
    dataset_type=config["stimulus_dataset"]["dataset_type"],
    stimulus_type=config["stimulus_dataset"]["stimulus_type"],
    seed=config["env"]["seed"],
    shuffle=config["training"]["shuffle"],
    num_workers=config["training"]["num_workers"],
    pin_memory=config["training"]["pin_memory"],
    drop_last=config["training"]["drop_last"],
)

optimizer = torch.optim.Adam(nn.parameters(), lr=config["training"]["lr"])
mtl = create_mtl_optimizer(modelTrainer, mtl_optimizer, training_config=config["training"])

saving_manager = SavingManager(
    f"outputs/{mtl_optimizer}", skip_keys=skip_keys, timestamp=True, use_tensorboard=use_tensorboard,
    task_names=list(modelTrainer.model_dict.keys()),
)

git_info = git_commit_info()
run_config = {
    "script": "scripts/scratch.py",
    "git": git_info,
    "config": config,
    "env": {
        "fs": fs,
        "device": str(device),
        "dtype": config["env"]["dtype"],
        "seed": config["env"]["seed"],
        "deterministic": config["env"].get("deterministic", False),
        "sofa_path": sofa_path,
        "gtfb_path": gtfb_path,
        "apply_distance_attenuation": apply_distance_attenuation,
    },
    "geometry": {
        "d": d,
        "speaker_angles_deg": speaker_angles_deg,
        "listeners": listeners,
    },
    "filter_estimator": {
        "n_samples": n_taps,
        "n_sources": n_sources,
        "filter_initialisation": filter_initialisation,
    },
    "models": [
        {"name": name, "weight": weight}
        for name, weight in zip(modelTrainer.model_dict.keys(), model_weights)
    ],
    "lambda_loc": lambda_loc,
    "targets": {
        "llado2025_grad_surgery": "target_angles",
        "LSD": 0.0,
    },
    "training": {
        **config["training"],
        "mtl_optimizer": mtl_optimizer,
        "optimizer": "Adam",
    },
    "stimulus_dataset": config["stimulus_dataset"],
    "llado2025_grad_surgery": config.get("llado2025_grad_surgery", {}),
    "lag_window_s": lag_window_s,
    "tau_iacc": tau_iacc,
}

write_run_config(saving_manager.output_dir, run_config)
saving_manager.save("run_config", run_config)
saving_manager.save("git_commit", git_info["commit"])
saving_manager.save("git_dirty", git_info["dirty"])
saving_manager.save("gtfb", gtfb)
saving_manager.save("lag_values", lag_values)
saving_manager.save("tau_iacc", tau_iacc)
saving_manager.save("target_angle", target_angle)
saving_manager.save("target_position", target_position)
saving_manager.save("target_angles", target_angles)
saving_manager.save("lstn_pos", lstn_pos)
saving_manager.save("spk_pos", spk_pos)
saving_manager.save("sofaile", sofaile)
saving_manager.save("target_cues", target_cues)
saving_manager.save("stimulus_dataset", config["stimulus_dataset"])
saving_manager.save("filter_initialisation", filter_initialisation)
saving_manager.save("seed", config["env"]["seed"])
saving_manager.save("device", str(device))
saving_manager.save("dtype", config["env"]["dtype"])
saving_manager.save("fs", fs)
saving_manager.save("sofa_path", sofa_path)
saving_manager.save("gtfb_path", gtfb_path)
saving_manager.save("apply_distance_attenuation", apply_distance_attenuation)
saving_manager.save("mtl_optimizer", mtl_optimizer)
saving_manager.save("n_taps", n_taps)
saving_manager.save("n_sources", n_sources)
saving_manager.save("model_weights", model_weights)
saving_manager.save("lambda_loc", lambda_loc)
saving_manager.save("lr", config["training"]["lr"])
saving_manager.save("num_epochs", config["training"]["num_epochs"])
saving_manager.save("batch_size", config["training"]["batch_size"])


dummy_input = torch.ones(1, 1, dtype=dtype, device=device)
n_listeners = len(lstn_pos)
for epoch in range(config["training"]["num_epochs"]):
    batch_loss_dicts = []
    batch_estimates = []
    batch_coherences = []
    batch_cues = {}
    batch_mtl_saved = []

    for stim in stim_loader:
        stim = stim.to(device=device, dtype=dtype)
        spk_filters = nn(dummy_input)
        binaural_signals = prop(stim, spk_filters)

        modelTrainer.compute_estimates(binaural_signals, spk_filters, stim)
        loss_dict = modelTrainer.compute_loss(targets)
        batch_loss_dicts.append(loss_dict)
        batch_estimates.append(modelTrainer.estimates)

        binaural_flat = binaural_signals.view(-1, 2, binaural_signals.shape[-1])
        # _, mean_coherence = magnitude_squared_coherence(binaural_flat[..., 0, :], binaural_flat[..., 1, :])
        # batch_coherences.append(mean_coherence)

        for model_name, kwargs in modelTrainer.kwargs_output.items():
            if "interauralcues" in kwargs:
                cues = kwargs["interauralcues"]
                batch_cues.setdefault(model_name, []).append(
                    {"itd": cues["itd"].detach(), "ild": cues["ild"].detach()}
                )

        optimizer.zero_grad()
        batch_mtl_saved.append(
            run_mtl_optimizer_step(
                mtl_optimizer, mtl, loss_dict, nn, optimizer, saving_manager, modelTrainer, save=False
            )
        )

    saving_manager.set_step(epoch)
    epoch_loss_dict = average_loss_dicts(batch_loss_dicts)
    epoch_estimates = average_estimates_dicts(batch_estimates)
    saving_manager.save("loss_dict", epoch_loss_dict)
    saving_manager.save("modelTrainer.estimates", epoch_estimates)
    # saving_manager.save("ear_coherence", average_coherence_values(batch_coherences, n_listeners))
    for model_name, cues_dicts in batch_cues.items():
        saving_manager.save(
            f"{model_name}_cues", average_interaural_cues_dicts(cues_dicts, n_listeners)
        )
    for key, value in average_mtl_saved_values(batch_mtl_saved).items():
        saving_manager.save(key, value)

    saving_manager.flush_tensorboard()
    print_loss_and_estimates(epoch_loss_dict, epoch_estimates, epoch)

with torch.no_grad():
    spk_filters = nn(dummy_input)
    saving_manager.save("spk_filters", spk_filters)
    plot_outputs = compute_white_noise_plot_outputs(saving_manager)
    if plot_outputs is not None:
        saving_manager.save("binaural_signals", plot_outputs["binaural_signals"])
        saving_manager.save("localisation_angles", plot_outputs["localisation_angles"])
        saving_manager.save("localisation_uncertainty", plot_outputs["localisation_uncertainty"])
        saving_manager.save("lsd", plot_outputs["lsd"])
        if "llado2025_grad_surgery_cues" in plot_outputs:
            saving_manager.save("llado2025_grad_surgery_cues", plot_outputs["llado2025_grad_surgery_cues"])

saving_manager.export_to_disk()
print(f"Saved outputs to {saving_manager.output_dir}")

ALL_PLOTTING_FUNCTIONS(saving_manager, show_plots=False)
saving_manager.close()
