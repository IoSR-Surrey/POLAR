import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import json

import torch

from utils.seed import set_seed
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import load_npy_to_torch

from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.train import ModelTrainer
from logger.printing import print_loss_and_estimates


# Get the hyperparameters
with open("config/hyperpameters.json", "r") as f:
    hyperpameters = json.load(f)

env_config = hyperpameters["env"]
training_config = hyperpameters["training"]
geometry_config = hyperpameters["geometry"]
filter_config = hyperpameters["filter_estimator"]


# Set the seed
set_seed(env_config["seed"])


# Get the dtype, device and fs
fs = env_config["fs"]
dtype = getattr(torch, env_config["dtype"])
device = torch.device(env_config["device"])


# Other configurations
apply_distance_attenuation = env_config.get("apply_distance_attenuation", False)
target_angle = training_config.get("target_angle", 15.0)
sofa_path = env_config["sofa_path"]
if "..." in sofa_path:
    sofa_path = "data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"


# Create the geometry
speakers = torch.tensor(geometry_config["speakers"], dtype=dtype, device=device)
listeners = torch.tensor(geometry_config["listeners"], dtype=dtype, device=device)


# Load the sofa file and localisation model inputs
sofaile = load_sofa(sofa_path)
hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)


# Create filter estimator
filter_estimator = MultiSourceFIR(
    n_samples=filter_config["n_samples"],
    n_sources=len(speakers),
    init_weight=filter_config["init_weight"],
)
if filter_config["init_weight"] is None:
    filter_estimator.set_filter_initialisation("middle")


# Create propagator
propagator = BinauralFreeFieldPropagation(
    fs=fs,
    lstn_pos=listeners,
    spk_pos=speakers,
    sofaile=sofaile,
    apply_distance_attenuation=apply_distance_attenuation,
    device=device,
    dtype=dtype,
)


# Create localisation model and trainer
localisation_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
model_trainer = ModelTrainer([localisation_model], weights=[1.0])
targets = {"llado2025_grad_surgery": torch.tensor([target_angle], dtype=dtype, device=device)}


# Create optimizer (optimise speaker filters, not the localisation model)
optimizer = torch.optim.Adam(filter_estimator.parameters(), lr=training_config["lr"])


# Monophonic impulse stimulus and filter input (replace with a dataloader for batch training)
stim = torch.zeros(1, 1, 1, dtype=dtype, device=device)
stim[:, :, 0] = 1.0
filter_input = torch.ones(1, 1, dtype=dtype, device=device)


# Optimisation loop
num_epochs = training_config["num_epochs"]
for epoch in range(num_epochs):
    spk_filters = filter_estimator(filter_input)
    binaural_signals = propagator(stim, spk_filters)

    model_trainer.compute_estimates(binaural_signals, spk_filters)
    loss_dict = model_trainer.compute_loss(targets)

    optimizer.zero_grad()
    loss_dict["total"].backward()
    optimizer.step()

    print_loss_and_estimates(loss_dict, model_trainer.estimates, epoch)


# Final estimate after optimisation
with torch.no_grad():
    spk_filters = filter_estimator(filter_input)
    binaural_signals = propagator(stim, spk_filters)
    model_trainer.compute_estimates(binaural_signals, spk_filters)

final_estimate = model_trainer.estimates["llado2025_grad_surgery"].item()
print("--------------------------------")
print(f"Target azimuth: {target_angle:.2f} deg")
print(f"Final estimated azimuth: {final_estimate:.2f} deg")
