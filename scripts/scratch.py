import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import load_npy_to_torch

from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.desena2020.model import desena2020
from models.train import ModelTrainer

from filter_estimation.model import MultiSourceFIR

from propagation.model import BinauralFreeFieldPropagation

from plotting.miscellaneous import ALL_PLOTTING_FUNCTIONS

from logger.printing import print_loss_and_estimates
from logger.saving import SavingManager
    
sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
hrirs = get_template_for_llado2025_desena_2020(sofaile)
gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
lag_values = torch.arange(-0.001*48000, 0.001*48000 + 1, 1, dtype=torch.float32, device=hrirs.device) / 48000

d = 2
lstn_pos = torch.tensor([[0, 0]])
spk_pos = torch.tensor([[d*torch.cos(torch.deg2rad(torch.tensor(30))), d*torch.sin(torch.deg2rad(torch.tensor(30)))], 
                        [d*torch.cos(torch.deg2rad(torch.tensor(30))), -d*torch.sin(torch.deg2rad(torch.tensor(30)))]])

llado_model = llado2025_grad_surgery(48000, hrirs, gtfb)
desena_model = desena2020(48000, hrirs, gtfb)
modelTrainer = ModelTrainer([desena_model,], weights=[1.0, 0.0])
target_angle = 15.0
targets = {"desena2020": torch.tensor([target_angle])}

prop = BinauralFreeFieldPropagation(48000, lstn_pos, spk_pos, sofaile)

nn = MultiSourceFIR(32, 2)
nn.set_filter_initialisation("middle")

stim = torch.ones(1,1,1)

optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)

saving_manager = SavingManager()
saving_manager.save("gtfb", gtfb)
saving_manager.save("lag_values", lag_values)
saving_manager.save("tau_iacc", 1e-2)
saving_manager.save("target_angle", target_angle)
saving_manager.save("lstn_pos", lstn_pos)
saving_manager.save("spk_pos", spk_pos)
saving_manager.save("sofaile", sofaile)


dummy_input = torch.ones(1,1)
for i in range(100):

    spk_filters = nn(dummy_input)

    binaural_signals = prop(stim, spk_filters)

    N_signals = binaural_signals.shape[0]
    N_listeners = binaural_signals.shape[1]
    binaural_signals = binaural_signals.view(N_signals*N_listeners, 2, -1)

    modelTrainer.compute_estimates(binaural_signals)
    loss_dict = modelTrainer.compute_loss(targets)

    ## DEBUG
    LOSS_DEBUG = loss_dict['total'] / 3600

    optimizer.zero_grad()
    LOSS_DEBUG.backward(retain_graph=True)
    
    # Saving after backward to have the gradients available
    saving_manager.save_branch_grads(modelTrainer)
    saving_manager.save_filter_tap_branch_grads(modelTrainer, nn)
    
    optimizer.step()
    
    print_loss_and_estimates(loss_dict, modelTrainer.estimates, i)

with torch.no_grad():
    spk_filters = nn(dummy_input)
    binaural_signals = prop(stim, spk_filters)
    binaural_signals = binaural_signals.view(
        binaural_signals.shape[0] * binaural_signals.shape[1], 2, -1
    )

saving_manager.save("binaural_signals", binaural_signals)
saving_manager.save("spk_filters", spk_filters)

ALL_PLOTTING_FUNCTIONS(saving_manager)
