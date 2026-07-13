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
from utils.miscellaneous import (load_npy_to_torch, 
                                target_angle_for_listeners, 
                                create_mtl_optimizer, 
                                run_mtl_optimizer_step)
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
from logger.printing import print_loss_and_estimates

with open('config/hyperpameters.json', 'r') as f:
    config = json.load(f)
mtl_optimizer = "cagrad" #config["training"]["mtl_optimizer"]
skip_keys = config["saving"]["skip_keys"]
use_tensorboard = config["saving"]["use_tensorboard"]
    
sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
hrirs = get_template_for_llado2025_desena_2020(sofaile)
gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
lag_values = torch.arange(-0.001*48000, 0.001*48000 + 1, 1, dtype=torch.float32, device=hrirs.device) / 48000

d = 2
lstn_pos = torch.tensor([[0,0],[0, 0.5], [0, -0.5]])
spk_pos = torch.tensor([[d*torch.cos(torch.deg2rad(torch.tensor(30))), d*torch.sin(torch.deg2rad(torch.tensor(30)))], 
                        [d*torch.cos(torch.deg2rad(torch.tensor(30))), -d*torch.sin(torch.deg2rad(torch.tensor(30)))]])

target_angle = 0.0
target_position = torch.tensor([d * torch.cos(torch.deg2rad(torch.tensor(target_angle))), d * torch.sin(torch.deg2rad(torch.tensor(target_angle)))])
target_angles = target_angle_for_listeners(lstn_pos, target_position)
target_cues = get_target_cues(target_angles.unsqueeze(-1), sofaile, gtfb)

llado_model = llado2025_grad_surgery(48000, hrirs, gtfb)
desena_model = desena2020(48000, hrirs, gtfb)
lsd_model = LSD(48000, sofaile, target_angles.unsqueeze(-1))
cue_model = CueMatchingModel(48000, hrirs, gtfb)

prop = BinauralFreeFieldPropagation(48000, lstn_pos, spk_pos, sofaile)

n_sources = len(spk_pos)
n_taps = 32
nn = MultiSourceFIR(n_taps, n_sources)
# nn.set_filter_initialisation("middle")
second_moment = SecondMoment(n_taps, n_sources)

modelTrainer = ModelTrainer(
    [llado_model, desena_model, lsd_model, second_moment],
    weights=[1.0, 1.0, 1.0, 1.0],
)

targets = {
    "llado2025_grad_surgery": torch.ones(1, len(lstn_pos)),
    "desena2020": torch.zeros(1, len(lstn_pos)),
    "LSD": torch.zeros(1, len(lstn_pos)),
    "SecondMoment": torch.zeros(1, len(spk_pos)),
}

stim = torch.ones(1,1,1)

optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
mtl = create_mtl_optimizer(modelTrainer, mtl_optimizer)

saving_manager = SavingManager(
    f"outputs/{mtl_optimizer}", skip_keys=skip_keys, timestamp=True, use_tensorboard=use_tensorboard,
    task_names=list(modelTrainer.model_dict.keys()),
)

saving_manager.save("gtfb", gtfb)
saving_manager.save("lag_values", lag_values)
saving_manager.save("tau_iacc", 1e-2)
saving_manager.save("target_angle", target_angle)
saving_manager.save("target_position", target_position)
saving_manager.save("lstn_pos", lstn_pos)
saving_manager.save("spk_pos", spk_pos)
saving_manager.save("sofaile", sofaile)
saving_manager.save("target_cues", target_cues)


dummy_input = torch.ones(1,1)
for i in range(1220):

    saving_manager.set_step(i)
    spk_filters = nn(dummy_input)

    binaural_signals = prop(stim, spk_filters)

    modelTrainer.compute_estimates(binaural_signals, spk_filters, stim)
    loss_dict = modelTrainer.compute_loss(targets)
    saving_manager.save("loss_dict", loss_dict)
    saving_manager.save("modelTrainer.estimates", modelTrainer.estimates)

    binaural_flat = binaural_signals.view(-1, 2, binaural_signals.shape[-1])
    _, mean_coherence = magnitude_squared_coherence(binaural_flat[..., 0, :], binaural_flat[..., 1, :])
    saving_manager.save("ear_coherence", mean_coherence.detach())

    for model_name, kwargs in modelTrainer.kwargs_output.items():
        if "interauralcues" in kwargs:
            cues = kwargs["interauralcues"]
            saving_manager.save(f"{model_name}_cues", {"itd": cues["itd"].detach(), "ild": cues["ild"].detach()})

    optimizer.zero_grad()
    run_mtl_optimizer_step(mtl_optimizer, mtl, loss_dict, nn, optimizer, saving_manager, modelTrainer)

    saving_manager.flush_tensorboard()
    print_loss_and_estimates(loss_dict, modelTrainer.estimates, i)

with torch.no_grad():
    spk_filters = nn(dummy_input)
    binaural_signals = prop(stim, spk_filters)
    binaural_signals = binaural_signals.view(
        binaural_signals.shape[0] * binaural_signals.shape[1], 2, -1
    )
    localisation_angles = llado_model(binaural_signals, stim)["output"]
    localisation_uncertainty = desena_model(binaural_signals, stim)["output"]
    lsd = lsd_model(binaural_signals, stim)["output"]

saving_manager.save("binaural_signals", binaural_signals)
saving_manager.save("localisation_angles", localisation_angles)
saving_manager.save("localisation_uncertainty", localisation_uncertainty)
saving_manager.save("lsd", lsd)
saving_manager.save("spk_filters", spk_filters)

saving_manager.export_to_disk()
print(f"Saved outputs to {saving_manager.output_dir}")

ALL_PLOTTING_FUNCTIONS(saving_manager, show_plots=True)
saving_manager.close()
