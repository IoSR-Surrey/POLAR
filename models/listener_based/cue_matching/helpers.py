import torch
import numpy as np
from utils.sofa_functions import sofa_find, get_IRs
from utils.peripheral_processing import estimate_interaural_cues

def get_target_cues(target_angles, sofaile, gtfb, fs=48000):

    target_idx = sofa_find(target_angles, sofaile)
    target_hrirs = get_IRs(target_idx, sofaile)

    lag_values = torch.arange(-0.001*fs, 0.001*fs + 1, 1, dtype=torch.float32, device=target_hrirs.device) / fs

    cues = estimate_interaural_cues(target_hrirs, gtfb, lag_values)

    target_cues = torch.stack([cues['itd'], cues['ild']], dim=1)

    return target_cues


def create_interaural_template(hrirs, gtfb, lag_values, tau_iacc, device='cpu', dtype=torch.float32):
    template = estimate_interaural_cues(hrirs, gtfb, lag_values, tau_iacc)
    template['weigthing_function'] = torch.linspace(0, 1, template['itd'].shape[1], device=device, dtype=dtype)
    template['itd_max'] = torch.max(torch.abs(template['itd']), dim = 0).values
    template['ild_max'] = torch.max(torch.abs(template['ild']), dim = 0).values
    template['itd_normed'] = template['itd'] / template['itd_max']
    template['ild_normed'] = template['ild'] / template['ild_max']
    template['azimuths'] = torch.arange(-90, 95, 5, dtype=dtype, device=device) # Hard Coded from -90 to 90 in 5 degree steps
    return template
