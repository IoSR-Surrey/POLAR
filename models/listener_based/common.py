import torch

from utils.peripheral_processing import estimate_interaural_cues
from utils.dsp import soft_argmax


def create_interaural_template(hrirs, gtfb, lag_values, tau_iacc, device='cpu', dtype=torch.float32):
    template = estimate_interaural_cues(hrirs, gtfb, lag_values, tau_iacc)
    template['weigthing_function'] = torch.linspace(0, 1, template['itd'].shape[1], device=device, dtype=dtype)
    template['itd_max'] = torch.max(torch.abs(template['itd']), dim = 0).values
    template['ild_max'] = torch.max(torch.abs(template['ild']), dim = 0).values
    template['itd_normed'] = template['itd'] / template['itd_max']
    template['ild_normed'] = template['ild'] / template['ild_max']
    template['azimuths'] = torch.arange(-90, 95, 5, dtype=dtype, device=device) # Hard Coded from -90 to 90 in 5 degree steps
    return template


def estimate_azimuth_from_distance(itd_ild_distance, azimuths):
    distance_combined = itd_ild_distance['distance_combined']
    estimated_azimuth_per_band = soft_argmax(-1*distance_combined, azimuths[None,:,None], dim=1)
    return torch.mean(estimated_azimuth_per_band, dim=-1)
