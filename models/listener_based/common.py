import torch

from utils.peripheral_processing import estimate_interaural_cues
from utils.dsp import rms, soft_argmax


def create_interaural_template(hrirs, gtfb, lag_values, tau_iacc, device='cpu', dtype=torch.float32):
    template = estimate_interaural_cues(hrirs, gtfb, lag_values, tau_iacc)
    template['weigthing_function'] = torch.linspace(0, 1, template['itd'].shape[1], device=device, dtype=dtype)
    template['itd_max'] = torch.max(torch.abs(template['itd']), dim = 0).values
    template['ild_max'] = torch.max(torch.abs(template['ild']), dim = 0).values
    template['itd_normed'] = template['itd'] / template['itd_max']
    template['ild_normed'] = template['ild'] / template['ild_max']
    template['azimuths'] = torch.arange(-90, 95, 5, dtype=dtype, device=device) # Hard Coded from -90 to 90 in 5 degree steps
    return template


def compute_loudness_weights(interauralcues):
    gtfbout_left = interauralcues['gtfb_signals'][:,:,0,:]
    gtfbout_right = interauralcues['gtfb_signals'][:,:,1,:]

    signal_rms = (rms(gtfbout_left,dim=-1) + rms(gtfbout_right,dim=-1))/2
    signal_dB = 20*torch.log10(signal_rms.clamp(min=1e-12))

    spl_cfs = torch.tensor([
    80.8040974369988, 69.7152778466767, 63.3217162662439, 58.8466830780741,
    55.7072986018014, 53.3475571597242, 51.6321495968390, 50.3987218539694,
    49.9996318902603, 51.0802858850330, 52.3491290304042, 52.0779324379977,
    49.2861519859267, 46.9451007372847, 46.2399898195513, 46.6506947407588,
    48.2807108683239, 51.3531651995338, 55.7893247423705, 59.9041927864145,
    62.6548923203973, 62.8950513166133, 60.1353379120974, 60.1353379120974],
    dtype=gtfbout_left.dtype, device=gtfbout_left.device).view(1, -1)

    spl_cfs_0 = spl_cfs - spl_cfs[0, 9]
    signal_dB_0 = signal_dB - signal_dB[:, 9].unsqueeze(1)

    return torch.softmax((signal_dB_0 - spl_cfs_0)/10, dim=-1)


def estimate_azimuth_from_distance(itd_ild_distance, azimuths):
    distance_combined = itd_ild_distance['distance_combined']
    loudness_weighting = itd_ild_distance['weighting_function_SPL']
    estimated_azimuth_per_band = soft_argmax(-1*distance_combined, azimuths[None,:,None], dim=1)

    estimated_azimuth = torch.sum(estimated_azimuth_per_band*loudness_weighting, dim=-1)
    return estimated_azimuth.clamp(min=azimuths.min(), max=azimuths.max())
