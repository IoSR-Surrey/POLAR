import torch
import torch.nn as nn
import numpy as np 

from utils.peripheral_processing import estimate_interaural_cues
from utils.itd_ild_distance import itd_ild_template_distance
from utils.dsp import rms
from models.listener_based.common import create_interaural_template


class desena2020(nn.Module):
    def __init__(self, fs, hrirs, gtfb, tau_iacc=1e-2, device='cpu', dtype=torch.float32):
        super(desena2020, self).__init__()

        self.model_type = 'listener_based'

        self.fs = fs
        self.hrirs = hrirs
        self.gtfb = gtfb
        self.device = device
        self.dtype = dtype

        max_lag = np.floor(0.001*fs)
        self.lag_values = torch.arange(-max_lag, max_lag + 1, 1, dtype=dtype, device=device) / fs
        self.tau_iacc = tau_iacc    

        self._create_template(hrirs)
        self.Hmin = 0
        self._set_Hmin()


    def forward(self, bin_stim, _):

        interauralcues = estimate_interaural_cues(bin_stim, self.gtfb, self.lag_values, self.tau_iacc)

        itd_ild_distance = itd_ild_template_distance(interauralcues, self.template)

        distance = self._loudness_weighting(interauralcues, itd_ild_distance)

        estimated_uncertainty = self._estimate_uncertainty(distance)

        kwargs = {"interauralcues": interauralcues,
                "itd_ild_distance": itd_ild_distance,
                "distance": distance}

        return {"output": estimated_uncertainty, "kwargs": kwargs}
    

    def _estimate_uncertainty(self, distance):

        weighted_likelihood = distance['weightedLikelihood']
        n_angles = weighted_likelihood.shape[1]
        natural_angles = self.template['azimuths']

        cos = torch.cos(torch.deg2rad(natural_angles*2)).view(1,n_angles)
        sin = torch.sin(torch.deg2rad(natural_angles*2)).view(1,n_angles)

        alpha = torch.sum(weighted_likelihood*cos, dim = 1)
        beta = torch.sum(weighted_likelihood*sin, dim = 1)

        H = 1 - torch.sqrt(torch.pow(alpha,2) + torch.pow(beta,2))

        Hnorm = (H - self.Hmin)/(1-self.Hmin)

        return(Hnorm)
    

    def _loudness_weighting(self, interauralcues, distance):
        
        gtfbout_left = interauralcues['gtfb_signals'][:,:,0,:]
        gtfbout_right = interauralcues['gtfb_signals'][:,:,1,:]
        likelihood_f = torch.softmax(-distance['distance_combined'], dim=1)
            
        dtype = gtfbout_left.dtype
        device = gtfbout_right.device

        signal_rms = (rms(gtfbout_left,dim=-1) + rms(gtfbout_right,dim=-1))/2
        signal_dB = 20*torch.log10(signal_rms)

        spl_cfs = torch.tensor([
        80.8040974369988, 69.7152778466767, 63.3217162662439, 58.8466830780741,
        55.7072986018014, 53.3475571597242, 51.6321495968390, 50.3987218539694,
        49.9996318902603, 51.0802858850330, 52.3491290304042, 52.0779324379977,
        49.2861519859267, 46.9451007372847, 46.2399898195513, 46.6506947407588,
        48.2807108683239, 51.3531651995338, 55.7893247423705, 59.9041927864145,
        62.6548923203973, 62.8950513166133, 60.1353379120974, 60.1353379120974], 
        dtype=dtype, device=device).view(1, -1)

        spl_cfs_0 = spl_cfs - spl_cfs[0, 9]
        signal_dB_0 = signal_dB - signal_dB[:, 9].unsqueeze(1)

        dB_diff = signal_dB_0 - spl_cfs_0

        distance['weighting_function_SPL'] = torch.softmax(dB_diff/10, dim=-1) 
        distance['weightedLikelihood'] = torch.einsum('cmd,cd->cm', likelihood_f, distance['weighting_function_SPL'])

        return distance


    def _set_Hmin(self):

        distance = itd_ild_template_distance(self.template, self.template)

        distance = self._loudness_weighting(self.template, distance)

        estimated_uncertainty = self._estimate_uncertainty(distance)

        self.Hmin = torch.min(estimated_uncertainty)

    def _create_template(self, hrirs):
        self.template = create_interaural_template(hrirs, self.gtfb, self.lag_values, self.tau_iacc, self.device, self.dtype)


    

        
    

