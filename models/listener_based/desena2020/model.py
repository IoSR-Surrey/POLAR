import torch
import torch.nn as nn
import numpy as np 

from utils.peripheral_processing import estimate_interaural_cues
from utils.itd_ild_distance import itd_ild_template_distance
from models.listener_based.common import create_interaural_template, compute_loudness_weights


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
        
        likelihood_f = torch.softmax(-distance['distance_combined'], dim=1)
        distance['weighting_function_SPL'] = compute_loudness_weights(interauralcues)
        distance['weightedLikelihood'] = torch.einsum('cmd,cd->cm', likelihood_f, distance['weighting_function_SPL'])

        return distance


    def _set_Hmin(self):

        distance = itd_ild_template_distance(self.template, self.template)

        distance = self._loudness_weighting(self.template, distance)

        estimated_uncertainty = self._estimate_uncertainty(distance)

        self.Hmin = torch.min(estimated_uncertainty)

    def _create_template(self, hrirs):
        self.template = create_interaural_template(hrirs, self.gtfb, self.lag_values, self.tau_iacc, self.device, self.dtype)


    

        
    

