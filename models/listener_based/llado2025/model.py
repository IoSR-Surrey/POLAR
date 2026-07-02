import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TF

from utils.peripheral_processing import estimate_interaural_cues
from utils.itd_ild_distance import itd_ild_template_distance
from utils.dsp import soft_argmax, hilbert_transform, compute_energy


class llado2025(nn.Module):

    def __init__(self, fs, hrirs, gtfb, tau_iacc=1e-2, device='cpu', dtype=torch.float32):
        super(llado2025, self).__init__()

        self.fs = fs
        self.hrirs = hrirs
        self.gtfb = gtfb
        self.device = device
        self.dtype = dtype

        max_lag = np.floor(0.001*fs)
        self.lag_values = torch.arange(-max_lag, max_lag + 1, 1, dtype=dtype, device=device) / fs
        self.tau_iacc = tau_iacc    

        self._create_template(hrirs)


    def forward(self, binaural_signals):   

        interauralcues = estimate_interaural_cues(binaural_signals, self.gtfb, self.lag_values, self.tau_iacc)

        itd_ild_distance = itd_ild_template_distance(interauralcues, self.template)

        estimated_azimuth = self._estimate_azimuth(itd_ild_distance)

        kwargs = {"interauralcues": interauralcues,
                "itd_ild_distance": itd_ild_distance}

        return {"output": estimated_azimuth, "kwargs": kwargs}

    def _estimate_azimuth(self, itd_ild_distance):

        distance_combined = itd_ild_distance['distance_combined']

        estimated_azimuth_per_band = soft_argmax(-1*distance_combined, self.template['azimuths'][None,:,None], dim=1)

        estimated_azimuth = torch.mean(estimated_azimuth_per_band, dim=-1)

        return estimated_azimuth

    def _create_template(self, hrirs):

        self.template = estimate_interaural_cues(hrirs, self.gtfb, self.lag_values, self.tau_iacc)
        self.template['weigthing_function'] = torch.linspace(0, 1, self.template['itd'].shape[1], device=self.device, dtype=self.dtype)
        self.template['itd_max'] = torch.max(torch.abs(self.template['itd']), dim = 0).values
        self.template['ild_max'] = torch.max(torch.abs(self.template['ild']), dim = 0).values
        self.template['itd_normed'] = self.template['itd'] / self.template['itd_max']
        self.template['ild_normed'] = self.template['ild'] / self.template['ild_max']
        self.template['azimuths'] = torch.arange(-90, 95, 5, dtype=self.dtype, device=self.device) # Hard Coded from -90 to 90 in 5 degree steps


#####################################################################################################################    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

from models.listener_based.llado2025.grad_surgery import ItdIldDBranchGradNormFn

class llado2025_grad_surgery(nn.Module):

    def __init__(self, fs, hrirs, gtfb, tau_iacc=1e-2, device='cpu', dtype=torch.float32):
        super(llado2025_grad_surgery, self).__init__()

        self.fs = fs
        self.hrirs = hrirs
        self.gtfb, self.N_banks = gtfb, gtfb.shape[0]
        self.device = device
        self.dtype = dtype

        max_lag = np.floor(0.001*fs)
        self.lag_values = torch.arange(-max_lag, max_lag + 1, 1, dtype=dtype, device=device) / fs
        self.tau_iacc = tau_iacc    

        self._create_template(hrirs)


    def forward(self, binaural_signals):   

        interauralcues = self._estimate_interaural_cues(binaural_signals)

        itd_ild_distance = self._itd_ild_template_distance(interauralcues)

        estimated_azimuth = self._estimate_azimuth(itd_ild_distance)

        kwargs = {"interauralcues": interauralcues,
                "itd_ild_distance": itd_ild_distance}

        return {"output": estimated_azimuth, "kwargs": kwargs}

    def _estimate_azimuth(self, itd_ild_distance):

        distance_combined = itd_ild_distance['distance_combined']

        estimated_azimuth_per_band = soft_argmax(-1*distance_combined, self.template['azimuths'][None,:,None], dim=1)

        estimated_azimuth = torch.mean(estimated_azimuth_per_band, dim=-1)

        return estimated_azimuth

    def _create_template(self, hrirs):

        self.template = self._estimate_interaural_cues(hrirs)
        self.template['weigthing_function'] = torch.linspace(0, 1, self.template['itd'].shape[1], device=self.device, dtype=self.dtype)
        self.template['itd_max'] = torch.max(torch.abs(self.template['itd']), dim = 0).values
        self.template['ild_max'] = torch.max(torch.abs(self.template['ild']), dim = 0).values
        self.template['itd_normed'] = self.template['itd'] / self.template['itd_max']
        self.template['ild_normed'] = self.template['ild'] / self.template['ild_max']
        self.template['azimuths'] = torch.arange(-90, 95, 5, dtype=self.dtype, device=self.device) # Hard Coded from -90 to 90 in 5 degree steps

    def _rectify_signals(self, gtfb_signals, split_band = 11):
        LF_rectified_signals = torch.max(gtfb_signals[:,:split_band,:,:], torch.zeros_like(gtfb_signals[:,:split_band,:,:]))
        HF_rectified_signals = torch.abs(hilbert_transform(gtfb_signals[:,split_band:,:,:]))
        rectified_signals = torch.cat([LF_rectified_signals, HF_rectified_signals], dim=1)
        return rectified_signals

    def _compute_iacc(self, rectified_signals, lag_values):
        pad_length =lag_values.shape[0]//2

        left_signals = F.pad(rectified_signals[:,:,0,:], (pad_length, pad_length), mode='constant', value=0)
        right_signals = torch.flip(rectified_signals[:,:,1,:], dims=[-1])

        left_energy = compute_energy(left_signals)
        right_energy = compute_energy(right_signals)

        norm = torch.sqrt(left_energy*right_energy).clamp(min=1e-12).unsqueeze(-1) 

        iacc = TF.fftconvolve(left_signals, right_signals, mode='valid')/norm 
        return iacc

    def _compute_itd(self, iacc, tau_iacc, lag_values):

        # Earlier left signals results in positive ITD, later right signals results in negative ITD
        itd = -soft_argmax(iacc, lag_values, dim=-1, tau=tau_iacc) 

        return itd

    def _compute_ild(self, rectified_signals):
        left_signals = rectified_signals[:,:,0,:]
        right_signals = torch.flip(rectified_signals[:,:,1,:], dims=[-1])
        
        # Louder left signals results in positive ILD, louder right signals results in negative ILD
        ild = (
            20 * torch.log10(torch.sqrt(torch.sum(left_signals**2, dim=-1) + 1e-12))
            - 20 * torch.log10(torch.sqrt(torch.sum(right_signals**2, dim=-1) + 1e-12))
        )
        return ild

    def _estimate_interaural_cues(self, binaural_signals):

        N_signals, N_ears, N_samples = binaural_signals.shape

        # Reshape the binaural signals to (N_signals, N_listeners, 1, N_ears, N_samples)
        binaural_signals = binaural_signals.view(N_signals, 1, N_ears, N_samples)
        gtfb = self.gtfb.view(1, self.N_banks, 1, -1)

        # Convolve the binaural signals with the gtfb || gtfb_signals.shape = (N_signals, N_listeners, N_banks, N_ears, N_samples)
        gtfb_signals = TF.fftconvolve(binaural_signals, gtfb)

        # Rectify the gtfb signals
        rectified_signals = self._rectify_signals(gtfb_signals)

        rectified_itd, rectified_ild = ItdIldDBranchGradNormFn.apply(rectified_signals)

        iacc = self._compute_iacc(rectified_itd, self.lag_values)

        # Estimate the interaural cues
        itd = self._compute_itd(iacc, self.tau_iacc, self.lag_values)
        ild = self._compute_ild(rectified_ild)

        return {"itd": itd, 
                "ild": ild,
                "gtfb_signals": gtfb_signals,
                "rectified_signals_itd": rectified_itd,
                "rectified_signals_ild": rectified_ild,
                "rectified_signals_grad_itd": ItdIldDBranchGradNormFn.last_itd_grad,
                "rectified_signals_grad_ild": ItdIldDBranchGradNormFn.last_ild_grad,
                "iacc": iacc}
    

    
    def _itd_ild_template_distance(self, interauralcues):

        weigthing_function = self.template['weigthing_function'][None, None, :]
        max_template_itd = self.template['itd_max']
        max_template_ild = self.template['ild_max']
        template_normed_itd = self.template['itd_normed']
        template_normed_ild = self.template['ild_normed']

        est_normed_itd = interauralcues['itd'][:, None, :] / max_template_itd[None, None, :]
        est_normed_ild = interauralcues['ild'][:, None, :] / max_template_ild[None, None, :]
        
        itd_distance = torch.pow(torch.abs(est_normed_itd - template_normed_itd).clamp(min=1e-8), 0.7)
        ild_distance = torch.pow(torch.abs(est_normed_ild - template_normed_ild).clamp(min=1e-8), 0.7)

        distance_combined = torch.pow(itd_distance * (1 - weigthing_function) + ild_distance * weigthing_function, 1)

        distance = {
            "itd_distance": itd_distance,
            "ild_distance": ild_distance,
            "distance_combined": distance_combined,
        }

        return distance




