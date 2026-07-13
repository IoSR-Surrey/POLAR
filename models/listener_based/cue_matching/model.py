import torch
import numpy as np
import torch.nn as nn
import torchaudio.functional as TF

from utils.peripheral_processing import rectify_signals, compute_iacc, compute_itd, compute_ild
from .helpers import create_interaural_template
from ..llado2025.grad_surgery import ItdIldDBranchGradNormFn

class CueMatchingModel(nn.Module):
    def __init__(self, fs, hrirs, gtfb, tau_iacc=1e-2, device='cpu', dtype=torch.float32):
            super(CueMatchingModel, self).__init__()

            self.model_type = 'listener_based'

            self.fs = fs
            self.hrirs = hrirs
            self.gtfb, self.N_banks = gtfb, gtfb.shape[0]
            self.device = device
            self.dtype = dtype

            max_lag = np.floor(0.001*fs)
            self.lag_values = torch.arange(-max_lag, max_lag + 1, 1, dtype=dtype, device=device) / fs
            self.tau_iacc = tau_iacc    

            self.template = create_interaural_template(hrirs, gtfb, self.lag_values, self.tau_iacc, self.device, self.dtype)


    def forward(self, binaural_signals, reference_stim):   

        interauralcues = self._estimate_interaural_cues(binaural_signals)

        cues = torch.stack([interauralcues['itd'], interauralcues['ild']], dim=1)

        return {"output": cues, "kwargs": {"interauralcues": interauralcues}}

    
    def _estimate_interaural_cues(self, binaural_signals):

        N_signals, N_ears, N_samples = binaural_signals.shape

        # Reshape the binaural signals to (N_signals, N_listeners, 1, N_ears, N_samples)
        binaural_signals = binaural_signals.view(N_signals, 1, N_ears, N_samples)
        gtfb = self.gtfb.view(1, self.N_banks, 1, -1)

        # Convolve the binaural signals with the gtfb || gtfb_signals.shape = (N_signals, N_listeners, N_banks, N_ears, N_samples)
        gtfb_signals = TF.fftconvolve(binaural_signals, gtfb)

        # Rectify the gtfb signals
        rectified_signals = rectify_signals(gtfb_signals)

        # Split into ITD/ILD branches so backward can route separate gradients
        rectified_itd, rectified_ild = ItdIldDBranchGradNormFn.apply(rectified_signals)

        iacc = compute_iacc(rectified_itd, self.lag_values)

        # Estimate the interaural cues
        itd = compute_itd(iacc, self.tau_iacc, self.lag_values)
        ild = compute_ild(rectified_ild)

        return {"itd": itd, 
                "ild": ild,
                "gtfb_signals": gtfb_signals,
                "rectified_signals": rectified_signals,
                "rectified_signals_itd": rectified_itd,
                "rectified_signals_ild": rectified_ild,
                "rectified_signals_grad_itd": ItdIldDBranchGradNormFn.last_itd_grad,
                "rectified_signals_grad_ild": ItdIldDBranchGradNormFn.last_ild_grad,
                "iacc": iacc}