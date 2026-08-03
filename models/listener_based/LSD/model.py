import torch
import torch.nn as nn

import torchaudio.functional as TF

from utils.sofa_functions import sofa_find, get_IRs

class LSD(nn.Module):
    def __init__(self, fs, sofaile, target_angles, n_fft = 4096, erb_weighting = True, f_min = 20.0, f_max = 20000.0, device='cpu', dtype=torch.float32):
        super(LSD, self).__init__()

        self.model_type = 'listener_based'

        self.fs = fs
        self.n_fft = n_fft
        self.target_angles = target_angles
        self.erb_weighting = erb_weighting
        self.f_min = f_min
        self.f_max = f_max
        self.device = device
        self.dtype = dtype

        target_idx = sofa_find(self.target_angles, sofaile)
        self.target_hrirs = get_IRs(target_idx, sofaile, device=device, dtype=dtype)

        self.fft_frequencies = torch.fft.rfftfreq(n_fft, 1/fs, device=device, dtype=dtype)
        self.erb_frequency_bin_weights = self._erb_frequency_bin_weights()

        self.freq_bin_mask = (self.fft_frequencies >= self.f_min) & (self.fft_frequencies <= self.f_max)


    def forward(self, binaural_signals, reference_signals, listener_indices = None):

        binaural_reference_signals = self.compute_binaural_reference_signals(reference_signals, listener_indices)
        lsd = self.log_spectral_distortion(binaural_signals, binaural_reference_signals)

        return {"output": lsd, "kwargs": {"binaural_reference_signals": binaural_reference_signals}}

    def log_spectral_distortion(self, binaural_signals, binaural_reference_signals):

        X = torch.fft.rfft(binaural_signals, n=self.n_fft, dim=-1)
        Y = torch.fft.rfft(binaural_reference_signals, n=self.n_fft, dim=-1)

        X_mag = torch.abs(X).clamp(min=1e-12)
        Y_mag = torch.abs(Y).clamp(min=1e-12)

        X_dB= 20*torch.log10(X_mag)
        Y_dB= 20*torch.log10(Y_mag)

        squared_error = (X_dB - Y_dB) ** 2
        squared_error = squared_error[..., self.freq_bin_mask]

        if self.erb_weighting:
            squared_error = squared_error * self.erb_frequency_bin_weights[..., self.freq_bin_mask]
            squared_error = squared_error.sum(dim=-1)
        else:
            squared_error = squared_error.mean(dim=-1)

        lsd_freq = torch.sqrt(squared_error)

        lsd = torch.mean(lsd_freq, dim=-1)

        return lsd


    def compute_binaural_reference_signals(self, reference_signals, listener_indices = None):

        if listener_indices is None:
            listener_indices = torch.arange(len(self.target_hrirs), device=self.device)
        
        n_stim = len(reference_signals)
        n_lstn = len(listener_indices)
        hrirs = self.target_hrirs[listener_indices].reshape(1, 2*n_lstn, -1)

        ear_signals = TF.convolve(reference_signals, hrirs, mode='full').view(n_stim, n_lstn, 2, -1)

        return ear_signals.view(n_stim*n_lstn, 2, -1)

    
    def _erb_frequency_bin_weights(self, eps=1e-12):
        """Per-FFT-bin ERB weights with sum(weights) == M (number of bins).

        Relative shape is (1/ERB) / max(1/ERB); ERB = 0.108 * freq + 24.7.
        """
        erb = 0.108 * self.fft_frequencies + 24.7
        inv_erb = 1.0 / erb.clamp(min=eps)
        w = inv_erb / inv_erb.max()

        return w / w.sum()

        