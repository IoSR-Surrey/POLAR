import torch
import torch.nn as nn
import sofar as sf
import torchaudio.functional as TF

from propagation.helpers import (_calculate_source_angle_2D, 
                                _calculate_distance_gain, 
                                _apply_delay, 
                                _calculate_delay_samples)

from utils.sofa_functions import sofa_find, get_IRs

class SingleSourceBinauralFreeFieldPropagation(nn.Module):
    """One source, multiple listeners. Returns binaural (L/R ear) per listener."""

    def __init__(self, 
        lstn_pos, 
        source_pos, 
        sofaile, 
        fs=48000, 
        delay_samples=None, 
        apply_distance_attenuation=False, 
        device = 'cpu', 
        dtype = torch.float32
        ):

        super().__init__()
        
        self.fs = fs
        self.device = device
        self.dtype = dtype
        self.apply_distance_attenuation = apply_distance_attenuation

        self.lstn_pos = lstn_pos
        self.n_listeners = len(self.lstn_pos)
        self.source_pos = source_pos
        self.sofaile = sofaile

        # Calculate the source angle and distance gain
        source_angle = _calculate_source_angle_2D(lstn_pos, source_pos)
        self.distance_gain = _calculate_distance_gain(lstn_pos, source_pos)

        # SOFA index per listener and HRIRs: (n_listeners, 2, ir_len) — 2 = left ear, right ear
        src_idx = sofa_find(source_angle.T, self.sofaile)
        self.hrirs = get_IRs(src_idx, sofaile=self.sofaile, device=self.device, dtype=self.dtype)
        
        # Optional per-listener delay (samples) to align with another source
        if delay_samples is not None:
            self.hrirs = _apply_delay(self.hrirs, delay_samples)
                    

    def forward(self, stim, source_filter, listener_indices=None):
        """
        stim: (batch, 1, stim_len)
        source_filter: (1, 1, filter_len)
        Returns: (batch, n_listeners, 2, T + filter_len + ir_len - 2) — 2 = L/R ear.
        """
        if listener_indices is None:
            listener_indices = torch.arange(self.n_listeners, device=self.device)
        
        n_stim = len(stim)
        n_lstn = len(listener_indices)
        hrirs = self.hrirs[listener_indices].reshape(1, 2*n_lstn, -1)

        src_signal = TF.convolve(stim, source_filter, mode='full')
        ear_signals = TF.convolve(src_signal, hrirs, mode='full').reshape(n_stim, n_lstn, 2, -1)

        if self.apply_distance_attenuation:
            distance_gain = self.distance_gain[listener_indices].view(1, n_lstn, 1, 1)
            ear_signals = ear_signals * distance_gain
        return ear_signals



## Multiple source anechoic class definition (omitting the "MultipleSource" prefix for brevity)
class BinauralFreeFieldPropagation(nn.Module):
    """Generic multi-source anechoic propagation with summed binaural output."""

    def __init__(self, 
        fs, 
        lstn_pos, 
        spk_pos, 
        sofaile, 
        apply_distance_attenuation=False, 
        device = 'cpu', 
        dtype = torch.float32
        ):

        super().__init__()
        
        self.fs = fs
        self.sofaile = sofaile
        self.apply_distance_attenuation = apply_distance_attenuation
        self.dtype = dtype
        self.device = device

        self.lstn_pos = lstn_pos
        self.n_listeners = len(self.lstn_pos)
        self.spk_pos = spk_pos
        self.n_sources = self.spk_pos.shape[0]

        # Time of arrival at each listener: (n_sources, n_listeners)
        self.delay_samples, self.max_delay_samples = _calculate_delay_samples(lstn_pos, spk_pos, fs)
        
        self.sources = self._create_multi_sources()

    def forward(self, stim, spk_filters, listener_indices=None):
        if listener_indices is None:
            listener_indices = torch.arange(self.n_listeners, device=self.device)

        if spk_filters.shape[1] != self.n_sources:
            raise ValueError(
                f"Expected spk_filters with {self.n_sources} sources on dim=1, "
                f"got {spk_filters.shape[1]}"
            )

        binaural_sum = 0
        for source_idx, source in enumerate(self.sources):
            source_filter = spk_filters[:, source_idx:source_idx + 1, :]
            binaural_sum = binaural_sum + source(stim, source_filter, listener_indices=listener_indices)

        return binaural_sum

    def _create_multi_sources(self):
        sources = nn.ModuleList()
        for source_idx in range(self.n_sources):
            source_delay_samples = self.delay_samples[source_idx]
            sources.append(
                SingleSourceBinauralFreeFieldPropagation(
                    lstn_pos=self.lstn_pos,
                    source_pos=self.spk_pos[source_idx:source_idx+1],
                    sofaile=self.sofaile,
                    fs=self.fs,
                    delay_samples=[source_delay_samples, self.max_delay_samples],
                    apply_distance_attenuation=self.apply_distance_attenuation,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        return sources