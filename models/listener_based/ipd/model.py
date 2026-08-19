import torch
import torch.nn as nn
import torchaudio.functional as TF

from utils.sofa_functions import sofa_find, get_IRs


def  _wrap_to_pi(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


class IPD(nn.Module):
    """Interaural phase difference mismatch relative to a binaural reference.

    Expected inputs (same contract as LSD):
        binaural_signals: (n_stim * n_listeners, 2, T)
        reference_signals: (n_stim, 1, T)

    Output:
        {"output": (n_stim * n_listeners,) IPD error, "kwargs": ...}

    Notes:
        - Natural training target is 0 (difference-to-reference score).
        - ``use_mse=False`` (default) returns mean absolute wrapped phase error;
          ``use_mse=True`` returns mean squared wrapped phase error.
        - Optional ``erb_weighting`` uses the same per-bin ERB weights as LSD.
    """

    def __init__(
        self,
        fs,
        sofaile,
        target_angles,
        n_fft=4096,
        erb_weighting=False,
        f_min=200.0,
        f_max=7000.0,
        use_mse=False,
        device="cpu",
        dtype=torch.float32,
    ):
        super(IPD, self).__init__()

        self.model_type = "listener_based"

        self.fs = fs
        self.n_fft = int(n_fft)
        self.target_angles = target_angles
        self.erb_weighting = erb_weighting
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.use_mse = use_mse
        self.device = device
        self.dtype = dtype

        target_idx = sofa_find(self.target_angles, sofaile)
        self.target_hrirs = get_IRs(target_idx, sofaile, device=device, dtype=dtype)

        self.fft_frequencies = torch.fft.rfftfreq(self.n_fft, 1 / fs, device=device, dtype=dtype)
        self.erb_frequency_bin_weights = self._erb_frequency_bin_weights(self.fft_frequencies)

        self.freq_bin_mask = (self.fft_frequencies >= self.f_min) & (self.fft_frequencies <= self.f_max)
        if not self.freq_bin_mask.any():
            raise ValueError(
                f"No FFT bins in [{self.f_min}, {self.f_max}] Hz for fs={fs} and n_fft={self.n_fft}."
            )

    def forward(self, binaural_signals, reference_signals, listener_indices=None):
        binaural_reference_signals = self.compute_binaural_reference_signals(
            reference_signals, listener_indices
        )
        ipd_error = self.interaural_phase_difference_error(
            binaural_signals, binaural_reference_signals
        )
        return {
            "output": ipd_error,
            "kwargs": {"binaural_reference_signals": binaural_reference_signals},
        }

    def interaural_phase_difference_error(self, binaural_signals, binaural_reference_signals):
        if binaural_signals.shape[-2] != 2:
            raise ValueError("Expected binaural_signals with 2 ears on dim=-2.")
        if binaural_reference_signals.shape[-2] != 2:
            raise ValueError("Expected binaural_reference_signals with 2 ears on dim=-2.")
        if binaural_signals.shape[:-1] != binaural_reference_signals.shape[:-1]:
            raise ValueError(
                "Expected binaural_signals and binaural_reference_signals to match on all but the time dim, "
                f"got {binaural_signals.shape} and {binaural_reference_signals.shape}."
            )

        # Pad to at least n_fft, or to the longer ear signal, matching LSD.
        n_fft = max(self.n_fft, binaural_signals.shape[-1], binaural_reference_signals.shape[-1])
        _, freq_bin_mask, erb_weights = self._fft_bin_metadata(n_fft)

        bin_f = torch.fft.rfft(binaural_signals, n=n_fft, dim=-1)
        ref_f = torch.fft.rfft(binaural_reference_signals, n=n_fft, dim=-1)

        # IPD := phase(L) - phase(R); positive = left leads
        ipd_bin = _wrap_to_pi(torch.angle(bin_f[..., 0, :]) - torch.angle(bin_f[..., 1, :]))
        ipd_ref = _wrap_to_pi(torch.angle(ref_f[..., 0, :]) - torch.angle(ref_f[..., 1, :]))

        ipd_err = _wrap_to_pi(ipd_bin - ipd_ref)[..., freq_bin_mask]
        bin_err = ipd_err ** 2 if self.use_mse else torch.abs(ipd_err)

        if self.erb_weighting:
            weights = erb_weights[..., freq_bin_mask]
            return (bin_err * weights).sum(dim=-1)

        return bin_err.mean(dim=-1)

    def compute_binaural_reference_signals(self, reference_signals, listener_indices=None):
        if listener_indices is None:
            listener_indices = torch.arange(len(self.target_hrirs), device=self.device)

        n_stim = len(reference_signals)
        n_lstn = len(listener_indices)
        hrirs = self.target_hrirs[listener_indices].reshape(1, 2 * n_lstn, -1)

        ear_signals = TF.convolve(reference_signals, hrirs, mode="full").view(n_stim, n_lstn, 2, -1)

        return ear_signals.view(n_stim * n_lstn, 2, -1)

    def _fft_bin_metadata(self, n_fft):
        if n_fft == self.n_fft:
            return self.fft_frequencies, self.freq_bin_mask, self.erb_frequency_bin_weights
        fft_frequencies = torch.fft.rfftfreq(n_fft, 1 / self.fs, device=self.device, dtype=self.dtype)
        freq_bin_mask = (fft_frequencies >= self.f_min) & (fft_frequencies <= self.f_max)
        return fft_frequencies, freq_bin_mask, self._erb_frequency_bin_weights(fft_frequencies)

    def _erb_frequency_bin_weights(self, fft_frequencies, eps=1e-12):
        """Per-FFT-bin ERB weights with sum(weights) == 1 (same as LSD)."""
        erb = 0.108 * fft_frequencies + 24.7
        inv_erb = 1.0 / erb.clamp(min=eps)
        w = inv_erb / inv_erb.max()
        return w / w.sum()
