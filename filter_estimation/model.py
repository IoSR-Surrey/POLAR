import numpy as np
import torch
import torch.nn as nn
from scipy.signal import firwin


def _build_split_cues_weights(n_taps, n_sources, fs=48000, n_bands=4, band_mode="linear", f_min=20.0):
    if band_mode not in ("linear", "log"):
        raise ValueError(f"band_mode must be 'linear' or 'log', got {band_mode!r}.")

    fir_taps = n_taps if n_taps % 2 == 1 else n_taps + 1
    if band_mode == "linear":
        edges = np.linspace(0, fs / 2, n_bands + 1)
    else:
        edges = np.geomspace(f_min, fs / 2, n_bands + 1)
    band_filters = []
    for band_idx in range(n_bands):
        low_hz, high_hz = edges[band_idx], edges[band_idx + 1]
        if low_hz <= 0:
            band_filter = firwin(fir_taps, cutoff=high_hz, fs=fs, pass_zero=True)
        elif high_hz >= fs / 2:
            band_filter = firwin(fir_taps, cutoff=low_hz, fs=fs, pass_zero=False)
        else:
            band_filter = firwin(fir_taps, cutoff=[low_hz, high_hz], fs=fs, pass_zero=False)
        if len(band_filter) != n_taps:
            start = (len(band_filter) - n_taps) // 2
            band_filter = band_filter[start:start + n_taps]
        band_filters.append(band_filter)

    filters = [np.zeros(n_taps, dtype=np.float64) for _ in range(n_sources)]
    for band_idx, band_filter in enumerate(band_filters):
        filters[band_idx % n_sources] += band_filter
    return filters


class MultiSourceFIR(nn.Module):
    """One FIR (Linear) layer per source; forward returns (batch, source, taps)."""

    def __init__(self, n_samples, n_sources, init_weight=None):
        super().__init__()
        self.n_sources = n_sources
        self.n_samples = n_samples
        self.branches = nn.ModuleList([
            nn.Linear(1, n_samples, bias=False)
            for _ in range(n_sources)
        ])

        if init_weight is not None:
            self.set_filter_initialisation(init_weight)

    def set_filter_initialisation(self, init_weight, branch_index = None, fs=48000, n_bands=4, band_mode="linear", f_min=20.0) -> None:
        """
        Sets the filter weights for all branches in-place.

        - If init_weight is a Tensor: copy it into each branch's weights.
        - If init_weight == "middle": set the weight to an impulse at the middle index.
        - If init_weight == "split_cues": bands over [0, fs/2] or [f_min, fs/2] assigned round-robin to sources.
        - If init_weight == "middle_opp": set weights to an positive and negative impulse at the middle index round-robin to sources.
          band_mode "linear" (default) or "log" (geomspace, matching stimulus_dataset log bands).
        """
        with torch.no_grad():
            if isinstance(init_weight, str) and init_weight.lower() == "split_cues":
                split_weights = _build_split_cues_weights(
                    self.n_samples, self.n_sources, fs=fs, n_bands=n_bands, band_mode=band_mode, f_min=f_min
                )
                branch_indices = range(self.n_sources) if branch_index is None else [branch_index]
                for branch_idx in branch_indices:
                    weight = self.branches[branch_idx].weight
                    weight.copy_(
                        torch.as_tensor(split_weights[branch_idx], device=weight.device, dtype=weight.dtype).reshape_as(weight)
                    )
                return

            branches = self.branches if branch_index is None else [self.branches[branch_index]]
            for i, branch in enumerate(branches):
                weight = branch.weight
                if isinstance(init_weight, str) and init_weight.lower() == "middle":
                    weight.zero_()
                    middle_idx = weight.shape[0] // 2
                    weight[middle_idx, 0] = 1.0
                elif isinstance(init_weight, str) and init_weight.lower() == "random":
                    weight.uniform_(-0.5, 0.5)
                elif isinstance(init_weight, str) and init_weight.lower() == "one_on":
                    weight.zero_()
                    if i == 0:
                        middle_idx = weight.shape[0] // 2
                        weight[middle_idx, 0] = 1.0
                elif isinstance(init_weight, str) and init_weight.lower() == "middle_opp":
                    weight.zero_()
                    middle_idx = weight.shape[0] // 2
                    weight[middle_idx, 0] = (-1.0)**i
                else:
                    weight.copy_(init_weight.reshape_as(weight))

       
       

    def forward(self, x):
        # Expects x shaped (batch, 1); returns stacked FIRs shaped (batch, n_sources, n_samples).
        return torch.stack([b(x) for b in self.branches], dim=1)

    def __getitem__(self, i):
        """So downstream can use nn[0], nn[-1] for device/dtype or first/last branch."""
        return self.branches[i]

