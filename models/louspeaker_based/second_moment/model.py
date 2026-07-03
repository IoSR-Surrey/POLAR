import torch
import torch.nn as nn

class SecondMoment(nn.Module):
    def __init__(self, n_samples, n_sources, device='cpu', dtype=torch.float32):
        super().__init__()

        self.model_type = 'loudspeaker_based'

        self.n_samples = n_samples
        self.n_sources = n_sources
        
        self.indices = torch.arange(n_samples, dtype=dtype, device=device)

    def forward(self, spk_filters):

        filter_energy = torch.sum(spk_filters**2, dim=-1).clamp(min=1e-12)

        filter_energy_mean = torch.sum(self.indices.view(1, 1, self.n_samples)  * spk_filters**2, dim=-1) / filter_energy

        second_moment = torch.sum((self.indices.view(1, 1, self.n_samples) - filter_energy_mean.view(1, self.n_sources, 1))**2 * spk_filters**2, dim=-1) / filter_energy

        kwargs = {"filter_energy": filter_energy, "filter_energy_mean": filter_energy_mean}

        return {"output": second_moment, "kwargs": kwargs}

