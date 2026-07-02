import torch
import torch.nn as nn



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

    def set_filter_initialisation(self, init_weight, branch_index = None) -> None:
        """
        Sets the filter weights for all branches in-place.

        - If init_weight is a Tensor: copy it into each branch's weights.
        - If init_weight == "middle": set the weight to an impulse at the middle index.
        """
        with torch.no_grad():
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
                else:
                    weight.copy_(init_weight.reshape_as(weight))

       
       

    def forward(self, x):
        # Expects x shaped (batch, 1); returns stacked FIRs shaped (batch, n_sources, n_samples).
        return torch.stack([b(x) for b in self.branches], dim=1)

    def __getitem__(self, i):
        """So downstream can use nn[0], nn[-1] for device/dtype or first/last branch."""
        return self.branches[i]

