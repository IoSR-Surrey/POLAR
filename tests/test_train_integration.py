from __future__ import annotations

import pytest

pytest.importorskip("torch.utils.tensorboard")

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.datasets.listener_dataset import get_listener_loader
from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.optimisation.custom_modules import MultiSourceFIR
from scripts.train.train import ListenerBasedLoss, Train


class IdentityPropagation(nn.Module):
    def forward(self, stimulus, spk_filters, listener_indices=None) -> torch.Tensor:
        n_listeners = 1 if listener_indices is None else len(listener_indices)
        left = spk_filters[:, 0, :]
        right = spk_filters[:, 1, :]
        bin_stim = torch.stack([left, right], dim=1).unsqueeze(1)
        return bin_stim.expand(-1, n_listeners, -1, -1)


class DifferenceModel(nn.Module):
    def __init__(self, target: float) -> None:
        super().__init__()
        self.target = torch.tensor([target], dtype=torch.float32)
        self.loss = F.mse_loss
        self.loss_norm = 1.0
        self.ref_flag = 0

    def forward(self, bin_stim: torch.Tensor) -> torch.Tensor:
        if bin_stim.ndim == 4:
            bin_stim = bin_stim[:, 0]
        left = bin_stim[:, 0:1, :].mean(dim=-1)
        right = bin_stim[:, 1:2, :].mean(dim=-1)
        return (left - right).view(bin_stim.shape[0], 1)


LOSS_KEY = "difference"


def _estimate_localisation(trainer: Train) -> float:
    unit_input = torch.ones(1, 1, device=trainer.device, dtype=trainer.dtype)
    with torch.no_grad():
        spk_filters = trainer.nn(unit_input)
        listener_idx = torch.arange(len(trainer.lstn_loader.dataset), device=trainer.device)
        stimulus = next(iter(trainer.stim_loader)).to(device=trainer.device, dtype=trainer.dtype)
        bin_stim = trainer.prop(stimulus, spk_filters, listener_idx)
        _, estimation = trainer.loss_dict[LOSS_KEY].compute_loss(
            bin_stim=bin_stim,
            spk_filters=spk_filters,
            stim=stimulus,
            listener_indices=listener_idx,
        )
    return float(estimation.mean())


def test_train_moves_estimate_towards_target() -> None:
    # This is a lightweight integration test for the actual Train loop:
    # - the project NN is used unchanged,
    # - the optimiser is the real project optimiser,
    # - the Train class performs real forward/backward updates.
    # We start from init=3, which creates all-zero filters, so the initial estimate is exactly 0.
    nn_model = MultiSourceFIR(
        n_samples=8,
        n_sources=2,
    ).to(device="cpu", dtype=torch.float32)

    with torch.no_grad():
        for branch in nn_model.branches:
            branch.weight.fill_(0.0)
            branch.weight[0, 0] = 1.0

    optimiser = torch.optim.Adam(nn_model.parameters(), lr=0.05)

    localisation_model = DifferenceModel(target=1.0)
    stim_loader, _ = get_stim_loader(
        fs=48_000,
        batch_size=1,
        n_stimulus=1,
        stimulus_type="impulse",
        device="cpu",
        dtype=torch.float32,
        shuffle=False,
    )
    lstn_pos = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    lstn_loader, _ = get_listener_loader(lstn_pos, batch_size=1, symmetric=False)

    trainer = Train(
        n_epoch=80,
        stim_loader=stim_loader,
        lstn_loader=lstn_loader,
        nn=nn_model,
        prop=IdentityPropagation(),
        loss_dict={
            LOSS_KEY: ListenerBasedLoss(
                model=localisation_model,
                weight=1.0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ),
        },
        optimiser=optimiser,
        logger=None,
        scheduler=None,
    )

    initial_error = abs(_estimate_localisation(trainer) - 1.0)
    trainer.train()
    final_error = abs(_estimate_localisation(trainer) - 1.0)

    assert initial_error > final_error
    assert 0.15 > final_error
