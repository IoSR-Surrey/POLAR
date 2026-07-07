import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cagrad import CAGrad
from models.train import ModelTrainer
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.louspeaker_based.second_moment.model import SecondMoment
from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import load_npy_to_torch


class _SingleTaskNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.5, -0.5]))

    def loss_dict(self) -> dict[str, torch.Tensor]:
        loss = (self.weight[0] - 1.0) ** 2 + (self.weight[1] + 2.0) ** 2
        return {"task": loss, "total": loss}


class _SharedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weight[0] ** 2, (self.weight[1] - 1.0) ** 2


def _flatten_grads(params):
    return torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
def test_cagrad_single_task_matches_plain_backward(c: float) -> None:
    torch.manual_seed(0)
    net_plain = _SingleTaskNet()
    net_cagrad = _SingleTaskNet()
    net_cagrad.load_state_dict(net_plain.state_dict())

    optimizer_plain = torch.optim.SGD(net_plain.parameters(), lr=0.1)
    optimizer_cagrad = torch.optim.SGD(net_cagrad.parameters(), lr=0.1)
    cagrad = CAGrad(["task"], c=c)

    for _ in range(20):
        loss_dict_plain = net_plain.loss_dict()
        loss_dict_cagrad = net_cagrad.loss_dict()

        optimizer_plain.zero_grad()
        loss_dict_plain["total"].backward()
        optimizer_plain.step()

        optimizer_cagrad.zero_grad()
        cagrad.step(loss_dict_cagrad, list(net_cagrad.parameters()), optimizer_cagrad)

    assert torch.allclose(net_plain.weight, net_cagrad.weight)
    assert torch.allclose(
        loss_dict_plain["total"],
        loss_dict_cagrad["total"],
    )


def test_cagrad_improves_worst_task_alignment_on_conflicting_tasks() -> None:
    net = _SharedNet()
    shared_params = list(net.parameters())
    loss_a, loss_b = net()
    loss_dict = {"task_a": loss_a, "task_b": loss_b, "total": loss_a + loss_b}

    grads_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True)
    grads_b = torch.autograd.grad(loss_b, shared_params)
    g_a = torch.cat([g.reshape(-1) for g in grads_a])
    g_b = torch.cat([g.reshape(-1) for g in grads_b])
    g0 = (g_a + g_b) / 2.0

    cagrad = CAGrad(["task_a", "task_b"], c=0.5)
    combined_grad, _, _ = cagrad._solve_dual([g_a, g_b])

    plain_alignment = min(torch.dot(g_a, g0).item(), torch.dot(g_b, g0).item())
    cagrad_alignment = min(torch.dot(g_a, combined_grad).item(), torch.dot(g_b, combined_grad).item())

    assert cagrad_alignment >= plain_alignment


def test_cagrad_step_updates_shared_params() -> None:
    net = _SharedNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    cagrad = CAGrad(["task_a", "task_b"], c=0.5)

    loss_a, loss_b = net()
    loss_dict = {"task_a": loss_a, "task_b": loss_b, "total": loss_a + loss_b}
    initial_weight = net.weight.detach().clone()

    optimizer.zero_grad()
    info = cagrad.step(loss_dict, list(net.parameters()), optimizer)

    assert net.weight.detach().equal(initial_weight) is False
    assert info["task_coeffs"].shape == (2,)
    assert info["task_weights"].shape == (2,)
    assert info["effective_grad"].shape == (2,)
    assert torch.isfinite(info["effective_grad"]).all()
    assert set(info["raw_grads"]) == {"task_a", "task_b"}


def test_cagrad_integration_smoke() -> None:
    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
    d = 2.0
    lstn_pos = torch.tensor([[0.0, 0.0]])
    spk_pos = torch.tensor(
        [
            [d * torch.cos(torch.deg2rad(torch.tensor(30.0))), d * torch.sin(torch.deg2rad(torch.tensor(30.0)))],
            [d * torch.cos(torch.deg2rad(torch.tensor(30.0))), -d * torch.sin(torch.deg2rad(torch.tensor(30.0)))],
        ]
    )

    llado = llado2025_grad_surgery(48000, hrirs, gtfb)
    second = SecondMoment(32, 2)
    trainer = ModelTrainer([llado, second], weights=[1.0, 1.0])
    prop = BinauralFreeFieldPropagation(48000, lstn_pos, spk_pos, sofaile)
    nn = MultiSourceFIR(32, 2)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    cagrad = CAGrad(list(trainer.model_dict.keys()), c=0.5)
    targets = {
        "llado2025_grad_surgery": torch.tensor([[15.0]]),
        "SecondMoment": torch.tensor([[0.0, 0.0]]),
    }

    for _ in range(3):
        spk_filters = nn(torch.ones(1, 1))
        binaural = prop(torch.ones(1, 1, 1), spk_filters)
        trainer.compute_estimates(binaural, spk_filters)
        loss_dict = trainer.compute_loss(targets)

        optimizer.zero_grad()
        info = cagrad.step(loss_dict, list(nn.parameters()), optimizer)

        assert torch.isfinite(info["effective_grad"]).all()
        assert info["task_coeffs"].shape == (2,)
