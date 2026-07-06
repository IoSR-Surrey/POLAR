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

from models.gradnorm import GradNorm


class _SharedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weight[0] ** 2, (self.weight[1] - 1.0) ** 2


class _SingleTaskNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.5, -0.5]))

    def loss_dict(self) -> dict[str, torch.Tensor]:
        loss = (self.weight[0] - 1.0) ** 2 + (self.weight[1] + 2.0) ** 2
        return {"task": loss, "total": loss}


class _TwoSlopeLinearNet(nn.Module):
    """Shared scalar theta with task losses L_i = slope_i * theta."""

    def __init__(self, flat_slope: float = 1.0, steep_slope: float = 10.0) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(1.0))
        self.flat_slope = flat_slope
        self.steep_slope = steep_slope

    def loss_dict(self) -> dict[str, torch.Tensor]:
        flat = self.flat_slope * self.theta
        steep = self.steep_slope * self.theta
        return {"flat": flat, "steep": steep, "total": flat + steep}


def _unweighted_grad_norm_ratio(loss_dict, task_names, shared_params) -> float:
    norms = []
    for name in task_names:
        grads = torch.autograd.grad(
            loss_dict[name],
            shared_params,
            retain_graph=True,
            allow_unused=True,
        )
        norms.append(torch.norm(torch.cat([g.reshape(-1) for g in grads if g is not None])))
    return (norms[1] / norms[0]).item()


def test_gradnorm_single_task_matches_plain_backward() -> None:
    torch.manual_seed(0)
    net_plain = _SingleTaskNet()
    net_gradnorm = _SingleTaskNet()
    net_gradnorm.load_state_dict(net_plain.state_dict())

    optimizer_plain = torch.optim.SGD(net_plain.parameters(), lr=0.1)
    optimizer_gradnorm = torch.optim.SGD(net_gradnorm.parameters(), lr=0.1)
    gradnorm = GradNorm(["task"], alpha=1.5, lr=0.1)

    for _ in range(20):
        loss_dict_plain = net_plain.loss_dict()
        loss_dict_gradnorm = net_gradnorm.loss_dict()

        optimizer_plain.zero_grad()
        loss_dict_plain["total"].backward()
        optimizer_plain.step()

        optimizer_gradnorm.zero_grad()
        gradnorm.step(loss_dict_gradnorm, list(net_gradnorm.parameters()), optimizer_gradnorm)

    assert torch.allclose(net_plain.weight, net_gradnorm.weight)
    assert torch.allclose(
        loss_dict_plain["total"],
        loss_dict_gradnorm["total"],
    )


def test_gradnorm_step_updates_shared_params_and_weights() -> None:
    net = _SharedNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    gradnorm = GradNorm(["task_a", "task_b"], alpha=1.5, lr=0.025)

    loss_dict = {"task_a": net()[0], "task_b": net()[1], "total": net()[0] + net()[1]}
    initial_weight = net.weight.detach().clone()

    optimizer.zero_grad()
    info = gradnorm.step(loss_dict, list(net.parameters()), optimizer)

    assert net.weight.detach().equal(initial_weight) is False
    assert info["task_weights"].shape == (2,)
    assert info["grad_norms"].shape == (2,)
    assert torch.isfinite(info["weighted_total"])
    assert gradnorm.initial_losses is not None


def test_gradnorm_balances_linear_task_slopes() -> None:
    flat_slope, steep_slope = 1.0, 10.0
    task_names = ["flat", "steep"]
    net = _TwoSlopeLinearNet(flat_slope, steep_slope)
    shared_params = list(net.parameters())
    optimizer = torch.optim.SGD(shared_params, lr=0.0)
    gradnorm = GradNorm(task_names, alpha=1.5, lr=0.1)

    baseline_ratio = _unweighted_grad_norm_ratio(net.loss_dict(), task_names, shared_params)
    assert baseline_ratio == pytest.approx(steep_slope / flat_slope)

    info = None
    for _ in range(200):
        optimizer.zero_grad()
        info = gradnorm.step(net.loss_dict(), shared_params, optimizer)

    balanced_ratio = (info["grad_norms"][1] / info["grad_norms"][0]).item()

    assert balanced_ratio < baseline_ratio
    assert balanced_ratio == pytest.approx(1.0, rel=0.15)
    assert info["task_weights"][1].item() < info["task_weights"][0].item()


def test_gradnorm_respects_task_importance() -> None:
    flat_slope, steep_slope = 1.0, 10.0
    importance_ratio = 3.0
    task_names = ["flat", "steep"]
    net = _TwoSlopeLinearNet(flat_slope, steep_slope)
    shared_params = list(net.parameters())
    optimizer = torch.optim.SGD(shared_params, lr=0.0)
    gradnorm = GradNorm(
        task_names,
        alpha=1.5,
        lr=0.1,
        task_importance=[importance_ratio, 1.0],
    )

    info = None
    for _ in range(100):
        optimizer.zero_grad()
        info = gradnorm.step(net.loss_dict(), shared_params, optimizer)

    balanced_ratio = (info["grad_norms"][1] / info["grad_norms"][0]).item()
    assert balanced_ratio == pytest.approx(1.0 / importance_ratio, rel=0.2)
