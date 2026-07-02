import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from filter_estimation.model import MultiSourceFIR

N_SAMPLES = 64
N_SOURCES = 3


def test_forward_output_shape() -> None:
    model = MultiSourceFIR(N_SAMPLES, N_SOURCES)
    x = torch.ones(4, 1)

    filters = model(x)

    assert filters.shape == (4, N_SOURCES, N_SAMPLES)


@pytest.mark.parametrize("init_mode", ["middle", "random", "one_on"])
def test_set_filter_initialisation_modes(init_mode: str) -> None:
    model = MultiSourceFIR(N_SAMPLES, N_SOURCES)
    model.set_filter_initialisation(init_mode)

    for branch_idx, branch in enumerate(model.branches):
        weight = branch.weight.detach()
        if init_mode == "middle":
            middle = N_SAMPLES // 2
            assert weight[middle, 0].item() == pytest.approx(1.0)
            assert weight.abs().sum().item() == pytest.approx(1.0)
        elif init_mode == "one_on":
            if branch_idx == 0:
                middle = N_SAMPLES // 2
                assert weight[middle, 0].item() == pytest.approx(1.0)
            else:
                assert weight.abs().sum().item() == pytest.approx(0.0)
        else:
            assert weight.abs().max().item() <= 0.5


def test_set_filter_initialisation_copies_tensor() -> None:
    model = MultiSourceFIR(N_SAMPLES, 1)
    target = torch.linspace(0.0, 1.0, N_SAMPLES).view(-1, 1)
    model.set_filter_initialisation(target)

    assert torch.allclose(model.branches[0].weight.detach(), target)


def test_set_filter_initialisation_branch_index_updates_single_branch() -> None:
    model = MultiSourceFIR(N_SAMPLES, N_SOURCES)
    target = torch.ones(N_SAMPLES, 1)
    model.set_filter_initialisation(target, branch_index=1)

    assert torch.allclose(model.branches[1].weight.detach(), target)
    assert model.branches[0].weight.detach().abs().sum() > 0.0
    assert model.branches[2].weight.detach().abs().sum() > 0.0


def test_init_weight_applies_initialisation() -> None:
    model = MultiSourceFIR(N_SAMPLES, 2, init_weight="middle")
    middle = N_SAMPLES // 2

    for branch in model.branches:
        weight = branch.weight.detach()
        assert weight[middle, 0].item() == pytest.approx(1.0)


def test_getitem_returns_expected_branch() -> None:
    model = MultiSourceFIR(N_SAMPLES, N_SOURCES)

    assert model[0] is model.branches[0]
    assert model[-1] is model.branches[-1]
