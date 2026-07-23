import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch
import numpy as np

from filter_estimation.model import MultiSourceFIR, _build_split_cues_weights

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


def test_split_cues_initialisation_assigns_bands_round_robin() -> None:
    fs = 48000
    n_bands = 4
    n_sources = 2
    model = MultiSourceFIR(N_SAMPLES, n_sources)
    model.set_filter_initialisation("split_cues", fs=fs, n_bands=n_bands)

    expected = _build_split_cues_weights(N_SAMPLES, n_sources, fs=fs, n_bands=n_bands)
    for branch_idx, branch in enumerate(model.branches):
        weight = branch.weight.detach().squeeze(1).numpy()
        assert weight == pytest.approx(expected[branch_idx])
        assert np.abs(weight).sum() > 0.0


def test_split_cues_log_band_edges_cover_frequency_range() -> None:
    fs = 48000
    n_bands = 8
    f_min = 20.0
    edges = np.geomspace(f_min, fs / 2, n_bands + 1)

    low_freqs, high_freqs = edges[:-1], edges[1:]

    assert low_freqs[0] == pytest.approx(f_min)
    assert high_freqs[-1] == pytest.approx(fs / 2)
    assert np.allclose(high_freqs[:-1], low_freqs[1:])


def test_split_cues_log_initialisation_matches_builder() -> None:
    fs = 48000
    n_bands = 4
    n_sources = 2
    model = MultiSourceFIR(N_SAMPLES, n_sources)
    model.set_filter_initialisation("split_cues", fs=fs, n_bands=n_bands, band_mode="log")

    expected = _build_split_cues_weights(N_SAMPLES, n_sources, fs=fs, n_bands=n_bands, band_mode="log")
    for branch_idx, branch in enumerate(model.branches):
        weight = branch.weight.detach().squeeze(1).numpy()
        assert weight == pytest.approx(expected[branch_idx])
    fs = 48000
    n_bands = 4
    model = MultiSourceFIR(N_SAMPLES, 2)
    model.set_filter_initialisation("middle")
    model.set_filter_initialisation("split_cues", branch_index=1, fs=fs, n_bands=n_bands)

    expected = _build_split_cues_weights(N_SAMPLES, 2, fs=fs, n_bands=n_bands)
    assert model.branches[1].weight.detach().squeeze(1).numpy() == pytest.approx(expected[1])
    middle = N_SAMPLES // 2
    assert model.branches[0].weight.detach()[middle, 0].item() == pytest.approx(1.0)
