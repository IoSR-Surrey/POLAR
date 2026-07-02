import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from propagation.helpers import (
    _apply_delay,
    _calculate_delay_samples,
    _calculate_distance_gain,
    _calculate_source_angle_2D,
)
from propagation.model import BinauralFreeFieldPropagation, SingleSourceBinauralFreeFieldPropagation
from utils.sofa_functions import load_sofa

FS = 48000
FILTER_LEN = 256


@pytest.fixture
def sofaile():
    return load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")


@pytest.fixture
def lstn_pos():
    return torch.tensor([[0.0, 0.0]])


def impulse_stim(batch: int = 1) -> torch.Tensor:
    stim = torch.zeros(batch, 1, 1)
    stim[:, 0, 0] = 1.0
    return stim


def impulse_filter() -> torch.Tensor:
    filt = torch.zeros(1, 1, 1)
    filt[0, 0, 0] = 1.0
    return filt


def unit_filters(n_sources: int, tap_len: int = FILTER_LEN) -> torch.Tensor:
    return torch.ones(1, n_sources, tap_len)


def make_single_source(lstn_pos, source_pos, sofaile, **kwargs) -> SingleSourceBinauralFreeFieldPropagation:
    return SingleSourceBinauralFreeFieldPropagation(lstn_pos, source_pos, sofaile, fs=FS, **kwargs)


def make_multi_source(lstn_pos, spk_pos, sofaile, **kwargs) -> BinauralFreeFieldPropagation:
    return BinauralFreeFieldPropagation(FS, lstn_pos, spk_pos, sofaile, **kwargs)


def left_right_energy_ratio(binaural: torch.Tensor) -> torch.Tensor:
    left_energy = binaural[..., 0, :].pow(2).sum(dim=-1)
    right_energy = binaural[..., 1, :].pow(2).sum(dim=-1)
    return left_energy / right_energy


# --- 1. propagation/helpers.py ------------------------------------------------


def test_calculate_source_angle_front_and_sides():
    lstn = torch.tensor([[0.0, 0.0]])
    assert _calculate_source_angle_2D(lstn, torch.tensor([[1.0, 0.0]])).item() == pytest.approx(0.0)
    assert _calculate_source_angle_2D(lstn, torch.tensor([[0.0, 1.0]])).item() == pytest.approx(90.0)
    assert _calculate_source_angle_2D(lstn, torch.tensor([[0.0, -1.0]])).item() == pytest.approx(-90.0)
    assert _calculate_source_angle_2D(lstn, torch.tensor([[-1.0, 0.0]])).item() == pytest.approx(180.0)


def test_calculate_source_angle_batched_listeners():
    lstn = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    source = torch.tensor([[1.0, 0.0]])
    angles = _calculate_source_angle_2D(lstn, source)

    assert angles.shape == (1, 2)
    assert angles[0, 0].item() == pytest.approx(0.0)
    assert angles[0, 1].item() == pytest.approx(-45.0)


def test_calculate_distance_gain_scales_with_distance():
    lstn = torch.tensor([[0.0, 0.0]])
    near = _calculate_distance_gain(lstn, torch.tensor([[1.0, 0.0]])) # 1/r where r = 1
    far = _calculate_distance_gain(lstn, torch.tensor([[2.0, 0.0]])) # 1/r where r = 2

    assert near.item() == pytest.approx(1.0)
    assert far.item() == pytest.approx(0.5)


def test_calculate_distance_gain_clamps_coincident_source():
    lstn = torch.tensor([[0.0, 0.0]])
    gain = _calculate_distance_gain(lstn, torch.tensor([[0.0, 0.0]]))

    assert gain.item() == pytest.approx(1e6)


def test_calculate_delay_samples_zero_for_equidistant_sources():
    lstn = torch.tensor([[0.0, 0.0]])
    spk_pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    delay_samples, max_delay_samples = _calculate_delay_samples(lstn, spk_pos, FS)

    assert torch.all(delay_samples == 0)
    assert max_delay_samples == 0


def test_calculate_delay_samples_increase_with_distance():
    lstn = torch.tensor([[0.0, 0.0]])
    spk_pos = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    delay_samples, max_delay_samples = _calculate_delay_samples(lstn, spk_pos, FS)

    assert delay_samples[0, 0].item() == 0
    assert delay_samples[1, 0].item() > 0
    assert max_delay_samples == int(delay_samples.max().item())


def test_apply_delay_leaves_impulse_in_place_when_zero():
    hrirs = torch.zeros(1, 2, 4)
    hrirs[0, 0, 0] = 1.0
    hrirs[0, 1, 0] = 1.0
    delay_samples = [torch.tensor([0]), 0]

    delayed = _apply_delay(hrirs.clone(), delay_samples)

    assert torch.allclose(delayed, hrirs)


def test_apply_delay_shifts_impulse_by_requested_samples():
    hrirs = torch.zeros(1, 2, 4)
    hrirs[0, 0, 0] = 1.0
    hrirs[0, 1, 0] = 1.0
    delay_samples = [torch.tensor([2]), 2]

    delayed = _apply_delay(hrirs.clone(), delay_samples)

    assert delayed.shape == (1, 2, 6)
    assert delayed[0, 0, 2].item() == pytest.approx(1.0)
    assert delayed[0, 1, 2].item() == pytest.approx(1.0)


# --- 2. SingleSourceBinauralFreeFieldPropagation ------------------------------


def test_single_source_output_shape(lstn_pos, sofaile):
    stim = torch.ones(2, 1, 128)
    source_filter = torch.ones(1, 1, FILTER_LEN)
    source_pos = torch.tensor([[1.0, 0.0]])
    prop = make_single_source(lstn_pos, source_pos, sofaile)

    output = prop(stim, source_filter)
    ir_len = prop.hrirs.shape[-1]
    expected_len = stim.shape[-1] + source_filter.shape[-1] + ir_len - 2

    assert output.shape == (2, 1, 2, expected_len)


def test_single_source_impulse_chain_recovers_hrir(lstn_pos, sofaile):
    source_pos = torch.tensor([[1.0, 0.0]])
    prop = make_single_source(lstn_pos, source_pos, sofaile)

    output = prop(impulse_stim(), impulse_filter())

    assert output.shape == (1, 1, 2, prop.hrirs.shape[-1])
    assert torch.allclose(output[0, 0], prop.hrirs[0], atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_single_source_batch_dimension_independent(batch_size, lstn_pos, sofaile):
    source_pos = torch.tensor([[1.0, 0.0]])
    prop = make_single_source(lstn_pos, source_pos, sofaile)
    stim = impulse_stim(batch_size)

    output = prop(stim, impulse_filter())

    assert output.shape[0] == batch_size
    if batch_size > 1:
        assert torch.allclose(output[0], output[1])
        assert torch.allclose(output[0], output[2])


def test_single_source_multiple_listeners_differ(lstn_pos, sofaile):
    # Collinear listeners share the same source angle so sofa_find stays valid.
    listeners = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    source_pos = torch.tensor([[3.0, 0.0]])
    prop = make_single_source(listeners, source_pos, sofaile, apply_distance_attenuation=True)

    output = prop(impulse_stim(), impulse_filter())

    assert output.shape[1] == 2
    assert not torch.allclose(output[0, 0], output[0, 1])


def test_single_source_listener_indices_subset(lstn_pos, sofaile):
    listeners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    source_pos = torch.tensor([[5.0, 0.0]])
    prop = make_single_source(listeners, source_pos, sofaile)
    listener_indices = torch.tensor([0, 2])

    subset = prop(impulse_stim(), impulse_filter(), listener_indices=listener_indices)
    full = prop(impulse_stim(), impulse_filter())

    assert subset.shape[1] == 2
    assert torch.allclose(subset[:, 0], full[:, 0])
    assert torch.allclose(subset[:, 1], full[:, 2])


def test_single_source_distance_attenuation_scales_energy(lstn_pos, sofaile):
    near_pos = torch.tensor([[1.0, 0.0]])
    far_pos = torch.tensor([[2.0, 0.0]])
    near = make_single_source(lstn_pos, near_pos, sofaile, apply_distance_attenuation=True)
    far = make_single_source(lstn_pos, far_pos, sofaile, apply_distance_attenuation=True)

    near_energy = near(impulse_stim(), impulse_filter()).pow(2).sum()
    far_energy = far(impulse_stim(), impulse_filter()).pow(2).sum()

    assert near_energy.item() == pytest.approx(4.0 * far_energy.item(), rel=1e-3)


def test_single_source_forward_is_differentiable(lstn_pos, sofaile):
    source_pos = torch.tensor([[1.0, 0.0]])
    prop = make_single_source(lstn_pos, source_pos, sofaile)
    stim = impulse_stim().requires_grad_(True)

    output = prop(stim, impulse_filter())
    output.sum().backward()

    assert stim.grad is not None
    assert stim.grad.abs().sum() > 0.0


def test_single_source_frontal_source_has_balanced_ears(lstn_pos, sofaile):
    source_pos = torch.tensor([[1.0, 0.0]])
    prop = make_single_source(lstn_pos, source_pos, sofaile)

    ratio = left_right_energy_ratio(prop(torch.ones(1, 1, 1), unit_filters(1)))

    assert ratio.item() == pytest.approx(1.0, abs=0.2)


def test_ear_energy_ratio_over_angle(lstn_pos, sofaile):
    stim = torch.ones(1, 1, 1)
    filters = unit_filters(1)
    angles = torch.arange(-90, 95, 5)
    ear_energy_ratios = []

    for angle in angles:
        source_pos = torch.tensor(
            [[torch.cos(torch.deg2rad(angle)), torch.sin(torch.deg2rad(angle))]]
        )
        prop = make_single_source(lstn_pos, source_pos, sofaile)
        binaural_signals = prop(stim, filters).squeeze()
        ear_energy_ratios.append(left_right_energy_ratio(binaural_signals.unsqueeze(0)))

    ear_energy_ratios = torch.stack(ear_energy_ratios).squeeze()

    assert ear_energy_ratios[0] < 1.0
    assert ear_energy_ratios[len(ear_energy_ratios) // 2].item() == pytest.approx(1.0, abs=0.2)
    assert ear_energy_ratios[-1] > 1.0
    assert torch.all(ear_energy_ratios[1:] >= ear_energy_ratios[:-1] - 1.0)


# --- 3. BinauralFreeFieldPropagation ------------------------------------------


def test_multi_source_single_speaker_matches_single_source_class(lstn_pos, sofaile):
    spk_pos = torch.tensor([[1.0, 0.0]])
    multi = make_multi_source(lstn_pos, spk_pos, sofaile)
    single = make_single_source(lstn_pos, spk_pos, sofaile)
    stim = impulse_stim()

    assert torch.allclose(multi(stim, unit_filters(1)), single(stim, unit_filters(1)))


def test_multi_source_one_active_source_equals_isolated_contribution(lstn_pos, sofaile):
    spk_pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prop = make_multi_source(lstn_pos, spk_pos, sofaile)
    stim = impulse_stim()

    source_zero = torch.zeros(1, 2, 1)
    source_zero[0, 0, 0] = 1.0
    source_one = torch.zeros(1, 2, 1)
    source_one[0, 1, 0] = 1.0

    only_first = prop(stim, source_zero)
    only_second = prop(stim, source_one)

    assert only_first.abs().sum() > 0.0
    assert only_second.abs().sum() > 0.0
    assert not torch.allclose(only_first, only_second)


def test_multi_source_superposition(lstn_pos, sofaile):
    spk_pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prop = make_multi_source(lstn_pos, spk_pos, sofaile)
    stim = impulse_stim()

    source_zero = torch.zeros(1, 2, 1)
    source_zero[0, 0, 0] = 1.0
    source_one = torch.zeros(1, 2, 1)
    source_one[0, 1, 0] = 1.0
    both = source_zero + source_one

    combined = prop(stim, both)
    summed = prop(stim, source_zero) + prop(stim, source_one)

    assert torch.allclose(combined, summed, atol=1e-5)


def test_multi_source_closer_source_arrives_before_farther_one(lstn_pos, sofaile):
    spk_pos = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    prop = make_multi_source(lstn_pos, spk_pos, sofaile)
    stim = impulse_stim()

    source_zero = torch.zeros(1, 2, 1)
    source_zero[0, 0, 0] = 1.0
    source_one = torch.zeros(1, 2, 1)
    source_one[0, 1, 0] = 1.0

    near_peak = prop(stim, source_zero)[0, 0, 0].abs().argmax().item()
    far_peak = prop(stim, source_one)[0, 0, 0].abs().argmax().item()

    assert near_peak < far_peak


def test_multi_source_rejects_wrong_filter_count(lstn_pos, sofaile):
    spk_pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prop = make_multi_source(lstn_pos, spk_pos, sofaile)

    with pytest.raises(ValueError, match="Expected spk_filters with 2 sources"):
        prop(impulse_stim(), unit_filters(1))


def test_multi_source_listener_indices_subset(lstn_pos, sofaile):
    listeners = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    spk_pos = torch.tensor([[5.0, 0.0], [-5.0, 0.0]])
    prop = make_multi_source(listeners, spk_pos, sofaile)
    listener_indices = torch.tensor([0, 2])
    stim = impulse_stim()

    subset = prop(stim, unit_filters(2), listener_indices=listener_indices)
    full = prop(stim, unit_filters(2))

    assert subset.shape[1] == 2
    assert torch.allclose(subset[:, 0], full[:, 0])
    assert torch.allclose(subset[:, 1], full[:, 2])


def test_multi_source_distance_attenuation_scales_energy(lstn_pos, sofaile):
    spk_pos = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    near_only = torch.zeros(1, 2, 1)
    near_only[0, 0, 0] = 1.0
    far_only = torch.zeros(1, 2, 1)
    far_only[0, 1, 0] = 1.0

    without_attenuation = make_multi_source(lstn_pos, spk_pos, sofaile, apply_distance_attenuation=False)
    with_attenuation = make_multi_source(lstn_pos, spk_pos, sofaile, apply_distance_attenuation=True)
    stim = impulse_stim()

    near_raw = without_attenuation(stim, near_only).pow(2).sum()
    far_raw = without_attenuation(stim, far_only).pow(2).sum()
    near_att = with_attenuation(stim, near_only).pow(2).sum()
    far_att = with_attenuation(stim, far_only).pow(2).sum()

    assert near_raw.item() == pytest.approx(far_raw.item(), rel=1e-3)
    assert near_att.item() == pytest.approx(4.0 * far_att.item(), rel=1e-3)
