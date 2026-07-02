import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import math

import pytest
import torch
import torch.nn.functional as F

from utils.dsp import (
    compute_energy,
    cosine_similarity,
    hilbert_transform,
    rms,
    soft_argmax,
    softmax_st,
)


def test_compute_energy_sums_squares_along_last_dim() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    energy = compute_energy(x)

    assert energy.shape == (1, 2)
    assert energy[0, 0].item() == pytest.approx(5.0)
    assert energy[0, 1].item() == pytest.approx(25.0)


def test_rms_matches_manual_calculation() -> None:
    x = torch.tensor([1.0, 3.0])
    assert rms(x).item() == pytest.approx(math.sqrt((1.0 + 9.0) / 2.0))


def test_cosine_similarity_for_parallel_vectors() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 4.0, 6.0])

    assert cosine_similarity(a, b).item() == pytest.approx(1.0)


def test_softmax_st_forward_equals_hard_selection() -> None:
    z = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    ste = softmax_st(z, dim=-1)
    hard = torch.zeros_like(z).scatter_(-1, z.argmax(dim=-1, keepdim=True), 1.0)

    assert torch.allclose(ste, hard)


def test_softmax_st_backward_matches_softmax_at_non_maximum() -> None:
    z = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    ste = softmax_st(z, dim=-1)
    soft = F.softmax(z, dim=-1)
    ste.sum().backward()
    z_ste = z.grad.clone()
    z.grad = None
    soft.sum().backward()

    assert torch.allclose(z_ste, z.grad)


def test_soft_argmax_peaks_at_maximum_index() -> None:
    indices = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    z = torch.tensor([[-5.0, -1.0, 4.0, -1.0, -5.0]])

    estimate = soft_argmax(z, indices, dim=-1, tau=0.01)

    assert estimate.item() == pytest.approx(0.0, abs=0.1)


def test_hilbert_transform_shifts_sine_by_quarter_period() -> None:
    n_samples = 256
    t = torch.arange(n_samples, dtype=torch.float32) / 48_000
    sine = torch.sin(2 * math.pi * 1_000 * t).view(1, 1, -1)

    analytic = hilbert_transform(sine)
    real_part = analytic.real
    imag_part = analytic.imag

    # sin -> Hilbert -> -cos; peak of imag should align with zero crossing of real.
    peak_imag = imag_part.abs().argmax(dim=-1).item()
    zero_crossing = (real_part[0, 0, 1:] * real_part[0, 0, :-1] < 0).nonzero()[0].item()
    assert abs(peak_imag - zero_crossing) <= 2
