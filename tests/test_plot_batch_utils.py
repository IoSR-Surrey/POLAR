import numpy as np
import pytest
import torch

from plotting.batch_utils import (
    average_estimates_dicts,
    average_loss_dicts,
    mean_over_stimulus_batch,
    reduce_batch_dimension,
)


def test_mean_over_stimulus_batch_no_op_when_already_per_listener() -> None:
    values = np.arange(6).reshape(3, 2)
    result = mean_over_stimulus_batch(values, n_listeners=3)
    np.testing.assert_allclose(result, values)


def test_mean_over_stimulus_batch_averages_flat_rows() -> None:
    values = np.array([
        [0.0, 10.0],
        [1.0, 11.0],
        [2.0, 12.0],
        [4.0, 14.0],
        [5.0, 15.0],
        [6.0, 16.0],
    ])
    result = mean_over_stimulus_batch(values, n_listeners=3)
    np.testing.assert_allclose(result, [[2.0, 12.0], [3.0, 13.0], [4.0, 14.0]])


def test_reduce_batch_dimension_averages_middle_axis() -> None:
    values = np.array([
        [[0.0, 10.0], [2.0, 12.0]],
        [[4.0, 14.0], [6.0, 16.0]],
    ])
    result = reduce_batch_dimension(values, n_listeners=2)
    np.testing.assert_allclose(result, [[1.0, 11.0], [5.0, 15.0]])


def test_reduce_batch_dimension_leaves_two_dimensional_series_unchanged() -> None:
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = reduce_batch_dimension(values, n_listeners=3)
    np.testing.assert_allclose(result, values)


def test_mean_over_stimulus_batch_accepts_torch() -> None:
    values = torch.tensor([
        [0.0, 10.0],
        [2.0, 12.0],
        [4.0, 14.0],
        [6.0, 16.0],
    ])
    result = mean_over_stimulus_batch(values, n_listeners=2)
    np.testing.assert_allclose(result, [[2.0, 12.0], [4.0, 14.0]])


def test_average_loss_dicts() -> None:
    loss_dicts = [
        {"total": torch.tensor(1.0), "task": torch.tensor(2.0)},
        {"total": torch.tensor(3.0), "task": torch.tensor(4.0)},
    ]
    averaged = average_loss_dicts(loss_dicts)
    assert averaged["total"].item() == pytest.approx(2.0)
    assert averaged["task"].item() == pytest.approx(3.0)


def test_average_estimates_dicts() -> None:
    estimates = [
        {"llado2025_grad_surgery": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        {"llado2025_grad_surgery": torch.tensor([[5.0, 6.0], [7.0, 8.0]])},
    ]
    averaged = average_estimates_dicts(estimates)
    np.testing.assert_allclose(averaged["llado2025_grad_surgery"], torch.tensor([4.0, 5.0]))


def test_average_estimates_dicts_unequal_batch_sizes() -> None:
    estimates = [
        {"llado2025_grad_surgery": torch.zeros(64, 3)},
        {"llado2025_grad_surgery": torch.ones(32, 3)},
    ]
    averaged = average_estimates_dicts(estimates)
    # (64*0 + 32*1) / 96
    np.testing.assert_allclose(averaged["llado2025_grad_surgery"], torch.full((3,), 32 / 96))
