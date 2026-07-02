import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pytest
import torch

from utils.miscellaneous import _flatten_tap_grads, load_npy_to_torch, ready_for_plotting


def test_load_npy_to_torch_round_trip() -> None:
    array = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "sample.npy"
        np.save(path, array)

        tensor = load_npy_to_torch(path)

    assert torch.allclose(tensor, torch.from_numpy(array))


def test_ready_for_plotting_tensor() -> None:
    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    array = ready_for_plotting(tensor)

    assert isinstance(array, np.ndarray)
    assert np.allclose(array, [1.0, 2.0])


def test_ready_for_plotting_list() -> None:
    values = [torch.tensor([1.0]), np.array([2.0])]
    array = ready_for_plotting(values)

    assert array.shape == (2, 1)
    assert array[0, 0] == pytest.approx(1.0)
    assert array[1, 0] == pytest.approx(2.0)


def test_ready_for_plotting_rejects_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported type"):
        ready_for_plotting({"a": 1})


def test_flatten_tap_grads_returns_none_for_empty_inputs() -> None:
    assert _flatten_tap_grads(None) is None
    assert _flatten_tap_grads([None, None]) is None


def test_flatten_tap_grads_concatenates_non_none_gradients() -> None:
    grads = [torch.tensor([1.0, 2.0]), None, torch.tensor([3.0])]
    flat = _flatten_tap_grads(grads)

    assert flat is not None
    assert flat.tolist() == [1.0, 2.0, 3.0]
