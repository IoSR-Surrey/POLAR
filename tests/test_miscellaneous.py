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

from utils.miscellaneous import (
    _flatten_tap_grads,
    create_mtl_optimizer,
    load_npy_to_torch,
    ready_for_plotting,
    run_mtl_optimizer_step,
    target_angle_for_listeners,
)
from logger.saving import SavingManager, task_grad_pairwise_cosine_similarity_stats
from models.train import ModelTrainer


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


def test_target_angle_for_listeners_from_target_position():
    lstn = torch.tensor([[0.0, 0.0], [0.0, 0.25], [0.0, -0.25]])
    target_pos = torch.tensor([2.0, 0.0])
    angles = target_angle_for_listeners(lstn, target_pos)

    assert angles.shape == (3,)
    assert angles[0].item() == pytest.approx(0.0)
    assert angles[1].item() == pytest.approx(-7.125, abs=0.01)
    assert angles[2].item() == pytest.approx(7.125, abs=0.01)


def test_target_angle_for_listeners_matches_propagation_helper():
    from propagation.helpers import _calculate_source_angle_2D

    lstn = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    target_pos = torch.tensor([2.0 * torch.cos(torch.deg2rad(torch.tensor(30.0))), 2.0 * torch.sin(torch.deg2rad(torch.tensor(30.0)))])
    expected = _calculate_source_angle_2D(lstn, target_pos.unsqueeze(0)).squeeze(0)
    assert torch.allclose(target_angle_for_listeners(lstn, target_pos), expected)


def test_create_mtl_optimizer_returns_none_for_plain() -> None:
    trainer = ModelTrainer([_MockModel()])
    assert create_mtl_optimizer(trainer, "plain") is None


def test_create_mtl_optimizer_returns_gradnorm() -> None:
    from models.gradnorm import GradNorm

    trainer = ModelTrainer([_MockModel()], weights=[2.0])
    mtl = create_mtl_optimizer(trainer, "gradnorm")

    assert isinstance(mtl, GradNorm)
    assert mtl.task_names == ["_MockModel"]


def test_create_mtl_optimizer_returns_cagrad() -> None:
    from models.cagrad import CAGrad

    trainer = ModelTrainer([_MockModel()])
    mtl = create_mtl_optimizer(trainer, "cagrad")

    assert isinstance(mtl, CAGrad)
    assert mtl.task_names == ["_MockModel"]


def test_create_mtl_optimizer_returns_gradnorm_cagrad() -> None:
    from models.cagrad import CAGrad
    from models.gradnorm import GradNorm

    class _TaskAModel(_MockModel):
        pass

    class _TaskBModel(_MockModel):
        pass

    trainer = ModelTrainer([_TaskAModel(), _TaskBModel()], weights=[1.0, 2.0])
    mtl = create_mtl_optimizer(trainer, "gradnorm+cagrad")

    assert isinstance(mtl, dict)
    assert isinstance(mtl["gradnorm"], GradNorm)
    assert isinstance(mtl["cagrad"], CAGrad)
    assert mtl["gradnorm"].task_names == ["_TaskAModel", "_TaskBModel"]
    assert mtl["cagrad"].task_names == ["_TaskAModel", "_TaskBModel"]


def test_run_mtl_optimizer_step_gradnorm_cagrad_saves_cagrad_diagnostics() -> None:
    class _TaskAModel(torch.nn.Module):
        model_type = "listener_based"

        def forward(self, x):
            return {"output": x.sum(dim=-1), "kwargs": {}}

    class _TaskBModel(torch.nn.Module):
        model_type = "listener_based"

        def forward(self, x):
            return {"output": x.sum(dim=-1), "kwargs": {}}

    class _SharedNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self.weight[0] ** 2, (self.weight[1] - 1.0) ** 2

    trainer = ModelTrainer([_TaskAModel(), _TaskBModel()])
    mtl = create_mtl_optimizer(trainer, "gradnorm+cagrad")
    net = _SharedNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    saving_manager = SavingManager(task_names=list(trainer.model_dict.keys()))

    loss_a, loss_b = net()
    loss_dict = {
        "_TaskAModel": loss_a,
        "_TaskBModel": loss_b,
        "total": loss_a + loss_b,
    }

    run_mtl_optimizer_step(
        "gradnorm+cagrad", mtl, loss_dict, net, optimizer, saving_manager, trainer,
    )

    assert "effective_grad" in saving_manager.saved_data
    assert "cagrad_dual_weights" in saving_manager.saved_data
    assert "cagrad_task_coeffs" in saving_manager.saved_data
    assert "gradnorm_task_weights" in saving_manager.saved_data
    assert set(saving_manager.saved_data["raw_grads"][0].keys()) == set(trainer.model_dict.keys())


class _MockModel(torch.nn.Module):
    model_type = "listener_based"

    def forward(self, x):
        return {"output": x.sum(dim=-1), "kwargs": {}}


def test_task_grad_pairwise_cosine_similarity_stats_with_combined_reference() -> None:
    g_a = torch.tensor([1.0, 0.0])
    g_b = torch.tensor([1.0, 0.0])
    g_combined = torch.tensor([1.0, 0.0])
    loss_grad_vectors = [
        {"task_a": g_a, "task_b": g_b},
        {"task_a": g_a, "task_b": g_b},
    ]
    combined_grad_vectors = [g_combined, g_combined]

    stats = task_grad_pairwise_cosine_similarity_stats(
        loss_grad_vectors, combined_grad_vectors=combined_grad_vectors,
    )

    assert stats["use_combined_reference"] is True
    assert stats["ref_cosines_over_epochs"]["task_a"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert stats["ref_cosines_over_epochs"]["task_b"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert stats["combined_norm_over_epochs"] == [pytest.approx(1.0), pytest.approx(1.0)]
