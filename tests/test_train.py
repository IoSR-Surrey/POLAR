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

from models.train import ModelTrainer, create_model_dict


class _MockListenerModel(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.model_type = "listener_based"
        self.scale = scale

    def forward(self, binaural_signals: torch.Tensor, _) -> dict:
        batch = binaural_signals.shape[0]
        return {
            "output": torch.full((batch,), self.scale, dtype=binaural_signals.dtype),
            "kwargs": {"tag": self.__class__.__name__},
        }


def test_create_model_dict_structure() -> None:
    model = _MockListenerModel(1.0)
    model_dict = create_model_dict([model], weights=[0.5])

    assert set(model_dict.keys()) == {"_MockListenerModel"}
    assert model_dict["_MockListenerModel"]["model"] is model
    assert model_dict["_MockListenerModel"]["weight"] == 0.5
    assert model_dict["_MockListenerModel"]["criterion"] is F.mse_loss


def test_create_model_dict_assigns_unique_class_names() -> None:
    class Alpha(_MockListenerModel):
        pass

    class Beta(_MockListenerModel):
        pass

    model_dict = create_model_dict([Alpha(1.0), Beta(2.0)])

    assert set(model_dict.keys()) == {"Alpha", "Beta"}


def test_model_trainer_compute_estimates() -> None:
    trainer = ModelTrainer([_MockListenerModel(3.0)])
    binaural = torch.ones(2, 1, 2, 8)
    spk_filters = torch.ones(1, 2, 32)

    trainer.compute_estimates(binaural, spk_filters, None)

    assert trainer.estimates["_MockListenerModel"].shape == (2, 1)
    assert trainer.estimates["_MockListenerModel"][0, 0].item() == pytest.approx(3.0)
    assert trainer.kwargs_output["_MockListenerModel"]["tag"] == "_MockListenerModel"


def test_model_trainer_compute_loss_applies_weights() -> None:
    trainer = ModelTrainer([_MockListenerModel(2.0)], weights=[2.0])
    binaural = torch.ones(1, 1, 2, 8)
    spk_filters = torch.ones(1, 2, 32)
    trainer.compute_estimates(binaural, spk_filters, None)

    loss_dict = trainer.compute_loss({"_MockListenerModel": torch.tensor([[0.0]])})

    per_model_loss = F.mse_loss(torch.tensor([[2.0]]), torch.tensor([[0.0]]))
    assert loss_dict["_MockListenerModel"].item() == pytest.approx(per_model_loss * 2.0)
    assert loss_dict["total"].item() == pytest.approx(per_model_loss * 2.0)
