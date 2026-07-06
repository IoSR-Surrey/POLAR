import json
from pathlib import Path

import numpy as np
import torch

from logger.saving import SavingManager


def test_export_to_disk_mixed_formats(tmp_path):
    manager = SavingManager()
    manager.save("target_angle", 0.0)
    manager.save("gtfb", torch.ones(2, 3))
    for epoch in range(3):
        manager.save("loss_dict", {
            "llado2025_grad_surgery": torch.tensor(float(epoch)),
            "total": torch.tensor(float(epoch + 1)),
        })
    manager.save("sofaile", object())

    manager.export_to_disk(tmp_path)

    with open(tmp_path / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["sofaile"]["skipped"] is True
    assert (tmp_path / "target_angle.json").exists()
    assert json.load(open(tmp_path / "target_angle.json", encoding="utf-8")) == 0.0

    gtfb = np.load(tmp_path / "gtfb.npy")
    assert gtfb.shape == (2, 3)

    loss = np.load(tmp_path / "loss_dict.llado2025_grad_surgery.npy")
    assert np.allclose(loss, [0.0, 1.0, 2.0])

    total = np.load(tmp_path / "loss_dict.total.npy")
    assert np.allclose(total, [1.0, 2.0, 3.0])
