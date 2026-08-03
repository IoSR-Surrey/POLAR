import json
from pathlib import Path

import numpy as np
import torch

from logger.saving import SavingManager


def test_export_to_disk_mixed_formats(tmp_path):
    manager = SavingManager()
    manager.save("target_angle", 0.0)
    manager.save("gtfb", torch.ones(2, 3))
    manager.save("stimulus_dataset", {
        "dataset_type": "bandpass",
        "stimulus_type": "noise",
        "n_bands": 16,
        "f_max": 20000.0,
    })
    manager.save("filter_initialisation", {
        "init_weight": "middle",
        "fs": 48000,
        "n_bands": 8,
        "band_mode": "log",
    })
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
    assert (tmp_path / "json" / "target_angle.json").exists()
    assert json.load(open(tmp_path / "json" / "target_angle.json", encoding="utf-8")) == 0.0

    stimulus = json.load(open(tmp_path / "json" / "stimulus_dataset.json", encoding="utf-8"))
    assert stimulus == {
        "dataset_type": "bandpass",
        "stimulus_type": "noise",
        "n_bands": 16,
        "f_max": 20000.0,
    }
    assert manifest["stimulus_dataset"]["format"] == "json"

    filter_init = json.load(open(tmp_path / "json" / "filter_initialisation.json", encoding="utf-8"))
    assert filter_init["init_weight"] == "middle"
    assert not (tmp_path / "json" / "filter_initialisation.init_weight.json").exists()

    gtfb = np.load(tmp_path / "npy" / "gtfb.npy")
    assert gtfb.shape == (2, 3)

    loss = np.load(tmp_path / "npy" / "loss_dict.llado2025_grad_surgery.npy")
    assert np.allclose(loss, [0.0, 1.0, 2.0])

    total = np.load(tmp_path / "npy" / "loss_dict.total.npy")
    assert np.allclose(total, [1.0, 2.0, 3.0])
