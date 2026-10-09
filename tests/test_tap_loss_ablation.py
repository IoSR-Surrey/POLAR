import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pytest

from scripts.tap_loss_ablation import (
    FILTER_INITIALISATION,
    LEARNING_RATE,
    MTL_OPTIMIZER,
    build_conditions,
    load_shared_assets,
    run_condition,
    run_study,
)

# Tiny overrides so the full pipeline runs in seconds while exercising every step.
FAST_OVERRIDES = {
    # band_mode must not stay as linear_erb: that mode needs n_bands>=12 with erb_factor=0.25.
    "stimulus_dataset": {"n_bands": 2, "n_samples_per_band": 1, "band_mode": "erb"},
    "training": {"batch_size": 2, "shuffle": True, "num_workers": 0, "drop_last": False},
}


def _load_config():
    with open("config/hyperpameters.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    for section, values in FAST_OVERRIDES.items():
        config[section].update(values)
    return config


def test_build_conditions_grid():
    conditions = build_conditions()
    assert [c["name"] for c in conditions] == [
        "taps32_llado_only",
        "taps32_llado_desena",
        "taps1024_llado_only",
        "taps1024_llado_desena",
    ]
    assert {c["n_taps"] for c in conditions} == {32, 1024}
    assert {c["loss_set"] for c in conditions} == {"llado_only", "llado_desena"}


def test_run_condition_enforces_middle_and_saves_filters(tmp_path):
    config = _load_config()
    shared = load_shared_assets(config)
    condition = {"name": "taps32_llado_desena", "n_taps": 32, "loss_set": "llado_desena"}

    result = run_condition(condition, tmp_path, config, shared, seed=42, num_epochs=1, run_plots=False)

    output_dir = Path(result["output_dir"])
    spk_filters = np.load(output_dir / "npy" / "spk_filters.npy")
    assert spk_filters.shape == (1, len(shared["spk_pos"]), 32)

    with open(output_dir / "json" / "filter_initialisation.json", "r", encoding="utf-8") as f:
        saved_init = json.load(f)
    assert saved_init["init_weight"] == "middle"
    assert FILTER_INITIALISATION["init_weight"] == "middle"

    with open(output_dir / "json" / "seed.json", "r", encoding="utf-8") as f:
        assert json.load(f) == 42

    assert result["n_taps"] == 32
    assert result["loss_set"] == "llado_desena"
    assert result["final_total_loss"] is not None
    assert result["final_desena_loss"] is not None


def test_llado_only_has_no_desena_loss(tmp_path):
    config = _load_config()
    shared = load_shared_assets(config)
    condition = {"name": "taps32_llado_only", "n_taps": 32, "loss_set": "llado_only"}

    result = run_condition(condition, tmp_path, config, shared, seed=42, num_epochs=1, run_plots=False)
    assert result["final_llado_loss"] is not None
    assert result["final_desena_loss"] is None


def test_per_condition_seed_reset_is_deterministic(tmp_path):
    config = _load_config()
    condition = {"name": "taps32_llado_desena", "n_taps": 32, "loss_set": "llado_desena"}

    first = run_condition(condition, tmp_path / "a", config, load_shared_assets(config), seed=42, num_epochs=2, run_plots=False)
    second = run_condition(condition, tmp_path / "b", config, load_shared_assets(config), seed=42, num_epochs=2, run_plots=False)

    filters_a = np.load(Path(first["output_dir"]) / "npy" / "spk_filters.npy")
    filters_b = np.load(Path(second["output_dir"]) / "npy" / "spk_filters.npy")
    assert np.allclose(filters_a, filters_b)


def test_run_study_writes_summary_and_playback(tmp_path):
    outcome = run_study(
        config_overrides=FAST_OVERRIDES,
        output_root=str(tmp_path / "outputs"),
        num_epochs=1,
        run_plots=False,
        study_name="unit_test",
    )

    study_root = Path(outcome["study_root"])
    assert (study_root / "summary.csv").exists()
    assert (study_root / "summary.json").exists()
    assert (study_root / "study_config.json").exists()

    playback_config = Path(outcome["playback_config"])
    with open(playback_config, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert len(payload["folders"]) == 4
    assert len(outcome["results"]) == 4

    with open(study_root / "study_config.json", "r", encoding="utf-8") as f:
        study_config = json.load(f)
    assert study_config["mtl_optimizer"] == MTL_OPTIMIZER
    assert study_config["learning_rate"] == LEARNING_RATE
