from __future__ import annotations
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
_cpu_count = os.cpu_count() or 1
os.environ.setdefault('OMP_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('MKL_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_cpu_count))

import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from scripts.optimisation.custom_modules import MultiSourceFIR

import numpy as np
import pytest


from scripts.datasets.stimulus_dataset import get_stim_loader
from scripts.datasets.listener_dataset import get_listener_loader
from scripts.propagation.free_field_propagation import BinauralFreeFieldPropagation
from scripts.perceptual_models.llado2025.llado2025 import llado2025
from scripts.perceptual_models.desena2020.desena2020 import desena2020
from scripts.perceptual_models.spectral_matching.spectral_matching import spectral_matching
from scripts.train.train import ListenerBasedLoss, Train


# Import setup functions
from scripts.setup.setup import set_seed


pytest.importorskip("sofar")
pytest.importorskip("torchaudio")

REFERENCE_PATH = Path(__file__).resolve().parent / "reference_outputs" / "test_reference_100_epoch_matches_known_output.json"


def _reference_payload(est_filters, estimation_dict):
    filters_np = np.asarray(est_filters, dtype=np.float32)
    return {
        "est_filters": filters_np.tolist(),
        "localisation": np.asarray(estimation_dict["localisation"]).tolist(),
        "colouration": np.asarray(estimation_dict["colouration"]).tolist(),
        "uncertainty": np.asarray(estimation_dict["uncertainty"]).tolist(),
    }


def test_reference_100_epoch_matches_known_output() -> None:

    """
    Runs the legacy 15 degree reference case on CPU for determinism and compares
    trained filters and perceptual metrics against a stored JSON baseline.

    To regenerate the baseline after an intentional code change:

        UPDATE_REFERENCE=1 python -m pytest tests/test_reference_regression.py::test_reference_100_epoch_matches_known_output -s
    """

    # Default paths relative to this repository (portable across machines/OSes)
    repo_root = Path(__file__).resolve().parents[1]
    d = 100
    fs = 48000
    env = SimpleNamespace( # Include here all variables that you want stored into config.json
        epochs=100,
        num_lstn_pos=1,
        listener_batch_size=1,  # 5 mirrored pairs plus the centre listener
        symmetric_listener_sampling=False,
        selected_stim="impulse",
        stimulus_batch_size = 1,
        n_stimulus = 1,
        n_filter_taps=32,
        target_pos=[d, float(np.tan(np.deg2rad(15.0)) * d)],
        sofa_path=str(repo_root / "data" / "impulses" / "SOFA" / "SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"),
        lr=0.1,
        seed=46,
        log=1,
        fs=fs,
    )
    # Backwards-compatible alias for helpers that still expect `n_samples`.
    env.n_samples = env.n_filter_taps
    env.dtype = torch.float64
    env.device_str = "cpu"
    env.device = torch.device(env.device_str)
    env.num_threads = _cpu_count
    env.num_interop_threads = min(8, _cpu_count)

    # Maximize CPU-side parallelism for PyTorch ops.
    torch.set_num_threads(env.num_threads)
    torch.set_num_interop_threads(env.num_interop_threads)
    torch.backends.mkldnn.enabled = True

    assert isinstance(env.listener_batch_size, int), "listener_batch_size must be set to positive integer"
    assert env.listener_batch_size > 0 , "listener_batch_size must be set to positive integer"

## Initial setup
    print(f"Using device: {env.device}")

    # Loudspkeaker and listener positions
    env.lstn_pos = torch.tensor([[0.0, 0.0]], device = env.device, dtype = env.dtype)
    base_angle_rad = np.deg2rad(30.0)
    env.speaker_distance = 2.0  # Distance from each loudspeaker to the reference-system center (0, 0)
    env.spk_pos = [
        [env.speaker_distance * np.cos(base_angle_rad), env.speaker_distance * np.sin(base_angle_rad)],
        [env.speaker_distance * np.cos(base_angle_rad), -env.speaker_distance * np.sin(base_angle_rad)],
    ]
    env.spk_pos = torch.tensor(env.spk_pos, device = env.device, dtype = env.dtype)
    env.n_sources = len(env.spk_pos)

    # Reset the seed so each run starts from the same point.
    set_seed(seed=env.seed)

## Load data
    training_stimulus_data_loader, stimulus_dataset = get_stim_loader(env.fs, env.stimulus_batch_size, env.n_stimulus, stimulus_type=env.selected_stim, device=env.device, dtype=env.dtype) # Get stimulus dataloader
    training_listener_data_loader, _ = get_listener_loader(env.lstn_pos, env.listener_batch_size, env.symmetric_listener_sampling) # Get listener dataloader

## Initialize model
    # Initialize sound propagation module
    stereo_prop = BinauralFreeFieldPropagation(fs=env.fs, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, sofa_file_path=env.sofa_path, device=env.device, dtype=env.dtype) # Load stereo anechoic propagation module

    # Initiate perceptual models
    loc_model = llado2025(loss = F.mse_loss, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, spk_pos=env.spk_pos, device=env.device, dtype=env.dtype, balance_interaural_cue_grads=False) # Llado et al. 2025 model
    col_model = spectral_matching(loss = F.l1_loss, target = 0, target_pos=env.target_pos, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype) # Spectral Matching model
    unc_model = desena2020(loss = F.l1_loss, target = 0, fs=env.fs, sofa_path=env.sofa_path, lstn_pos=env.lstn_pos, device=env.device, dtype=env.dtype, balance_interaural_cue_grads=False) # De Sena et al. 2020 model

    loss_dict = {
        "localisation": ListenerBasedLoss(model=loc_model, weight=1.0, device=env.device, dtype=env.dtype),
        "colouration": ListenerBasedLoss(model=col_model, weight=1.0e-3, device=env.device, dtype=env.dtype),
        "uncertainty": ListenerBasedLoss(model=unc_model, weight=1.0e-3, device=env.device, dtype=env.dtype),
    }

    # Initialise neural network
    nn_model = MultiSourceFIR(
        n_samples=env.n_filter_taps,
        n_sources=env.n_sources,
    ).to(device=env.device, dtype=env.dtype)
    # Initialise FIR filter weights
    centre_tap_idx = env.n_filter_taps // 2
    with torch.no_grad():
        for i, branch in enumerate(nn_model.branches):
            branch.weight.fill_(0.0)
            branch.weight[centre_tap_idx] = 1.0
        # nn_model.branches[0].weight[0, 0] = 1.0

    nn_optimiser = torch.optim.Adam(nn_model.parameters(), lr=env.lr)

## Training loop
    # Train the neural network
    trainer = Train(n_epoch=env.epochs,
                    stim_loader = training_stimulus_data_loader,
                    lstn_loader = training_listener_data_loader,
                    nn=nn_model,
                    prop=stereo_prop,
                    loss_dict=loss_dict,
                    optimiser=nn_optimiser,
                    logger=None
                    ) # Initialize training class
    

## TRAINING LOOP
    trainer.train()

## Print and Plot results
    # Evaluate final neural network FIRs
    est_filters, estimation_dict, bin_stim = trainer.evaluate(eval_lstn_pos=env.lstn_pos, eval_stimulus_data_loader=training_stimulus_data_loader)
    
    # Normalise filters to prevent clipping 
    est_filters /= np.max(np.abs(est_filters)) / 0.9 # TODO: move this to a more appropriate place at the end of the training loop

    # Print estimated perceptual metrics
    for key, value in estimation_dict.items():
        print(f"Estimated {key}: {value}")

    reference = _reference_payload(est_filters, estimation_dict)
    if os.environ.get("UPDATE_REFERENCE") == "1":
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REFERENCE_PATH, "w") as f:
            json.dump(reference, f, indent=4)
            f.write("\n")
        print(f"Wrote reference to {REFERENCE_PATH}")
        return

    with open(REFERENCE_PATH) as f:
        expected = json.load(f)

    expected_filters = np.array(expected["est_filters"], dtype=np.float32)
    expected_localisation = expected["localisation"]
    expected_colouration = expected["colouration"]
    expected_uncertainty = expected["uncertainty"]
    filters_np = np.asarray(est_filters, dtype=np.float32)
    _ci = os.environ.get("GITHUB_ACTIONS") == "true"
    np.testing.assert_allclose(filters_np, expected_filters, rtol=1e-2 if _ci else 1e-5, atol=1e-2 if _ci else 1e-5)
    np.testing.assert_allclose(np.asarray(estimation_dict["localisation"]), expected_localisation, rtol=1e-2 if _ci else 1e-4, atol=1e-0 if _ci else 1e-2)
    np.testing.assert_allclose(np.asarray(estimation_dict["colouration"]), expected_colouration, rtol=1e-2 if _ci else 1e-4, atol=1e-0 if _ci else 1e-2)
    np.testing.assert_allclose(np.asarray(estimation_dict["uncertainty"]), expected_uncertainty, rtol=1e-2 if _ci else 1e-5, atol=1e-2 if _ci else 1e-5)
