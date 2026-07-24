"""Sanity checks: localisation depends only on the mono stimulus, not unrelated RNG state."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.IKO_functions import num_hoa_channels
from utils.miscellaneous import load_npy_to_torch
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from models.listener_based.llado2025.model import llado2025_grad_surgery

SIGNAL_LENGTH = 256
SEED_A = 42
SEED_B = 12345


def load_iko_brir_optimisation_module():
    module_path = ROOT / "scripts" / "IKO" / "IKO_brir_optimisation.py"
    spec = importlib.util.spec_from_file_location("iko_brir_optimisation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(iko_brir):
    with open(ROOT / "config" / "hyperpameters.json", encoding="utf-8") as f:
        config = json.load(f)

    fs = config["env"]["fs"]
    device = config["env"]["device"]
    dtype = torch.float32
    target_angle = float(config["training"]["target_angle"])
    n_eigenbeams = num_hoa_channels(3)

    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    llado_model.eval()

    brirs = iko_brir.load_iko_brirs(iko_brir.BRIR_NPY, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype)
    eigenbeam_to_loudspeaker = iko_brir.load_sph_to_driver(
        iko_brir.SPH_TO_DRIVER_NPY, n_eigenbeams, iko_brir.N_IKO_LOUDSPEAKERS, device, dtype,
    )
    weights = torch.zeros(n_eigenbeams, device=device, dtype=dtype)
    weights[0] = 1.0
    target = torch.tensor(target_angle, device=device, dtype=dtype)
    return {
        "device": device,
        "dtype": dtype,
        "llado_model": llado_model,
        "brirs": brirs,
        "eigenbeam_to_loudspeaker": eigenbeam_to_loudspeaker,
        "weights": weights,
        "target": target,
    }


def estimate_azimuth(iko_brir, shared, signal):
    with torch.no_grad():
        binaural, _ = iko_brir.render_binaural_brir(
            signal, shared["weights"], shared["eigenbeam_to_loudspeaker"], shared["brirs"],
        )
        azimuth, loss = iko_brir.evaluate_localisation(
            shared["llado_model"], binaural, shared["target"],
        )
    return float(azimuth.item()), float(loss.item()), binaural.detach().cpu().clone()


def make_noise(seed, length, device, dtype):
    torch.manual_seed(seed)
    return torch.randn(1, length, device=device, dtype=dtype)


def main():
    iko_brir = load_iko_brir_optimisation_module()
    shared = setup(iko_brir)
    device, dtype = shared["device"], shared["dtype"]
    atol = 0.0
    rtol = 0.0

    print("Test 1: same seed twice → identical localisation")
    signal_1a = make_noise(SEED_A, SIGNAL_LENGTH, device, dtype)
    az_1a, loss_1a, bin_1a = estimate_azimuth(iko_brir, shared, signal_1a)
    signal_1b = make_noise(SEED_A, SIGNAL_LENGTH, device, dtype)
    az_1b, loss_1b, bin_1b = estimate_azimuth(iko_brir, shared, signal_1b)
    same_signal = torch.equal(signal_1a, signal_1b)
    same_binaural = torch.allclose(bin_1a, bin_1b, rtol=rtol, atol=atol)
    same_az = az_1a == az_1b
    same_loss = loss_1a == loss_1b
    pass_1 = same_signal and same_binaural and same_az and same_loss
    print(f"  seed={SEED_A}")
    print(f"  signal equal: {same_signal}")
    print(f"  binaural equal: {same_binaural}")
    print(f"  azimuth: {az_1a:+.6f}° vs {az_1b:+.6f}° → equal={same_az}")
    print(f"  loc_loss: {loss_1a:.6f} vs {loss_1b:.6f} → equal={same_loss}")
    print(f"  PASS={pass_1}")
    print()

    print("Test 2: different seeds for torch, but load the same saved noise → identical localisation")
    signal_saved = make_noise(SEED_A, SIGNAL_LENGTH, device, dtype)
    with tempfile.TemporaryDirectory() as tmp:
        npy_path = Path(tmp) / "mono_noise.npy"
        np.save(npy_path, signal_saved.detach().cpu().numpy())

        torch.manual_seed(SEED_A)
        loaded_a = torch.from_numpy(np.load(npy_path)).to(device=device, dtype=dtype)
        # Burn some RNG so global state differs before localisation
        _ = torch.randn(8, device=device, dtype=dtype)
        az_2a, loss_2a, bin_2a = estimate_azimuth(iko_brir, shared, loaded_a)

        torch.manual_seed(SEED_B)
        loaded_b = torch.from_numpy(np.load(npy_path)).to(device=device, dtype=dtype)
        _ = torch.randn(8, device=device, dtype=dtype)
        az_2b, loss_2b, bin_2b = estimate_azimuth(iko_brir, shared, loaded_b)

    same_loaded = torch.equal(loaded_a, loaded_b) and torch.equal(loaded_a, signal_saved)
    same_binaural_2 = torch.allclose(bin_2a, bin_2b, rtol=rtol, atol=atol)
    same_az_2 = az_2a == az_2b
    same_loss_2 = loss_2a == loss_2b
    pass_2 = same_loaded and same_binaural_2 and same_az_2 and same_loss_2
    print(f"  saved with seed={SEED_A}; reload under seeds {SEED_A} and {SEED_B}")
    print(f"  loaded signals equal (and match saved): {same_loaded}")
    print(f"  binaural equal: {same_binaural_2}")
    print(f"  azimuth: {az_2a:+.6f}° vs {az_2b:+.6f}° → equal={same_az_2}")
    print(f"  loc_loss: {loss_2a:.6f} vs {loss_2b:.6f} → equal={same_loss_2}")
    print(f"  PASS={pass_2}")
    print()

    all_pass = pass_1 and pass_2
    print(f"ALL PASS={all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
