import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from models.listener_based.desena2020.model import desena2020
from utils.miscellaneous import load_npy_to_torch
from utils.sofa_functions import get_template_for_llado2025_desena_2020, load_sofa

N_AZIMUTHS = 37
N_BANDS = 24


@pytest.fixture
def model() -> desena2020:
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile)
    return desena2020(fs=48000, hrirs=hrirs, gtfb=gtfb)


def _azimuth_index(azimuth_deg: float) -> int:
    return int((azimuth_deg + 90) / 5)


def test_template_has_expected_structure(model: desena2020) -> None:
    template = model.template

    assert template["itd"].shape == (N_AZIMUTHS, N_BANDS)
    assert template["ild"].shape == (N_AZIMUTHS, N_BANDS)
    assert template["azimuths"].shape == (N_AZIMUTHS,)
    assert template["weigthing_function"].shape == (N_BANDS,)
    assert template["azimuths"][0].item() == pytest.approx(-90.0)
    assert template["azimuths"][-1].item() == pytest.approx(90.0)


def test_forward_output_shape_for_batch(model: desena2020) -> None:
    batch = model.hrirs[12:15]
    output = model(batch, None)["output"]

    assert output.shape == (3,)


def test_forward_returns_expected_kwargs(model: desena2020) -> None:
    result = model(model.hrirs[18:19], None)

    assert set(result.keys()) == {"output", "kwargs"}
    assert set(result["kwargs"].keys()) == {"interauralcues", "itd_ild_distance", "distance"}
    assert "weightedLikelihood" in result["kwargs"]["distance"]
    assert "weighting_function_SPL" in result["kwargs"]["distance"]


def test_output_uncertainty_within_unit_interval(model: desena2020) -> None:
    uncertainty = model(model.hrirs, None)["output"]

    assert torch.all(uncertainty >= 0.0)
    assert torch.all(uncertainty <= 1.0)


def test_template_hrir_has_lower_uncertainty_than_noise(model: desena2020) -> None:
    template_stimulus = model.hrirs[_azimuth_index(0.0) : _azimuth_index(0.0) + 1]
    noise = torch.randn(1, 2, 256) * 0.01

    template_uncertainty = model(template_stimulus, None)["output"].item()
    noise_uncertainty = model(noise, None)["output"].item()

    assert template_uncertainty < noise_uncertainty


def test_loudness_weighting_produces_expected_shapes(model: desena2020) -> None:
    stimulus = model.hrirs[18:19]
    result = model(stimulus, None)
    distance = result["kwargs"]["distance"]

    assert distance["weightedLikelihood"].shape == (1, N_AZIMUTHS)
    assert distance["weighting_function_SPL"].shape == (1, N_BANDS)


@pytest.mark.parametrize("azimuth_deg", [0.0, -45.0, 45.0])
def test_template_hrir_uncertainty_below_midpoint(model: desena2020, azimuth_deg: float) -> None:
    idx = _azimuth_index(azimuth_deg)
    uncertainty = model(model.hrirs[idx : idx + 1], None)["output"].item()

    assert uncertainty < 0.5
