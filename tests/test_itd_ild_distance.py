import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch

from utils.itd_ild_distance import itd_ild_template_distance

N_AZIMUTHS = 37
N_BANDS = 24


def make_template() -> dict[str, torch.Tensor]:
    angles = torch.arange(-90, 95, 5, dtype=torch.float32)
    cues = (angles / 90.0).unsqueeze(1).expand(-1, N_BANDS).clone()
    weigthing_function = torch.linspace(0, 1, N_BANDS)

    itd_max = torch.max(torch.abs(cues), dim=0).values
    ild_max = torch.max(torch.abs(cues), dim=0).values

    return {
        "itd": cues.clone(),
        "ild": cues.clone(),
        "azimuths": angles.clone(),
        "weigthing_function": weigthing_function,
        "itd_max": itd_max,
        "ild_max": ild_max,
        "itd_normed": cues / itd_max,
        "ild_normed": cues / ild_max,
    }


def test_distance_returns_expected_keys() -> None:
    template = make_template()
    estimated = {"itd": template["itd"][18:19], "ild": template["ild"][18:19]}
    distance = itd_ild_template_distance(estimated, template)

    assert set(distance.keys()) == {"itd_distance", "ild_distance", "distance_combined"}


def test_distance_tensors_have_expected_shape() -> None:
    template = make_template()
    estimated = {"itd": template["itd"][18:19], "ild": template["ild"][18:19]}
    distance = itd_ild_template_distance(estimated, template)

    assert distance["itd_distance"].shape == (1, N_AZIMUTHS, N_BANDS)
    assert distance["ild_distance"].shape == (1, N_AZIMUTHS, N_BANDS)
    assert distance["distance_combined"].shape == (1, N_AZIMUTHS, N_BANDS)


def test_matching_template_row_has_minimum_combined_distance() -> None:
    template = make_template()
    row_idx = 18
    estimated = {"itd": template["itd"][row_idx : row_idx + 1], "ild": template["ild"][row_idx : row_idx + 1]}
    distance = itd_ild_template_distance(estimated, template)

    combined = distance["distance_combined"][0]
    min_per_band = combined.min(dim=0).values
    matched_per_band = combined[row_idx]

    assert torch.allclose(matched_per_band, min_per_band, atol=1e-5)


def test_weighting_function_selects_itd_or_ild_branch() -> None:
    template = make_template()
    estimated = {"itd": torch.zeros(1, N_BANDS), "ild": torch.ones(1, N_BANDS)}

    template["weigthing_function"] = torch.zeros(N_BANDS)
    distance_itd_only = itd_ild_template_distance(estimated, template)

    template["weigthing_function"] = torch.ones(N_BANDS)
    distance_ild_only = itd_ild_template_distance(estimated, template)

    assert torch.allclose(
        distance_itd_only["distance_combined"], distance_itd_only["itd_distance"]
    )
    assert torch.allclose(
        distance_ild_only["distance_combined"], distance_ild_only["ild_distance"]
    )
