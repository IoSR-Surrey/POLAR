import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pytest
import torch

from utils.sofa_functions import (
    get_IRs,
    get_template_for_llado2025_desena_2020,
    load_sofa,
    sofa_find,
)

SOFA_PATH = "data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
N_AZIMUTHS = 37
IR_LEN = 256


@pytest.fixture
def sofaile():
    return load_sofa(SOFA_PATH)


@pytest.mark.parametrize("azimuth_deg", [-90.0, 0.0, 90.0])
def test_sofa_find_returns_valid_index_for_known_azimuths(sofaile, azimuth_deg: float) -> None:
    idx = sofa_find(np.array([[azimuth_deg]]), sofaile)

    assert idx.shape == (1,)
    assert 0 <= idx[0] < sofaile.Data_IR.shape[0]


def test_sofa_find_tensor_matches_numpy(sofaile) -> None:
    angles = torch.tensor([[-30.0], [0.0], [30.0]])
    idx_tensor = sofa_find(angles, sofaile)
    idx_numpy = sofa_find(angles.detach().cpu().numpy(), sofaile)

    assert np.array_equal(idx_tensor, idx_numpy)


def test_sofa_find_template_azimuths_returns_unique_indices(sofaile) -> None:
    azimuths = np.arange(-90, 95, 5).reshape(-1, 1)
    indices = sofa_find(azimuths, sofaile)

    assert indices.shape == (N_AZIMUTHS,)
    assert len(np.unique(indices)) == N_AZIMUTHS


def test_get_irs_shape_and_dtype(sofaile) -> None:
    idx = sofa_find(np.array([[0.0]]), sofaile)
    hrirs = get_IRs(idx, sofaile, device="cpu", dtype=torch.float32)

    assert hrirs.shape == (1, 2, IR_LEN)
    assert hrirs.dtype == torch.float32


def test_get_template_for_llado2025_desena_2020_shape(sofaile) -> None:
    template = get_template_for_llado2025_desena_2020(sofaile)

    assert template.shape == (N_AZIMUTHS, 2, IR_LEN)


def test_symmetric_sofa_swaps_ears_at_opposite_azimuths(sofaile) -> None:
    hrirs = get_template_for_llado2025_desena_2020(sofaile)

    left_at_minus_90 = hrirs[0, 0]
    right_at_minus_90 = hrirs[0, 1]
    left_at_plus_90 = hrirs[-1, 0]
    right_at_plus_90 = hrirs[-1, 1]

    assert torch.allclose(left_at_minus_90, right_at_plus_90, atol=1e-4)
    assert torch.allclose(right_at_minus_90, left_at_plus_90, atol=1e-4)


def test_load_sofa_returns_sofa_object(sofaile) -> None:
    assert hasattr(sofaile, "Data_IR")
    assert hasattr(sofaile, "SourcePosition")
