import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pytest
import torch
import torch.nn.functional as F
import torchaudio.functional as TF

from datasets.stimulus_dataset.dataset import BandpassWhiteNoiseDataset
from models.listener_based.llado2025 import grad_surgery as grad_surgery_module
from models.listener_based.llado2025.grad_surgery import ItdIldDBranchGradNormFn
from models.listener_based.llado2025.model import llado2025, llado2025_grad_surgery
from utils.peripheral_processing import estimate_interaural_cues
from utils.itd_ild_distance import itd_ild_template_distance
from utils.miscellaneous import load_npy_to_torch
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020

MODEL_VARIANTS = [
    pytest.param(False, id="llado2025"),
    pytest.param(True, id="llado2025_grad_surgery"),
]

TEMPLATE_AZIMUTHS = [0.0, -30.0, 30.0, -90.0, 90.0]
N_AZIMUTHS = 37
N_BANDS = 24


@pytest.fixture(autouse=True)
def _reset_grad_surgery_branch_state() -> None:
    ItdIldDBranchGradNormFn.itd_init_branch_grad = None
    ItdIldDBranchGradNormFn.ild_init_branch_grad = None
    ItdIldDBranchGradNormFn.last_itd_grad = None
    ItdIldDBranchGradNormFn.last_ild_grad = None


def make_model(grad_surgery: bool = False) -> llado2025 | llado2025_grad_surgery:
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile)
    if grad_surgery:
        return llado2025_grad_surgery(fs=48000, hrirs=hrirs, gtfb=gtfb)
    return llado2025(fs=48000, hrirs=hrirs, gtfb=gtfb)


def make_template(dtype: torch.dtype = torch.float32, device: str = "cpu") -> dict[str, torch.Tensor]:
    angles = torch.arange(-90, 95, 5, dtype=dtype, device=device)
    cues = (angles / 90.0).unsqueeze(1).expand(-1, N_BANDS).clone()

    weigthing_function = torch.linspace(0, 1, cues.shape[1], dtype=dtype, device=device)
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


def make_localisation_model(grad_surgery: bool = False) -> llado2025 | llado2025_grad_surgery:
    model_cls = llado2025_grad_surgery if grad_surgery else llado2025
    model = model_cls.__new__(model_cls)
    model.template = make_template()
    return model


def estimate_localisation(target_value: float, grad_surgery: bool = False) -> torch.Tensor:
    model = make_localisation_model(grad_surgery=grad_surgery)
    template = model.template
    target = {
        "itd": torch.full((1, N_BANDS), target_value, dtype=template["itd"].dtype, device=template["itd"].device),
        "ild": torch.full((1, N_BANDS), target_value, dtype=template["ild"].dtype, device=template["ild"].device),
    }
    distance = itd_ild_template_distance(target, template)
    distance["weighting_function_SPL"] = torch.full(
        (1, N_BANDS), 1/N_BANDS, dtype=template["itd"].dtype, device=template["itd"].device
    )
    return model._estimate_azimuth(distance)


def _azimuth_index(azimuth_deg: float) -> int:
    return int((azimuth_deg + 90) / 5)


def _symmetric_impulse_binaural_signal() -> torch.Tensor:
    binaural = torch.ones(1, 2, 1)
    return binaural.requires_grad_(True)


def _impulse_panning_ear_gradients(target_deg: float, grad_surgery: bool) -> tuple[float, float, float]:
    model = make_model(grad_surgery=grad_surgery)
    binaural = _symmetric_impulse_binaural_signal()
    estimate = model(binaural, None)["output"]
    loss = F.mse_loss(estimate, torch.tensor([target_deg], dtype=estimate.dtype))
    loss.backward()
    return (
        estimate.item(),
        binaural.grad[0, 0, 0].item(),
        binaural.grad[0, 1, 0].item(),
    )


def _branch_combined_grad_norm(
    monkeypatch: pytest.MonkeyPatch, apply_norm: bool, target_deg: float = 45.0
) -> float:
    ItdIldDBranchGradNormFn.itd_init_branch_grad = None
    ItdIldDBranchGradNormFn.ild_init_branch_grad = None
    ItdIldDBranchGradNormFn.last_itd_grad = None
    ItdIldDBranchGradNormFn.last_ild_grad = None
    monkeypatch.setattr(grad_surgery_module, "apply_init_branch_grad_norm", apply_norm)

    model = make_model(grad_surgery=True)
    binaural = torch.ones(1, 2, 1, requires_grad=True)
    loss = F.mse_loss(model(binaural, None)["output"], torch.tensor([target_deg], dtype=binaural.dtype))
    loss.backward()
    combined = ItdIldDBranchGradNormFn.last_itd_grad + ItdIldDBranchGradNormFn.last_ild_grad
    return torch.norm(combined).item()


class _EarGains(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gains = torch.nn.Parameter(torch.ones(2))

    def forward(self) -> torch.Tensor:
        return torch.stack([self.gains[0], self.gains[1]]).view(1, 2, 1)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_localisation_estimator_returns_middle_azimuth_for_middle_template(grad_surgery: bool) -> None:
    estimated = estimate_localisation(0.0, grad_surgery=grad_surgery)

    assert estimated.shape == (1,)
    assert estimated.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_localisation_estimator_changes_sign_with_target_cues(grad_surgery: bool) -> None:
    left_estimate = estimate_localisation(-1.0, grad_surgery=grad_surgery).item()
    right_estimate = estimate_localisation(1.0, grad_surgery=grad_surgery).item()

    assert 0.0 > left_estimate
    assert right_estimate > 0.0
    assert left_estimate == pytest.approx(-90.0)
    assert right_estimate == pytest.approx(90.0)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
@pytest.mark.parametrize("azimuth_deg", TEMPLATE_AZIMUTHS)
def test_returns_corresponding_template_angle_with_template_hrirs(
    grad_surgery: bool, azimuth_deg: float
) -> None:
    model = make_model(grad_surgery=grad_surgery)
    hrir_stimulus = model.hrirs[_azimuth_index(azimuth_deg) : _azimuth_index(azimuth_deg) + 1]

    estimated = model(hrir_stimulus, None)["output"]

    assert estimated.shape == (1,)
    assert estimated.item() == pytest.approx(azimuth_deg, abs=0.5) # abs = 0.5 due to problems with argmax when there are multiple max values


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_louder_bandpass_source_dominates_localisation(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    noise_dataset = BandpassWhiteNoiseDataset(
        fs=48000, n_bands=2, n_samples_per_band=1, stimulus_length_ms=200, ramp_time_ms=5
    )
    loud_noise = torch.from_numpy(noise_dataset._bandpass_white_noise(500, 700, seed=11)).view(1, 1, -1)
    quiet_noise = torch.from_numpy(noise_dataset._bandpass_white_noise(3000, 4000, seed=22)).view(1, 1, -1)

    loud_angle = 45.0
    quiet_angle = -45.0
    loud_source = TF.fftconvolve(
        loud_noise, model.hrirs[_azimuth_index(loud_angle) : _azimuth_index(loud_angle) + 1]
    )
    quiet_source = TF.fftconvolve(
        quiet_noise, model.hrirs[_azimuth_index(quiet_angle) : _azimuth_index(quiet_angle) + 1]
    )

    estimated = model(loud_source + quiet_source*10**(-40/20), None)["output"]

    assert estimated.item() == pytest.approx(loud_angle, abs=5.0)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_template_has_expected_structure(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    template = model.template

    assert template["itd"].shape == (N_AZIMUTHS, N_BANDS)
    assert template["ild"].shape == (N_AZIMUTHS, N_BANDS)
    assert template["azimuths"].shape == (N_AZIMUTHS,)
    assert template["weigthing_function"].shape == (N_BANDS,)
    assert template["itd_max"].shape == (N_BANDS,)
    assert template["ild_max"].shape == (N_BANDS,)
    assert template["azimuths"][0].item() == pytest.approx(-90.0)
    assert template["azimuths"][-1].item() == pytest.approx(90.0)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_forward_output_shape_for_batch(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    batch = model.hrirs[12:15]

    output = model(batch, None)["output"]

    assert output.shape == (3,)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_forward_returns_expected_kwargs(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    result = model(model.hrirs[18:19], None)

    assert set(result.keys()) == {"output", "kwargs"}
    assert set(result["kwargs"].keys()) == {"interauralcues", "itd_ild_distance"}
    assert set(result["kwargs"]["interauralcues"].keys()) >= {"itd", "ild", "gtfb_signals", "iacc"}
    assert set(result["kwargs"]["itd_ild_distance"].keys()) == {
        "itd_distance",
        "ild_distance",
        "distance_combined",
        "weighting_function_SPL",
    }


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_distance_tensors_have_expected_shape(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    distance = model(model.hrirs[18:19], None)["kwargs"]["itd_ild_distance"]

    assert distance["itd_distance"].shape == (1, N_AZIMUTHS, N_BANDS)
    assert distance["ild_distance"].shape == (1, N_AZIMUTHS, N_BANDS)
    assert distance["distance_combined"].shape == (1, N_AZIMUTHS, N_BANDS)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_azimuth_estimates_within_valid_range(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    estimates = model(model.hrirs, None)["output"]

    assert estimates.shape == (N_AZIMUTHS,)
    assert torch.all(estimates >= -90.0)
    assert torch.all(estimates <= 90.0)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
def test_template_hrir_estimates_increase_with_azimuth_index(grad_surgery: bool) -> None:
    model = make_model(grad_surgery=grad_surgery)
    estimates = model(model.hrirs, None)["output"]

    # Allow small local non-monotonicity from soft-argmax ties at extreme angles.
    assert torch.all(estimates[1:] >= estimates[:-1] - 1.0)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
@pytest.mark.parametrize("n_samples", [1, 256, 4096], ids=["1tap", "256tap", "4096tap"])
def test_forward_accepts_variable_input_lengths(grad_surgery: bool, n_samples: int) -> None:
    model = make_model(grad_surgery=grad_surgery)
    stimulus = torch.ones(1, 2, n_samples)

    output = model(stimulus, None)["output"]

    assert output.shape == (1,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
@pytest.mark.parametrize("azimuth_deg", [30.0, -30.0])
def test_swapping_ears_flips_estimated_azimuth_sign(grad_surgery: bool, azimuth_deg: float) -> None:
    model = make_model(grad_surgery=grad_surgery)
    stimulus = model.hrirs[_azimuth_index(azimuth_deg) : _azimuth_index(azimuth_deg) + 1]

    original = model(stimulus, None)["output"].item()
    swapped = model(stimulus[:, [1, 0], :], None)["output"].item()

    assert original == pytest.approx(azimuth_deg, abs=0.5)
    assert swapped == pytest.approx(-azimuth_deg, abs=0.5)


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
@pytest.mark.parametrize("target_deg", [45.0, -45.0])
def test_one_gradient_descent_step_moves_estimate_toward_target(
    grad_surgery: bool, target_deg: float
) -> None:
    model = make_model(grad_surgery=grad_surgery)
    binaural = torch.ones(1, 2, 1, requires_grad=True)
    learning_rate = 0.01

    initial_estimate = model(binaural, None)["output"].detach().item()
    loss = F.mse_loss(model(binaural, None)["output"], torch.tensor([target_deg], dtype=binaural.dtype))
    loss.backward()
    with torch.no_grad():
        binaural -= learning_rate * binaural.grad

    updated_estimate = model(binaural, None)["output"].detach().item()
    initial_error = abs(initial_estimate - target_deg)
    updated_error = abs(updated_estimate - target_deg)

    assert initial_estimate == pytest.approx(0.0, abs=1e-4)
    assert updated_error < initial_error


@pytest.mark.parametrize("target_deg", [45.0, -45.0])
def test_grad_surgery_input_gradients_differ_from_base_when_branch_norm_enabled(
    monkeypatch: pytest.MonkeyPatch, target_deg: float
) -> None:
    monkeypatch.setattr(grad_surgery_module, "apply_init_branch_grad_norm", True)

    base_model = make_model(grad_surgery=False)
    grad_model = make_model(grad_surgery=True)
    base_input = torch.ones(1, 2, 1, requires_grad=True)
    grad_input = torch.ones(1, 2, 1, requires_grad=True)
    target = torch.tensor([target_deg], dtype=base_input.dtype)

    F.mse_loss(base_model(base_input, None)["output"], target).backward()
    F.mse_loss(grad_model(grad_input, None)["output"], target).backward()

    assert not torch.allclose(base_input.grad, grad_input.grad)


def test_grad_surgery_branch_norm_preserves_combined_gradient_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_norm = _branch_combined_grad_norm(monkeypatch, apply_norm=False)
    normed_norm = _branch_combined_grad_norm(monkeypatch, apply_norm=True)

    assert raw_norm == pytest.approx(normed_norm, rel=1e-3)


@pytest.mark.parametrize("target_deg", [45.0, -45.0])
def test_grad_surgery_itd_and_ild_branches_receive_nonzero_gradients(target_deg: float) -> None:
    model = make_model(grad_surgery=True)
    binaural = torch.ones(1, 2, 1, requires_grad=True)

    F.mse_loss(model(binaural, None)["output"], torch.tensor([target_deg], dtype=binaural.dtype)).backward()

    assert ItdIldDBranchGradNormFn.last_itd_grad is not None
    assert ItdIldDBranchGradNormFn.last_ild_grad is not None
    assert ItdIldDBranchGradNormFn.last_itd_grad.abs().sum() > 0.0
    assert ItdIldDBranchGradNormFn.last_ild_grad.abs().sum() > 0.0


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
@pytest.mark.parametrize("target_deg", [45.0, -45.0])
def test_mini_optimisation_moves_impulse_estimate_toward_target(
    grad_surgery: bool, target_deg: float
) -> None:
    model = make_model(grad_surgery=grad_surgery)
    ear_gains = _EarGains()
    optimizer = torch.optim.Adam(ear_gains.parameters(), lr=0.05)

    with torch.no_grad():
        initial_estimate = model(ear_gains(), None)["output"].item()

    for _ in range(20):
        optimizer.zero_grad()
        loss = F.mse_loss(model(ear_gains(), None)["output"], torch.tensor([target_deg]))
        loss.backward()
        optimizer.step()

    final_estimate = model(ear_gains(), None)["output"].item()
    initial_error = abs(initial_estimate - target_deg)
    final_error = abs(final_estimate - target_deg)

    assert initial_estimate == pytest.approx(0.0, abs=1e-4)
    assert final_error < initial_error
    assert final_error < 10.0
    if target_deg > 0.0:
        assert ear_gains.gains[0].item() > ear_gains.gains[1].item()
    else:
        assert ear_gains.gains[1].item() > ear_gains.gains[0].item()


def test_grad_surgery_forward_matches_base_model() -> None:
    base_model = make_model(grad_surgery=False)
    grad_model = make_model(grad_surgery=True)
    stimulus = base_model.hrirs[18:19]

    base_result = base_model(stimulus, None)
    grad_result = grad_model(stimulus, None)

    assert torch.allclose(base_result["output"], grad_result["output"])
    assert torch.allclose(
        base_result["kwargs"]["interauralcues"]["itd"],
        grad_result["kwargs"]["interauralcues"]["itd"],
    )
    assert torch.allclose(
        base_result["kwargs"]["interauralcues"]["ild"],
        grad_result["kwargs"]["interauralcues"]["ild"],
    )
    assert torch.allclose(
        base_result["kwargs"]["itd_ild_distance"]["distance_combined"],
        grad_result["kwargs"]["itd_ild_distance"]["distance_combined"],
    )


@pytest.mark.parametrize("target_deg", [0.0, 45.0, -45.0])
def test_grad_surgery_input_gradients_match_base_when_branch_norm_disabled(
    monkeypatch: pytest.MonkeyPatch, target_deg: float
) -> None:
    monkeypatch.setattr(grad_surgery_module, "apply_init_branch_grad_norm", False)

    base_model = make_model(grad_surgery=False)
    grad_model = make_model(grad_surgery=True)
    base_input = torch.ones(1, 2, 1, requires_grad=True)
    grad_input = torch.ones(1, 2, 1, requires_grad=True)
    target = torch.tensor([target_deg], dtype=base_input.dtype)

    base_loss = F.mse_loss(base_model(base_input, None)["output"], target)
    grad_loss = F.mse_loss(grad_model(grad_input, None)["output"], target)
    base_loss.backward()
    grad_loss.backward()

    assert torch.allclose(base_input.grad, grad_input.grad)


def test_grad_surgery_backward_flows_to_input() -> None:
    model = make_model(grad_surgery=True)
    stimulus = model.hrirs[18:19].clone().requires_grad_(True)

    model(stimulus, None)["output"].sum().backward()

    assert stimulus.grad is not None
    assert stimulus.grad.shape == stimulus.shape
    assert stimulus.grad.abs().sum() > 0.0


@pytest.mark.parametrize("grad_surgery", MODEL_VARIANTS)
@pytest.mark.parametrize(
    ("target_deg", "left_sign", "right_sign"),
    [
        # Positive azimuth is to the left: +45 should raise the left ear and lower the right.
        (45.0, -1, 1),
        (-45.0, 1, -1),
    ],
    ids=["target_+45", "target_-45"],
)
def test_impulse_panning_gradients_push_correct_ear_levels(
    grad_surgery: bool, target_deg: float, left_sign: int, right_sign: int
) -> None:
    # Symmetric 1-tap impulse is centred at 0 degrees. Under gradient descent
    # (param -= lr * grad), a negative grad increases amplitude and a positive grad decreases it.
    estimate, left_grad, right_grad = _impulse_panning_ear_gradients(target_deg, grad_surgery)

    assert estimate == pytest.approx(0.0, abs=1e-4)
    assert left_grad * left_sign > 0.0
    assert right_grad * right_sign > 0.0


def test_grad_surgery_backward_stores_branch_gradients() -> None:
    model = make_model(grad_surgery=True)
    stimulus = model.hrirs[18:19].clone().requires_grad_(True)

    result = model(stimulus, None)
    cues = result["kwargs"]["interauralcues"]
    result["output"].sum().backward()

    assert ItdIldDBranchGradNormFn.last_itd_grad is not None
    assert ItdIldDBranchGradNormFn.last_ild_grad is not None
    assert ItdIldDBranchGradNormFn.last_itd_grad.shape == cues["rectified_signals_itd"].shape
    assert ItdIldDBranchGradNormFn.last_ild_grad.shape == cues["rectified_signals_ild"].shape


def test_MATLAB_same_interauralcues_on_test_signal() -> None:
    ref_gtfb_left = load_npy_to_torch("data/test_data/test_data_llado2025/gtfb_left.npy")
    ref_gtfb_right = load_npy_to_torch("data/test_data/test_data_llado2025/gtfb_right.npy")
    ref_itd = load_npy_to_torch("data/test_data/test_data_llado2025/itd.npy")
    ref_ild = load_npy_to_torch("data/test_data/test_data_llado2025/ild.npy")
    input_data = load_npy_to_torch("data/test_data/test_data_llado2025/input.npy")
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
    lag_values = torch.arange(-0.001 * 48000, 0.001 * 48000 + 1, 1, dtype=torch.float32) / 48000
    interauralcues = estimate_interaural_cues(input_data, gtfb, lag_values)

    estimated_gtfb = interauralcues["gtfb_signals"]
    assert torch.allclose(estimated_gtfb[:, :, 0, :4831], ref_gtfb_left)
    assert torch.allclose(estimated_gtfb[:, :, 1, :4831], ref_gtfb_right)
    assert torch.allclose(interauralcues["itd"], -1 * ref_itd, atol=1e-3)
    assert torch.allclose(interauralcues["ild"], -1 * ref_ild, atol=1e-1)


def test_grad_surgery_interaural_cues_match_peripheral_processing() -> None:
    ref_gtfb_left = load_npy_to_torch("data/test_data/test_data_llado2025/gtfb_left.npy")
    ref_gtfb_right = load_npy_to_torch("data/test_data/test_data_llado2025/gtfb_right.npy")
    ref_itd = load_npy_to_torch("data/test_data/test_data_llado2025/itd.npy")
    ref_ild = load_npy_to_torch("data/test_data/test_data_llado2025/ild.npy")
    input_data = load_npy_to_torch("data/test_data/test_data_llado2025/input.npy")
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
    lag_values = torch.arange(-0.001 * 48000, 0.001 * 48000 + 1, 1, dtype=torch.float32) / 48000

    reference = estimate_interaural_cues(input_data, gtfb, lag_values)
    model = make_model(grad_surgery=True)
    estimated = model._estimate_interaural_cues(input_data)

    assert torch.allclose(estimated["gtfb_signals"][:, :, 0, :4831], ref_gtfb_left)
    assert torch.allclose(estimated["gtfb_signals"][:, :, 1, :4831], ref_gtfb_right)
    assert torch.allclose(estimated["itd"], -1 * ref_itd, atol=1e-3)
    assert torch.allclose(estimated["ild"], -1 * ref_ild, atol=1e-1)
    assert torch.allclose(estimated["itd"], reference["itd"])
    assert torch.allclose(estimated["ild"], reference["ild"])
