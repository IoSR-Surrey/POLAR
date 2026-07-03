"""Verify filter-tap ITD / ILD branch gradient decomposition."""

import torch

from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import load_npy_to_torch
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.llado2025.grad_surgery import ItdIldDBranchGradNormFn
from models.louspeaker_based.second_moment.model import SecondMoment
from models.train import ModelTrainer
from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from logger.saving import extract_filter_tap_branch_grads


def _reset_grad_surgery_state() -> None:
    ItdIldDBranchGradNormFn.itd_init_branch_grad = None
    ItdIldDBranchGradNormFn.ild_init_branch_grad = None
    ItdIldDBranchGradNormFn.last_itd_grad = None
    ItdIldDBranchGradNormFn.last_ild_grad = None


def _flatten_tap_list(grad_on_taps):
    return torch.cat([g.flatten() for g in grad_on_taps if g is not None])


def _training_setup(*, include_second_moment: bool):
    _reset_grad_surgery_state()

    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy")
    d = 2.0
    lstn_pos = torch.tensor([[0.0, 0.0]])
    spk_pos = torch.tensor(
        [
            [d * torch.cos(torch.deg2rad(torch.tensor(30.0))), d * torch.sin(torch.deg2rad(torch.tensor(30.0)))],
            [d * torch.cos(torch.deg2rad(torch.tensor(30.0))), -d * torch.sin(torch.deg2rad(torch.tensor(30.0)))],
        ]
    )

    llado = llado2025_grad_surgery(48000, hrirs, gtfb)
    second = SecondMoment(32, 2)
    models = [llado, second] if include_second_moment else [llado]
    weights = [1.0, 1.0] if include_second_moment else [1.0]
    targets = {"llado2025_grad_surgery": torch.tensor([[15.0]])}
    if include_second_moment:
        targets["SecondMoment"] = torch.tensor([[0.0, 0.0]])

    prop = BinauralFreeFieldPropagation(48000, lstn_pos, spk_pos, sofaile)
    nn = MultiSourceFIR(32, 2)
    trainer = ModelTrainer(models, weights=weights)
    spk_filters = nn(torch.ones(1, 1))
    binaural = prop(torch.ones(1, 1, 1), spk_filters)
    trainer.compute_estimates(binaural, spk_filters, listener_idx=0)
    trainer.compute_loss(targets)
    return nn, trainer


def test_llado_itd_and_ild_sum_to_llado_tap_grad() -> None:
    nn, trainer = _training_setup(include_second_moment=False)
    grad_itd_on_taps, grad_ild_on_taps = extract_filter_tap_branch_grads(trainer, nn)
    llado_on_taps = _flatten_tap_list(grad_itd_on_taps) + _flatten_tap_list(grad_ild_on_taps)
    llado_reference = _flatten_tap_list(
        torch.autograd.grad(
            trainer.loss_dict["llado2025_grad_surgery"],
            list(nn.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
    )
    assert torch.norm(_flatten_tap_list(grad_itd_on_taps)) > 0.0
    assert torch.norm(_flatten_tap_list(grad_ild_on_taps)) > 0.0
    assert torch.allclose(llado_on_taps, llado_reference, rtol=1e-4, atol=1e-3)


def test_branch_grads_exclude_second_moment() -> None:
    nn, trainer = _training_setup(include_second_moment=True)
    grad_itd_on_taps, grad_ild_on_taps = extract_filter_tap_branch_grads(trainer, nn)
    llado_on_taps = _flatten_tap_list(grad_itd_on_taps) + _flatten_tap_list(grad_ild_on_taps)
    llado_reference = _flatten_tap_list(
        torch.autograd.grad(
            trainer.loss_dict["llado2025_grad_surgery"],
            list(nn.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
    )
    assert torch.allclose(llado_on_taps, llado_reference, rtol=1e-4, atol=1e-3)
