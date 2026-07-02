import json
import torch
import torch.nn as nn

with open('utils/hyperpameters.json', 'r') as f:
    hyperpameters = json.load(f)
    apply_init_branch_grad_norm = hyperpameters["llado2025_grad_surgery"]["apply_init_branch_grad_norm"]

class ItdIldDBranchGradNormFn(torch.autograd.Function):

    itd_init_branch_grad = None
    ild_init_branch_grad = None

    last_itd_grad = None
    last_ild_grad = None
    
    @staticmethod
    def forward(ctx, input):

        input_itd = input.clone()
        input_ild = input.clone()

        return input_itd, input_ild
    
    @staticmethod
    def backward(ctx, grad_itd, grad_ild):

        if ItdIldDBranchGradNormFn.itd_init_branch_grad is None:
            ItdIldDBranchGradNormFn.itd_init_branch_grad = torch.norm(grad_itd) + 1e-8
            ItdIldDBranchGradNormFn.ild_init_branch_grad = torch.norm(grad_ild) + 1e-8

            print(f"Initial branch grad norms: ITD = {ItdIldDBranchGradNormFn.itd_init_branch_grad}, ILD = {ItdIldDBranchGradNormFn.ild_init_branch_grad}")

        if not apply_init_branch_grad_norm:

            ItdIldDBranchGradNormFn.last_itd_grad = grad_itd
            ItdIldDBranchGradNormFn.last_ild_grad = grad_ild

            grad_input = grad_itd + grad_ild

            return grad_input

        grad_norm_sum = torch.norm(grad_itd + grad_ild).detach()

        # Normalise each branch by its *initial* norm,
        # then rescale by current total magnitude

        itd_grad = grad_itd / ItdIldDBranchGradNormFn.itd_init_branch_grad
        ild_grad = grad_ild / ItdIldDBranchGradNormFn.ild_init_branch_grad

        grad_combined = itd_grad + ild_grad
        scale = grad_norm_sum / (torch.norm(grad_combined).detach() + 1e-8)

        ItdIldDBranchGradNormFn.last_itd_grad = itd_grad * scale
        ItdIldDBranchGradNormFn.last_ild_grad = ild_grad * scale

        grad_input = ItdIldDBranchGradNormFn.last_itd_grad + ItdIldDBranchGradNormFn.last_ild_grad

        # Only one input to forward (input) → return one gradient
        return grad_input
