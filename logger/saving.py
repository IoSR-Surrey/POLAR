


class SavingManager:

    def __init__(self):
        
        self.saved_data = {}

    def save(self, key, value):
        if key not in self.saved_data:
            self.saved_data[key] = []
        self.saved_data[key].append(value)

    def save_branch_grads(self, model_trainer):
        rectified_signals_grad_itd, rectified_signals_grad_ild = extract_branch_grads(model_trainer)
        self.save("rectified_signals_grad_itd", rectified_signals_grad_itd)
        self.save("rectified_signals_grad_ild", rectified_signals_grad_ild)

    def save_filter_tap_branch_grads(self, model_trainer, filter_module):
        grad_itd_on_taps, grad_ild_on_taps = extract_filter_tap_branch_grads(model_trainer, filter_module)
        self.save("grad_itd_on_taps", grad_itd_on_taps)
        self.save("grad_ild_on_taps", grad_ild_on_taps)

        grad_itd_on_taps_norm, grad_ild_on_taps_norm, grad_itd_on_taps_norm_ratio = filter_tap_branch_grad_norms(grad_itd_on_taps, grad_ild_on_taps)
        self.save("grad_itd_on_taps_norm", grad_itd_on_taps_norm)
        self.save("grad_ild_on_taps_norm", grad_ild_on_taps_norm)
        self.save("grad_itd_on_taps_norm_ratio", grad_itd_on_taps_norm_ratio)




############################################################################################################################################################
############################################################################################################################################################
                                                    # EXTRA/HELPER FUNCTIONS #
############################################################################################################################################################
############################################################################################################################################################

import torch
from models.listener_based.llado2025.grad_surgery import ItdIldDBranchGradNormFn


def extract_branch_grads(model_trainer):

    if "llado2025_grad_surgery" in model_trainer.model_dict:

        rectified_signals_grad_itd = ItdIldDBranchGradNormFn.last_itd_grad
        rectified_signals_grad_ild = ItdIldDBranchGradNormFn.last_ild_grad

        return rectified_signals_grad_itd, rectified_signals_grad_ild



def extract_filter_tap_branch_grads(model_trainer, filter_module):
    """
    ITD / ILD contribution to FIR filter taps after loss.backward().
    Call before optimizer.step(). Requires loss.backward(retain_graph=True).
    """
    if "llado2025_grad_surgery" not in model_trainer.model_dict:
        return None, None
    cues = model_trainer.kwargs_output["llado2025_grad_surgery"]["interauralcues"]
    filter_params = list(filter_module.parameters())
    grad_itd_on_taps = torch.autograd.grad(
        cues["rectified_signals_itd"],
        filter_params,
        grad_outputs=ItdIldDBranchGradNormFn.last_itd_grad,
        retain_graph=False,
        allow_unused=True,
    )
    with torch.no_grad():
        grad_ild_on_taps = [
            (p.grad.detach().clone() - g.detach().clone()) if g is not None else p.grad.detach().clone()
            for p, g in zip(filter_params, grad_itd_on_taps)
        ]
        grad_itd_on_taps = [
            g.detach().clone() if g is not None else None
            for g in grad_itd_on_taps
        ]
    return grad_itd_on_taps, grad_ild_on_taps

def filter_tap_branch_grad_norms(grad_itd_on_taps, grad_ild_on_taps):
    norm_itd = torch.norm(torch.cat([g.flatten() for g in grad_itd_on_taps if g is not None]))
    norm_ild = torch.norm(torch.cat([g.flatten() for g in grad_ild_on_taps if g is not None]))
    return norm_itd, norm_ild, norm_itd / (norm_itd + norm_ild + 1e-12)