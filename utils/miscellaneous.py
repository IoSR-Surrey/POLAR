import torch
import numpy as np

from models.cagrad import CAGrad
from models.gradnorm import GradNorm

from logger.saving import extract_loss_grad_and_norms

def load_npy_to_torch(npy_file, device='cpu', dtype=torch.float32):
    return torch.from_numpy(np.load(npy_file)).to(device=device, dtype=dtype)

def ready_for_plotting(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    elif isinstance(tensor, list):
        return np.array([ready_for_plotting(item) for item in tensor])
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unsupported type: {type(tensor)}")

def target_angle_for_listeners(lstn_pos, target_pos):
    """Azimuth (degrees) of target_pos as seen from each listener."""
    target_pos = torch.as_tensor(target_pos, device=lstn_pos.device, dtype=lstn_pos.dtype)
    source_vec = target_pos - lstn_pos
    return torch.rad2deg(torch.atan2(source_vec[..., 1], source_vec[..., 0]))


def create_mtl_optimizer(model_trainer, mtl_optimizer):
    task_names = list(model_trainer.model_dict.keys())
    if mtl_optimizer == "gradnorm":
        return GradNorm(
            task_names,
            task_importance=[model["weight"] for model in model_trainer.model_dict.values()],
        )
    if mtl_optimizer == "cagrad":
        return CAGrad(task_names)
    if mtl_optimizer == "plain":
        return None
    raise ValueError(f"Unsupported mtl_optimizer: {mtl_optimizer}")

def run_mtl_optimizer_step(mtl_optimizer, mtl, loss_dict, filter_module, optimizer, saving_manager, model_trainer):
    shared_params = list(filter_module.parameters())

    if mtl_optimizer == "gradnorm":
        gradnorm_info = mtl.step(loss_dict, shared_params, optimizer)
        saving_manager.save("raw_grads", gradnorm_info["raw_grads"])
        saving_manager.save("effective_grads", gradnorm_info["effective_grads"])
        saving_manager.save("weighted_total", gradnorm_info["weighted_total"])
        saving_manager.save("gradnorm_task_weights", gradnorm_info["task_weights"])
    elif mtl_optimizer == "cagrad":
        cagrad_info = mtl.step(loss_dict, shared_params, optimizer)
        saving_manager.save("raw_grads", cagrad_info["raw_grads"])
        saving_manager.save("effective_grad", cagrad_info["effective_grad"])
        saving_manager.save("cagrad_dual_weights", cagrad_info["task_weights"])
        saving_manager.save("cagrad_task_coeffs", cagrad_info["task_coeffs"])
    else:
        loss_grad_vectors, loss_grad_norms = extract_loss_grad_and_norms(loss_dict, filter_module)
        loss_dict['total'].backward()
        optimizer.step()
        saving_manager.save("raw_grads", loss_grad_vectors)
        saving_manager.save("raw_grad_norms", loss_grad_norms)

    
def _flatten_tap_grads(grad_on_taps):
    if grad_on_taps is None:
        return None
    grads = [g for g in grad_on_taps if g is not None]
    if not grads:
        return None
    return torch.cat([g.flatten() for g in grads])