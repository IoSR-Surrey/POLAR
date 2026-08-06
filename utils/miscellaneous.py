import json
import subprocess
from pathlib import Path

import torch
import numpy as np

from models.cagrad import CAGrad
from models.gradnorm import GradNorm

from logger.saving import extract_loss_grad_and_norms

from scipy.signal import firwin, lfilter


def git_commit_info(cwd=None):
    """Return {"commit": sha_or_None, "dirty": bool_or_None} for the repo at cwd."""
    if cwd is None:
        cwd = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"commit": None, "dirty": None}


def write_run_config(output_dir, run_config, filename="run_config.json"):
    """Write run_config.json, attaching git commit info if missing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(run_config)
    if "git" not in payload:
        payload["git"] = git_commit_info()
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


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


def create_mtl_optimizer(model_trainer, mtl_optimizer, training_config=None):
    from models.mtl_base import load_training_config

    if training_config is None:
        training_config = load_training_config()

    task_names = list(model_trainer.model_dict.keys())
    if mtl_optimizer == "gradnorm":
        return GradNorm(
            task_names,
            alpha=training_config["gradnorm_alpha"],
            lr=training_config["gradnorm_lr"],
            task_importance=[model["weight"] for model in model_trainer.model_dict.values()],
        )
    if mtl_optimizer == "cagrad":
        return CAGrad(task_names, c=training_config["cagrad_c"])
    if mtl_optimizer == "gradnorm+cagrad":
        return {
            "gradnorm": GradNorm(
                task_names,
                alpha=training_config["gradnorm_alpha"],
                lr=training_config["gradnorm_lr"],
                task_importance=[model["weight"] for model in model_trainer.model_dict.values()],
            ),
            "cagrad": CAGrad(task_names, c=training_config["cagrad_c"]),
        }
    if mtl_optimizer == "plain":
        return None
    raise ValueError(f"Unsupported mtl_optimizer: {mtl_optimizer}")

def run_mtl_optimizer_step(mtl_optimizer, mtl, loss_dict, filter_module, optimizer, saving_manager, model_trainer, save=True):
    shared_params = list(filter_module.parameters())
    saved = {}

    if mtl_optimizer == "gradnorm":
        gradnorm_info = mtl.step(loss_dict, shared_params, optimizer)
        saved = {
            "raw_grads": gradnorm_info["raw_grads"],
            "effective_grads": gradnorm_info["effective_grads"],
            "weighted_total": gradnorm_info["weighted_total"],
            "gradnorm_task_weights": gradnorm_info["task_weights"],
        }
    elif mtl_optimizer == "cagrad":
        cagrad_info = mtl.step(loss_dict, shared_params, optimizer)
        saved = {
            "raw_grads": cagrad_info["raw_grads"],
            "effective_grad": cagrad_info["effective_grad"],
            "cagrad_dual_weights": cagrad_info["task_weights"],
            "cagrad_task_coeffs": cagrad_info["task_coeffs"],
        }

    elif mtl_optimizer == "gradnorm+cagrad":
        gradnorm, cagrad = mtl["gradnorm"], mtl["cagrad"]
        gradnorm_info = gradnorm.step(loss_dict, shared_params, optimizer, step=False)
        task_weights = gradnorm_info["task_weights"]
        for i, name in enumerate(gradnorm.task_names):
            loss_dict[name] = loss_dict[name] * task_weights[i]

        cagrad_info = cagrad.step(loss_dict, shared_params, optimizer)
        saved = {
            "raw_grads": cagrad_info["raw_grads"],
            "effective_grad": cagrad_info["effective_grad"],
            "cagrad_dual_weights": cagrad_info["task_weights"],
            "cagrad_task_coeffs": cagrad_info["task_coeffs"],
            "gradnorm_task_weights": gradnorm_info["task_weights"],
        }
    else:
        loss_grad_vectors, loss_grad_norms = extract_loss_grad_and_norms(loss_dict, filter_module)
        loss_dict['total'].backward()
        optimizer.step()
        saved = {
            "raw_grads": loss_grad_vectors,
            "raw_grad_norms": loss_grad_norms,
        }

    if save:
        for key, value in saved.items():
            saving_manager.save(key, value)
    return saved

    
def _flatten_tap_grads(grad_on_taps):
    if grad_on_taps is None:
        return None
    grads = [g for g in grad_on_taps if g is not None]
    if not grads:
        return None
    return torch.cat([g.flatten() for g in grads])

def generate_low_pass_stimulus(cutoff_hz=1000, fs=48000, length=512, device='cpu', dtype=torch.float32):
    fir_coeff = firwin(numtaps=101, cutoff=cutoff_hz, fs=fs)
    noise = np.random.randn(length)
    lowpass_signal = lfilter(fir_coeff, 1.0, noise)
    return torch.from_numpy(lowpass_signal).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)