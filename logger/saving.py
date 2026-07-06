


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

    def save_effective_loss_grad_norms(self, loss_grad_norms, model_trainer, gradnorm_task_weights=None):
        effective_loss_grad_norms = to_effective_loss_grad_norms(loss_grad_norms, model_trainer, gradnorm_task_weights)
        self.save("loss_grad_norms", effective_loss_grad_norms)

    def save_loss_grad_vectors(self, loss_dict, filter_module):
        self.save("loss_grad_vectors", extract_loss_grad_vectors(loss_dict, filter_module))

    def export_to_disk(self, output_dir, skip_keys=None, timestamp=False):
        """Write saved_data to output_dir as .npy and .json files."""
        from datetime import datetime
        from pathlib import Path
        import json

        output_dir = Path(output_dir)
        if timestamp:
            output_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        skip_keys = set(skip_keys) if skip_keys is not None else {"sofaile"}

        manifest = {}
        for key, values in self.saved_data.items():
            if key in skip_keys:
                manifest[key] = {"skipped": True, "reason": "in skip_keys"}
                continue
            try:
                manifest[key] = _export_saved_values(key, values, output_dir)
            except Exception as exc:
                manifest[key] = {"skipped": True, "reason": str(exc)}

        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return output_dir

############################################################################################################################################################
############################################################################################################################################################
                                                    # EXTRA/HELPER FUNCTIONS #
############################################################################################################################################################
############################################################################################################################################################

import json
from pathlib import Path

import numpy as np
import torch
from utils.dsp import cosine_similarity
from models.listener_based.llado2025.grad_surgery import ItdIldDBranchGradNormFn


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if all(isinstance(x, (torch.Tensor, np.ndarray, type(None))) for x in value):
            parts = [_to_numpy(x).ravel() for x in value if x is not None]
            if not parts:
                return None
            return np.concatenate(parts)
        return [_to_numpy(x) for x in value]
    raise TypeError(f"Unsupported type for export: {type(value)}")


def _is_scalar_like(value):
    if isinstance(value, (int, float, bool, str)) or value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return False


def _json_ready(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _can_stack(arrays):
    if not arrays:
        return False
    if all(_is_scalar_like(x) for x in arrays):
        return True
    if not all(isinstance(x, np.ndarray) for x in arrays):
        return False
    shapes = {x.shape for x in arrays}
    return len(shapes) == 1


def _stack_values(values):
    converted = [_to_numpy(v) for v in values]
    if _can_stack(converted):
        if all(_is_scalar_like(x) for x in converted):
            return np.array([x.item() if isinstance(x, np.ndarray) else x for x in converted])
        return np.stack(converted)
    return converted


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(value), f, indent=2)


def _write_npy(path, value):
    np.save(path, value)


def _export_saved_values(key, values, output_dir):
    converted = [_to_numpy(v) for v in values]

    if converted and all(isinstance(v, dict) for v in converted):
        return _export_dict_series(key, converted, output_dir)

    if len(converted) == 1:
        single = converted[0]
        if _is_scalar_like(single):
            path = output_dir / f"{key}.json"
            _write_json(path, single.item() if isinstance(single, np.ndarray) else single)
            return {"format": "json", "path": path.name, "n_entries": 1}
        if isinstance(single, np.ndarray):
            path = output_dir / f"{key}.npy"
            _write_npy(path, single)
            return {"format": "npy", "path": path.name, "shape": list(single.shape), "n_entries": 1}
        path = output_dir / f"{key}.json"
        _write_json(path, single)
        return {"format": "json", "path": path.name, "n_entries": 1}

    stacked = _stack_values(values)
    if isinstance(stacked, np.ndarray):
        path = output_dir / f"{key}.npy"
        _write_npy(path, stacked)
        return {"format": "npy", "path": path.name, "shape": list(stacked.shape), "n_entries": len(values)}

    path = output_dir / f"{key}.json"
    _write_json(path, converted)
    return {"format": "json", "path": path.name, "n_entries": len(values)}


def _export_dict_series(key, dict_series, output_dir):
    subkeys = dict_series[0].keys()
    entries = {}
    for subkey in subkeys:
        sub_values = [entry[subkey] for entry in dict_series]
        sub_key = f"{key}.{subkey}"
        entries[subkey] = _export_saved_values(sub_key, sub_values, output_dir)
    return {"format": "dict_series", "subkeys": entries, "n_entries": len(dict_series)}


def _populate_llado_branch_grads(model_trainer):
    """Run Llado loss backward to grad-surgery input so last_itd/ild_grad are set."""
    cues = model_trainer.kwargs_output["llado2025_grad_surgery"]["interauralcues"]
    llado_loss = model_trainer.loss_dict["llado2025_grad_surgery"]
    torch.autograd.grad(llado_loss, cues["rectified_signals"], retain_graph=True)


def extract_branch_grads(model_trainer):

    if "llado2025_grad_surgery" in model_trainer.model_dict:
        _populate_llado_branch_grads(model_trainer)
        rectified_signals_grad_itd = ItdIldDBranchGradNormFn.last_itd_grad
        rectified_signals_grad_ild = ItdIldDBranchGradNormFn.last_ild_grad

        return rectified_signals_grad_itd, rectified_signals_grad_ild



def extract_filter_tap_branch_grads(model_trainer, filter_module):
    """
    Llado ITD / ILD branch contributions to FIR filter taps.

    Call before loss.backward(). Branch upstream grads are populated via partial
    backward on the Llado loss; ILD uses an explicit VJP and ITD is the remaining
    Llado-only gradient on the filter taps.
    """
    if "llado2025_grad_surgery" not in model_trainer.model_dict:
        return None, None
    _populate_llado_branch_grads(model_trainer)
    cues = model_trainer.kwargs_output["llado2025_grad_surgery"]["interauralcues"]
    filter_params = list(filter_module.parameters())
    llado_loss = model_trainer.loss_dict["llado2025_grad_surgery"]
    grad_ild_on_taps = torch.autograd.grad(
        cues["rectified_signals_ild"],
        filter_params,
        grad_outputs=ItdIldDBranchGradNormFn.last_ild_grad,
        retain_graph=True,
        allow_unused=True,
    )
    grad_llado_on_taps = torch.autograd.grad(
        llado_loss,
        filter_params,
        retain_graph=True,
        allow_unused=True,
    )
    with torch.no_grad():
        grad_ild_on_taps = [
            g.detach().clone() if g is not None else None
            for g in grad_ild_on_taps
        ]
        grad_itd_on_taps = []
        for g_llado, g_ild in zip(grad_llado_on_taps, grad_ild_on_taps):
            if g_llado is None:
                grad_itd_on_taps.append(None)
            elif g_ild is None:
                grad_itd_on_taps.append(g_llado.detach().clone())
            else:
                grad_itd_on_taps.append(g_llado.detach().clone() - g_ild.detach())
    return grad_itd_on_taps, grad_ild_on_taps

def filter_tap_branch_grad_norms(grad_itd_on_taps, grad_ild_on_taps):
    norm_itd = torch.norm(torch.cat([g.flatten() for g in grad_itd_on_taps if g is not None]))
    norm_ild = torch.norm(torch.cat([g.flatten() for g in grad_ild_on_taps if g is not None]))
    return norm_itd, norm_ild, norm_itd / (norm_itd + norm_ild + 1e-12)

def extract_loss_grad_norms(loss_dict, filter_module):
    filter_params = list(filter_module.parameters())
    loss_grad_norms = {}
    for key, loss in loss_dict.items():
        if key == "total":
            continue
        grads = torch.autograd.grad(loss, filter_params, retain_graph=True, allow_unused=True)
        flat_grads = [g.reshape(-1) for g in grads if g is not None]
        if not flat_grads:
            loss_grad_norms[key] = torch.zeros((), device=loss.device, dtype=loss.dtype)
        else:
            loss_grad_norms[key] = torch.norm(torch.cat(flat_grads)).detach()
    return loss_grad_norms

def extract_loss_grad_vectors(loss_dict, filter_module):
    filter_params = list(filter_module.parameters())
    loss_grad_vectors = {}
    for key, loss in loss_dict.items():
        if key == "total":
            continue
        grads = torch.autograd.grad(loss, filter_params, retain_graph=True, allow_unused=True)
        flat_grads = [g.reshape(-1) for g in grads if g is not None]
        if not flat_grads:
            loss_grad_vectors[key] = None
        else:
            loss_grad_vectors[key] = torch.cat(flat_grads).detach()
    return loss_grad_vectors

def to_effective_loss_grad_norms(loss_grad_norms, model_trainer, gradnorm_task_weights=None):
    effective = {}
    for i, name in enumerate(model_trainer.model_dict):
        scale = gradnorm_task_weights[i] if gradnorm_task_weights is not None else model_trainer.model_dict[name]["weight"]
        effective[name] = loss_grad_norms[name] * scale
    return effective

def task_grad_pairwise_cosine_similarity_stats(loss_grad_vectors):
    n_epochs = len(loss_grad_vectors)
    task_names = list(loss_grad_vectors[0].keys())
    n_tasks = len(task_names)

    g0 = loss_grad_vectors[0][task_names[0]]
    ref = torch.zeros(g0.numel(), device=g0.device, dtype=g0.dtype)
    ref[0] = 1.0

    matrices = []
    ref_cosines_over_epochs = {name: [] for name in task_names}
    for epoch in range(n_epochs):
        mat = np.eye(n_tasks)
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                a = loss_grad_vectors[epoch][task_names[i]]
                b = loss_grad_vectors[epoch][task_names[j]]
                if a is None or b is None:
                    cos = float("nan")
                else:
                    cos = float(cosine_similarity(a, b))
                mat[i, j] = cos
                mat[j, i] = cos
        matrices.append(mat)
        for name in task_names:
            g = loss_grad_vectors[epoch][name]
            if g is None:
                ref_cosines_over_epochs[name].append(float("nan"))
            else:
                ref_cosines_over_epochs[name].append(float(cosine_similarity(g, ref)))

    matrices = np.asarray(matrices)
    pairwise_labels = []
    pairwise_over_epochs = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            pairwise_labels.append(f"{task_names[i]} vs {task_names[j]}")
            pairwise_over_epochs.append(matrices[:, i, j])

    return {
        "task_names": task_names,
        "mean_matrix": np.nanmean(matrices, axis=0),
        "pairwise_labels": pairwise_labels,
        "pairwise_over_epochs": pairwise_over_epochs,
        "ref_cosines_over_epochs": ref_cosines_over_epochs,
    }
