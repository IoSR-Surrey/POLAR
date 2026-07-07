
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from utils.dsp import cosine_similarity
from models.listener_based.llado2025.grad_surgery import ItdIldDBranchGradNormFn


_TENSORBOARD_SKIP_KEYS = {
    "sofaile", "gtfb", "lag_values", "binaural_signals", "spk_filters",
    "grad_itd_on_taps", "grad_ild_on_taps", "loss_grad_vectors", "effective_loss_grad_vectors", "combined_grad_vector",
    "rectified_signals_grad_itd", "rectified_signals_grad_ild",
    "grad_itd_on_taps_norm", "grad_ild_on_taps_norm", "grad_itd_on_taps_norm_ratio",
    "lstn_pos", "spk_pos", "target_angle", "tau_iacc", "interaural_cues"
}


class SavingManager:

    def __init__(self, output_dir=None, skip_keys=None, timestamp=False, use_tensorboard=False, task_names=None):
        
        self.saved_data = {}
        self.output_dir = None
        self._seen_fignums = set()
        self._writer = None
        self._tensorboard_proc = None
        self._tensorboard_url = None
        self._step = None
        self._task_names = list(task_names) if task_names is not None else None
        self._skip_keys = set(skip_keys) if skip_keys is not None else {"sofaile"}
        self._tensorboard_skip_keys = self._skip_keys | _TENSORBOARD_SKIP_KEYS

        if output_dir is not None:
            self._setup_output_dir(output_dir, timestamp=timestamp)
            if use_tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                logdir = self.output_dir / "tensorboard"
                self._writer = SummaryWriter(logdir)
                self._tensorboard_proc, self._tensorboard_url = _launch_tensorboard(logdir)
                print(f"TensorBoard running at {self._tensorboard_url}")

    def set_step(self, step):
        self._step = step

    def _setup_output_dir(self, output_dir, timestamp=False, skip_keys=None):
        output_dir = Path(output_dir)
        if timestamp:
            output_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "npy").mkdir(exist_ok=True)
        (output_dir / "json").mkdir(exist_ok=True)
        self.output_dir = output_dir
        if skip_keys is not None:
            self._skip_keys = set(skip_keys)
            self._tensorboard_skip_keys = self._skip_keys | _TENSORBOARD_SKIP_KEYS
        self._seen_fignums = set()

    def save(self, key, value):
        if key not in self.saved_data:
            self.saved_data[key] = []
        self.saved_data[key].append(value)
        if self._writer is not None and key not in self._tensorboard_skip_keys:
            step = self._step if self._step is not None else len(self.saved_data[key]) - 1
            try:
                _log_to_tensorboard(self._writer, key, _tensorboard_value(key, value, self._task_names), step)
            except (TypeError, ValueError):
                pass

    def flush_tensorboard(self):
        if self._writer is not None:
            self._writer.flush()

    def export_to_disk(self, output_dir=None, skip_keys=None, timestamp=False):
        """Write saved_data to output_dir as .npy and .json files."""
        if self.output_dir is None:
            self._setup_output_dir(output_dir, timestamp=timestamp, skip_keys=skip_keys)

        manifest = {}
        for key, values in self.saved_data.items():
            if key in self._skip_keys:
                manifest[key] = {"skipped": True, "reason": "in skip_keys"}
                continue
            try:
                manifest[key] = _export_saved_values(key, values, self.output_dir)
            except Exception as exc:
                manifest[key] = {"skipped": True, "reason": str(exc)}

        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return self.output_dir

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._tensorboard_proc is not None:
            self._tensorboard_proc.terminate()
            self._tensorboard_proc = None

    def save_new_plots(self, basename, names=None):
        import io

        import matplotlib.pyplot as plt
        from PIL import Image

        if self.output_dir is None:
            self._seen_fignums = set(plt.get_fignums())
            return

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        new_fignums = sorted(n for n in plt.get_fignums() if n not in self._seen_fignums)
        for i, fig_num in enumerate(new_fignums):
            if names is not None and i < len(names):
                filename = names[i]
            elif len(new_fignums) == 1:
                filename = basename
            else:
                filename = f"{basename}_{i}"
            fig = plt.figure(fig_num)
            path = plot_dir / f"{filename}.png"
            if self._writer is not None:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                png_bytes = buf.getvalue()
                path.write_bytes(png_bytes)
                img = np.array(Image.open(io.BytesIO(png_bytes)))
                self._writer.add_image(f"plots/{filename}", img, global_step=0, dataformats="HWC")
            else:
                fig.savefig(path, bbox_inches="tight", dpi=150)

        self._seen_fignums |= set(new_fignums)

############################################################################################################################################################
############################################################################################################################################################
                                                    # EXTRA/HELPER FUNCTIONS #
############################################################################################################################################################
############################################################################################################################################################


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


def _tensorboard_value(key, value, task_names):
    if task_names is not None and key in ("gradnorm_task_weights", "gradnorm_grad_norms", "cagrad_task_coeffs", "cagrad_dual_weights", "cagrad_grad_norms"):
        array = np.asarray(_to_numpy(value)).reshape(-1)
        return {name: float(v) for name, v in zip(task_names, array)}
    return value


def _scalar_value(value):
    if _is_scalar_like(value):
        return float(value.item() if isinstance(value, np.ndarray) else value)
    return None


def _series_dict_from_array(value):
    return {str(i): float(v) for i, v in enumerate(np.asarray(value).reshape(-1))}


def _log_to_tensorboard(writer, tag, value, step):
    value = _to_numpy(value)
    if isinstance(value, dict):
        scalar_dict = {}
        series_items = []
        for key, item in value.items():
            scalar = _scalar_value(item)
            if scalar is not None:
                scalar_dict[key] = scalar
            elif isinstance(item, np.ndarray) and item.size <= 16:
                series_items.append((key, item))
            else:
                _log_to_tensorboard(writer, f"{tag}/{key}", item, step)
        if scalar_dict:
            writer.add_scalars(tag, scalar_dict, step)
        for key, item in series_items:
            writer.add_scalars(f"{tag}/{key}", _series_dict_from_array(item), step)
        return
    scalar = _scalar_value(value)
    if scalar is not None:
        writer.add_scalar(tag, scalar, step)
        return
    if isinstance(value, np.ndarray) and value.size <= 16:
        writer.add_scalars(tag, _series_dict_from_array(value), step)


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


def _npy_path(output_dir, key):
    return output_dir / "npy" / f"{key}.npy"


def _json_path(output_dir, key):
    return output_dir / "json" / f"{key}.json"


def _export_saved_values(key, values, output_dir):
    converted = [_to_numpy(v) for v in values]

    if converted and all(isinstance(v, dict) for v in converted):
        return _export_dict_series(key, converted, output_dir)

    if len(converted) == 1:
        single = converted[0]
        if _is_scalar_like(single):
            path = _json_path(output_dir, key)
            _write_json(path, single.item() if isinstance(single, np.ndarray) else single)
            return {"format": "json", "path": path.relative_to(output_dir).as_posix(), "n_entries": 1}
        if isinstance(single, np.ndarray):
            path = _npy_path(output_dir, key)
            _write_npy(path, single)
            return {"format": "npy", "path": path.relative_to(output_dir).as_posix(), "shape": list(single.shape), "n_entries": 1}
        path = _json_path(output_dir, key)
        _write_json(path, single)
        return {"format": "json", "path": path.relative_to(output_dir).as_posix(), "n_entries": 1}

    stacked = _stack_values(values)
    if isinstance(stacked, np.ndarray):
        path = _npy_path(output_dir, key)
        _write_npy(path, stacked)
        return {"format": "npy", "path": path.relative_to(output_dir).as_posix(), "shape": list(stacked.shape), "n_entries": len(values)}

    path = _json_path(output_dir, key)
    _write_json(path, converted)
    return {"format": "json", "path": path.relative_to(output_dir).as_posix(), "n_entries": len(values)}


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

def extract_loss_grad_and_norms(loss_dict, filter_module):
    filter_params = list(filter_module.parameters())
    loss_grad_vectors = {}
    loss_grad_norms = {}
    for key, loss in loss_dict.items():
        grads = torch.autograd.grad(loss, filter_params, retain_graph=True, allow_unused=True)
        flat_grads = [g.reshape(-1) for g in grads if g is not None]
        if not flat_grads:
            loss_grad_vectors[key] = None
            loss_grad_norms[key] = torch.zeros((), device=loss.device, dtype=loss.dtype)
        else:
            loss_grad_vectors[key] = torch.cat(flat_grads).detach()
            loss_grad_norms[key] = torch.norm(loss_grad_vectors[key]).detach()
    return loss_grad_vectors, loss_grad_norms

def task_grad_pairwise_cosine_similarity_stats(loss_grad_vectors, combined_grad_vectors=None):
    n_epochs = len(loss_grad_vectors)
    task_names = list(loss_grad_vectors[0].keys())
    n_tasks = len(task_names)
    use_combined_reference = combined_grad_vectors is not None

    if not use_combined_reference:
        g0 = loss_grad_vectors[0][task_names[0]]
        ref = torch.zeros(g0.numel(), device=g0.device, dtype=g0.dtype)
        ref[0] = 1.0

    matrices = []
    ref_cosines_over_epochs = {name: [] for name in task_names}
    combined_norm_over_epochs = [] if use_combined_reference else None
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
        if use_combined_reference:
            g_combined = combined_grad_vectors[epoch]
            combined_norm_over_epochs.append(
                float(torch.norm(g_combined)) if g_combined is not None else float("nan")
            )
            for name in task_names:
                g = loss_grad_vectors[epoch][name]
                if g is None or g_combined is None:
                    ref_cosines_over_epochs[name].append(float("nan"))
                else:
                    ref_cosines_over_epochs[name].append(float(cosine_similarity(g, g_combined)))
        else:
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

    stats = {
        "task_names": task_names,
        "mean_matrix": np.nanmean(matrices, axis=0),
        "pairwise_labels": pairwise_labels,
        "pairwise_over_epochs": pairwise_over_epochs,
        "ref_cosines_over_epochs": ref_cosines_over_epochs,
        "use_combined_reference": use_combined_reference,
    }
    if use_combined_reference:
        stats["combined_norm_over_epochs"] = combined_norm_over_epochs
    return stats


def _find_free_port(start=6006):
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _wait_for_tensorboard(url, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.5)
    return False


def _launch_tensorboard(logdir):
    port = _find_free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", f"--logdir={logdir}", f"--port={port}", "--reload_interval=5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    url = f"http://localhost:{port}"
    if not _wait_for_tensorboard(url):
        err = proc.stderr.read().decode(errors="replace")
        proc.terminate()
        hint = " Try: pip install 'setuptools>=65,<82'" if "pkg_resources" in err else ""
        raise RuntimeError(f"TensorBoard failed to start on {url}: {err}{hint}")
    webbrowser.open(url)
    return proc, url
