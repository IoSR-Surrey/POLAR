import numpy as np
import torch
from scipy.optimize import minimize

from models.mtl_base import MultiTaskOptimizer, load_training_config

cagrad_c = load_training_config()["cagrad_c"]


class CAGrad(MultiTaskOptimizer):
    """Conflict-Averse Gradient descent (Liu et al., NeurIPS 2021)."""

    def __init__(self, task_names, c=cagrad_c, norm_eps=1e-8):
        super().__init__(task_names)
        self.c = c
        self.norm_eps = norm_eps

    def _num_shared_params(self, shared_params):
        return sum(param.numel() for param in shared_params if param.requires_grad)

    def _task_grads(self, losses, shared_params):
        num_params = self._num_shared_params(shared_params)
        task_grads = []
        for task_idx, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=task_idx < len(losses) - 1,
                allow_unused=True,
            )
            flat_grads = [g.reshape(-1) for g in grads if g is not None]
            if not flat_grads:
                task_grads.append(torch.zeros(num_params, device=loss.device, dtype=loss.dtype))
            else:
                task_grads.append(torch.cat(flat_grads))
        return task_grads

    def _solve_dual(self, task_grads):
        if len(task_grads) == 1:
            d = task_grads[0]
            coeffs = torch.ones(1, device=d.device, dtype=d.dtype)
            return d, coeffs.clone(), coeffs

        G = torch.stack(task_grads, dim=1)
        g0 = G.mean(dim=1)
        g0_norm = torch.norm(g0).item()

        if g0_norm < self.norm_eps or self.c == 0:
            coeffs = torch.ones(len(task_grads), device=G.device, dtype=G.dtype) / len(task_grads)
            d = g0
            dual_weights = coeffs.clone()
            return d, dual_weights, coeffs

        G_np = G.detach().cpu().numpy()
        g0_np = g0.detach().cpu().numpy()
        phi_sqrt = self.c * g0_norm
        n_tasks = len(task_grads)

        def objective(w):
            gw = G_np @ w
            return g0_np @ gw + phi_sqrt * np.linalg.norm(gw)

        w0 = np.ones(n_tasks) / n_tasks
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_tasks
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        dual_weights_np = result.x
        gw = G_np @ dual_weights_np
        gw_norm = np.linalg.norm(gw)

        if gw_norm < self.norm_eps:
            coeffs_np = np.ones(n_tasks) / n_tasks
        else:
            coeffs_np = np.ones(n_tasks) / n_tasks + (phi_sqrt / gw_norm) * dual_weights_np

        d = torch.from_numpy(G_np @ coeffs_np).to(device=G.device, dtype=G.dtype)
        dual_weights = torch.from_numpy(dual_weights_np).to(device=G.device, dtype=G.dtype)
        coeffs = torch.from_numpy(coeffs_np).to(device=G.device, dtype=G.dtype)
        return d, dual_weights, coeffs

    def _set_grad_from_flat(self, flat_grad, shared_params):
        offset = 0
        for param in shared_params:
            if not param.requires_grad:
                continue
            numel = param.numel()
            param.grad = flat_grad[offset:offset + numel].reshape(param.shape).clone()
            offset += numel

    def step(self, loss_dict, shared_params, model_optimizer):
        losses = self._task_losses(loss_dict)
        task_grads = self._task_grads(losses, shared_params)
        grad_norms = torch.stack([torch.norm(g) for g in task_grads])
        combined_grad, dual_weights, task_coeffs = self._solve_dual(task_grads)

        model_optimizer.zero_grad()
        self._set_grad_from_flat(combined_grad, shared_params)
        model_optimizer.step()

        return {
            "raw_grads": {name: task_grads[i].detach() for i, name in enumerate(self.task_names)},
            "effective_grad": combined_grad.detach(),
            "task_weights": dual_weights.detach(),
            "task_coeffs": task_coeffs.detach(),
        }
