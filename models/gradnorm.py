import json

import torch
import torch.nn as nn

with open('config/hyperpameters.json', 'r') as f:
    hyperpameters = json.load(f)
    gradnorm_alpha = hyperpameters["training"]["gradnorm_alpha"]
    gradnorm_lr = hyperpameters["training"]["gradnorm_lr"]


class GradNorm:
    """Adaptive task weighting via gradient norms (Chen et al., 2018)."""

    def __init__(self, task_names, alpha=gradnorm_alpha, lr=gradnorm_lr, task_importance=None, device='cpu', dtype=torch.float32):
        self.task_names = list(task_names)
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(len(self.task_names), dtype=torch.float32))
        self.weight_optimizer = torch.optim.Adam([self.weights], lr=lr)
        self.initial_losses = None

        if task_importance is None:
            task_importance = [1.0] * len(self.task_names)
        if len(task_importance) != len(self.task_names):
            raise ValueError(
                f"task_importance length ({len(task_importance)}) must match "
                f"task_names length ({len(self.task_names)})"
            )
        importance = torch.tensor(task_importance, dtype=torch.float32)
        self.task_importance = (importance / importance.mean()).to(device=device, dtype=dtype)

    def _renormalize_weights(self):
        with torch.no_grad():
            self.weights.data = self.weights.data * len(self.task_names) / self.weights.data.sum()

    def _task_losses(self, loss_dict):
        return [loss_dict[name] for name in self.task_names]

    def _grad_norm(self, weighted_loss, shared_params):
        grads = torch.autograd.grad(
            weighted_loss,
            shared_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )
        flat_grads = [g.reshape(-1) for g in grads if g is not None]
        if not flat_grads:
            return torch.zeros((), device=weighted_loss.device, dtype=weighted_loss.dtype)
        return torch.norm(torch.cat(flat_grads))

    def step(self, loss_dict, shared_params, model_optimizer):
        losses = self._task_losses(loss_dict)

        if self.initial_losses is None:
            self.initial_losses = torch.stack([loss.detach() for loss in losses])

        grad_norms = []
        for task_idx, loss in enumerate(losses):
            grad_norms.append(self._grad_norm(self.weights[task_idx] * loss, shared_params))
        grad_norms = torch.stack(grad_norms)

        importance = self.task_importance
        mean_grad_norm = (grad_norms * importance).sum() / importance.sum()
        mean_grad_norm = mean_grad_norm.detach()
        inverse_train_rates = torch.stack(
            [losses[i] / self.initial_losses[i] for i in range(len(losses))]
        ).pow(self.alpha).detach()
        target_grad_norms = mean_grad_norm * inverse_train_rates * importance

        gradnorm_loss = (importance * (grad_norms - target_grad_norms).abs()).sum() / importance.sum()

        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward(retain_graph=True)
        self.weight_optimizer.step()
        self._renormalize_weights()

        model_optimizer.zero_grad()
        total_loss = sum(self.weights[i] * losses[i] for i in range(len(losses)))
        total_loss.backward()
        model_optimizer.step()

        return {
            "grad_norms": grad_norms.detach(),
            "task_weights": self.weights.detach().clone(),
            "weighted_total": total_loss.detach(),
        }
