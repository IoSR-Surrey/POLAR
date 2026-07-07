import json


def load_training_config():
    with open('config/hyperpameters.json', 'r') as f:
        return json.load(f)["training"]


class MultiTaskOptimizer:
    """Shared state for multi-task optimizers (task names + per-task loss lookup)."""

    def __init__(self, task_names):
        self.task_names = list(task_names)

    def _task_losses(self, loss_dict):
        return [loss_dict[name] for name in self.task_names]
