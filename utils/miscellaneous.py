import torch
import numpy as np


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

    
def _flatten_tap_grads(grad_on_taps):
    if grad_on_taps is None:
        return None
    grads = [g for g in grad_on_taps if g is not None]
    if not grads:
        return None
    return torch.cat([g.flatten() for g in grads])