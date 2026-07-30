import torch
import torchaudio.functional as TF
import numpy as np
import sofar as sf

def sofa_find(angle, sofaile, elv=0):

    # Handle tensor
    if isinstance(angle, torch.Tensor):
        angle = angle.detach().cpu().numpy()

    # Convert int/float/array to numpy array, then ensure 1D
    angle = np.atleast_2d(np.asarray(angle, dtype=float))

    # Find index of closest angle in sofaile.SourcePosition
    sofa_angles = sofaile.SourcePosition[:, :2] % 360  # N_y x (az, elv)
    angles = np.concatenate((angle, elv * np.ones_like(angle)), axis=1) % 360  # N_x x (az, elv)
    diff = sofa_angles[:, None, :] - angles[None, :, :]  # N_y x N_x x (az, elv)
    dist = np.linalg.norm(diff, axis=-1)  # N_y x N_x
    idx = np.argmin(dist, axis=0)  # N_x

    return idx

def get_IRs(idx, sofaile, device='cpu', dtype=torch.float32):
    return torch.from_numpy(sofaile.Data_IR[idx, :, :]).to(device=device, dtype=dtype)


def load_sofa(sofa_file_path):
    return sf.read_sofa(sofa_file_path, verbose=False)

def resample_sofa_ir(sofaile, target_fs):
    """Resample a SOFA object's HRIRs in place to target_fs."""
    source_fs = float(np.asarray(sofaile.Data_SamplingRate).reshape(-1)[0])
    if source_fs == target_fs:
        return sofaile

    ir_dtype = sofaile.Data_IR.dtype
    sofaile.Data_IR = TF.resample(
        torch.from_numpy(sofaile.Data_IR),
        source_fs,
        target_fs,
    ).numpy().astype(ir_dtype, copy=False)
    sofaile.Data_SamplingRate = np.asarray(target_fs, dtype=np.asarray(sofaile.Data_SamplingRate).dtype)
    return sofaile


def get_template_for_llado2025_desena_2020(sofaile, device='cpu', dtype=torch.float32):
    angles = torch.arange(-90, 95, 5, device=device, dtype=dtype).view(-1,1)
    
    idx = sofa_find(angles, sofaile)

    out = get_IRs(idx, sofaile, device=device, dtype=dtype)
    return out

