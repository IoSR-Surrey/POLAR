import torch

def _calculate_source_angle_2D(lstn_pos, source_pos):
    source_vec = source_pos[:, None, :] - lstn_pos[None, :, :]
    return torch.rad2deg(torch.atan2(source_vec[..., 1], source_vec[..., 0]))

def _calculate_distance_gain(lstn_pos, source_pos):
    source_vec = source_pos[:, None, :] - lstn_pos[None, :, :]
    gain = torch.reciprocal(torch.clamp(torch.linalg.vector_norm(source_vec.float(), dim=-1), min=1e-6))
    return gain.reshape(-1)

def _apply_delay(hrirs, delay_samples):
    max_delay_samples = delay_samples[1]
    hrirs = torch.nn.functional.pad(hrirs, (0, max_delay_samples))
    for i in range(len(hrirs)):
        d = int(delay_samples[0][i])
        if d != 0:
            hrirs[i, 0, :] = torch.roll(hrirs[i, 0, :], d, dims=0)
            hrirs[i, 1, :] = torch.roll(hrirs[i, 1, :], d, dims=0)
    return hrirs

def _calculate_delay_samples(lstn_pos, spk_pos, fs):
    source_vec = spk_pos[:, None, :] - lstn_pos[None, :, :]
    tau_sources = torch.linalg.vector_norm(source_vec.float(), dim=-1) / 343
    min_tau_per_listener = torch.min(tau_sources, dim=0).values
    delay_samples = torch.round((tau_sources - min_tau_per_listener[None, :]) * fs)
    max_delay_samples = int(torch.max(delay_samples).item())
    return delay_samples, max_delay_samples
