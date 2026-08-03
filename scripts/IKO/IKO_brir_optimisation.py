import os
import sys
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as TF
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from propagation.helpers import _calculate_source_angle_2D
from utils.IKO_functions import compute_real_sh_matrix, num_hoa_channels
from utils.miscellaneous import load_npy_to_torch
from utils.seed import set_seed
from utils.sofa_functions import load_sofa, resample_sofa_ir, get_template_for_llado2025_desena_2020
from logger.saving import extract_loss_grad_and_norms, task_grad_pairwise_cosine_similarity_stats
from models.cagrad import CAGrad
from models.listener_based.llado2025.model import llado2025_grad_surgery
from plotting.scatter_plots import _add_critical_band_colorbar, _compute_target_itd_ild, _scatter_itd_ild_from_arrays
from plotting.XY_plots import (
    loss_over_epochs_plot,
    loss_grad_norms_over_epochs_plot,
    raw_grad_cosine_to_reference_plot,
    task_grad_pairwise_cosine_similarity_plot,
)


# Optional extra rotation for beam-pattern polar plots only (does not affect SH evaluation).
_BEAM_PATTERN_PLOT_ROTATION_RAD = 0.0
# hoa_to_driver physical radiation is π out from raw scipy real-SH azimuths (listening tests:
# ACN3+ beams to IKO back, ACN1+ to IKO right). Evaluate Y at physical_az + offset so plots
# and SH losses match the physical IKO frame (az 0 = look / front).
_PHYSICAL_AZ_TO_MATH_SH_OFFSET_RAD = np.pi
_BEAM_PATTERN_DB_FLOOR = -40.0
N_IKO_LOUDSPEAKERS = 20
IKO_DATA_DIR = Path("data/IKO")
SPH_TO_DRIVER_NPY = IKO_DATA_DIR / "hoa_to_driver" / "hoa_to_driver.npy"
BRIR_NPY = IKO_DATA_DIR / "driver_to_vl" / "brir_centre_v2.npy"


def physical_azimuths_to_math_sh(azimuths):
    """Map IKO-frame (physical) azimuths to the math SH angles used by compute_real_sh_matrix."""
    return (np.asarray(azimuths, dtype=float) + _PHYSICAL_AZ_TO_MATH_SH_OFFSET_RAD) % (2 * np.pi)


def build_iko_weighted_loss_dict(lambdas, losses):
    """Build lambda-weighted task losses, dropping tasks with zero weight."""
    return {name: lambdas[name] * losses[name] for name in losses if lambdas[name] != 0.0}


def iko_total_loss(lambdas, losses):
    """Sum of lambda-weighted losses (includes zero-weight terms for logging)."""
    total = None
    for name in losses:
        term = lambdas[name] * losses[name]
        total = term if total is None else total + term
    return total


def get_active_harmonic_indices(order, use_2d_harmonics, device):
    """Return full HOA indices, or only sectoral harmonics (|m| = n) for 2D playback."""
    indices = []
    idx = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            if not use_2d_harmonics or abs(m) == n:
                indices.append(idx)
            idx += 1
    return torch.tensor(indices, device=device, dtype=torch.long)


def expand_active_harmonic_weights(weight_model, active_indices, n_channels, device, dtype):
    """Expand optimised active coefficients into a full HOA vector with inactive channels at zero."""
    active_weights = weight_model(torch.ones(1, 1, device=device, dtype=dtype)).reshape(-1)
    return torch.zeros(n_channels, device=device, dtype=dtype).scatter(0, active_indices, active_weights)


def apply_iko_optim_step(optimizer, weight_model, weighted_losses, total_loss, cagrad=None):
    """Update weights via CAGrad (if given) or plain summed-loss backward.

    Returns dict with per-task raw_grads (and effective_grad when using CAGrad).
    """
    if cagrad is not None:
        return cagrad.step(weighted_losses, list(weight_model.parameters()), optimizer)
    raw_grads, _ = extract_loss_grad_and_norms(weighted_losses, weight_model)
    total_loss.backward()
    optimizer.step()
    return {"raw_grads": raw_grads}


def _beam_pattern_plot_azimuths(azimuths):
    """Rotate IKO-frame azimuth angles for beam-pattern polar display only."""
    return (np.asarray(azimuths, dtype=float) + _BEAM_PATTERN_PLOT_ROTATION_RAD) % (2 * np.pi)


def _beam_pattern_db(magnitude, reference):
    """Convert beam-pattern magnitude to dB with a fixed display floor."""
    magnitude = np.asarray(magnitude, dtype=float)
    return np.maximum(20.0 * np.log10(np.maximum(magnitude, 1e-12) / max(float(reference), 1e-12)), _BEAM_PATTERN_DB_FLOOR)


def _format_db_polar_axis(ax):
    """Format offset non-negative polar radii as dB labels."""
    tick_db = np.arange(_BEAM_PATTERN_DB_FLOOR, 0.1, 10.0)
    ax.set_ylim(0.0, -_BEAM_PATTERN_DB_FLOOR)
    ax.set_yticks(tick_db - _BEAM_PATTERN_DB_FLOOR)
    ax.set_yticklabels([f"{value:.0f} dB" for value in tick_db])


def load_iko_brirs(brir_path, n_speakers, device, dtype):
    """Load measured loudspeaker BRIRs from npy. Returns (n_speakers, 2, T)."""
    brirs = np.load(brir_path)
    if brirs.ndim != 3:
        raise ValueError(f"Expected BRIRs with shape (n_speakers, 2, T), got {brirs.shape}.")
    if brirs.shape[0] != n_speakers or brirs.shape[1] != 2:
        raise ValueError(
            f"Expected BRIRs shape ({n_speakers}, 2, T), got {brirs.shape}."
        )
    return torch.from_numpy(np.asarray(brirs, dtype=np.float32)).to(device=device, dtype=dtype)


def load_sph_to_driver(path, n_eigenbeams, n_speakers, device, dtype):
    """Load SH-to-driver impulse responses from npy. Returns (L, n_speakers, taps)."""
    sph_to_driver = np.load(path)
    if sph_to_driver.ndim != 3:
        raise ValueError(
            f"Expected sph_to_driver with shape (L, n_speakers, taps), got {sph_to_driver.shape}."
        )
    if sph_to_driver.shape[0] != n_eigenbeams or sph_to_driver.shape[1] != n_speakers:
        raise ValueError(
            f"Expected sph_to_driver shape ({n_eigenbeams}, {n_speakers}, taps), "
            f"got {sph_to_driver.shape}."
        )
    return torch.from_numpy(np.asarray(sph_to_driver, dtype=np.float32)).to(device=device, dtype=dtype)


def setup_geometry(d, iko_angle_deg, reflector_angle_deg, listener_positions, fs, c=343.0, iko_look_direction_deg=180.0):
    """Build scene geometry, SH directions (from IKO), SOFA azimuths (from listener), and reflection delay.

    SH azimuths are expressed in the IKO frame: 0 rad aligns with iko_look_direction_deg in world
    coordinates (the orientation in which sph_to_driver / eigenbeam patterns were measured).
    """
    iko_position = torch.tensor(
        [[d * torch.cos(torch.deg2rad(torch.tensor(iko_angle_deg))), d * torch.sin(torch.deg2rad(torch.tensor(iko_angle_deg)))]],
        dtype=torch.float32,
    )
    reflector_position = torch.tensor(
        [[d * torch.cos(torch.deg2rad(torch.tensor(reflector_angle_deg))), d * torch.sin(torch.deg2rad(torch.tensor(reflector_angle_deg)))]],
        dtype=torch.float32,
    )

    iko_to_listener = listener_positions - iko_position
    iko_to_reflector = reflector_position - iko_position
    world_azimuths_rad = torch.cat([
        torch.atan2(iko_to_listener[:, 1], iko_to_listener[:, 0]),
        torch.atan2(iko_to_reflector[:, 1], iko_to_reflector[:, 0]),
    ], dim=0)
    look_rad = torch.deg2rad(torch.tensor(iko_look_direction_deg, dtype=torch.float32))
    sh_azimuths_rad = torch.remainder(world_azimuths_rad - look_rad, 2 * torch.pi)
    sh_elevations_rad = torch.zeros(2, dtype=torch.float32)

    iko_azimuth_deg = _calculate_source_angle_2D(listener_positions, iko_position)
    reflector_azimuth_deg = _calculate_source_angle_2D(listener_positions, reflector_position)

    direct_dist = torch.norm(iko_position - listener_positions, dim=-1)
    reflect_dist = torch.norm(reflector_position - iko_position, dim=-1) + torch.norm(reflector_position - listener_positions, dim=-1)
    delay_samples = int(torch.round((reflect_dist - direct_dist) / c * fs).item())

    return {
        "iko_position": iko_position,
        "reflector_position": reflector_position,
        "sh_azimuths_rad": sh_azimuths_rad,
        "sh_elevations_rad": sh_elevations_rad,
        "iko_azimuth_deg": iko_azimuth_deg,
        "reflector_azimuth_deg": reflector_azimuth_deg,
        "delay_samples": delay_samples,
    }


def combine_speaker_impulse_responses(weights, eigenbeam_to_loudspeaker):
    """Combine weighted eigenbeam loudspeaker IRs. Returns (n_speakers, taps)."""
    return torch.einsum("l,lst->st", weights, eigenbeam_to_loudspeaker)


def driver_excitation_energy(speaker_ir):
    """Per-driver energy from combined speaker IRs. Returns (n_speakers,)."""
    return speaker_ir.pow(2).sum(dim=-1)


def fingerprint_driver_energies(sph_to_driver, channel_idx):
    """Driver energy profile for a one-hot HOA channel."""
    weights = torch.zeros(sph_to_driver.shape[0], device=sph_to_driver.device, dtype=sph_to_driver.dtype)
    weights[channel_idx] = 1.0
    speaker_ir = combine_speaker_impulse_responses(weights, sph_to_driver)
    return driver_excitation_energy(speaker_ir)


def render_binaural_brir(signal, weights, eigenbeam_to_loudspeaker, brirs, mode="full"):
    """Render binaural signal through weighted eigenbeams, loudspeaker IRs, and BRIRs.

    signal: (1, T) mono stimulus
    weights: (L,) HOA/eigenbeam weights
    eigenbeam_to_loudspeaker: (L, n_speakers, taps)
    brirs: (n_speakers, 2, T_brir)
    Returns (1, 2, T_out), speaker_ir (n_speakers, taps).
    """
    mono = signal.view(1, 1, -1)
    speaker_ir = combine_speaker_impulse_responses(weights, eigenbeam_to_loudspeaker)
    n_speakers = brirs.shape[0]
    loudspeaker_signals = TF.convolve(mono, speaker_ir.unsqueeze(0), mode=mode)
    ear_signals = TF.convolve(
        loudspeaker_signals.repeat_interleave(2, dim=1),
        brirs.reshape(1, 2 * n_speakers, -1),
        mode=mode,
    )
    binaural = ear_signals.view(1, n_speakers, 2, -1).sum(dim=1)
    return binaural, speaker_ir


def pink_noise(n, random_state=None):
    """Generate pink noise (1/f) of length n."""
    rng = np.random.default_rng(random_state)
    num_bins = n // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, num_bins)
    magnitudes = rng.standard_normal(num_bins)
    spectrum = magnitudes * np.exp(1j * phases)
    frequencies = np.arange(num_bins, dtype=float)
    frequencies[0] = 1.0
    spectrum /= np.sqrt(frequencies)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=n)
    signal -= signal.mean()
    signal /= signal.std()
    return signal


def make_pink_noise_burst(fs, burst_s, silence_s, seed):
    """Pink-noise burst with leading and trailing silence."""
    burst_len = int(round(burst_s * fs))
    silence_len = int(round(silence_s * fs))
    burst = pink_noise(burst_len, random_state=seed)
    return np.concatenate([
        np.zeros(silence_len, dtype=np.float64),
        burst,
        np.zeros(silence_len, dtype=np.float64),
    ])


def mono_to_hoa_channels(mono, weights):
    """Scale a mono signal by HOA weights. Returns (T, L)."""
    mono = np.asarray(mono, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return mono[:, np.newaxis] * w[np.newaxis, :]


def normalize_wav_audio(audio, target_peak=0.95):
    """Peak-normalise multi-channel audio for WAV export."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def save_hoa_pink_noise_wav(
    output_dir,
    weights,
    fs,
    order=3,
    stage="final",
    duration_s=2.0,
    burst_s=0.25,
    silence_s=0.25,
    seed=0,
    target_peak=0.95,
):
    """Save continuous pink noise and pink-noise bursts as multi-channel HOA WAV files."""
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_channels = num_hoa_channels(order)
    if weights.shape[0] != n_channels:
        raise ValueError(f"Expected {n_channels} HOA weights, got {weights.shape[0]}.")

    wav_dir = Path(output_dir) / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    mono = pink_noise(int(round(duration_s * fs)), random_state=seed)
    hoa_audio = normalize_wav_audio(mono_to_hoa_channels(mono, weights), target_peak=target_peak)
    continuous_path = wav_dir / f"{stage}_hoa_pink_noise.wav"
    wavfile.write(continuous_path, fs, hoa_audio)

    mono_burst = make_pink_noise_burst(fs, burst_s, silence_s, seed)
    hoa_burst = normalize_wav_audio(mono_to_hoa_channels(mono_burst, weights), target_peak=target_peak)
    burst_path = wav_dir / f"{stage}_hoa_pink_noise_burst.wav"
    wavfile.write(burst_path, fs, hoa_burst)

    print(
        f"Saved {stage} HOA pink-noise WAVs ({hoa_audio.shape[1]} ch): "
        f"{continuous_path.name}, {burst_path.name}"
    )
    return continuous_path, burst_path


def save_run_artifacts(output_dir, stage, mono_signal, weights, speaker_ir, binaural_signal):
    """Save mono input, HOA weights, combined speaker IRs, and rendered binaural signal."""
    npy_dir = output_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_dir / f"{stage}_mono_signal.npy", mono_signal.detach().cpu().numpy())
    np.save(npy_dir / f"{stage}_hoa_weights.npy", weights.detach().cpu().numpy())
    np.save(npy_dir / f"{stage}_speaker_ir.npy", speaker_ir.detach().cpu().numpy())
    np.save(npy_dir / f"{stage}_binaural_signal.npy", binaural_signal.detach().cpu().numpy())


def log_signal_summary(stage, weights, speaker_ir):
    """Print HOA weights and combined loudspeaker IR summary."""
    print(f"{stage} HOA weights: {weights.detach().cpu().numpy()}")
    print(
        f"{stage} speaker IR shape: {tuple(speaker_ir.shape)}, "
        f"speaker IR energy: {speaker_ir.pow(2).sum().item():.4f}"
    )


def evaluate_localisation(llado_model, binaural_signal, target_angle):
    """Forward pass through localisation model; return azimuth and MSE loss."""
    estimated_azimuth = llado_model(binaural_signal, None)["output"]
    loss = torch.mean((estimated_azimuth - target_angle) ** 2)
    return estimated_azimuth, loss


def build_pattern_sampling(order, listener_az_rad, reflector_az_rad, half_width_deg, num_points, device, dtype):
    """Precompute full-circle SH matrix and separate listener/reflector lobe masks.

    Azimuths are in the IKO/physical frame; SH rows are evaluated at math angles
    (physical + offset) so lobe energy matches hoa_to_driver radiation.
    Returns (sh_matrix_full, listener_mask, reflector_mask).
    """
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    elevations = np.zeros(num_points)
    sh_matrix_full = torch.from_numpy(
        compute_real_sh_matrix(order, physical_azimuths_to_math_sh(azimuths), elevations)
    ).to(device=device, dtype=dtype)
    azimuth_samples = torch.from_numpy(azimuths).to(device=device, dtype=dtype)
    listener_az = torch.tensor(float(listener_az_rad), device=device, dtype=dtype)
    reflector_az = torch.tensor(float(reflector_az_rad), device=device, dtype=dtype)
    half_width = torch.deg2rad(torch.tensor(half_width_deg, device=device, dtype=dtype))
    dist_listener = torch.atan2(
        torch.sin(azimuth_samples - listener_az), torch.cos(azimuth_samples - listener_az),
    ).abs()
    dist_reflector = torch.atan2(
        torch.sin(azimuth_samples - reflector_az), torch.cos(azimuth_samples - reflector_az),
    ).abs()
    listener_mask = dist_listener <= half_width
    reflector_mask = dist_reflector <= half_width
    return sh_matrix_full, listener_mask, reflector_mask


def compute_directivity_loss(weights, sh_matrix_full, listener_mask, reflector_mask, sh_matrix):
    """Separate listener/reflector directivity losses with a shared sidelobe region.

    Sidelobes are angles outside both windows, so each direction's in-lobe energy
    ignores the other window and neither loss treats the other direction as a sidelobe.
    Matrices must be built in the physical IKO frame (see physical_azimuths_to_math_sh).
    Returns sidelobe_energy, listener_directivity, reflector_directivity, gains,
    listener_lobe_energy, reflector_lobe_energy.
    """
    pattern = sh_matrix_full @ weights
    listener_lobe_energy = pattern[listener_mask].pow(2).mean()
    reflector_lobe_energy = pattern[reflector_mask].pow(2).mean()
    sidelobe_energy = pattern[~(listener_mask | reflector_mask)].pow(2).mean()
    listener_directivity_loss = sidelobe_energy / (listener_lobe_energy + 1e-6)
    reflector_directivity_loss = sidelobe_energy / (reflector_lobe_energy + 1e-6)
    directional_gains = sh_matrix @ weights
    return (
        sidelobe_energy,
        listener_directivity_loss,
        reflector_directivity_loss,
        directional_gains,
        listener_lobe_energy,
        reflector_lobe_energy,
    )


def compute_pattern_sparsity_loss(weights, sh_matrix_full, listener_mask, reflector_mask):
    """Scale-invariant angular sparsity ||p||_1 / ||p||_2 plus target-lobe energy loss.

    Sparsity encourages few active azimuths. Target energy anchors energy at the
    listener and reflector windows so sparsity does not collapse the pattern to zero
    or a single unrelated peak.
    Returns sparsity_loss, target_energy_loss.
    """
    pattern = sh_matrix_full @ weights
    pattern_l2 = pattern.pow(2).sum().sqrt().clamp_min(1e-6)
    sparsity_loss = pattern.abs().sum() / pattern_l2
    listener_lobe_energy = pattern[listener_mask].pow(2).mean()
    reflector_lobe_energy = pattern[reflector_mask].pow(2).mean()
    target_energy_loss = 1.0 / (listener_lobe_energy + 1e-6) + 1.0 / (reflector_lobe_energy + 1e-6)
    return sparsity_loss, target_energy_loss


def build_vertical_sampling(order, elevation_deg, num_azimuths, device, dtype):
    """Precompute SH matrix at non-zero elevations over full azimuth (physical IKO frame)."""
    elevation_deg = np.asarray(elevation_deg, dtype=float).reshape(-1)
    elevations_rad = np.deg2rad(elevation_deg)
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_azimuths, endpoint=False)
    azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations_rad, indexing="ij")
    sh_matrix_vertical = torch.from_numpy(
        compute_real_sh_matrix(
            order,
            physical_azimuths_to_math_sh(azimuth_grid.ravel()),
            elevation_grid.ravel(),
        )
    ).to(device=device, dtype=dtype)
    return sh_matrix_vertical


def expand_vertical_elevations_deg(elevation_deg):
    """Mirror positive elevation angles to include symmetric below-horizon samples."""
    elevation_deg = np.asarray(elevation_deg, dtype=float).reshape(-1)
    elevations = []
    for elevation in elevation_deg:
        if elevation > 0:
            elevations.extend([elevation, -elevation])
        elif elevation < 0:
            elevations.append(elevation)
        else:
            elevations.append(0.0)
    return np.array(sorted(set(elevations)), dtype=float)


def compute_vertical_radiation_loss(weights, sh_matrix_vertical):
    """Penalise mean squared beam-pattern energy at non-zero elevations."""
    pattern = sh_matrix_vertical @ weights
    return pattern.pow(2).mean()


def beam_pattern_over_circle(order, weights, num_points=360):
    """Full horizontal beam pattern in the physical IKO frame (az 0 = look / front)."""
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    elevations = np.zeros(num_points)
    Y = compute_real_sh_matrix(order, physical_azimuths_to_math_sh(azimuths), elevations)
    w = weights.detach().cpu().numpy().reshape(-1)
    return azimuths, Y @ w


def beam_pattern_over_sphere(order, weights, num_azimuths=180, num_elevations=91):
    """Full 3D beam pattern in the physical IKO frame (az 0 = look / front)."""
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_azimuths, endpoint=False)
    elevations = np.linspace(-np.pi / 2.0, np.pi / 2.0, num_elevations)
    azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations)
    Y = compute_real_sh_matrix(
        order,
        physical_azimuths_to_math_sh(azimuth_grid.ravel()),
        elevation_grid.ravel(),
    )
    w = weights.detach().cpu().numpy().reshape(-1)
    return azimuth_grid, elevation_grid, (Y @ w).reshape(azimuth_grid.shape)


def sh_weights_to_nm(order, weights):
    """Map flat HOA weight vector to (n, m, value) using the same ordering as compute_real_sh_matrix."""
    w = weights.detach().cpu().numpy().reshape(-1)
    nm_values = []
    idx = 0
    for n in range(order + 1):
        for m in range(-n, n + 1):
            nm_values.append((n, m, w[idx]))
            idx += 1
    return nm_values


def extract_interaural_cues(llado_model, binaural_signal):
    """Extract ITD/ILD per critical band from a binaural signal."""
    with torch.no_grad():
        cues = llado_model(binaural_signal, None)["kwargs"]["interauralcues"]
    return cues["itd"], cues["ild"]


def plot_itd_ild_scatter_comparison(plots_dir, initial_binaural, final_binaural, llado_model, sofaile, gtfb, target_angle):
    """Save side-by-side initial and final ITD/ILD scatter plots against the target template."""
    target_itd, target_ild = _compute_target_itd_ild(
        sofaile, target_angle, gtfb, llado_model.lag_values, llado_model.tau_iacc,
    )
    stage_signals = [("Initial", initial_binaural), ("Optimised", final_binaural)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    estimated_scatter = None
    for ax, (label, binaural_signal) in zip(axes, stage_signals):
        estimated_itd, estimated_ild = extract_interaural_cues(llado_model, binaural_signal)
        estimated_scatter = _scatter_itd_ild_from_arrays(
            ax,
            np.ravel(estimated_itd.detach().cpu().numpy()),
            np.ravel(estimated_ild.detach().cpu().numpy()),
            target_itd,
            target_ild,
            f"{label} ITD-ILD ({target_angle}°)",
        )
    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    _add_critical_band_colorbar(fig, estimated_scatter, axes)
    fig.savefig(plots_dir / "itd_ild_scatter.png", dpi=150)
    plt.close(fig)


def plot_weight_decomposition(order, initial_weights, final_weights, plots_dir):
    """Plot HOA weights on an (m, n) grid with the weight value at each valid (n, m) cell."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    m_ticks = list(range(-order, order + 1))
    n_ticks = list(range(order + 1))
    all_weights = np.concatenate([
        initial_weights.detach().cpu().numpy().reshape(-1),
        final_weights.detach().cpu().numpy().reshape(-1),
    ])
    vmax = max(float(np.max(np.abs(all_weights))), 1e-6)

    im = None
    for ax, weights, label in zip(axes, [initial_weights, final_weights], ["Initial", "Optimised"]):
        grid = np.full((order + 1, 2 * order + 1), np.nan)
        for n, m, w in sh_weights_to_nm(order, weights):
            grid[n, m + order] = w

        im = ax.imshow(
            grid, origin="lower", aspect="equal", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            extent=[-order - 0.5, order + 0.5, -0.5, order + 0.5],
        )
        for n, m, w in sh_weights_to_nm(order, weights):
            ax.text(m, n, f"{w:.3f}", ha="center", va="center", fontsize=8, color="black")

        ax.set_xticks(m_ticks)
        ax.set_yticks(n_ticks)
        ax.set_xlabel("m")
        ax.set_ylabel("n (order)")
        ax.set_title(f"{label} HOA weights")
        ax.grid(True, color="white", linewidth=0.5, alpha=0.5)

    fig.subplots_adjust(left=0.06, right=0.86, wspace=0.35)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Weight")
    fig.savefig(plots_dir / "hoa_weights.png", dpi=150)
    plt.close(fig)


def plot_beam_pattern_overlay(order, weights, listener_az, reflector_az, plots_dir, stage="final", num_points=360):
    """Save polar dB subplots (one per harmonic order) with weighted SH patterns overlaid."""
    azimuths = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    plot_azimuths = _beam_pattern_plot_azimuths(azimuths)
    elevations = np.zeros(num_points)
    sh_matrix = compute_real_sh_matrix(order, physical_azimuths_to_math_sh(azimuths), elevations)

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "polar"}, figsize=(12, 12))
    axes = axes.ravel()
    for ax in axes[order + 1:]:
        ax.set_visible(False)
    plot_listener_az = float(_beam_pattern_plot_azimuths(listener_az))
    plot_reflector_az = float(_beam_pattern_plot_azimuths(reflector_az))

    for n, ax in enumerate(axes):
        order_sum = np.zeros(num_points, dtype=np.float64)
        weighted_patterns = []
        for idx, (harmonic_n, m, w) in enumerate(sh_weights_to_nm(order, weights)):
            if harmonic_n != n:
                continue
            weighted_pattern = sh_matrix[:, idx] * w
            order_sum += weighted_pattern
            weighted_patterns.append((m, np.abs(weighted_pattern)))
        reference = max(
            [float(np.max(pattern)) for _, pattern in weighted_patterns]
            + [float(np.max(np.abs(order_sum))), 1e-12]
        )
        for m, magnitude in weighted_patterns:
            ax.plot(plot_azimuths, _beam_pattern_db(magnitude, reference) - _BEAM_PATTERN_DB_FLOOR, label=f"m={m:+d}")

        ax.plot(
            plot_azimuths,
            _beam_pattern_db(np.abs(order_sum), reference) - _BEAM_PATTERN_DB_FLOOR,
            color="black", linewidth=2.0, label="Σ order",
        )
        ax.plot([plot_listener_az, plot_listener_az], [0, -_BEAM_PATTERN_DB_FLOOR], color="blue", linestyle="--", linewidth=1.5, label="Listener")
        ax.plot([plot_reflector_az, plot_reflector_az], [0, -_BEAM_PATTERN_DB_FLOOR], color="gray", linestyle="--", linewidth=1.5, label="Reflector")
        _format_db_polar_axis(ax)
        ax.set_title(f"Order n={n} ({stage})")
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"Weighted SH patterns by order in dB ({stage})", y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / f"beam_pattern_overlay_{stage}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beam_pattern_3d(order, initial_weights, final_weights, listener_az, reflector_az, plots_dir):
    """Save initial and optimised 3D beam-pattern surfaces from several viewpoints."""
    patterns = []
    for weights in [initial_weights, final_weights]:
        azimuths, elevations, pattern = beam_pattern_over_sphere(order, weights)
        patterns.append((azimuths, elevations, np.abs(pattern)))

    reference = max(max(float(np.max(pattern)) for _, _, pattern in patterns), 1e-12)
    patterns = [
        (azimuths, elevations, _beam_pattern_db(magnitude, reference))
        for azimuths, elevations, magnitude in patterns
    ]
    radial_limit = -_BEAM_PATTERN_DB_FLOOR
    norm = colors.Normalize(vmin=_BEAM_PATTERN_DB_FLOOR, vmax=0.0)
    view_azimuths = [-60, 60, 180]
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    axes = [
        [fig.add_subplot(2, len(view_azimuths), row * len(view_azimuths) + column + 1, projection="3d")
         for column in range(len(view_azimuths))]
        for row in range(2)
    ]
    plot_listener_az = float(_beam_pattern_plot_azimuths(listener_az))
    plot_reflector_az = float(_beam_pattern_plot_azimuths(reflector_az))

    for row, ((azimuths, elevations, pattern_db), label) in enumerate(zip(patterns, ["Initial", "Optimised"])):
        plot_azimuths = _beam_pattern_plot_azimuths(azimuths)
        radius = pattern_db - _BEAM_PATTERN_DB_FLOOR
        x = radius * np.cos(elevations) * np.cos(plot_azimuths)
        y = radius * np.cos(elevations) * np.sin(plot_azimuths)
        z = radius * np.sin(elevations)
        for ax, view_azimuth in zip(axes[row], view_azimuths):
            ax.plot_surface(
                x, y, z, facecolors=cm.viridis(norm(pattern_db)),
                linewidth=0, antialiased=True, shade=False,
            )
            ax.plot(
                [0, radial_limit * np.cos(plot_listener_az)],
                [0, radial_limit * np.sin(plot_listener_az)],
                [0, 0], color="blue", linestyle="--", label="Listener",
            )
            ax.plot(
                [0, radial_limit * np.cos(plot_reflector_az)],
                [0, radial_limit * np.sin(plot_reflector_az)],
                [0, 0], color="gray", linestyle="--", label="Reflector",
            )
            ax.set(xlim=(-radial_limit, radial_limit), ylim=(-radial_limit, radial_limit), zlim=(-radial_limit, radial_limit))
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("x (azimuth 0°)")
            ax.set_ylabel("y (azimuth +90°)")
            ax.set_zlabel("z")
            ax.set_title(f"{label} dB — view azimuth {view_azimuth}°")
            ax.view_init(elev=25, azim=view_azimuth)
            ax.legend(loc="upper right", fontsize=8)

    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=[ax for row_axes in axes for ax in row_axes], shrink=0.7, pad=0.04, label="Level relative to shared peak (dB)",
    )
    fig.savefig(plots_dir / "beam_pattern_3d.png", dpi=150)
    plt.close(fig)


def _cpu_grad_dict(raw_grads):
    """Detach per-task gradients to CPU for epoch history storage."""
    return {
        name: (grad.detach().cpu() if grad is not None else None)
        for name, grad in raw_grads.items()
    }


def plot_task_grad_diagnostics(plots_dir, raw_grads_history, effective_grad_history=None):
    """Save per-task gradient norm and cosine-similarity diagnostic plots."""
    if not raw_grads_history:
        return
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    task_names = list(raw_grads_history[0].keys())

    raw_grad_norms = [
        {key: torch.norm(epoch_grads[key]) for key in epoch_grads}
        for epoch_grads in raw_grads_history
    ]
    loss_grad_norms_over_epochs_plot(raw_grad_norms, "Per-task raw gradient norm", norm_kind="Raw")
    plt.savefig(plots_dir / "raw_grad_norms.png", dpi=150)
    plt.close()

    summed_raw_grad = [
        torch.stack([epoch_grads[key] for key in epoch_grads]).sum(dim=0)
        for epoch_grads in raw_grads_history
    ]
    raw_grad_cosine_to_reference_plot(
        raw_grads_history, summed_raw_grad, task_names,
        "Task gradient vs plain (summed) direction",
    )
    plt.savefig(plots_dir / "raw_vs_plain_cosine_similarity.png", dpi=150)
    plt.close()

    if effective_grad_history is not None:
        raw_grad_cosine_to_reference_plot(
            raw_grads_history, effective_grad_history, task_names,
            "Task gradient vs CAGrad update direction",
        )
        plt.savefig(plots_dir / "raw_vs_cagrad_cosine_similarity.png", dpi=150)
        plt.close()

    task_grad_pairwise_cosine_similarity_plot(
        **task_grad_pairwise_cosine_similarity_stats(raw_grads_history),
        title="Task Gradients on HOA Weights",
        pairwise_only=True,
    )
    plt.savefig(plots_dir / "task_grad_pairwise_cosine_similarity.png", dpi=150)
    plt.close()


def plot_results(output_dir, order, initial_weights, final_weights, geometry, initial_azimuth, final_azimuth, target_angle, initial_binaural, final_binaural, llado_model, sofaile, gtfb):
    """Save polar beam-pattern and scene localisation plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    listener_az = float(geometry["sh_azimuths_rad"][0])
    reflector_az = float(geometry["sh_azimuths_rad"][1])

    plot_beam_pattern_overlay(order, final_weights, listener_az, reflector_az, plots_dir, stage="final")
    plot_beam_pattern_3d(order, initial_weights, final_weights, listener_az, reflector_az, plots_dir)

    beam_patterns = [beam_pattern_over_circle(order, weights) for weights in [initial_weights, final_weights]]
    reference = max(max(float(np.max(np.abs(pattern))) for _, pattern in beam_patterns), 1e-12)
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 5))
    plot_listener_az = float(_beam_pattern_plot_azimuths(listener_az))
    plot_reflector_az = float(_beam_pattern_plot_azimuths(reflector_az))
    for ax, (azimuths, pattern), label in zip(axes, beam_patterns, ["Initial", "Optimised"]):
        plot_azimuths = _beam_pattern_plot_azimuths(azimuths)
        ax.plot(plot_azimuths, _beam_pattern_db(np.abs(pattern), reference) - _BEAM_PATTERN_DB_FLOOR, label="Pattern")
        ax.plot([plot_listener_az, plot_listener_az], [0, -_BEAM_PATTERN_DB_FLOOR], color="blue", linestyle="--", linewidth=1.5, label="Listener")
        ax.plot([plot_reflector_az, plot_reflector_az], [0, -_BEAM_PATTERN_DB_FLOOR], color="gray", linestyle="--", linewidth=1.5, label="Reflector")
        _format_db_polar_axis(ax)
        ax.set_title(f"{label} beam pattern (dB)")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "beam_pattern.png", dpi=150)
    plt.close(fig)

    listener_pos = geometry["listener_positions"].numpy()
    iko_pos = geometry["iko_position"].numpy()
    reflector_pos = geometry["reflector_position"].numpy()
    arrow_len = 1.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, azimuth, label in zip(axes, [initial_azimuth, final_azimuth], ["Initial", "Optimised"]):
        ax.scatter(listener_pos[:, 0], listener_pos[:, 1], c="blue", marker="o", label="Listener", zorder=3)
        ax.scatter(iko_pos[:, 0], iko_pos[:, 1], c="black", marker="x", label="IKO", zorder=3)
        ax.scatter(reflector_pos[:, 0], reflector_pos[:, 1], c="gray", marker="s", label="Reflector", zorder=3)
        az_rad = np.deg2rad(float(azimuth))
        ax.arrow(
            listener_pos[0, 0], listener_pos[0, 1],
            arrow_len * np.cos(az_rad), arrow_len * np.sin(az_rad),
            head_width=0.15, head_length=0.1, fc="red", ec="red", label="Estimated",
        )
        target_rad = np.deg2rad(target_angle)
        ax.arrow(
            listener_pos[0, 0], listener_pos[0, 1],
            arrow_len * np.cos(target_rad), arrow_len * np.sin(target_rad),
            head_width=0.15, head_length=0.1, fc="green", ec="green", linestyle="--", label="Target",
        )
        ax.set_aspect("equal")
        ax.set_title(f"{label} localisation ({float(azimuth):.1f}°)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "localisation_scene.png", dpi=150)
    plt.close(fig)

    plot_weight_decomposition(order, initial_weights, final_weights, plots_dir)
    plot_itd_ild_scatter_comparison(plots_dir, initial_binaural, final_binaural, llado_model, sofaile, gtfb, target_angle)


def main():
    with open("config/hyperpameters.json", "r") as f:
        config = json.load(f)
    set_seed(config["env"]["seed"])

    fs = config["env"]["fs"]
    device = config["env"]["device"]
    dtype = torch.float32
    n_epochs = config["training"]["num_epochs"]
    lr = 0.01
    target_angle = config["training"]["target_angle"]
    iko_config = config["iko"]
    lambda_loc = iko_config["lambda_loc"]
    lambda_l1 = iko_config["lambda_l1"]
    lambda_sidelobe = iko_config["lambda_sidelobe"]
    lambda_directivity = iko_config["lambda_directivity"]
    lambda_sparsity = iko_config["lambda_sparsity"]
    lambda_target_energy = iko_config["lambda_target_energy"]
    lobe_half_width_deg = iko_config["lobe_half_width_deg"]
    pattern_num_points = iko_config["pattern_num_points"]
    lambda_vertical = iko_config.get("lambda_vertical", 0.0)
    vertical_elevation_deg = iko_config.get("vertical_elevation_deg", [30.0, 60.0, 90.0])
    vertical_num_azimuths = iko_config.get("vertical_num_azimuths", pattern_num_points)
    use_cagrad = iko_config.get("use_cagrad", False)
    use_2d_harmonics = iko_config.get("use_2d_harmonics", False)
    cagrad_c = float(config["training"].get("cagrad_c", 0.5))
    lambdas = {
        "loc": lambda_loc,
        "l1": lambda_l1,
        "sidelobe": lambda_sidelobe,
        "directivity": lambda_directivity,
        "sparsity": lambda_sparsity,
        "target_energy": lambda_target_energy,
        "vertical": lambda_vertical,
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path("outputs") / "iko_brir" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    sofaile = resample_sofa_ir(load_sofa("data/SOFA/mit_kemar_large_pinna.sofa"), fs)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)

    d = 2.0
    iko_order = 3
    n_eigenbeams = num_hoa_channels(iko_order)
    active_harmonic_indices = get_active_harmonic_indices(iko_order, use_2d_harmonics, device)
    listener_positions = torch.tensor([[0.0, 0.0]], dtype=dtype)

    brirs = load_iko_brirs(BRIR_NPY, N_IKO_LOUDSPEAKERS, device, dtype)
    eigenbeam_to_loudspeaker = load_sph_to_driver(
        SPH_TO_DRIVER_NPY, n_eigenbeams, N_IKO_LOUDSPEAKERS, device, dtype,
    )

    geometry = setup_geometry(d, 30.0, -30.0, listener_positions, fs)
    geometry["listener_positions"] = listener_positions

    sh_matrix = torch.from_numpy(
        compute_real_sh_matrix(
            iko_order,
            physical_azimuths_to_math_sh(geometry["sh_azimuths_rad"].numpy()),
            geometry["sh_elevations_rad"].numpy(),
        )
    ).to(device=device, dtype=dtype)
    sh_matrix_full, listener_mask, reflector_mask = build_pattern_sampling(
        iko_order,
        float(geometry["sh_azimuths_rad"][0]),
        float(geometry["sh_azimuths_rad"][1]),
        lobe_half_width_deg,
        pattern_num_points,
        device,
        dtype,
    )
    sh_matrix_vertical = build_vertical_sampling(
        iko_order,
        expand_vertical_elevations_deg(vertical_elevation_deg),
        vertical_num_azimuths,
        device,
        dtype,
    )

    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    llado_model.eval()

    weight_model = nn.Linear(1, len(active_harmonic_indices), bias=False, device=device, dtype=dtype)
    weight_model.weight.data.zero_()
    weight_model.weight.data[0, 0] = 1.0  # omnidirectional (0th-order) SH only
    optimizer = torch.optim.Adam(weight_model.parameters(), lr=lr)
    active_tasks = [name for name, weight in lambdas.items() if weight != 0.0]
    cagrad = CAGrad(active_tasks, c=cagrad_c) if use_cagrad else None
    print(f"Optimiser step: {'CAGrad (c=' + str(cagrad_c) + ', tasks=' + str(active_tasks) + ')' if use_cagrad else 'plain weighted sum'}")
    print(f"HOA components: {'2D sectoral only (|m| = n)' if use_2d_harmonics else 'full 3D'}")
    print(f"Loss weights: {lambdas}")
    with open(output_dir / "optim_config.json", "w") as f:
        json.dump({"use_cagrad": use_cagrad, "use_2d_harmonics": use_2d_harmonics, "cagrad_c": cagrad_c, "loss_weights": lambdas}, f, indent=2)

    signal = torch.zeros(1, 256, device=device, dtype=dtype)
    signal[0, 0] = 1.0  # unit impulse (deterministic stimulus)
    target = torch.tensor(target_angle, device=device, dtype=dtype)

    with torch.no_grad():
        initial_weights = expand_active_harmonic_weights(
            weight_model, active_harmonic_indices, n_eigenbeams, device, dtype,
        )
        initial_binaural, initial_speaker_ir = render_binaural_brir(
            signal, initial_weights, eigenbeam_to_loudspeaker, brirs,
        )
        initial_azimuth, initial_loss = evaluate_localisation(llado_model, initial_binaural, target)
        initial_l1_loss = torch.nn.functional.l1_loss(initial_weights, torch.zeros_like(initial_weights))
        (
            initial_sidelobe_loss,
            initial_listener_directivity_loss,
            initial_reflector_directivity_loss,
            initial_gains,
            initial_listener_lobe_energy,
            initial_reflector_lobe_energy,
        ) = compute_directivity_loss(
            initial_weights, sh_matrix_full, listener_mask, reflector_mask, sh_matrix,
        )
        initial_sparsity_loss, initial_target_energy_loss = compute_pattern_sparsity_loss(
            initial_weights, sh_matrix_full, listener_mask, reflector_mask,
        )
        initial_vertical_loss = compute_vertical_radiation_loss(initial_weights, sh_matrix_vertical)
        save_run_artifacts(output_dir, "initial", signal, initial_weights, initial_speaker_ir, initial_binaural)
        log_signal_summary("Initial", initial_weights, initial_speaker_ir)

    print(
        f"Initial azimuth: {initial_azimuth.item():.2f}°, loc loss: {initial_loss.item():.4f}, "
        f"L1 loss: {initial_l1_loss.item():.4f}, "
        f"sidelobe loss: {initial_sidelobe_loss.item():.4f}, "
        f"listener dir: {initial_listener_directivity_loss.item():.4f}, "
        f"reflector dir: {initial_reflector_directivity_loss.item():.4f}, "
        f"sparsity loss: {initial_sparsity_loss.item():.4f}, "
        f"target energy loss: {initial_target_energy_loss.item():.4f}, "
        f"vertical loss: {initial_vertical_loss.item():.4f}, "
        f"beam gains: listener={initial_gains[0].item():.3f}, reflector={initial_gains[1].item():.3f}"
    )

    raw_grads_history = []
    effective_grad_history = [] if use_cagrad else None
    loss_history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        weights = expand_active_harmonic_weights(
            weight_model, active_harmonic_indices, n_eigenbeams, device, dtype,
        )
        binaural_signal, _ = render_binaural_brir(
            signal, weights, eigenbeam_to_loudspeaker, brirs,
        )
        _, loss = evaluate_localisation(llado_model, binaural_signal, target)

        l1_loss = torch.nn.functional.l1_loss(weights, torch.zeros_like(weights))
        (
            sidelobe_loss,
            listener_directivity_loss,
            reflector_directivity_loss,
            _,
            _,
            _,
        ) = compute_directivity_loss(
            weights, sh_matrix_full, listener_mask, reflector_mask, sh_matrix,
        )
        sparsity_loss, target_energy_loss = compute_pattern_sparsity_loss(
            weights, sh_matrix_full, listener_mask, reflector_mask,
        )
        vertical_loss = compute_vertical_radiation_loss(weights, sh_matrix_vertical)
        losses = {
            "loc": loss,
            "l1": l1_loss,
            "sidelobe": sidelobe_loss,
            "directivity": listener_directivity_loss + reflector_directivity_loss,
            "sparsity": sparsity_loss,
            "target_energy": target_energy_loss,
            "vertical": vertical_loss,
        }
        weighted_losses = build_iko_weighted_loss_dict(lambdas, losses)
        total_loss = iko_total_loss(lambdas, losses)
        loss_history.append({
            **{name: value.detach().cpu() for name, value in losses.items()},
            "total": total_loss.detach().cpu(),
        })
        step_info = apply_iko_optim_step(optimizer, weight_model, weighted_losses, total_loss, cagrad)
        raw_grads_history.append(_cpu_grad_dict(step_info["raw_grads"]))
        if effective_grad_history is not None:
            effective_grad_history.append(step_info["effective_grad"].detach().cpu())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                azimuth, _ = evaluate_localisation(llado_model, binaural_signal, target)
                (
                    _,
                    _,
                    _,
                    gains,
                    listener_lobe_energy,
                    reflector_lobe_energy,
                ) = compute_directivity_loss(
                    weights, sh_matrix_full, listener_mask, reflector_mask, sh_matrix,
                )
            print(
                f"Epoch {epoch + 1}/{n_epochs}, loc loss: {loss.item():.4f}, "
                f"L1 loss: {l1_loss.item():.4f}, "
                f"sidelobe loss: {sidelobe_loss.item():.4f}, "
                f"listener dir: {listener_directivity_loss.item():.4f}, "
                f"reflector dir: {reflector_directivity_loss.item():.4f}, "
                f"sparsity loss: {sparsity_loss.item():.4f}, "
                f"target energy loss: {target_energy_loss.item():.4f}, "
                f"vertical loss: {vertical_loss.item():.4f}, "
                f"total loss: {total_loss.item():.4f}, azimuth: {azimuth.item():.2f}°, "
                f"beam gains: listener={gains[0].item():.3f}, reflector={gains[1].item():.3f}, "
                f"lobe energy: listener={listener_lobe_energy.item():.4f}, "
                f"reflector={reflector_lobe_energy.item():.4f}"
            )

    with torch.no_grad():
        final_weights = expand_active_harmonic_weights(
            weight_model, active_harmonic_indices, n_eigenbeams, device, dtype,
        )
        final_binaural, final_speaker_ir = render_binaural_brir(
            signal, final_weights, eigenbeam_to_loudspeaker, brirs,
        )
        final_azimuth, final_loss = evaluate_localisation(llado_model, final_binaural, target)
        final_l1_loss = torch.nn.functional.l1_loss(final_weights, torch.zeros_like(final_weights))
        (
            final_sidelobe_loss,
            final_listener_directivity_loss,
            final_reflector_directivity_loss,
            final_gains,
            final_listener_lobe_energy,
            final_reflector_lobe_energy,
        ) = compute_directivity_loss(
            final_weights, sh_matrix_full, listener_mask, reflector_mask, sh_matrix,
        )
        final_sparsity_loss, final_target_energy_loss = compute_pattern_sparsity_loss(
            final_weights, sh_matrix_full, listener_mask, reflector_mask,
        )
        final_vertical_loss = compute_vertical_radiation_loss(final_weights, sh_matrix_vertical)
        save_run_artifacts(output_dir, "final", signal, final_weights, final_speaker_ir, final_binaural)
        log_signal_summary("Final", final_weights, final_speaker_ir)

    print(
        f"Final azimuth: {final_azimuth.item():.2f}°, loc loss: {final_loss.item():.4f}, "
        f"L1 loss: {final_l1_loss.item():.4f}, "
        f"sidelobe loss: {final_sidelobe_loss.item():.4f}, "
        f"listener dir: {final_listener_directivity_loss.item():.4f}, "
        f"reflector dir: {final_reflector_directivity_loss.item():.4f}, "
        f"sparsity loss: {final_sparsity_loss.item():.4f}, "
        f"target energy loss: {final_target_energy_loss.item():.4f}, "
        f"vertical loss: {final_vertical_loss.item():.4f} "
        f"(target: {target_angle}°), beam gains: listener={final_gains[0].item():.3f}, "
        f"reflector={final_gains[1].item():.3f}, "
        f"lobe energy: listener={final_listener_lobe_energy.item():.4f}, "
        f"reflector={final_reflector_lobe_energy.item():.4f}"
    )

    plot_results(
        output_dir, iko_order, initial_weights, final_weights, geometry,
        initial_azimuth.item(), final_azimuth.item(), target_angle,
        initial_binaural, final_binaural, llado_model, sofaile, gtfb,
    )
    loss_over_epochs_plot(loss_history, "IKO BRIR losses over epochs")
    plt.savefig(output_dir / "plots" / "loss_over_epochs.png", dpi=150)
    plt.close()
    plot_task_grad_diagnostics(output_dir / "plots", raw_grads_history, effective_grad_history)
    save_hoa_pink_noise_wav(
        output_dir, final_weights, fs, order=iko_order, stage="final", seed=config["env"].get("seed", 0),
    )
    print(f"Plots saved to {output_dir / 'plots'}")
    print(f"Signals and weights saved to {output_dir / 'npy'}")
    print(f"HOA pink-noise WAVs saved to {output_dir / 'wav'}")


if __name__ == "__main__":
    main()
