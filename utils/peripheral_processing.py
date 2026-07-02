import torch
import torch.nn.functional as F
import torchaudio.functional as TF

from utils.dsp import hilbert_transform, soft_argmax, compute_energy


def rectify_signals(gtfb_signals, split_band = 11):
    LF_rectified_signals = torch.max(gtfb_signals[:,:split_band,:,:], torch.zeros_like(gtfb_signals[:,:split_band,:,:]))
    HF_rectified_signals = torch.abs(hilbert_transform(gtfb_signals[:,split_band:,:,:]))
    rectified_signals = torch.cat([LF_rectified_signals, HF_rectified_signals], dim=1)
    return rectified_signals

def compute_iacc(rectified_signals, lag_values):
    pad_length =lag_values.shape[0]//2

    left_signals = F.pad(rectified_signals[:,:,0,:], (pad_length, pad_length), mode='constant', value=0)
    right_signals = torch.flip(rectified_signals[:,:,1,:], dims=[-1])

    left_energy = compute_energy(left_signals)
    right_energy = compute_energy(right_signals)

    norm = torch.sqrt(left_energy*right_energy).clamp(min=1e-12).unsqueeze(-1) 

    iacc = TF.fftconvolve(left_signals, right_signals, mode='valid')/norm 
    return iacc

def compute_itd(iacc, tau_iacc, lag_values):

    # Earlier left signals results in positive ITD, later right signals results in negative ITD
    itd = -soft_argmax(iacc, lag_values, dim=-1, tau=tau_iacc) 

    return itd

def compute_ild(rectified_signals):
    left_signals = rectified_signals[:,:,0,:]
    right_signals = torch.flip(rectified_signals[:,:,1,:], dims=[-1])
    
    # Louder left signals results in positive ILD, louder right signals results in negative ILD
    ild = (
        20 * torch.log10(torch.sqrt(torch.sum(left_signals**2, dim=-1) + 1e-12))
        - 20 * torch.log10(torch.sqrt(torch.sum(right_signals**2, dim=-1) + 1e-12))
    )
    return ild

def estimate_interaural_cues(binaural_signals, gtfb, lag_values, tau_iacc = 1e-2):

    N_signals, N_ears, N_samples = binaural_signals.shape
    N_banks = gtfb.shape[0]

    # Reshape the binaural signals to (N_signals, N_listeners, 1, N_ears, N_samples)
    binaural_signals = binaural_signals.view(N_signals, 1, N_ears, N_samples)
    gtfb = gtfb.view(1, N_banks, 1, -1)

    # Convolve the binaural signals with the gtfb || gtfb_signals.shape = (N_signals, N_listeners, N_banks, N_ears, N_samples)
    gtfb_signals = TF.fftconvolve(binaural_signals, gtfb)

    # Rectify the gtfb signals
    rectified_signals = rectify_signals(gtfb_signals)
    iacc = compute_iacc(rectified_signals, lag_values)

    # Estimate the interaural cues
    itd = compute_itd(iacc, tau_iacc, lag_values)
    ild = compute_ild(rectified_signals)

    return {"itd": itd, 
            "ild": ild,
            "gtfb_signals": gtfb_signals,
            "rectified_signals": rectified_signals,
            "iacc": iacc}
