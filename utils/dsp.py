import torch
import torch.nn.functional as F

def compute_energy(x):
    return torch.sum(x**2, dim=-1)

def softmax_st(z, dim=-1, tau=1.0):
    tau_t = torch.as_tensor(tau, device=z.device, dtype=z.dtype)
    if tau_t.ndim == 0:
        z_scaled = z / tau_t
    else:
        view_shape = [1] * z.ndim
        view_shape[-2] = tau_t.numel()
        z_scaled = z / tau_t.view(*view_shape)
    y_soft = F.softmax(z_scaled, dim=dim)
    index = z.max(dim=dim, keepdim=True)[1]
    y_hard = torch.zeros_like(z).scatter_(dim, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft

def soft_argmax(z, indices,dim=-1, tau=1.0):
    
    y = softmax_st(z, dim=dim, tau=tau)

    return torch.sum(y * indices, dim=dim)

def hilbert_transform(x):
    """
    Differentiable Hilbert transform for real input of shape (B, C, N).
    Returns the complex analytic signal of shape (B, C, N).
    """
    N = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)

    # Build the Hilbert filter mask
    h = torch.zeros(N, dtype=x.dtype, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    H = h.view(1, 1, N)  # Shape: [1, 1, N] to broadcast over B and C
    Xf_filtered = Xf * H

    x_analytic = torch.fft.ifft(Xf_filtered, dim=-1)
    return x_analytic  # Complex tensor: real + j*hilbert(real)

def rms(x, dim=-1):
    return torch.sqrt(torch.mean(x**2, dim=dim))

def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def magnitude_squared_coherence(x, y, nperseg=None):
    """
    Magnitude-squared coherence via Welch averaging over segments.

    Cxy(f) = |Pxy(f)|^2 / (Pxx(f) * Pyy(f))

    Single-segment MSC is identically 1; averaging across overlapped
    windowed segments is required for a meaningful estimate.

    Args:
        x: Signal 1, shape (..., N)
        y: Signal 2, shape (..., N)
        nperseg: Segment length (default: min(256, N // 4), at least 8)

    Returns:
        coherence: Coherence spectrum, shape (..., nperseg//2+1)
        mean_coherence: Mean over frequency, shape (...)
    """
    N = x.shape[-1]
    if nperseg is None:
        nperseg = min(256, max(8, N // 4))
    nperseg = min(nperseg, N)
    noverlap = nperseg // 2
    step = nperseg - noverlap

    window = torch.hann_window(nperseg, device=x.device, dtype=x.dtype)
    # Number of segments that fit with the chosen hop
    n_segments = 1 + (N - nperseg) // step

    # Gather segments: (..., n_segments, nperseg)
    starts = torch.arange(n_segments, device=x.device) * step
    idx = starts.unsqueeze(-1) + torch.arange(nperseg, device=x.device)
    x_segs = x[..., idx] * window
    y_segs = y[..., idx] * window

    X = torch.fft.rfft(x_segs, dim=-1)
    Y = torch.fft.rfft(y_segs, dim=-1)

    Pxx = (X * torch.conj(X)).real.mean(dim=-2)
    Pyy = (Y * torch.conj(Y)).real.mean(dim=-2)
    Pxy = (X * torch.conj(Y)).mean(dim=-2)

    coherence = torch.abs(Pxy) ** 2 / (Pxx * Pyy).clamp(min=1e-10)
    mean_coherence = coherence.mean(dim=-1)
    return coherence, mean_coherence


def interaural_cross_correlation(left, right, max_lag_samples=None):
    """
    Compute the Interaural Cross-Correlation Coefficient (IACC).
    
    IACC is the maximum of the normalized cross-correlation between
    left and right ear signals within a specified lag range.
    
    Args:
        left: Left ear signal, shape (..., N)
        right: Right ear signal, shape (..., N)
        max_lag_samples: Maximum lag to search (default: N // 4)
    
    Returns:
        iacc: Maximum cross-correlation value in [-1, 1], shape (...)
        iacc_lag: Lag (in samples) at which maximum occurs, shape (...)
    """
    N = left.shape[-1]
    if max_lag_samples is None:
        max_lag_samples = N // 4

    left_norm = left - left.mean(dim=-1, keepdim=True)
    right_norm = right - right.mean(dim=-1, keepdim=True)

    left_std = left_norm.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-10)
    right_std = right_norm.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-10)

    nfft = 2 * N - 1
    L = torch.fft.rfft(left_norm, n=nfft, dim=-1)
    R = torch.fft.rfft(right_norm, n=nfft, dim=-1)

    cross_corr = torch.fft.irfft(L * torch.conj(R), n=nfft, dim=-1)
    cross_corr = cross_corr / (left_std * right_std)

    # Extract lags within [-max_lag_samples, +max_lag_samples]
    # Cross-correlation output: [0, 1, ..., N-1, -(N-1), ..., -1]
    pos_lags = cross_corr[..., :max_lag_samples + 1]
    neg_lags = cross_corr[..., -(max_lag_samples):]
    relevant_corr = torch.cat([neg_lags, pos_lags], dim=-1)

    iacc, max_idx = relevant_corr.abs().max(dim=-1)
    iacc_lag = max_idx - max_lag_samples

    iacc_signed = torch.gather(relevant_corr, -1, max_idx.unsqueeze(-1)).squeeze(-1)

    return iacc_signed, iacc_lag