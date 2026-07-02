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