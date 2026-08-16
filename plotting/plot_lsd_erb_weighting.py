"""Plot the per-bin ERB weighting used by the LSD colouration model.

Weights follow ``LSD._erb_frequency_bin_weights``:
  ERB(f) = 0.108 f + 24.7,  w(f) ∝ 1/ERB(f),  then L1-normalised over all FFT bins.
Only bins in [f_min, f_max] are plotted (those applied in the forward pass).

Example:
  python plotting/plot_lsd_erb_weighting.py
  python plotting/plot_lsd_erb_weighting.py --fs 44100 --n-fft 2048
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def erb_frequency_bin_weights(fft_frequencies, eps=1e-12):
    """Match ``LSD._erb_frequency_bin_weights`` without constructing the full model."""
    erb = 0.108 * fft_frequencies + 24.7
    inv_erb = 1.0 / np.maximum(erb, eps)
    w = inv_erb / inv_erb.max()
    return erb, w, w / w.sum()


def plot_lsd_erb_weighting(fs=48000, n_fft=4096, f_min=20.0, f_max=20000.0):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    _, relative_w, weights = erb_frequency_bin_weights(freqs)
    used = (freqs >= f_min) & (freqs <= f_max)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    weight_sum_all = float(weights.sum())
    weight_sum_used = float(weights[used].sum())

    fig, (ax_w, ax_rel) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    ax_w.semilogx(freqs[used], weights[used], color="C0", lw=2.0)
    ax_w.set_ylabel("Per-bin weight")
    ax_w.set_title("LSD ERB weighting")
    ax_w.tick_params(axis="both", labelsize=12)
    ax_w.grid(True, which="both", alpha=0.35)
    ax_w.set_xlim(f_min, f_max)
    ax_w.set_ylim(bottom=0.0)
    # ax_w.text(
    #     0.97, 0.95,
    #     rf"$\sum w = {weight_sum_all:.3f}$ (all bins)" "\n"
    #     rf"$\sum w = {weight_sum_used:.3f}$ (applied)",
    #     transform=ax_w.transAxes,
    #     va="top", ha="right", fontsize=12,
    #     bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.92),
    # )

    ax_rel.semilogx(freqs[used], relative_w[used], color="C1", lw=2.0)
    ax_rel.set_xlabel("Frequency [Hz]")
    ax_rel.set_ylabel("Relative weight")
    ax_rel.set_ylim(0.0, 1.05)
    ax_rel.tick_params(axis="both", labelsize=12)
    ax_rel.grid(True, which="both", alpha=0.35)

    fig.tight_layout()
    return fig, (ax_w, ax_rel), {
        "n_bins": int(freqs.size),
        "n_used": int(used.sum()),
        "weight_sum_all": float(weights.sum()),
        "weight_sum_used": float(weights[used].sum()),
        "w_min_used": float(weights[used].min()) if used.any() else float("nan"),
        "w_max_used": float(weights[used].max()) if used.any() else float("nan"),
        "rel_min_used": float(relative_w[used].min()) if used.any() else float("nan"),
        "rel_max_used": float(relative_w[used].max()) if used.any() else float("nan"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LSD ERB frequency-bin weighting")
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=20000.0)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/lsd_erb_weighting/lsd_erb_weighting.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig, _, stats = plot_lsd_erb_weighting(
        fs=args.fs, n_fft=args.n_fft, f_min=args.f_min, f_max=args.f_max,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"FFT bins: {stats['n_bins']}  (used in [{args.f_min:g}, {args.f_max:g}] Hz: {stats['n_used']})")
    print(f"sum(weights) over all bins: {stats['weight_sum_all']:.6f}")
    print(f"sum(weights) in analysis band: {stats['weight_sum_used']:.6f}")
    print(f"relative 1/ERB in band: {stats['rel_min_used']:.3f} .. {stats['rel_max_used']:.3f}")
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
