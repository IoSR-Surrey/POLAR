"""Plot the band-pass tiling of BandpassWhiteNoiseDataset.

Defaults match the recent linear_erb training setup (12 bands, erb_factor=0.25).

Example:
  python plotting/plot_bandpass_dataset.py
  python plotting/plot_bandpass_dataset.py --n-samples-per-band 4 --output outputs/bandpass_dataset.png
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

from datasets.stimulus_dataset.dataset import BandpassWhiteNoiseDataset
from plotting.XY_plots import bandpass_dataset_plot


def parse_args():
    parser = argparse.ArgumentParser(description="Plot band-pass dataset coverage")
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--n-bands", type=int, default=12)
    parser.add_argument("--n-samples-per-band", type=int, default=4)
    parser.add_argument("--stimulus-length-ms", type=float, default=5.0)
    parser.add_argument("--ramp-time-ms", type=float, default=1.0)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=20000.0)
    parser.add_argument("--band-mode", type=str, default="linear_erb")
    parser.add_argument("--erb-factor", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/bandpass_dataset/bandpass_dataset.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = BandpassWhiteNoiseDataset(
        fs=args.fs,
        n_bands=args.n_bands,
        n_samples_per_band=args.n_samples_per_band,
        stimulus_length_ms=args.stimulus_length_ms,
        ramp_time_ms=args.ramp_time_ms,
        f_min=args.f_min,
        f_max=args.f_max,
        band_mode=args.band_mode,
        erb_factor=args.erb_factor,
    )
    fig, _ = bandpass_dataset_plot(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
