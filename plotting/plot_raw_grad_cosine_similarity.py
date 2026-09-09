"""Combined task-gradient cosine-similarity figure with a moving average.

Loads ``raw_grads.*`` and ``effective_grad`` from a CAGrad run folder and
plots each task vs the plain (summed) direction and vs the CAGrad update.

Example:
  python plotting/plot_raw_grad_cosine_similarity.py
  python plotting/plot_raw_grad_cosine_similarity.py --window 101
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

from plotting.XY_plots import raw_grad_cosine_to_references_plot

DEFAULT_RUN = Path("outputs/cagrad/2026-07-13_010939")
TASK_ORDER = ("llado2025_grad_surgery", "desena2020", "LSD", "SecondMoment")


def load_run_gradients(run_dir):
    npy_dir = Path(run_dir) / "npy"
    raw_grads = {}
    for name in TASK_ORDER:
        path = npy_dir / f"raw_grads.{name}.npy"
        if path.exists():
            raw_grads[name] = np.load(path)
    if not raw_grads:
        raise FileNotFoundError(f"No raw_grads.*.npy files in {npy_dir}")
    task_names = [name for name in TASK_ORDER if name in raw_grads]
    extra = sorted(p.name[len("raw_grads."):-len(".npy")]
                   for p in npy_dir.glob("raw_grads.*.npy")
                   if p.name[len("raw_grads."):-len(".npy")] not in raw_grads)
    for name in extra:
        raw_grads[name] = np.load(npy_dir / f"raw_grads.{name}.npy")
        task_names.append(name)
    plain = np.sum([raw_grads[name] for name in task_names], axis=0)
    cagrad = np.load(npy_dir / "effective_grad.npy")
    return raw_grads, plain, cagrad, task_names


def parse_args():
    parser = argparse.ArgumentParser(description="Combined MA cosine-similarity plot")
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN))
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--hide-raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    raw_grads, plain, cagrad, task_names = load_run_gradients(run_dir)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, _ = raw_grad_cosine_to_references_plot(
        raw_grads, [plain, cagrad], task_names,
        ["SGD (summed) direction", "CAGrad update direction"],
        window=args.window, show_raw=not args.hide_raw,
    )

    output_path = Path(args.output) if args.output else (
        run_dir / "plots" / "raw_vs_plain_and_cagrad_cosine_similarity.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Saved {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
