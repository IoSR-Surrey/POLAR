"""2-task illustration of CAGrad vs plain (averaged) gradient descent.

Task 1 has a stronger gradient, so the average update is dominated by it.
CAGrad searches a ball around that average and rotates toward the
worse-aligned task (Task 2).

Uses models.cagrad.CAGrad so the arrows match this repo's solver.

Example:
  python plotting/plot_cagrad_2task_example.py
  python plotting/plot_cagrad_2task_example.py --cagrad-c 0.25
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import torch

from models.cagrad import CAGrad

# Conflicting 2-D task gradients (Task 1 stronger, mild conflict in x).
G1 = np.array([2.5, 0.4])
G2 = np.array([-0.4, 1.2])

TASK1_COLOR = "C0"
TASK2_COLOR = "C1"
PLAIN_COLOR = "0.15"
CAGRAD_COLOR = "C3"


def _arrow(ax, start, vec, color, lw=2.0, mutation_scale=14, linestyle="-"):
    end = np.asarray(start) + np.asarray(vec)
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )
    return end


def _label_vec(ax, origin, vec, text, color, offset=(0.0, 0.0), loc=0.72):
    p = np.asarray(origin) + loc * np.asarray(vec) + np.asarray(offset)
    ax.text(p[0], p[1], text, color=color, fontsize=12, fontweight="bold", zorder=5)


def solve_cagrad(g1, g2, c):
    cagrad = CAGrad(["task_1", "task_2"], c=c)
    t1 = torch.tensor(g1, dtype=torch.float64)
    t2 = torch.tensor(g2, dtype=torch.float64)
    d, dual_weights, coeffs = cagrad._solve_dual([t1, t2])
    g0 = 0.5 * (t1 + t2)
    return {
        "g1": g1,
        "g2": g2,
        "g0": g0.numpy(),
        "d": d.numpy(),
        "w": dual_weights.numpy(),
        "coeffs": coeffs.numpy(),
        "c": c,
    }


def _alignment_stats(sol):
    g1, g2, g0, d = sol["g1"], sol["g2"], sol["g0"], sol["d"]
    return {
        "cos12": float(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))),
        "plain_min": float(min(np.dot(g1, g0), np.dot(g2, g0))),
        "cagrad_min": float(min(np.dot(g1, d), np.dot(g2, d))),
        "angle_deg": float(np.degrees(np.arccos(np.clip(
            np.dot(g0, d) / (np.linalg.norm(g0) * np.linalg.norm(d)), -1.0, 1.0
        )))),
    }


def plot_cagrad_2task_example(sol):
    g1, g2, g0, d = sol["g1"], sol["g2"], sol["g0"], sol["d"]
    stats = _alignment_stats(sol)
    radius = sol["c"] * np.linalg.norm(g0)

    fig, ax_g = plt.subplots(figsize=(6.4, 6.2))

    ax_g.axhline(0.0, color="0.8", lw=0.8, zorder=0)
    ax_g.axvline(0.0, color="0.8", lw=0.8, zorder=0)
    ax_g.add_patch(Circle(
        g0, radius, fill=False, ls="--", lw=1.2, color="0.45", zorder=1,
    ))
    ax_g.plot(*g0, "o", color=PLAIN_COLOR, ms=4.5, zorder=5)
    _arrow(ax_g, (0, 0), g1, TASK1_COLOR, lw=2.0)
    _arrow(ax_g, (0, 0), g2, TASK2_COLOR, lw=2.0)
    _arrow(ax_g, (0, 0), g0, PLAIN_COLOR, lw=2.2)
    _arrow(ax_g, (0, 0), d, CAGRAD_COLOR, lw=2.4)
    _arrow(ax_g, g0, d - g0, CAGRAD_COLOR, lw=1.2, mutation_scale=11, linestyle="--")
    _label_vec(ax_g, (0, 0), g1, r"$\mathbf{g}_1$", TASK1_COLOR, offset=(0.10, -0.20))
    _label_vec(ax_g, (0, 0), g2, r"$\mathbf{g}_2$", TASK2_COLOR, offset=(-0.42, 0.04))
    _label_vec(ax_g, (0, 0), g0, r"$\mathbf{d}_0$", PLAIN_COLOR, offset=(0.12, -0.20), loc=0.55)
    _label_vec(ax_g, (0, 0), d, r"$\mathbf{d}$", CAGRAD_COLOR, offset=(-0.30, 0.08), loc=0.62)

    pad = 0.55
    xs = np.array([0.0, g1[0], g2[0], g0[0], d[0], g0[0] - radius, g0[0] + radius])
    ys = np.array([0.0, g1[1], g2[1], g0[1], d[1], g0[1] - radius, g0[1] + radius])
    x_mid = 0.5 * (xs.min() + xs.max())
    y_mid = 0.5 * (ys.min() + ys.max())
    half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) + pad
    ax_g.set_xlim(x_mid - half, x_mid + half)
    ax_g.set_ylim(y_mid - half, y_mid + half)
    ax_g.set_aspect("equal", adjustable="box")
    ax_g.set_xlabel("Gradient component 1", fontsize=14)
    ax_g.set_ylabel("Gradient component 2", fontsize=14)
    ax_g.set_title("Two-task CAGrad: average gradient vs conflict-averse update", fontsize=16)
    ax_g.tick_params(axis="both", labelsize=12)
    ax_g.grid(True, alpha=0.35)
    ax_g.legend(
        handles=[
            Line2D([0], [0], color=TASK1_COLOR, lw=2, label="Task 1"),
            Line2D([0], [0], color=TASK2_COLOR, lw=2, label="Task 2"),
            Line2D([0], [0], color=PLAIN_COLOR, lw=2.2, label="Plain GD"),
            Line2D([0], [0], color=CAGRAD_COLOR, lw=2.4, label="CAGrad"),
            Line2D([0], [0], color="0.45", ls="--", lw=1.2, label="CAGrad ball"),
        ],
        loc="upper right", fontsize=12, framealpha=0.92,
    )
    fig.tight_layout()
    return fig, ax_g, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a 2-task CAGrad vs plain GD example")
    parser.add_argument("--cagrad-c", type=float, default=0.4)
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/cagrad_2task_example/cagrad_2task_example.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sol = solve_cagrad(G1, G2, args.cagrad_c)
    fig, _, stats = plot_cagrad_2task_example(sol)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    g0, d, w = sol["g0"], sol["d"], sol["w"]
    print(f"g1 = {G1}")
    print(f"g2 = {G2}")
    print(f"plain GD gradient g_bar = {np.round(g0, 3)}")
    print(f"CAGrad gradient     d   = {np.round(d, 3)}")
    print(f"dual weights w* = {np.round(w, 3)}")
    print(f"cos(g1, g2) = {stats['cos12']:+.3f}")
    print(f"angle(g_bar, d) = {stats['angle_deg']:.1f} deg")
    print(f"worst-task <gi, g_bar> = {stats['plain_min']:.3f}")
    print(f"worst-task <gi, d>     = {stats['cagrad_min']:.3f}")
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
