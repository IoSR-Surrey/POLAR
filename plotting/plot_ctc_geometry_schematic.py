"""Top-down schematic of the CTC stereo pair and target direction.

Two loudspeakers at ±30° on a 2 m radius, one central listener, and the
virtual-source azimuth used in the CTC optimisation.

Example:
  python plotting/plot_ctc_geometry_schematic.py
  python plotting/plot_ctc_geometry_schematic.py --target-angle 15
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
from matplotlib.patches import Arc, FancyArrowPatch
from matplotlib.transforms import Affine2D
import numpy as np


SPEAKER_DISTANCE = 2.0
SPEAKER_ANGLES_DEG = np.array([30.0, -30.0])
LISTENER_POS = np.array([0.0, 0.0])

SPEAKER_COLOR = "#e07a3d"
LISTENER_COLOR = "#2b6cb0"
TARGET_COLOR = "#c0392b"


def pos_from_angle(angle_deg, r=SPEAKER_DISTANCE):
    a = np.deg2rad(angle_deg)
    return np.array([r * np.cos(a), r * np.sin(a)])


def draw_speaker(ax, xy, angle_deg, color, alpha=1.0, zorder=4):
    w, h = 0.14, 0.22
    rect = plt.Rectangle(
        (-w / 2, -h / 2), w, h,
        facecolor=color, edgecolor="black", linewidth=1.0,
        alpha=alpha, zorder=zorder,
    )
    transform = Affine2D().rotate_deg(angle_deg).translate(xy[0], xy[1]) + ax.transData
    rect.set_transform(transform)
    ax.add_patch(rect)
    inward = -pos_from_angle(angle_deg, r=1.0)
    ax.annotate(
        "", xy=xy + inward * 0.28, xytext=xy + inward * 0.02,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, mutation_scale=12),
        zorder=zorder + 1,
    )


def plot_ctc_geometry_schematic(target_angle_deg=45.0):
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    ax.add_patch(Arc(
        (0, 0), 2 * SPEAKER_DISTANCE, 2 * SPEAKER_DISTANCE,
        theta1=-40, theta2=55, linestyle="--", linewidth=1.2,
        color="0.55", zorder=1,
    ))

    # 0° reference (no loudspeaker on-axis) for the target-angle wedge.
    ax.plot([0, SPEAKER_DISTANCE], [0, 0], color="0.75", linestyle=":", linewidth=1.0, zorder=1)

    label_radius = SPEAKER_DISTANCE + 0.28
    for a in SPEAKER_ANGLES_DEG:
        p = pos_from_angle(a)
        ax.plot([0, p[0]], [0, p[1]], color="0.75", linestyle=":", linewidth=1.0, zorder=1)
        lp = pos_from_angle(a, r=label_radius)
        ax.text(
            lp[0], lp[1], f"{a:+.0f}°",
            ha="left" if a > 0 else "right", va="center",
            fontsize=12, color="0.25", zorder=5,
        )
        draw_speaker(ax, p, a, color=SPEAKER_COLOR)

    ax.scatter(
        LISTENER_POS[0], LISTENER_POS[1], c=LISTENER_COLOR, s=110, marker="o",
        edgecolors="black", linewidths=0.8, zorder=6,
    )

    target_len = 1.45
    target_tip = LISTENER_POS + pos_from_angle(target_angle_deg, r=target_len)
    ax.add_patch(FancyArrowPatch(
        LISTENER_POS, target_tip,
        arrowstyle="-|>", mutation_scale=18, linewidth=2.4,
        color=TARGET_COLOR, shrinkA=0, shrinkB=0, zorder=8,
    ))

    wedge_r = 0.55
    ax.add_patch(Arc(
        tuple(LISTENER_POS), 2 * wedge_r, 2 * wedge_r,
        theta1=min(0.0, target_angle_deg), theta2=max(0.0, target_angle_deg),
        linewidth=1.6, color=TARGET_COLOR, zorder=7,
    ))
    mid = 0.5 * target_angle_deg
    mid_pos = pos_from_angle(mid, r=wedge_r + 0.12)
    ax.text(
        mid_pos[0], mid_pos[1], f"{target_angle_deg:+.0f}°",
        color=TARGET_COLOR, fontsize=13, fontweight="bold",
        ha="left" if target_angle_deg >= 0 else "right", va="bottom", zorder=8,
    )

    ax.annotate(
        "", xy=(SPEAKER_DISTANCE, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2),
    )
    ax.text(1.0, -0.12, "2 m", ha="center", va="top", fontsize=13, color="0.35")

    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-1.45, 1.70)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, color="0.88", zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=LISTENER_COLOR,
            markeredgecolor="black", markersize=10, label="Listening position",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=SPEAKER_COLOR,
            markeredgecolor="black", markersize=10, label="Loudspeaker (±30°)",
        ),
        Line2D(
            [0], [0], color=TARGET_COLOR, lw=2.4, label=f"Target direction ({target_angle_deg:+.0f}°)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.95, fontsize=12)

    fig.tight_layout()
    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-angle", type=float, default=45.0)
    parser.add_argument(
        "--output", type=str,
        default="outputs/ctc/ctc_geometry_schematic.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig, _ = plot_ctc_geometry_schematic(target_angle_deg=args.target_angle)

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
