"""Top-down schematic of the original Rodenas two-pair idea.

Two 2-loudspeaker arrays sit on a 2 m arc (inner ±30°, outer ±35°). Each pair
shows a 60° listening-facing directivity wedge (Rodenas 2003 Fig. 12 optimum),
quiet beside the array and louder toward y = 0. A phantom source sits between
the two arrays.

Example:
  python plotting/plot_rodenas_geometry_schematic.py
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
from matplotlib.patches import Rectangle
import numpy as np


SPEAKER_DISTANCE = 2.0
INNER_ANGLE_DEG = 30.0
OUTER_ANGLE_DEG = 35.0
LISTENING_AREA_WIDTH = 0.5
LISTENING_AREA_HEIGHT = 2.0

LISTENER_COLOR = "#2b6cb0"
TARGET_COLOR = "#c0392b"
BEAM_COLOR = "tab:blue"
DIRECTIVITY_SCALE = 0.8
DIRECTIVITY_FLOOR_DB = -20.0

LEFT_ANGLES_DEG = np.array([INNER_ANGLE_DEG, OUTER_ANGLE_DEG])
RIGHT_ANGLES_DEG = np.array([-INNER_ANGLE_DEG, -OUTER_ANGLE_DEG])
RODENAS_FIG12_OPT = ROOT / "data" / "miscellaneous" / "rodenas2003_fig12_opt.csv"


def pos_from_angle(angle_deg, r=SPEAKER_DISTANCE):
    a = np.deg2rad(angle_deg)
    return np.array([r * np.cos(a), r * np.sin(a)])


def pair_centroid(angles_deg):
    return np.mean([pos_from_angle(a) for a in angles_deg], axis=0)


def load_rodenas_fig12_opt():
    raw = np.loadtxt(RODENAS_FIG12_OPT, delimiter=",", skiprows=1)
    return np.array([[-((angle_deg + 180.0) % 360.0), level_db] for angle_deg, level_db in raw])


def _db_at_nearest_angle_deg(sample_angles_deg, levels_db, target_angle_deg):
    sample_angles_deg = np.asarray(sample_angles_deg, dtype=float).ravel()
    levels_db = np.asarray(levels_db, dtype=float).ravel()
    target_wrapped = float(np.mod(target_angle_deg, 360.0))
    samples_wrapped = np.mod(sample_angles_deg, 360.0)
    separation_deg = np.abs((samples_wrapped - target_wrapped + 180.0) % 360.0 - 180.0)
    return float(levels_db[int(np.argmin(separation_deg))])


def draw_directivity_wedge(ax, group_mid, overlay_curve, label=None):
    """60° listening-facing wedge, matching plot_arrows_with_directivity."""
    cx, cy = float(group_mid[0]), float(group_mid[1])
    if cy > 0.0:
        theta_wedge0 = np.pi
        theta_wedge1 = np.pi + np.pi / 3.0
    elif cy < 0.0:
        theta_wedge0 = np.pi - np.pi / 3.0
        theta_wedge1 = np.pi
    else:
        theta_wedge0 = np.pi
        theta_wedge1 = np.pi + np.pi / 3.0

    mid_rad = 0.5 * (theta_wedge0 + theta_wedge1)
    th_oblique_ray = theta_wedge1 if cy >= 0.0 else theta_wedge0
    n_arc = 72
    th_arc = np.linspace(theta_wedge0, theta_wedge1, n_arc)
    theta_label = 0.5 * (theta_wedge0 + theta_wedge1)
    cos_tl, sin_tl = np.cos(theta_label), np.sin(theta_label)

    line_extent = 0.0
    for db_ref in (-10.0, 0.0, 10.0):
        line_extent = max(
            line_extent,
            DIRECTIVITY_SCALE * (db_ref - DIRECTIVITY_FLOOR_DB) / (0.0 - DIRECTIVITY_FLOOR_DB),
        )
    label_radial_nudge = 0.04 * line_extent
    ref_line_kw = dict(color="0.45", linewidth=0.9, linestyle="-", zorder=35, alpha=0.75)
    ref_arc_kw = dict(color="0.45", linewidth=0.8, linestyle="--", zorder=34, alpha=0.75)

    ax.plot(
        [cx, cx + line_extent * np.cos(np.pi)],
        [cy, cy + line_extent * np.sin(np.pi)],
        **ref_line_kw,
    )
    ax.plot(
        [cx, cx + line_extent * np.cos(th_oblique_ray)],
        [cy, cy + line_extent * np.sin(th_oblique_ray)],
        **ref_line_kw,
    )
    for db_ref in (-10.0, 0.0, 10.0):
        circle_r = DIRECTIVITY_SCALE * (db_ref - DIRECTIVITY_FLOOR_DB) / (0.0 - DIRECTIVITY_FLOOR_DB)
        if circle_r > 0.0:
            ax.plot(cx + circle_r * np.cos(th_arc), cy + circle_r * np.sin(th_arc), **ref_arc_kw)
            if db_ref > 0.0:
                db_lbl = f"+{int(round(db_ref))} dB"
            elif db_ref < 0.0:
                db_lbl = f"{int(round(db_ref))} dB"
            else:
                db_lbl = "0 dB"
            r_lbl = circle_r + label_radial_nudge
            ax.text(
                cx + r_lbl * cos_tl, cy + r_lbl * sin_tl, db_lbl,
                ha="center", va="center", fontsize=12, color="0.35", zorder=36,
            )

    curve = np.asarray(overlay_curve, dtype=float)
    overlay_angle_offset_deg = 180.0 - float(np.min(np.mod(curve[:, 0], 360.0)))
    angles_deg_plot = np.mod(curve[:, 0] + overlay_angle_offset_deg, 360.0)
    db_sel = np.flip(curve[:, 1])
    if cy < 0.0:
        angles_deg_plot = np.mod(360.0 - angles_deg_plot, 360.0)
    mid_angle_deg = float(np.rad2deg(mid_rad) % 360.0)
    db_at_mid = _db_at_nearest_angle_deg(angles_deg_plot, db_sel, mid_angle_deg)
    db_aligned = db_sel - db_at_mid
    db_clipped = np.maximum(db_aligned, DIRECTIVITY_FLOOR_DB)
    r_overlay = DIRECTIVITY_SCALE * (db_clipped - DIRECTIVITY_FLOOR_DB) / (0.0 - DIRECTIVITY_FLOOR_DB)
    theta_overlay = np.deg2rad(angles_deg_plot)
    order = np.argsort(angles_deg_plot)
    ax.plot(
        cx + r_overlay[order] * np.cos(theta_overlay[order]),
        cy + r_overlay[order] * np.sin(theta_overlay[order]),
        color=BEAM_COLOR, linewidth=2.0, zorder=50, label=label,
    )


def plot_rodenas_geometry_schematic():
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

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    left_c = pair_centroid(LEFT_ANGLES_DEG)
    right_c = pair_centroid(RIGHT_ANGLES_DEG)
    phantom_xy = 0.5 * (left_c + right_c)
    spk_xy = np.array([pos_from_angle(a) for a in np.concatenate([LEFT_ANGLES_DEG, RIGHT_ANGLES_DEG])])
    fig12 = load_rodenas_fig12_opt()

    ax.add_patch(Rectangle(
        (-LISTENING_AREA_WIDTH / 2.0, -LISTENING_AREA_HEIGHT / 2.0),
        LISTENING_AREA_WIDTH, LISTENING_AREA_HEIGHT,
        facecolor=LISTENER_COLOR, edgecolor="black", linewidth=1.0,
        alpha=0.28, zorder=2,
    ))

    draw_directivity_wedge(ax, left_c, fig12, label="Pair directivity")
    draw_directivity_wedge(ax, right_c, fig12)

    ax.scatter(
        spk_xy[:, 0], spk_xy[:, 1], c="black", marker="x", s=55,
        linewidths=1.4, zorder=6, label="Loudspeakers",
    )
    ax.scatter(
        phantom_xy[0], phantom_xy[1], marker="^", c=TARGET_COLOR, s=70,
        zorder=7, label="Phantom source",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 2.05)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, color="0.88", zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        Rectangle(
            (0, 0), 1, 1, facecolor=LISTENER_COLOR, edgecolor="black",
            linewidth=1.0, alpha=0.28, label="Listening area",
        ),
        Line2D(
            [0], [0], marker="x", color="w", markeredgecolor="black",
            markersize=9, markeredgewidth=1.6, label="Loudspeakers",
        ),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor=TARGET_COLOR,
            markeredgecolor=TARGET_COLOR, markersize=10, label="Phantom source",
        ),
        Line2D([0], [0], color=BEAM_COLOR, lw=2.0, label="Pair directivity"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
        ncol=2, frameon=False, fontsize=12, borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.35, rect=[0.02, 0.02, 0.98, 0.91])
    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=str,
        default="outputs/rodenas/rodenas_geometry_schematic.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig, _ = plot_rodenas_geometry_schematic()

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
