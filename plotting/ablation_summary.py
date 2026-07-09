import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_summary_rows(summary_path):
    if not summary_path.exists():
        return []

    import csv

    with open(summary_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in (
                "run_id",
                "lr",
                "final_angle_error_mean",
                "final_desena_uncertainty_mean",
                "final_lsd_mean",
                "final_second_moment_mean",
                "final_total_loss",
            ):
                if row.get(key) not in (None, "", "None"):
                    if key == "run_id":
                        row[key] = int(row[key])
                    else:
                        row[key] = float(row[key])
            rows.append(row)
    return rows


def _successful_rows(rows):
    return [row for row in rows if row.get("status") == "success"]


def plot_ablation_summary(output_root):
    output_root = Path(output_root)
    summary_path = output_root / "summary.csv"
    rows = _successful_rows(_load_summary_rows(summary_path))
    if not rows:
        print("No successful ablation runs found; skipping summary plots.")
        return

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _plot_angle_error_heatmap(rows, plot_dir)
    _plot_total_loss_heatmap(rows, plot_dir)
    _plot_loss_curves(output_root, plot_dir)


def _plot_angle_error_heatmap(rows, plot_dir):
    loss_sets = sorted({row["loss_set"] for row in rows})
    optimizers = sorted({row["mtl_optimizer"] for row in rows})
    lrs = sorted({row["lr"] for row in rows})

    fig, axes = plt.subplots(
        len(loss_sets),
        len(optimizers),
        figsize=(4 * len(optimizers), 3.5 * len(loss_sets)),
        squeeze=False,
    )

    for i, loss_set in enumerate(loss_sets):
        for j, optimizer in enumerate(optimizers):
            ax = axes[i][j]
            matrix = np.full((len(lrs), 1), np.nan)
            for row in rows:
                if row["loss_set"] == loss_set and row["mtl_optimizer"] == optimizer:
                    lr_idx = lrs.index(row["lr"])
                    matrix[lr_idx, 0] = row["final_angle_error_mean"]

            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set_xticks([0])
            ax.set_xticklabels(["angle error"])
            ax.set_yticks(range(len(lrs)))
            ax.set_yticklabels([f"{lr:g}" for lr in lrs])
            ax.set_title(f"{loss_set}\n{optimizer}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Final mean angle error (deg)")
    fig.tight_layout()
    fig.savefig(plot_dir / "angle_error_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_total_loss_heatmap(rows, plot_dir):
    loss_sets = sorted({row["loss_set"] for row in rows})
    optimizers = sorted({row["mtl_optimizer"] for row in rows})
    lrs = sorted({row["lr"] for row in rows})

    fig, axes = plt.subplots(
        len(loss_sets),
        len(optimizers),
        figsize=(4 * len(optimizers), 3.5 * len(loss_sets)),
        squeeze=False,
    )

    for i, loss_set in enumerate(loss_sets):
        for j, optimizer in enumerate(optimizers):
            ax = axes[i][j]
            matrix = np.full((len(lrs), 1), np.nan)
            for row in rows:
                if row["loss_set"] == loss_set and row["mtl_optimizer"] == optimizer:
                    lr_idx = lrs.index(row["lr"])
                    matrix[lr_idx, 0] = row["final_total_loss"]

            im = ax.imshow(matrix, aspect="auto", cmap="magma")
            ax.set_xticks([0])
            ax.set_xticklabels(["total loss"])
            ax.set_yticks(range(len(lrs)))
            ax.set_yticklabels([f"{lr:g}" for lr in lrs])
            ax.set_title(f"{loss_set}\n{optimizer}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Final total loss")
    fig.tight_layout()
    fig.savefig(plot_dir / "total_loss_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_loss_curves(output_root, plot_dir):
    summary_json_path = output_root / "summary.json"
    if not summary_json_path.exists():
        return

    with open(summary_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), squeeze=False)
    loss_keys = [
        ("llado_loss", "Llado loss"),
        ("desena_loss", "Desena loss"),
        ("lsd_loss", "LSD loss"),
        ("second_moment_loss", "Second moment loss"),
    ]

    for ax, (key, title) in zip(axes[0], loss_keys):
        for result in results:
            if result.get("status") != "success":
                continue
            epoch_metrics = result.get("epoch_metrics", [])
            values = [step[key] for step in epoch_metrics if step.get(key) is not None]
            if not values:
                continue
            epochs = list(range(len(values)))
            label = f"{result['mtl_optimizer']} | lr={result['lr']} | {result['loss_set']}"
            ax.plot(epochs, values, alpha=0.5, label=label)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Per-epoch loss curves")
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_curves.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
