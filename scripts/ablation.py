import argparse
import csv
import json
import os
import sys
import traceback
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.seed import set_seed
from scripts.run_experiment import run_experiment
from plotting.ablation_summary import plot_ablation_summary

MTL_OPTIMIZERS = ["plain", "gradnorm", "cagrad", "gradnorm+cagrad"]
LEARNING_RATES = [0.01, 0.05, 0.1, 0.2]
LOSS_SETS = ["llado_only", "llado_desena", "llado_desena_lsd", "llado_desena_lsd_second"]
NUM_EPOCHS = 1000

SUMMARY_FIELDS = [
    "run_id",
    "mtl_optimizer",
    "lr",
    "loss_set",
    "status",
    "output_dir",
    "final_angle_error_mean",
    "final_desena_uncertainty_mean",
    "final_lsd_mean",
    "final_second_moment_mean",
    "final_total_loss",
    "error",
]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run MTL optimisation ablation sweeps.")
    parser.add_argument("--optimizers", nargs="+", default=MTL_OPTIMIZERS, choices=MTL_OPTIMIZERS)
    parser.add_argument("--lrs", nargs="+", type=float, default=LEARNING_RATES)
    parser.add_argument("--loss-sets", nargs="+", default=LOSS_SETS, choices=LOSS_SETS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-run plots (not recommended).")
    parser.add_argument("--no-summary-plots", action="store_true")
    parser.add_argument("--config", default="config/hyperpameters.json")
    return parser.parse_args()


def _format_lr(lr):
    return f"{lr:g}".replace(".", "p")


def _summary_row(run_id, result):
    return {
        "run_id": run_id,
        "mtl_optimizer": result.get("mtl_optimizer"),
        "lr": result.get("lr"),
        "loss_set": result.get("loss_set"),
        "status": result.get("status", "failed"),
        "output_dir": result.get("output_dir"),
        "final_angle_error_mean": result.get("final_angle_error_mean"),
        "final_desena_uncertainty_mean": result.get("final_desena_uncertainty_mean"),
        "final_lsd_mean": result.get("final_lsd_mean"),
        "final_second_moment_mean": result.get("final_second_moment_mean"),
        "final_total_loss": result.get("final_total_loss"),
        "error": result.get("error"),
    }


def _write_summary(output_root, rows, all_results):
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


def main():
    args = _parse_args()
    grid = list(product(args.loss_sets, args.optimizers, args.lrs))
    total_runs = len(grid)

    if args.dry_run:
        print(f"Dry run: {total_runs} experiments")
        for run_id, (loss_set, mtl_optimizer, lr) in enumerate(grid, start=1):
            print(f"[{run_id}/{total_runs}] {mtl_optimizer} | lr={lr} | {loss_set}")
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    set_seed(config["env"]["seed"])

    output_root = Path("outputs/ablation")
    rows = []
    all_results = []

    for run_id, (loss_set, mtl_optimizer, lr) in enumerate(grid, start=1):
        print(f"[{run_id}/{total_runs}] {mtl_optimizer} | lr={lr} | {loss_set}")
        output_dir = output_root / loss_set / mtl_optimizer / f"lr_{_format_lr(lr)}"

        try:
            result = run_experiment(
                mtl_optimizer=mtl_optimizer,
                loss_set=loss_set,
                lr=lr,
                output_dir=str(output_dir),
                use_tensorboard=False,
                run_plots=not args.no_plots,
                config_path=args.config,
                training_config_override={"num_epochs": NUM_EPOCHS},
            )
            result["run_id"] = run_id
            all_results.append(result)
            rows.append(_summary_row(run_id, result))
        except Exception as exc:
            error_result = {
                "run_id": run_id,
                "mtl_optimizer": mtl_optimizer,
                "lr": lr,
                "loss_set": loss_set,
                "status": "failed",
                "output_dir": None,
                "final_angle_error_mean": None,
                "final_desena_uncertainty_mean": None,
                "final_lsd_mean": None,
                "final_second_moment_mean": None,
                "final_total_loss": None,
                "error": traceback.format_exc(),
            }
            all_results.append(error_result)
            rows.append(_summary_row(run_id, error_result))
            print(f"Run {run_id} failed: {exc}")

        _write_summary(output_root, rows, all_results)

    if not args.no_summary_plots:
        plot_ablation_summary(output_root)

    print(f"Ablation complete. Summary written to {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
