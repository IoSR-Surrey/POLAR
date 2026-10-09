"""Tap-length x loss-set ablation.

Isolates two factors that differ between the early hand-run optimisations:
  - filter length (32 vs 1024 taps)
  - objective set (Llado localisation only vs Llado + De Sena)

Everything else (middle initialisation, seed, CAGrad, Adam lr, bandpass dataset,
geometry) is held constant so the two factors are the only intended variables.
Each condition trains over the full stimulus dataset for a fixed number of epochs
(averaging over every batch per epoch, matching scripts/scratch.py) and writes a
standard run folder (including npy/spk_filters.npy for scripts/listening/play_filtered.py).
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.seed import set_seed
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import (
    load_npy_to_torch,
    target_angle_for_listeners,
    create_mtl_optimizer,
    run_mtl_optimizer_step,
    git_commit_info,
    write_run_config,
)
from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from models.train import ModelTrainer
from logger.saving import SavingManager
from logger.printing import print_loss_and_estimates
from datasets.stimulus_dataset import get_stim_loader
from plotting.miscellaneous import ALL_PLOTTING_FUNCTIONS
from plotting.evaluation import compute_white_noise_plot_outputs
from plotting.batch_utils import (
    average_estimates_dicts,
    average_interaural_cues_dicts,
    average_loss_dicts,
    average_mtl_saved_values,
)
from scripts.run_experiment import LOSS_SETS, _build_models, _build_targets

MTL_OPTIMIZER = "cagrad"
TAP_COUNTS = [32, 1024]
LOSS_SET_NAMES = ["llado_only", "llado_desena"]
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
# Middle initialisation is enforced for every condition in this study.
FILTER_INITIALISATION = {"init_weight": "middle", "fs": 48000, "n_bands": 8, "band_mode": "log"}

SUMMARY_FIELDS = [
    "condition",
    "n_taps",
    "loss_set",
    "output_dir",
    "final_total_loss",
    "final_llado_loss",
    "final_desena_loss",
    "final_angle_error_mean",
    "final_desena_uncertainty_mean",
]


def build_conditions(tap_counts=TAP_COUNTS, loss_set_names=LOSS_SET_NAMES):
    """The ordered list of (n_taps, loss_set) conditions in the study grid."""
    conditions = []
    for n_taps, loss_set in product(tap_counts, loss_set_names):
        conditions.append({
            "name": f"taps{n_taps}_{loss_set}",
            "n_taps": n_taps,
            "loss_set": loss_set,
        })
    return conditions


def _merge_config(config, overrides):
    if not overrides:
        return config
    merged = json.loads(json.dumps(config))
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def load_shared_assets(config):
    """Load the condition-independent tensors once and reuse across conditions."""
    env_config = config["env"]
    device = torch.device(env_config["device"])
    dtype = getattr(torch, env_config["dtype"])
    fs = env_config["fs"]

    sofaile = load_sofa("data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa")
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)
    lag_values = torch.arange(-0.001 * fs, 0.001 * fs + 1, 1, dtype=dtype, device=device) / fs

    lstn_pos = torch.tensor(config["geometry"]["listeners"], dtype=dtype, device=device)
    spk_pos = torch.tensor(config["geometry"]["speakers"], dtype=dtype, device=device)
    target_angle = config["training"].get("target_angle", 0.0)
    target_distance = spk_pos[0].norm()
    target_position = torch.tensor(
        [
            target_distance * torch.cos(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))),
            target_distance * torch.sin(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))),
        ],
        dtype=dtype,
        device=device,
    )
    target_angles = target_angle_for_listeners(lstn_pos, target_position)

    return {
        "device": device,
        "dtype": dtype,
        "fs": fs,
        "sofaile": sofaile,
        "hrirs": hrirs,
        "gtfb": gtfb,
        "lag_values": lag_values,
        "lstn_pos": lstn_pos,
        "spk_pos": spk_pos,
        "target_angle": target_angle,
        "target_position": target_position,
        "target_angles": target_angles,
    }


def run_condition(condition, study_root, config, shared, seed, num_epochs, run_plots=True, show_plots=False):
    """Train a single (n_taps, loss_set) condition and write its run folder."""
    n_taps = condition["n_taps"]
    loss_set = condition["loss_set"]

    # Reset every RNG before each condition so the tap count and loss set are the
    # only intended differences between runs.
    set_seed(seed)

    device = shared["device"]
    dtype = shared["dtype"]
    fs = shared["fs"]
    sofaile = shared["sofaile"]
    hrirs = shared["hrirs"]
    gtfb = shared["gtfb"]
    lstn_pos = shared["lstn_pos"]
    spk_pos = shared["spk_pos"]
    n_sources = len(spk_pos)
    n_listeners = len(lstn_pos)

    models, _ = _build_models(
        loss_set, fs, hrirs, gtfb, n_taps, n_sources, device, dtype, sofaile, shared["target_angles"]
    )
    model_trainer = ModelTrainer(models, weights=[1.0] * len(models))
    targets = _build_targets(loss_set, lstn_pos, shared["target_position"], n_sources, dtype, device)

    filter_estimator = MultiSourceFIR(n_taps, n_sources)
    filter_estimator.set_filter_initialisation(**FILTER_INITIALISATION)
    filter_estimator = filter_estimator.to(device=device, dtype=dtype)

    stim_loader, _ = get_stim_loader(
        fs=fs,
        n_bands=config["stimulus_dataset"]["n_bands"],
        n_samples_per_band=config["stimulus_dataset"]["n_samples_per_band"],
        batch_size=config["training"]["batch_size"],
        stimulus_length_ms=config["stimulus_dataset"]["stimulus_length_ms"],
        ramp_time_ms=config["stimulus_dataset"]["ramp_time_ms"],
        f_min=config["stimulus_dataset"]["f_min"],
        f_max=config["stimulus_dataset"]["f_max"],
        band_mode=config["stimulus_dataset"]["band_mode"],
        erb_factor=config["stimulus_dataset"].get("erb_factor", 1.0),
        dataset_type=config["stimulus_dataset"]["dataset_type"],
        stimulus_type=config["stimulus_dataset"]["stimulus_type"],
        seed=config["env"]["seed"],
        shuffle=config["training"]["shuffle"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        drop_last=config["training"]["drop_last"],
    )

    optimizer = torch.optim.Adam(filter_estimator.parameters(), lr=LEARNING_RATE)
    mtl = create_mtl_optimizer(model_trainer, MTL_OPTIMIZER, training_config=config["training"])

    output_dir = Path(study_root) / condition["name"]
    saving_manager = SavingManager(
        str(output_dir), skip_keys=config["saving"]["skip_keys"], timestamp=False,
        use_tensorboard=False, task_names=list(model_trainer.model_dict.keys()),
    )

    git_info = git_commit_info()
    run_config = {
        "script": "scripts/tap_loss_ablation.py",
        "git": git_info,
        "condition": condition,
        "config": config,
        "mtl_optimizer": MTL_OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "num_epochs": num_epochs,
        "seed": seed,
        "filter_initialisation": FILTER_INITIALISATION,
        "n_taps": n_taps,
        "loss_set": loss_set,
    }
    write_run_config(output_dir, run_config)
    saving_manager.save("run_config", run_config)
    saving_manager.save("git_commit", git_info["commit"])
    saving_manager.save("git_dirty", git_info["dirty"])
    saving_manager.save("gtfb", gtfb)
    saving_manager.save("lag_values", shared["lag_values"])
    saving_manager.save("tau_iacc", 1e-2)
    saving_manager.save("target_angle", shared["target_angle"])
    saving_manager.save("target_position", shared["target_position"])
    saving_manager.save("lstn_pos", lstn_pos)
    saving_manager.save("spk_pos", spk_pos)
    saving_manager.save("sofaile", sofaile)
    saving_manager.save("stimulus_dataset", config["stimulus_dataset"])
    saving_manager.save("filter_initialisation", FILTER_INITIALISATION)
    saving_manager.save("seed", seed)
    saving_manager.save("n_taps", n_taps)
    saving_manager.save("loss_set", loss_set)

    dummy_input = torch.ones(1, 1, dtype=dtype, device=device)
    history = []

    for epoch in range(num_epochs):
        batch_loss_dicts = []
        batch_estimates = []
        batch_cues = {}
        batch_mtl_saved = []

        for stim in stim_loader:
            stim = stim.to(device=device, dtype=dtype)
            spk_filters = filter_estimator(dummy_input)
            binaural_signals = prop_forward(shared, config, spk_pos, lstn_pos, sofaile, fs, device, dtype, stim, spk_filters)

            model_trainer.compute_estimates(binaural_signals, spk_filters, stim)
            loss_dict = model_trainer.compute_loss(targets)
            batch_loss_dicts.append(loss_dict)
            batch_estimates.append(model_trainer.estimates)

            for model_name, kwargs in model_trainer.kwargs_output.items():
                if "interauralcues" in kwargs:
                    cues = kwargs["interauralcues"]
                    batch_cues.setdefault(model_name, []).append(
                        {"itd": cues["itd"].detach(), "ild": cues["ild"].detach()}
                    )

            optimizer.zero_grad()
            batch_mtl_saved.append(
                run_mtl_optimizer_step(
                    MTL_OPTIMIZER, mtl, loss_dict, filter_estimator, optimizer, saving_manager, model_trainer, save=False
                )
            )

        saving_manager.set_step(epoch)
        epoch_loss_dict = average_loss_dicts(batch_loss_dicts)
        epoch_estimates = average_estimates_dicts(batch_estimates)
        saving_manager.save("loss_dict", epoch_loss_dict)
        saving_manager.save("modelTrainer.estimates", epoch_estimates)
        for model_name, cues_dicts in batch_cues.items():
            saving_manager.save(
                f"{model_name}_cues", average_interaural_cues_dicts(cues_dicts, n_listeners)
            )
        for key, value in average_mtl_saved_values(batch_mtl_saved).items():
            saving_manager.save(key, value)

        history.append({
            "epoch": epoch,
            "loss": {key: float(value.detach().cpu()) for key, value in epoch_loss_dict.items()},
            "estimates": {key: value.detach().cpu().tolist() for key, value in epoch_estimates.items()},
        })
        print_loss_and_estimates(epoch_loss_dict, epoch_estimates, epoch)

    with torch.no_grad():
        spk_filters = filter_estimator(dummy_input)
        saving_manager.save("spk_filters", spk_filters)
        if run_plots:
            plot_outputs = compute_white_noise_plot_outputs(saving_manager)
            if plot_outputs is not None:
                saving_manager.save("binaural_signals", plot_outputs["binaural_signals"])
                saving_manager.save("localisation_angles", plot_outputs["localisation_angles"])
                saving_manager.save("localisation_uncertainty", plot_outputs["localisation_uncertainty"])
                saving_manager.save("lsd", plot_outputs["lsd"])
                if "llado2025_grad_surgery_cues" in plot_outputs:
                    saving_manager.save("llado2025_grad_surgery_cues", plot_outputs["llado2025_grad_surgery_cues"])

    saving_manager.export_to_disk()
    if run_plots:
        ALL_PLOTTING_FUNCTIONS(saving_manager, show_plots=show_plots)
    saving_manager.close()

    result = _condition_result(condition, output_dir, history, targets, shared)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "condition_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved condition '{condition['name']}' to {output_dir}")
    return result


def prop_forward(shared, config, spk_pos, lstn_pos, sofaile, fs, device, dtype, stim, spk_filters):
    """Build (once, cached on shared) the propagation model and run it."""
    prop = shared.get("prop")
    if prop is None:
        prop = BinauralFreeFieldPropagation(fs, lstn_pos, spk_pos, sofaile, device=device, dtype=dtype)
        shared["prop"] = prop
    return prop(stim, spk_filters)


def _condition_result(condition, output_dir, history, targets, shared):
    final = history[-1]["loss"]
    result = {
        "condition": condition["name"],
        "n_taps": condition["n_taps"],
        "loss_set": condition["loss_set"],
        "output_dir": str(output_dir).replace("\\", "/"),
        "final_total_loss": final.get("total"),
        "final_llado_loss": final.get("llado2025_grad_surgery"),
        "final_desena_loss": final.get("desena2020"),
        "epoch_metrics": history,
    }

    llado_estimates = history[-1]["estimates"].get("llado2025_grad_surgery")
    if llado_estimates is not None:
        estimates = np.asarray(llado_estimates).reshape(-1)
        target = shared["target_angles"].detach().cpu().numpy().reshape(-1)
        result["final_angle_error_mean"] = float(np.abs(estimates - target).mean())
    else:
        result["final_angle_error_mean"] = None

    desena_estimates = history[-1]["estimates"].get("desena2020")
    result["final_desena_uncertainty_mean"] = (
        float(np.asarray(desena_estimates).mean()) if desena_estimates is not None else None
    )
    return result


def _summary_row(result):
    return {field: result.get(field) for field in SUMMARY_FIELDS}


def write_summary(study_root, results):
    study_root = Path(study_root)
    with open(study_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(_summary_row(result) for result in results)
    with open(study_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def write_playback_config(study_root, results, config_path=None):
    study_root = Path(study_root)
    if config_path is None:
        config_path = ROOT / "config" / "tap_loss_ablation_play_config.json"
    folders = []
    for result in results:
        folder = Path(result["output_dir"])
        try:
            folder = folder.relative_to(ROOT)
        except ValueError:
            pass
        folders.append(str(folder).replace("\\", "/"))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"folders": folders}, f, indent=4)
        f.write("\n")
    return config_path


def _load_final_filters(result):
    path = Path(result["output_dir"]) / "npy" / "spk_filters.npy"
    filters = np.load(path)
    if filters.ndim == 3:
        filters = filters[0]
    return filters


def _magnitude_db(filter_taps, fs, n_fft=8192, f_min=20.0, f_max=20000.0):
    response = np.fft.rfft(filter_taps, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1 / fs)
    mask = (frequencies >= f_min) & (frequencies <= f_max)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(response[mask]), 1e-8))
    magnitude_db -= magnitude_db.max()
    return frequencies[mask], magnitude_db


def plot_comparisons(study_root, results, fs):
    study_root = Path(study_root)
    plot_dir = study_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _plot_loss_trajectories(plot_dir, results)
    _plot_localisation_trajectories(plot_dir, results)
    _plot_frequency_responses(plot_dir, results, fs)


def _condition_style(result):
    # Colour encodes tap length, linestyle encodes loss set: both effects readable at once.
    tap_colours = {32: "tab:blue", 1024: "tab:red"}
    loss_styles = {"llado_only": "-", "llado_desena": "--"}
    colour = tap_colours.get(result["n_taps"], "tab:green")
    style = loss_styles.get(result["loss_set"], ":")
    return colour, style


def _plot_loss_trajectories(plot_dir, results):
    loss_keys = [("total", "Total loss"), ("llado2025_grad_surgery", "Llado loss"), ("desena2020", "De Sena loss")]
    fig, axes = plt.subplots(1, len(loss_keys), figsize=(16, 4), squeeze=False)
    for ax, (key, title) in zip(axes[0], loss_keys):
        for result in results:
            values = [step["loss"].get(key) for step in result["epoch_metrics"]]
            values = [v for v in values if v is not None]
            if not values:
                continue
            colour, style = _condition_style(result)
            ax.plot(range(len(values)), values, color=colour, linestyle=style, label=result["condition"])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Loss trajectories (colour = taps, style = loss set)")
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_trajectories.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_localisation_trajectories(plot_dir, results):
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        errors = []
        for step in result["epoch_metrics"]:
            estimates = step["estimates"].get("llado2025_grad_surgery")
            if estimates is None:
                continue
            errors.append(float(np.abs(np.asarray(estimates)).mean()))
        if not errors:
            continue
        colour, style = _condition_style(result)
        ax.plot(range(len(errors)), errors, color=colour, linestyle=style, label=result["condition"])
    ax.set_title("Mean |Llado localisation estimate| over epochs (target 0 deg)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean |azimuth| (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "localisation_trajectories.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_frequency_responses(plot_dir, results, fs):
    filters = {result["condition"]: _load_final_filters(result) for result in results}
    n_sources = max(f.shape[0] for f in filters.values())
    fig, axes = plt.subplots(1, n_sources, figsize=(7 * n_sources, 5), squeeze=False, sharex=True, sharey=True)
    for source_idx in range(n_sources):
        ax = axes[0][source_idx]
        for result in results:
            condition_filters = filters[result["condition"]]
            if source_idx >= condition_filters.shape[0]:
                continue
            frequencies, magnitude_db = _magnitude_db(condition_filters[source_idx], fs)
            colour, style = _condition_style(result)
            ax.semilogx(frequencies, magnitude_db, color=colour, linestyle=style, label=result["condition"])
        ax.set_title(f"Speaker {source_idx + 1}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalised magnitude (dB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Final speaker frequency responses (colour = taps, style = loss set)")
    fig.tight_layout()
    fig.savefig(plot_dir / "frequency_response_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def run_study(
    *,
    config_path="config/hyperpameters.json",
    config_overrides=None,
    output_root="outputs/tap_loss_ablation",
    tap_counts=TAP_COUNTS,
    loss_set_names=LOSS_SET_NAMES,
    num_epochs=NUM_EPOCHS,
    seed=None,
    run_plots=True,
    study_name=None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = _merge_config(config, config_overrides)
    if seed is None:
        seed = config["env"]["seed"]

    if study_name is None:
        study_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    study_root = Path(output_root) / study_name
    study_root.mkdir(parents=True, exist_ok=True)

    shared = load_shared_assets(config)
    conditions = build_conditions(tap_counts, loss_set_names)

    write_run_config(study_root, {
        "script": "scripts/tap_loss_ablation.py",
        "conditions": conditions,
        "num_epochs": num_epochs,
        "seed": seed,
        "mtl_optimizer": MTL_OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "filter_initialisation": FILTER_INITIALISATION,
        "stimulus_dataset": config["stimulus_dataset"],
        "training": config["training"],
        "config": config,
    }, filename="study_config.json")

    results = []
    for index, condition in enumerate(conditions, start=1):
        print(f"[{index}/{len(conditions)}] {condition['name']}")
        results.append(
            run_condition(condition, study_root, config, shared, seed, num_epochs, run_plots=run_plots)
        )

    write_summary(study_root, results)
    if run_plots:
        plot_comparisons(study_root, results, shared["fs"])
    playback_config = write_playback_config(study_root, results)

    print(f"Study complete. Summary at {study_root / 'summary.csv'}")
    print(f"Playback config written to {playback_config}")
    return {"study_root": str(study_root), "results": results, "playback_config": str(playback_config)}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the tap-length x loss-set ablation study.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--tap-counts", nargs="+", type=int, default=TAP_COUNTS)
    parser.add_argument("--loss-sets", nargs="+", default=LOSS_SET_NAMES, choices=list(LOSS_SETS))
    parser.add_argument("--config", default="config/hyperpameters.json")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-run and comparison plots.")
    parser.add_argument("--smoke", action="store_true", help="One-epoch smoke study for a quick pipeline check.")
    return parser.parse_args()


def main():
    args = _parse_args()
    num_epochs = 1 if args.smoke else args.num_epochs
    study_name = "smoke" if args.smoke else None
    run_study(
        config_path=args.config,
        tap_counts=args.tap_counts,
        loss_set_names=args.loss_sets,
        num_epochs=num_epochs,
        run_plots=not args.no_plots,
        study_name=study_name,
    )


if __name__ == "__main__":
    main()
