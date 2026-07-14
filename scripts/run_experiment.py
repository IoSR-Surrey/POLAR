import json
from pathlib import Path

import torch

from utils.seed import set_seed
from utils.sofa_functions import load_sofa, get_template_for_llado2025_desena_2020
from utils.miscellaneous import (
    load_npy_to_torch,
    target_angle_for_listeners,
    create_mtl_optimizer,
    run_mtl_optimizer_step,
)

from filter_estimation.model import MultiSourceFIR
from propagation.model import BinauralFreeFieldPropagation
from models.listener_based.llado2025.model import llado2025_grad_surgery
from models.listener_based.desena2020.model import desena2020
from models.listener_based.LSD.model import LSD
from models.loudspeaker_based.second_moment.model import SecondMoment
from models.train import ModelTrainer
from logger.saving import SavingManager
from plotting.miscellaneous import ALL_PLOTTING_FUNCTIONS
from plotting.evaluation import compute_white_noise_plot_outputs
from logger.printing import print_loss_and_estimates
from datasets.stimulus_dataset import get_stim_loader

LOSS_SETS = {
    "llado_only": ["llado2025_grad_surgery"],
    "llado_desena": ["llado2025_grad_surgery", "desena2020"],
    "llado_desena_lsd": ["llado2025_grad_surgery", "desena2020", "LSD"],
    "llado_desena_lsd_second": ["llado2025_grad_surgery", "desena2020", "LSD", "SecondMoment"],
}


def _resolve_sofa_path(sofa_path):
    if "..." in sofa_path:
        return "data/SOFA/SYMMETRIC_D1_48K_24bit_256tap_FIR_SOFA.sofa"
    return sofa_path


def _build_models(loss_set, fs, hrirs, gtfb, n_taps, n_sources, device, dtype, sofaile, target_angles):
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    models = []
    for model_name in LOSS_SETS[loss_set]:
        if model_name == "llado2025_grad_surgery":
            models.append(llado_model)
        elif model_name == "desena2020":
            models.append(desena2020(fs, hrirs, gtfb, device=device, dtype=dtype))
        elif model_name == "LSD":
            models.append(LSD(fs, sofaile, target_angles.unsqueeze(-1), device=device, dtype=dtype))
        elif model_name == "SecondMoment":
            models.append(SecondMoment(n_taps, n_sources, device=device, dtype=dtype))
    return models, llado_model


def _build_targets(loss_set, lstn_pos, target_position, n_sources, dtype, device):
    targets = {
        "llado2025_grad_surgery": target_angle_for_listeners(lstn_pos, target_position).unsqueeze(0),
    }
    if "desena2020" in LOSS_SETS[loss_set]:
        targets["desena2020"] = torch.zeros(1, len(lstn_pos), dtype=dtype, device=device)
    if "LSD" in LOSS_SETS[loss_set]:
        targets["LSD"] = torch.zeros(1, len(lstn_pos), dtype=dtype, device=device)
    if "SecondMoment" in LOSS_SETS[loss_set]:
        targets["SecondMoment"] = torch.zeros(1, n_sources, dtype=dtype, device=device)
    return targets


def _compute_summary_metrics(model_trainer, targets, loss_dict, llado_model, desena_model=None):
    metrics = {
        "final_total_loss": float(loss_dict["total"].detach().cpu().item()),
    }

    llado_estimates = model_trainer.estimates["llado2025_grad_surgery"]
    llado_targets = targets["llado2025_grad_surgery"]
    angle_errors = (llado_estimates - llado_targets).abs()
    metrics["final_angle_error_mean"] = float(angle_errors.mean().detach().cpu().item())
    metrics["final_angle_error_per_listener"] = angle_errors.squeeze(0).detach().cpu().tolist()

    if desena_model is not None and "desena2020" in model_trainer.estimates:
        desena_estimates = model_trainer.estimates["desena2020"]
        metrics["final_desena_uncertainty_mean"] = float(desena_estimates.mean().detach().cpu().item())
    else:
        metrics["final_desena_uncertainty_mean"] = None

    if "LSD" in model_trainer.estimates:
        lsd = model_trainer.estimates["LSD"]
        metrics["final_lsd_mean"] = float(lsd.mean().detach().cpu().item())
    else:
        metrics["final_lsd_mean"] = None

    if "SecondMoment" in model_trainer.estimates:
        second_moment = model_trainer.estimates["SecondMoment"]
        metrics["final_second_moment_mean"] = float(second_moment.mean().detach().cpu().item())
    else:
        metrics["final_second_moment_mean"] = None

    return metrics


def run_experiment(
    *,
    mtl_optimizer,
    loss_set,
    lr=None,
    output_dir=None,
    use_tensorboard=False,
    run_plots=True,
    show_plots=False,
    config_path="config/hyperpameters.json",
    training_config_override=None,
):
    if loss_set not in LOSS_SETS:
        raise ValueError(f"Unsupported loss_set: {loss_set}. Choose from {list(LOSS_SETS)}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    env_config = config["env"]
    training_config = dict(config["training"])
    if training_config_override:
        training_config.update(training_config_override)
    if lr is not None:
        training_config["lr"] = lr

    geometry_config = config["geometry"]
    filter_config = config["filter_estimator"]
    stimulus_config = config["stimulus_dataset"]
    skip_keys = config["saving"]["skip_keys"]

    set_seed(env_config["seed"])

    fs = env_config["fs"]
    dtype = getattr(torch, env_config["dtype"])
    device = torch.device(env_config["device"])
    num_epochs = training_config["num_epochs"]
    target_angle = training_config.get("target_angle", 0.0)

    lstn_pos = torch.tensor(geometry_config["listeners"], dtype=dtype, device=device)
    spk_pos = torch.tensor(geometry_config["speakers"], dtype=dtype, device=device)
    target_distance = spk_pos[0].norm().item()
    target_position = torch.tensor(
        [
            target_distance * torch.cos(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))),
            target_distance * torch.sin(torch.deg2rad(torch.tensor(target_angle, dtype=dtype, device=device))),
        ],
        dtype=dtype,
        device=device,
    )

    sofaile = load_sofa(_resolve_sofa_path(env_config["sofa_path"]))
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)
    lag_values = torch.arange(-0.001 * fs, 0.001 * fs + 1, 1, dtype=dtype, device=device) / fs

    n_sources = len(spk_pos)
    n_taps = filter_config["n_samples"]
    target_angles = target_angle_for_listeners(lstn_pos, target_position)
    models, llado_model = _build_models(
        loss_set, fs, hrirs, gtfb, n_taps, n_sources, device, dtype, sofaile, target_angles
    )
    desena_model = next((m for m in models if m.__class__.__name__ == "desena2020"), None)
    lsd_model = next((m for m in models if isinstance(m, LSD)), None)

    prop = BinauralFreeFieldPropagation(fs, lstn_pos, spk_pos, sofaile, device=device, dtype=dtype)
    filter_estimator = MultiSourceFIR(n_taps, n_sources)
    if filter_config["init_weight"] is None:
        filter_estimator.set_filter_initialisation("middle")

    model_trainer = ModelTrainer(models, weights=[1.0] * len(models))
    targets = _build_targets(loss_set, lstn_pos, target_position, n_sources, dtype, device)

    optimizer = torch.optim.Adam(filter_estimator.parameters(), lr=training_config["lr"])
    mtl = create_mtl_optimizer(model_trainer, mtl_optimizer, training_config=training_config)

    if output_dir is None:
        output_dir = f"outputs/{mtl_optimizer}"

    saving_manager = SavingManager(
        output_dir,
        skip_keys=skip_keys,
        timestamp=True,
        use_tensorboard=use_tensorboard,
        task_names=list(model_trainer.model_dict.keys()),
    )

    run_config = {
        "mtl_optimizer": mtl_optimizer,
        "loss_set": loss_set,
        "lr": training_config["lr"],
        "num_epochs": num_epochs,
        "config": config,
    }
    with open(saving_manager.output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    saving_manager.save("gtfb", gtfb)
    saving_manager.save("lag_values", lag_values)
    saving_manager.save("tau_iacc", 1e-2)
    saving_manager.save("target_angle", target_angle)
    saving_manager.save("target_position", target_position)
    saving_manager.save("lstn_pos", lstn_pos)
    saving_manager.save("spk_pos", spk_pos)
    saving_manager.save("sofaile", sofaile)

    stim_loader, _ = get_stim_loader(
        fs=fs,
        n_bands=stimulus_config["n_bands"],
        n_samples_per_band=stimulus_config["n_samples_per_band"],
        batch_size=training_config["batch_size"],
        stimulus_length_ms=stimulus_config["stimulus_length_ms"],
        ramp_time_ms=stimulus_config["ramp_time_ms"],
        f_min=stimulus_config["f_min"],
        f_max=stimulus_config["f_max"],
        shuffle=training_config["shuffle"],
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        drop_last=training_config["drop_last"],
    )
    stim_iter = iter(stim_loader)
    dummy_input = torch.ones(1, 1, dtype=dtype, device=device)
    epoch_metrics = []

    for epoch in range(num_epochs):
        try:
            stim = next(stim_iter)
        except StopIteration:
            stim_iter = iter(stim_loader)
            stim = next(stim_iter)
        stim = stim.to(device=device, dtype=dtype)

        saving_manager.set_step(epoch)
        spk_filters = filter_estimator(dummy_input)
        binaural_signals = prop(stim, spk_filters)

        model_trainer.compute_estimates(binaural_signals, spk_filters, stim)
        loss_dict = model_trainer.compute_loss(targets)
        saving_manager.save("loss_dict", loss_dict)
        saving_manager.save("modelTrainer.estimates", model_trainer.estimates)

        if "llado2025_grad_surgery" in model_trainer.model_dict:
            cues = model_trainer.kwargs_output["llado2025_grad_surgery"]["interauralcues"]
            saving_manager.save("llado2025_grad_surgery_cues", {"itd": cues["itd"].detach(), "ild": cues["ild"].detach()})

        optimizer.zero_grad()
        run_mtl_optimizer_step(
            mtl_optimizer, mtl, loss_dict, filter_estimator, optimizer, saving_manager, model_trainer
        )

        saving_manager.flush_tensorboard()
        print_loss_and_estimates(loss_dict, model_trainer.estimates, epoch)

        epoch_metrics.append({
            "epoch": epoch,
            "total_loss": float(loss_dict["total"].detach().cpu().item()),
            "llado_loss": float(loss_dict["llado2025_grad_surgery"].detach().cpu().item())
            if "llado2025_grad_surgery" in loss_dict else None,
            "desena_loss": float(loss_dict["desena2020"].detach().cpu().item())
            if "desena2020" in loss_dict else None,
            "lsd_loss": float(loss_dict["LSD"].detach().cpu().item())
            if "LSD" in loss_dict else None,
            "second_moment_loss": float(loss_dict["SecondMoment"].detach().cpu().item())
            if "SecondMoment" in loss_dict else None,
        })

    with torch.no_grad():
        spk_filters = filter_estimator(dummy_input)
        saving_manager.save("spk_filters", spk_filters)
        plot_outputs = compute_white_noise_plot_outputs(saving_manager, config_path=config_path)
        if plot_outputs is not None:
            saving_manager.save("binaural_signals", plot_outputs["binaural_signals"])
            saving_manager.save("localisation_angles", plot_outputs["localisation_angles"])
            if "localisation_uncertainty" in plot_outputs:
                saving_manager.save("localisation_uncertainty", plot_outputs["localisation_uncertainty"])
            if "lsd" in plot_outputs:
                saving_manager.save("lsd", plot_outputs["lsd"])
            if "llado2025_grad_surgery_cues" in plot_outputs:
                saving_manager.save("llado2025_grad_surgery_cues", plot_outputs["llado2025_grad_surgery_cues"])
            stim = plot_outputs["stim"]
            binaural_signals = plot_outputs["binaural_signals"].unsqueeze(0)
        else:
            stim = next(iter(get_stim_loader(
                fs=fs,
                n_bands=stimulus_config["n_bands"],
                n_samples_per_band=stimulus_config["n_samples_per_band"],
                batch_size=training_config["batch_size"],
                stimulus_length_ms=stimulus_config["stimulus_length_ms"],
                ramp_time_ms=stimulus_config["ramp_time_ms"],
                f_min=stimulus_config["f_min"],
                f_max=stimulus_config["f_max"],
                shuffle=False,
                drop_last=training_config["drop_last"],
            )[0])).to(device=device, dtype=dtype)
            binaural_signals = prop(stim, spk_filters)

        model_trainer.compute_estimates(binaural_signals, spk_filters, stim)
    final_loss_dict = model_trainer.compute_loss(targets)
    summary = _compute_summary_metrics(model_trainer, targets, final_loss_dict, llado_model, desena_model)
    summary.update({
        "mtl_optimizer": mtl_optimizer,
        "loss_set": loss_set,
        "lr": training_config["lr"],
        "output_dir": str(saving_manager.output_dir),
        "epoch_metrics": epoch_metrics,
        "status": "success",
    })

    saving_manager.export_to_disk()
    print(f"Saved outputs to {saving_manager.output_dir}")

    if run_plots:
        ALL_PLOTTING_FUNCTIONS(saving_manager, show_plots=show_plots)

    saving_manager.close()
    return summary
