"""Replot Rodenas arrows+directivity figures using the optimisation bandpass set."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.stimulus_dataset import get_stim_loader
from plotting.evaluation import average_llado_over_stimulus_draws
from propagation.model import BinauralFreeFieldPropagation
from models.listener_based.llado2025.model import llado2025_grad_surgery
from utils.miscellaneous import load_npy_to_torch
from utils.sofa_functions import get_template_for_llado2025_desena_2020, load_sofa

RUNS = [
    ROOT / "outputs/rodenas/cagrad/2026-08-07_144847",
]


def _load_rodenas_module():
    spec = importlib.util.spec_from_file_location(
        "rodenas_optimisation", ROOT / "scripts" / "rodenas_optimisation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_value(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_journal_fig(fig, path):
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {path.with_suffix('.png')} and {path.with_suffix('.pdf')}")


def replot_run(run_dir, rodenas_mod, device, dtype):
    npy = run_dir / "npy"
    with open(run_dir / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)

    fs = int(_json_value(run_dir / "json" / "fs.json"))
    sofa_path = run_config["env"]["sofa_path"]
    apply_distance_attenuation = bool(run_config["env"]["apply_distance_attenuation"])
    upper_indices = tuple(run_config["upper_indices"])
    lower_indices = tuple(run_config["lower_indices"])

    lstn_pos = torch.as_tensor(np.load(npy / "lstn_pos.npy"), device=device, dtype=dtype)
    spk_pos = torch.as_tensor(np.load(npy / "spk_pos.npy"), device=device, dtype=dtype)
    target_position = torch.as_tensor(np.load(npy / "target_position.npy"), device=device, dtype=dtype)
    spk_filters = torch.as_tensor(np.load(npy / "spk_filters.npy"), device=device, dtype=dtype)
    gtfb_path = npy / "gtfb.npy"
    if gtfb_path.exists():
        gtfb = torch.as_tensor(np.load(gtfb_path), device=device, dtype=dtype)
    else:
        gtfb = load_npy_to_torch("data/banks/gtfb_48k.npy", device=device, dtype=dtype)

    sofaile = load_sofa(sofa_path)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    n_listeners = len(lstn_pos)

    prop = BinauralFreeFieldPropagation(
        fs, lstn_pos, spk_pos, sofaile,
        apply_distance_attenuation=apply_distance_attenuation,
        device=device, dtype=dtype,
    )
    stim_cfg = run_config["stimulus_dataset"]
    n_stim = int(stim_cfg["n_bands"]) * int(stim_cfg["n_samples_per_band"])
    stim_loader, _ = get_stim_loader(
        fs=fs,
        n_bands=stim_cfg["n_bands"],
        n_samples_per_band=stim_cfg["n_samples_per_band"],
        batch_size=n_stim,
        stimulus_length_ms=stim_cfg["stimulus_length_ms"],
        ramp_time_ms=stim_cfg["ramp_time_ms"],
        f_min=stim_cfg["f_min"],
        f_max=stim_cfg["f_max"],
        band_mode=stim_cfg.get("band_mode", "erb"),
        erb_factor=stim_cfg.get("erb_factor", 1.0),
        dataset_type=stim_cfg.get("dataset_type", "bandpass"),
        stimulus_type=stim_cfg.get("stimulus_type", "noise"),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    plot_stim = next(iter(stim_loader)).to(device=device, dtype=dtype)
    print(
        f"  evaluating 4-loudspeaker Llado on {plot_stim.shape[0]} "
        f"x {plot_stim.shape[-1] / fs * 1000:.0f} ms {stim_cfg.get('dataset_type', 'bandpass')} draws "
        f"(ramp {stim_cfg['ramp_time_ms']} ms)"
    )
    loc_plot = average_llado_over_stimulus_draws(
        llado_model, prop, spk_filters, plot_stim, n_listeners
    )

    est_filters = spk_filters.squeeze(0).detach().cpu().numpy()
    est_filters = est_filters / (np.max(np.abs(est_filters)) / 0.9)
    est_filters = est_filters * 1.41

    overlay_opt = np.loadtxt(ROOT / "data" / "miscellaneous" / "rodenas2003_fig12_opt.csv", delimiter=",", skiprows=1)
    overlay_opt = [[-((angle_deg + 180) % 360.0), level_db] for angle_deg, level_db in overlay_opt]
    overlay_1000 = np.loadtxt(ROOT / "data" / "miscellaneous" / "rodenas2003_fig12_1000Hz.csv", delimiter=",", skiprows=1)
    overlay_1000 = [[-((angle_deg + 180) % 360.0), level_db] for angle_deg, level_db in overlay_1000]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = rodenas_mod.plot_arrows_with_directivity(
        est_filters=est_filters, spk_pos=spk_pos.detach().cpu(), lstn_pos=lstn_pos.detach().cpu(),
        loc=loc_plot.detach().cpu(), fs=fs,
        group_idx=[upper_indices, lower_indices],
        target_pos=target_position, show_listener_target_lines=True, show=False,
        freqs_eval_hz=[250, 500, 1000, 2000, 4000],
        directivity_observation_radius_m=10.0, directivity_mode="freq",
        directivity_freq_idx=2,
        directivity_scale=0.8, directivity_floor_db=-20,
        overlay_curves=[overlay_opt, overlay_1000],
        overlay_labels=["rodenas2003 (optimum)", "rodenas2003 (1000 Hz)"],
        figsize=(5.2, 5.6),
    )
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    _save_journal_fig(fig, plot_dir / "arrows_and_directivity")

    est_filters_centre = np.stack([
        np.sum(est_filters[list(upper_indices)], axis=0),
        np.sum(est_filters[list(lower_indices)], axis=0),
    ], axis=0)
    spk_pos_np = spk_pos.detach().cpu().numpy()
    spk_pos_centre = np.stack([
        spk_pos_np[list(upper_indices)].mean(axis=0),
        spk_pos_np[list(lower_indices)].mean(axis=0),
    ], axis=0)
    prop_centre = BinauralFreeFieldPropagation(
        fs, lstn_pos, torch.as_tensor(spk_pos_centre, dtype=dtype, device=device), sofaile,
        apply_distance_attenuation=apply_distance_attenuation, device=device, dtype=dtype,
    )
    h_centre = torch.as_tensor(est_filters_centre, dtype=dtype, device=device).unsqueeze(0)
    print(f"  evaluating summed-pair Llado on {plot_stim.shape[0]} draws")
    loc_centre = average_llado_over_stimulus_draws(
        llado_model, prop_centre, h_centre, plot_stim, n_listeners
    )

    fig_centre = rodenas_mod.plot_arrows_with_directivity(
        est_filters=est_filters_centre,
        spk_pos=torch.as_tensor(spk_pos_centre, dtype=dtype),
        lstn_pos=lstn_pos.detach().cpu(), loc=loc_centre.detach().cpu(), fs=fs,
        group_idx=[(0,), (1,)],
        target_pos=target_position, show_listener_target_lines=True, show=False,
        freqs_eval_hz=[250, 500, 1000, 2000, 4000],
        directivity_observation_radius_m=10.0, directivity_mode="avg",
        directivity_scale=0.8, directivity_floor_db=-20,
        directivity_label="Omni source", directivity_color="#CC79A7",
        figsize=(5.2, 5.6),
    )
    _save_journal_fig(fig_centre, plot_dir / "arrows_and_directivity_centre")

    old_loc = np.asarray(np.load(npy / "localisation_angles.npy")).reshape(-1)
    loc = loc_plot.detach().cpu().numpy().reshape(-1)
    print(run_dir.name)
    lstn_np = np.asarray(lstn_pos.detach().cpu())
    for i, pos in enumerate(lstn_np):
        print(f"  ({pos[0]:.2f}, {pos[1]:.2f}): az {loc[i]:.1f} deg (old {old_loc[i]:.1f})")


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    print(f"Replotting Rodenas arrows on {device}")
    rodenas_mod = _load_rodenas_module()
    for run_dir in RUNS:
        replot_run(run_dir, rodenas_mod, device, dtype)


if __name__ == "__main__":
    main()
