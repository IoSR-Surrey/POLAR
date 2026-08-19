"""Replot all CTC figures from raw optimiser filters (spk_filters / spk_filters_col_only)."""
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
import torch.nn.functional as F

from models.listener_based.LSD.model import LSD
from propagation.model import BinauralFreeFieldPropagation
from utils.seed import set_seed
from utils.sofa_functions import load_sofa
from datasets.stimulus_dataset import get_stim_loader

CTC_ROOT = ROOT / "outputs/ctc"
RUNS = ["2026-08-17_173904", "2026-08-17_175118", "2026-08-18_184923"]
PLOT_NAMES = {
    "Proposed (Col. + IPD)": "localisation_proposed",
    "Proposed (Col. only)": "localisation_col_only",
    "CTC": "localisation_ctc",
}


def _load_ctc_module():
    spec = importlib.util.spec_from_file_location(
        "ctc_optimisation", ROOT / "scripts" / "ctc_optimisation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_2d_filters(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def recover_col_only_raw(run_dir, cfg, ctc_mod, prop, lsd_model, device, dtype):
    npy = run_dir / "npy"
    saved = npy / "spk_filters_col_only.npy"
    if saved.exists():
        return _as_2d_filters(np.load(saved))

    args = cfg["args"]
    set_seed(int(args["seed"]))
    n_listeners = 1
    n_taps = int(args["n_taps"])
    n_sources = 2
    fs = int(cfg["fs"])

    plot_stim = torch.zeros(1, 1, 1, dtype=dtype, device=device)
    plot_stim[:, :, 0] = 1.0
    filter_input = torch.ones(1, 1, dtype=dtype, device=device)

    stim_loader = None
    if args.get("stimulus") == "bandpass":
        stim_loader, _ = get_stim_loader(
            fs=fs,
            n_bands=int(args["n_bands"]),
            n_samples_per_band=int(args["n_samples_per_band"]),
            batch_size=int(args["batch_size"]),
            stimulus_length_ms=float(args["stimulus_length_ms"]),
            ramp_time_ms=1.0,
            f_min=20.0,
            f_max=20000.0,
            band_mode="linear_erb",
            erb_factor=0.25,
            dataset_type="bandpass",
            stimulus_type="noise",
            shuffle=True,
        )

    print(f"  Re-optimising colouration-only to recover raw filters for {run_dir.name}")
    spk_filters_col_only, _, _ = ctc_mod._optimise_filters(
        models=[lsd_model],
        criterions=[F.l1_loss],
        weights=[float(args["col_weight"])],
        targets={"LSD": torch.zeros(1, n_listeners, dtype=dtype, device=device)},
        prop=prop,
        stim=plot_stim,
        stim_loader=stim_loader,
        filter_input=filter_input,
        n_taps=n_taps,
        n_sources=n_sources,
        lr=float(args["lr"]),
        num_epochs=int(args["epochs"]),
        device=device,
        dtype=dtype,
        mtl_optimizer="plain",
        cagrad_c=float(args["cagrad_c"]),
        saving_manager=None,
    )
    raw = spk_filters_col_only.squeeze(0).detach().cpu().numpy()
    np.save(saved, raw)
    return raw


def _save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def _save_localisation_npy(npy, results):
    for name, res in results.items():
        for key, value in res.items():
            path = npy / f"localisation_results.{name}.{key}.npy"
            np.save(path, np.asarray(value))


def replot_run(run_dir, ctc_mod, device, dtype):
    npy = run_dir / "npy"
    if not (npy / "spk_filters.npy").exists():
        print(f"skip {run_dir.name}: no spk_filters.npy")
        return

    with open(run_dir / "run_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    fs = int(cfg["fs"])
    target_angle = float(cfg["args"]["target_angle"])
    sofaile = load_sofa(cfg["args"]["sofa_path"])
    apply_distance_attenuation = bool(cfg.get("apply_distance_attenuation", True))
    speaker_distance = float(cfg.get("speaker_distance", 2.0))

    lstn_pos, spk_pos, _, target_angles, _ = ctc_mod._build_geometry(
        dtype, device, target_angle, speaker_distance=speaker_distance,
    )
    prop = BinauralFreeFieldPropagation(
        fs, lstn_pos, spk_pos, sofaile,
        apply_distance_attenuation=apply_distance_attenuation,
        device=device, dtype=dtype,
    )
    lsd_model = LSD(fs, sofaile, target_angles, device=device, dtype=dtype)
    plot_stim = torch.zeros(1, 1, 1, dtype=dtype, device=device)
    plot_stim[:, :, 0] = 1.0
    reference_bin = lsd_model.compute_binaural_reference_signals(plot_stim)

    proposed = _as_2d_filters(np.load(npy / "spk_filters.npy"))
    col_only = recover_col_only_raw(run_dir, cfg, ctc_mod, prop, lsd_model, device, dtype)
    ctc_n_fft = cfg.get("ctc_n_fft") or cfg["args"].get("ctc_n_fft")
    if ctc_n_fft is not None:
        ctc_n_fft = int(ctc_n_fft)
        ctc = ctc_mod.compute_ctc_filters(sofaile, target_angle, n_fft=ctc_n_fft)
        if apply_distance_attenuation:
            ctc = ctc * ctc_mod._ctc_distance_attenuation_gain(lstn_pos, spk_pos)
        np.save(npy / "ctc_filters.npy", ctc)
        print(f"  CTC inverse n_fft={ctc_n_fft}, taps={ctc.shape[-1]}")
    else:
        ctc = np.load(npy / "ctc_filters.npy")
        print(f"  loaded saved CTC taps={ctc.shape[-1]}")

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    print(run_dir.name)

    fig_raw = ctc_mod.plot_ctc_magnitude(
        proposed=proposed,
        proposed_col_only=col_only,
        ctc=ctc,
        reference_bin=reference_bin,
        fs=fs,
        prop=prop,
        stim=plot_stim,
        smooth=False,
    )
    _save_fig(fig_raw, plot_dir / "ctc_comparison_raw.png")

    fig_mag, fig_phase = ctc_mod.plot_ctc_comparison(
        proposed=proposed,
        proposed_col_only=col_only,
        ctc=ctc,
        reference_bin=reference_bin,
        fs=fs,
        prop=prop,
        stim=plot_stim,
    )
    fig_mag.savefig(plot_dir / "ctc_comparison.png", bbox_inches="tight", dpi=150)
    fig_mag.savefig(plot_dir / "ctc_comparison_raw_third_octave.png", bbox_inches="tight", dpi=150)
    print(f"  saved {plot_dir / 'ctc_comparison.png'}")
    print(f"  saved {plot_dir / 'ctc_comparison_raw_third_octave.png'}")
    plt.close(fig_mag)
    _save_fig(fig_phase, plot_dir / "ctc_comparison_phase.png")

    fig_taps = ctc_mod.plot_ctc_time_domain(proposed, col_only, ctc, fs)
    _save_fig(fig_taps, plot_dir / "ctc_comparison_taps.png")
    fig_taps_raw = ctc_mod.plot_ctc_time_domain(proposed, col_only, ctc, fs)
    _save_fig(fig_taps_raw, plot_dir / "ctc_comparison_taps_raw.png")

    figs, results = ctc_mod.plot_ctc_localisation(
        proposed=proposed,
        proposed_col_only=col_only,
        ctc=ctc,
        prop=prop,
        lstn_pos=lstn_pos,
        spk_pos=spk_pos,
        target_angles=target_angles,
        target_angle_deg=target_angle,
        sofaile=sofaile,
        fs=fs,
        device=device,
        dtype=dtype,
    )
    for fig, (name, res) in zip(figs, results.items()):
        _save_fig(fig, plot_dir / f"{PLOT_NAMES[name]}.png")
        angle = float(np.atleast_1d(np.asarray(res["localisation_angles"]).reshape(-1))[0])
        print(f"    {PLOT_NAMES[name]}: {angle:.1f} deg")
    _save_localisation_npy(npy, results)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Replotting CTC from spk_filters on {device}")
    ctc_mod = _load_ctc_module()
    for name in RUNS:
        replot_run(CTC_ROOT / name, ctc_mod, device, dtype)


if __name__ == "__main__":
    main()
