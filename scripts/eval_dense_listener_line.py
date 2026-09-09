"""Evaluate saved speaker filters on a dense line of virtual listeners."""
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

from models.listener_based.LSD.model import LSD
from models.listener_based.desena2020.model import desena2020
from models.listener_based.llado2025.model import llado2025_grad_surgery
from plotting.batch_utils import mean_over_stimulus_batch
from plotting.spatial_plots import LSD_LOWPASS_REFERENCE_DB, localisation_arrows_plot
from propagation.model import BinauralFreeFieldPropagation
from utils.miscellaneous import target_angle_for_listeners
from utils.sofa_functions import get_template_for_llado2025_desena_2020, load_sofa


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vbap = _load_module(
    "create_vbap_stimuli",
    ROOT / "scripts" / "listening" / "create_vbap_stimuli.py",
)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RUNS = [
    {
        "path": ROOT / "outputs/cagrad/2026-08-05_045408",
        "label": r"$S_{0.5}$",
    },
    {
        "path": ROOT / "outputs/cagrad/2026-08-05_050357",
        "label": r"$S_{0.25}$",
    },
    {
        "path": ROOT / "outputs/cagrad/2026-08-20_111650",
        "label": r"$S_{0}$",
    },
]
N_LISTENERS = 51
N_ARROW_LISTENERS = 7
STIM_LENGTH_MS = 100.0
N_STIM = 8
SEED = 42
Y_START = 0.5
Y_END = -0.5
LISTENER_CHUNK = 17
SERIES_COLORS = ("#E69F00", "#56B4E9", "#000000", "#009E73")
SERIES_LINESTYLES = ("-", "-", "-", "--")


def _json_value(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run(run_dir):
    npy = run_dir / "npy"
    js = run_dir / "json"
    with open(run_dir / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)
    return {
        "run_dir": run_dir,
        "run_config": run_config,
        "spk_filters": np.load(npy / "spk_filters.npy"),
        "spk_pos": np.load(npy / "spk_pos.npy"),
        "lstn_pos": np.load(npy / "lstn_pos.npy"),
        "target_position": np.load(npy / "target_position.npy"),
        "gtfb": np.load(npy / "gtfb.npy"),
        "localisation_angles": np.load(npy / "localisation_angles.npy"),
        "localisation_uncertainty": np.load(npy / "localisation_uncertainty.npy"),
        "lsd": np.load(npy / "lsd.npy"),
        "fs": int(_json_value(js / "fs.json")),
        "sofa_path": _json_value(js / "sofa_path.json"),
        "apply_distance_attenuation": bool(_json_value(js / "apply_distance_attenuation.json")),
        "lambda_loc": float(_json_value(js / "lambda_loc.json")),
        "stimulus_dataset": _json_value(js / "stimulus_dataset.json"),
        "batch_size": int(_json_value(js / "batch_size.json")),
        "seed": int(_json_value(js / "seed.json")),
    }


def make_stim(fs, device, dtype):
    n_samples = int(round(STIM_LENGTH_MS * fs / 1000.0))
    rng = np.random.default_rng(SEED)
    stim = rng.standard_normal((N_STIM, 1, n_samples), dtype=np.float32)
    stim = stim / np.std(stim).clip(min=1e-8)
    return torch.as_tensor(stim, device=device, dtype=dtype)


def vbap_impulse_filters(spk_pos, target_angle_deg):
    """Constant-power 2D VBAP as a 1-tap filter per loudspeaker."""
    gains = vbap.vbap_gains_2d(spk_pos, target_angle_deg)
    return np.asarray(gains, dtype=np.float32).reshape(-1, 1)


def evaluate_filters(
    spk_filters,
    lstn_pos,
    spk_pos,
    sofaile,
    fs,
    stim,
    llado_model,
    desena_model,
    target_position,
    apply_distance_attenuation,
    device,
    dtype,
):
    n_listeners = len(lstn_pos)
    target_angles = target_angle_for_listeners(lstn_pos, target_position)
    lsd_model = LSD(fs, sofaile, target_angles.unsqueeze(-1), device=device, dtype=dtype)
    prop = BinauralFreeFieldPropagation(
        fs, lstn_pos, spk_pos, sofaile,
        apply_distance_attenuation=apply_distance_attenuation,
        device=device, dtype=dtype,
    )
    if spk_filters.ndim == 2:
        spk_filters = spk_filters.unsqueeze(0)

    loc_parts, unc_parts, lsd_parts = [], [], []
    with torch.no_grad():
        for start in range(0, n_listeners, LISTENER_CHUNK):
            end = min(start + LISTENER_CHUNK, n_listeners)
            idx = torch.arange(start, end, device=device)
            binaural = prop(stim, spk_filters, listener_indices=idx)
            binaural_flat = binaural.reshape(-1, 2, binaural.shape[-1])
            n_chunk = end - start
            loc_parts.append(mean_over_stimulus_batch(llado_model(binaural_flat, stim)["output"], n_chunk))
            unc_parts.append(mean_over_stimulus_batch(desena_model(binaural_flat, stim)["output"], n_chunk))
            lsd_parts.append(mean_over_stimulus_batch(lsd_model(binaural_flat, stim, listener_indices=idx)["output"], n_chunk))

    return {
        "localisation_angles": np.concatenate([np.atleast_1d(p).reshape(-1) for p in loc_parts]),
        "localisation_uncertainty": np.concatenate([np.atleast_1d(p).reshape(-1) for p in unc_parts]),
        "lsd": np.concatenate([np.atleast_1d(p).reshape(-1) for p in lsd_parts]),
        "target_angles": target_angles.detach().cpu().numpy(),
    }


def subsample_listeners(lstn_pos, results, n_listeners=N_ARROW_LISTENERS):
    if hasattr(lstn_pos, "detach"):
        lstn_pos = lstn_pos.detach().cpu().numpy()
    lstn_pos = np.asarray(lstn_pos).reshape(-1, 2)
    idx = np.unique(np.linspace(0, len(lstn_pos) - 1, n_listeners).round().astype(int))
    return lstn_pos[idx], {key: np.asarray(value)[idx] for key, value in results.items()}


def plot_dense_arrows(run, results, lstn_pos, out_path):
    lstn_pos, results = subsample_listeners(lstn_pos, results)
    training_y = np.array([-0.5, 0.0, 0.5])
    trained_mask = np.any(np.isclose(lstn_pos[:, 1:2], training_y[None, :]), axis=1)
    localisation_arrows_plot(
        lstn_pos,
        run["spk_pos"],
        results["localisation_angles"],
        results["localisation_uncertainty"],
        lsd=results["lsd"],
        target_pos=run["target_position"],
        max_arrow_length=1.1,
        arrow_head_width=0.12,
        arrow_head_length=0.14,
        arrow_linewidth=2,
        trained_listener_mask=trained_mask,
    )
    fig = plt.gcf()
    fig.axes[0].set_title(f"Estimated localisation ({run['label']})", fontsize=16)
    for ax in fig.axes:
        ax.tick_params(labelsize=12)
        if hasattr(ax, "yaxis") and ax.get_ylabel() == "LSD [dB]":
            ax.tick_params(labelsize=12)
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(runs_with_results, y, out_path, title="Dense virtual-listener line at $x=0$"):
    order = np.argsort(y)
    y_sorted = y[order]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.5), sharex=True)
    training_y = np.array([-0.5, 0.0, 0.5])

    for i, (run, results) in enumerate(runs_with_results):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        linestyle = SERIES_LINESTYLES[i % len(SERIES_LINESTYLES)]
        az = np.asarray(results["localisation_angles"])[order]
        target = np.asarray(results["target_angles"])[order]
        lsd = np.asarray(results["lsd"])[order]
        axes[0].plot(y_sorted, az, color=color, linestyle=linestyle, linewidth=2, label=run["label"])
        axes[1].plot(y_sorted, az - target, color=color, linestyle=linestyle, linewidth=2, label=run["label"])
        axes[2].plot(y_sorted, lsd, color=color, linestyle=linestyle, linewidth=2, label=run["label"])
        train_idx = [int(np.argmin(np.abs(y_sorted - yi))) for yi in training_y]
        axes[0].scatter(y_sorted[train_idx], az[train_idx], color=color, marker="o", s=36, zorder=3)
        axes[1].scatter(y_sorted[train_idx], (az - target)[train_idx], color=color, marker="o", s=36, zorder=3)
        axes[2].scatter(y_sorted[train_idx], lsd[train_idx], color=color, marker="o", s=36, zorder=3)

    target_sorted = np.asarray(runs_with_results[0][1]["target_angles"])[order]
    axes[0].plot(y_sorted, target_sorted, color="black", linestyle="--", linewidth=1.5, label="Geometric target")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].axhline(LSD_LOWPASS_REFERENCE_DB, color="black", linestyle="--", linewidth=1.0, label="7 kHz LP ref")

    for ax in axes:
        for y_train in training_y:
            ax.axvline(y_train, color="0.75", linestyle=":", linewidth=1.0, zorder=0)
        ax.grid(True, zorder=0)
        ax.legend()

    axes[0].set_ylabel("Estimated azimuth [deg]")
    axes[1].set_ylabel("Azimuth error [deg]")
    axes[2].set_ylabel("LSD [dB]")
    axes[2].set_xlabel("Listener y position [m]")
    axes[0].set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Evaluating dense listener line on {device}")

    runs = []
    for spec in RUNS:
        run = load_run(spec["path"])
        run["label"] = spec["label"]
        runs.append(run)
        print(
            f"{run['run_dir'].name}: lambda_loc={run['lambda_loc']}, "
            f"filters={tuple(run['spk_filters'].shape)}, sofa={run['sofa_path']}"
        )

    sofa_paths = {run["sofa_path"] for run in runs}
    fs_values = {run["fs"] for run in runs}
    if len(sofa_paths) != 1 or len(fs_values) != 1:
        raise ValueError("Runs do not share the same SOFA path / fs; cannot share models.")
    sofa_path = sofa_paths.pop()
    fs = fs_values.pop()

    sofaile = load_sofa(sofa_path)
    gtfb = torch.as_tensor(runs[0]["gtfb"], device=device, dtype=dtype)
    hrirs = get_template_for_llado2025_desena_2020(sofaile, device=device, dtype=dtype)
    print("Building localisation models...")
    llado_model = llado2025_grad_surgery(fs, hrirs, gtfb, device=device, dtype=dtype)
    desena_model = desena2020(fs, hrirs, gtfb, device=device, dtype=dtype)
    stim = make_stim(fs, device, dtype)

    y = np.linspace(Y_START, Y_END, N_LISTENERS)
    lstn_pos = torch.tensor([[0.0, float(yi)] for yi in y], device=device, dtype=dtype)
    spk_pos = torch.as_tensor(runs[0]["spk_pos"], device=device, dtype=dtype)
    target_position = torch.as_tensor(runs[0]["target_position"], device=device, dtype=dtype)

    runs_with_results = []
    for run in runs:
        print(f"Evaluating {run['run_dir'].name}...")
        filters = torch.as_tensor(run["spk_filters"], device=device, dtype=dtype)
        results = evaluate_filters(
            filters, lstn_pos, spk_pos, sofaile, fs, stim, llado_model, desena_model,
            target_position, run["apply_distance_attenuation"], device, dtype,
        )
        plot_dir = run["run_dir"] / "plots"
        plot_dir.mkdir(exist_ok=True)
        npy_dir = run["run_dir"] / "npy"
        np.save(npy_dir / "dense_lstn_pos.npy", lstn_pos.detach().cpu().numpy())
        np.save(npy_dir / "dense_localisation_angles.npy", results["localisation_angles"])
        np.save(npy_dir / "dense_localisation_uncertainty.npy", results["localisation_uncertainty"])
        np.save(npy_dir / "dense_lsd.npy", results["lsd"])
        np.save(npy_dir / "dense_target_angles.npy", results["target_angles"])
        plot_dense_arrows(run, results, lstn_pos, plot_dir / "localisation_arrows_dense.png")
        err = results["localisation_angles"] - results["target_angles"]
        print(
            f"  y in [{y.min():.2f}, {y.max():.2f}] m | "
            f"az [{results['localisation_angles'].min():.1f}, {results['localisation_angles'].max():.1f}] deg | "
            f"MAE {np.mean(np.abs(err)):.1f} deg | "
            f"LSD mean {results['lsd'].mean():.2f} dB "
            f"[{results['lsd'].min():.2f}, {results['lsd'].max():.2f}]"
        )
        print("  training-listener y (dense eval vs saved plot-time arrows):")
        for pos, saved_az, saved_lsd in zip(run["lstn_pos"], run["localisation_angles"], run["lsd"]):
            i = int(np.argmin(np.abs(y - pos[1])))
            print(
                f"    ({pos[0]:.2f}, {pos[1]:.2f}): "
                f"az {results['localisation_angles'][i]:.1f} deg "
                f"(saved {saved_az:.1f}), "
                f"LSD {results['lsd'][i]:.2f} dB "
                f"(saved {saved_lsd:.2f})"
            )
        runs_with_results.append((run, results))

    target_angle = float(runs[0]["run_config"]["training"]["target_angle"])
    vbap_filters = vbap_impulse_filters(runs[0]["spk_pos"], target_angle)
    print(
        f"Evaluating VBAP at {target_angle:g} deg, gains={np.round(vbap_filters[:, 0], 4).tolist()}..."
    )
    vbap_results = evaluate_filters(
        torch.as_tensor(vbap_filters, device=device, dtype=dtype),
        lstn_pos, spk_pos, sofaile, fs, stim, llado_model, desena_model,
        target_position, runs[0]["apply_distance_attenuation"], device, dtype,
    )
    vbap_run = {"label": "VBAP", "spk_pos": runs[0]["spk_pos"], "target_position": runs[0]["target_position"]}
    vbap_err = vbap_results["localisation_angles"] - vbap_results["target_angles"]
    print(
        f"  y in [{y.min():.2f}, {y.max():.2f}] m | "
        f"az [{vbap_results['localisation_angles'].min():.1f}, {vbap_results['localisation_angles'].max():.1f}] deg | "
        f"MAE {np.mean(np.abs(vbap_err)):.1f} deg | "
        f"LSD mean {vbap_results['lsd'].mean():.2f} dB "
        f"[{vbap_results['lsd'].min():.2f}, {vbap_results['lsd'].max():.2f}]"
    )
    comparison_dir = ROOT / "outputs/cagrad"
    np.save(comparison_dir / "dense_vbap_lstn_pos.npy", lstn_pos.detach().cpu().numpy())
    np.save(comparison_dir / "dense_vbap_localisation_angles.npy", vbap_results["localisation_angles"])
    np.save(comparison_dir / "dense_vbap_localisation_uncertainty.npy", vbap_results["localisation_uncertainty"])
    np.save(comparison_dir / "dense_vbap_lsd.npy", vbap_results["lsd"])
    np.save(comparison_dir / "dense_vbap_target_angles.npy", vbap_results["target_angles"])
    plot_dense_arrows(vbap_run, vbap_results, lstn_pos, comparison_dir / "localisation_arrows_dense_vbap.png")
    runs_with_results.append((vbap_run, vbap_results))

    comparison_path = ROOT / "outputs/cagrad/dense_listener_line_lam050_lam025_lam000.png"
    plot_comparison(
        runs_with_results, y, comparison_path,
        title="Dense virtual-listener line at $x=0$ (100 ms white noise)",
    )
    for run, _ in runs_with_results:
        if "run_dir" not in run:
            continue
        plot_comparison(
            runs_with_results, y, run["run_dir"] / "plots" / "localisation_dense_vs_y.png",
            title="Dense virtual-listener line at $x=0$ (100 ms white noise)",
        )
    print(f"Wrote {comparison_path}")


def replot_saved_dense_arrows():
    """Redraw arrow plots from saved dense arrays; does not overwrite the original 3-listener figures."""
    for spec in RUNS:
        run = load_run(spec["path"])
        run["label"] = spec["label"]
        npy = run["run_dir"] / "npy"
        lstn_pos = np.load(npy / "dense_lstn_pos.npy")
        results = {
            "localisation_angles": np.load(npy / "dense_localisation_angles.npy"),
            "localisation_uncertainty": np.load(npy / "dense_localisation_uncertainty.npy"),
            "lsd": np.load(npy / "dense_lsd.npy"),
            "target_angles": np.load(npy / "dense_target_angles.npy"),
        }
        out = run["run_dir"] / "plots" / "localisation_arrows_dense.png"
        plot_dense_arrows(run, results, lstn_pos, out)
        print(f"Wrote {out} ({N_ARROW_LISTENERS} listeners)")

    comparison_dir = ROOT / "outputs/cagrad"
    vbap_lstn = np.load(comparison_dir / "dense_vbap_lstn_pos.npy")
    vbap_results = {
        "localisation_angles": np.load(comparison_dir / "dense_vbap_localisation_angles.npy"),
        "localisation_uncertainty": np.load(comparison_dir / "dense_vbap_localisation_uncertainty.npy"),
        "lsd": np.load(comparison_dir / "dense_vbap_lsd.npy"),
        "target_angles": np.load(comparison_dir / "dense_vbap_target_angles.npy"),
    }
    vbap_run = {
        "label": "VBAP",
        "spk_pos": np.load(RUNS[0]["path"] / "npy" / "spk_pos.npy"),
        "target_position": np.load(RUNS[0]["path"] / "npy" / "target_position.npy"),
    }
    vbap_out = comparison_dir / "localisation_arrows_dense_vbap.png"
    plot_dense_arrows(vbap_run, vbap_results, vbap_lstn, vbap_out)
    print(f"Wrote {vbap_out} ({N_ARROW_LISTENERS} listeners)")

    runs_with_results = []
    y = None
    for spec in RUNS:
        run = load_run(spec["path"])
        run["label"] = spec["label"]
        npy = run["run_dir"] / "npy"
        lstn_pos = np.load(npy / "dense_lstn_pos.npy")
        if y is None:
            y = np.asarray(lstn_pos).reshape(-1, 2)[:, 1]
        results = {
            "localisation_angles": np.load(npy / "dense_localisation_angles.npy"),
            "localisation_uncertainty": np.load(npy / "dense_localisation_uncertainty.npy"),
            "lsd": np.load(npy / "dense_lsd.npy"),
            "target_angles": np.load(npy / "dense_target_angles.npy"),
        }
        runs_with_results.append((run, results))
    runs_with_results.append((vbap_run, vbap_results))
    comparison_path = comparison_dir / "dense_listener_line_lam050_lam025_lam000.png"
    plot_comparison(
        runs_with_results, y, comparison_path,
        title="Dense virtual-listener line at $x=0$ (100 ms white noise)",
    )
    print(f"Wrote {comparison_path}")
    for run, _ in runs_with_results:
        if "run_dir" not in run:
            continue
        plot_comparison(
            runs_with_results, y, run["run_dir"] / "plots" / "localisation_dense_vs_y.png",
            title="Dense virtual-listener line at $x=0$ (100 ms white noise)",
        )


if __name__ == "__main__":
    if "--replot-arrows" in sys.argv:
        replot_saved_dense_arrows()
    else:
        main()
