"""Wait for current grid cell to finish, then stop the grid driver (no next cell)."""
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "outputs" / "cagrad"
DRIVER_LOG = LOG_DIR / "grid_4param_driver.log"
GRID_LOG = LOG_DIR / "grid_4param_log.json"
HANDOFF = LOG_DIR / "GRID_4PARAM_HANDOFF.md"
CURRENT_ID = "t128_b12_lr0p0050_lam0p10"
CURRENT_LOG = LOG_DIR / f"grid_{CURRENT_ID}.log"


def ps(cmd: str) -> str:
    return subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", cmd],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def pids_matching(pattern: str):
    out = ps(
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
        "ForEach-Object { $_.ProcessId }"
    )
    if not out:
        return []
    return [int(x) for x in out.splitlines() if x.strip().isdigit()]


def kill_pids(pids):
    for pid in pids:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
            check=False,
        )


def current_saved():
    if not CURRENT_LOG.exists():
        return False
    text = CURRENT_LOG.read_text(encoding="utf-8", errors="replace")
    return "Saved outputs to" in text


def last_iter():
    if not CURRENT_LOG.exists():
        return None
    its = re.findall(r"Iteration (\d+)", CURRENT_LOG.read_text(encoding="utf-8", errors="replace"))
    return int(its[-1]) if its else None


def write_handoff(paused_after_path=None):
    status = {}
    if GRID_LOG.exists():
        status = json.loads(GRID_LOG.read_text(encoding="utf-8"))
    done = status.get("n_done")
    skipped = status.get("n_skipped")
    remaining = status.get("n_remaining")
    results = status.get("results") or []
    skipped_list = status.get("skipped") or []

    lines = [
        "# Grid 4-param search — pause / resume handoff",
        "",
        f"Paused at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## What this is",
        "Multi-day factorial-ish search over `n_taps`, `n_bands`, `lr`, `lambda_loc`.",
        "Driver script: `scripts/grid_4param_search.py`",
        "Status log: `outputs/cagrad/grid_4param_log.json`",
        "Driver stdout: `outputs/cagrad/grid_4param_driver.log`",
        "",
        "## Fixed settings",
        "- Distance attenuation: ON",
        "- cagrad_c: 1.0",
        "- num_epochs: 180",
        "- n_samples_per_band: 32",
        "- Losses: llado2025_grad_surgery + LSD via CAGrad",
        "- Centre: 128 taps · 12 bands · lr=0.0035 · low λ",
        "",
        "## Grid levels",
        "- n_taps: [64, 128, 256]",
        "- n_bands: [6, 12, 24] (erb 0.1 for 6b, 0.25 for 12/24)",
        "- lr: [0.0025, 0.0035, 0.005]",
        "- lambda_loc factorial: [0.05, 0.10, 0.15, 0.25]",
        "- Extra high-λ at centre only: 0.50 and 0.90",
        "- Total ~110 cells; near-centre first",
        "",
        "## Pause state",
        f"- Intentionally paused after cell: `{CURRENT_ID}`",
        f"- Last finished path (if known): `{paused_after_path}`",
        f"- Last `grid_4param_log.json` snapshot: n_done={done}, n_skipped={skipped}, n_remaining={remaining}",
        f"- Results recorded: {len(results)}; skipped recorded: {len(skipped_list)}",
        "",
        "## Resume (do this — do not start from scratch)",
        "1. Confirm no leftover `scratch.py` / `grid_4param_search.py` is running.",
        "2. From repo root:",
        "   ```",
        "   conda run -n polar-env --no-capture-output python scripts/grid_4param_search.py > outputs/cagrad/grid_4param_driver.log 2>&1",
        "   ```",
        "   (Or background with `start /B` on Windows as before.)",
        "3. The script **re-scans** `outputs/cagrad/*/json/run_config.json` and **skips** any cell whose",
        "   (taps, bands, lr, lambda, epochs, c, atten) already completed with `spk_filters.npy`.",
        "4. It will **not** redo tonight/centre cells that already finished.",
        "",
        "## Do not",
        "- Do not delete finished `outputs/cagrad/2026-08-*` folders used by the grid.",
        "- Do not wipe `grid_4param_log.json` (helpful but not required for skip logic).",
        "- Do not kill mid-cell unless necessary; wait for `Saved outputs` first.",
        "",
        "## Recent completed cells (from log)",
    ]
    for r in results[-12:]:
        lines.append(
            f"- {r.get('id')}: {r.get('status')} {r.get('elapsed_minutes')} min → {r.get('path')}"
        )
    lines.append("")
    lines.append("## Skipped (already done before / during tonight handoff)")
    for s in skipped_list:
        lines.append(f"- {s.get('id')}: {s.get('path')}")
    lines.append("")
    HANDOFF.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote handoff: {HANDOFF}", flush=True)


def main():
    print(f"Watching for finish of {CURRENT_ID} ...", flush=True)
    while not current_saved():
        print(f"  iter={last_iter()} @ {datetime.now().isoformat(timespec='seconds')}", flush=True)
        # if processes died unexpectedly, stop
        if not pids_matching(r"scratch\\.py") and not pids_matching(r"grid_4param_search"):
            print("No scratch/grid processes; writing handoff and exiting.", flush=True)
            write_handoff(None)
            return
        time.sleep(30)

    # parse saved path
    text = CURRENT_LOG.read_text(encoding="utf-8", errors="replace")
    m = None
    for m in re.finditer(r"Saved outputs to (.+)", text):
        pass
    saved = m.group(1).strip() if m else None
    print(f"Current cell saved: {saved}", flush=True)

    # Wait briefly for driver to record this cell in grid_4param_log.json, then stop
    # before the next cell trains for long. If next scratch already started, kill it too.
    for _ in range(60):
        if GRID_LOG.exists():
            try:
                d = json.loads(GRID_LOG.read_text(encoding="utf-8"))
                ids = {r.get("id") for r in (d.get("results") or [])}
                if CURRENT_ID in ids:
                    print("Driver logged current cell as done.", flush=True)
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(1)
    else:
        print("Driver log not updated yet; pausing anyway.", flush=True)

    time.sleep(2)

    # Kill grid driver tree and any NEW scratch that may have started for the next cell
    grid_pids = pids_matching(r"grid_4param_search")
    scratch_pids = pids_matching(r"scratch\\.py")
    # Also conda wrappers that mention grid
    conda_grid = pids_matching(r"grid_4param")
    print(f"Stopping grid pids={grid_pids + conda_grid} scratch pids={scratch_pids}", flush=True)
    kill_pids(set(grid_pids + conda_grid + scratch_pids))
    time.sleep(3)

    # If a next cell barely started, remove incomplete folder? Don't delete — incomplete won't have full run_config+filters matching skip key usually.
    left = pids_matching(r"scratch\\.py|grid_4param")
    print(f"Remaining matching pids: {left}", flush=True)

    # Refresh status from log if driver updated it
    write_handoff(saved.replace("\\", "/") if saved else None)
    print("Paused after current cell.", flush=True)


if __name__ == "__main__":
    main()
