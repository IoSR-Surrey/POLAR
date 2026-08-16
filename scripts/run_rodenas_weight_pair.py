"""Run two Rodenas CAGrad jobs: loc/col weights (0.75, 0.25) then (0.25, 0.75)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "rodenas" / "weight_pair_075_025.log"
RUNS = [
    ("loc075_col025", [0.75, 0.25]),
    ("loc025_col075", [0.25, 0.75]),
]


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    py = sys.executable

    with open(LOG, "w", encoding="utf-8") as log:
        for run_id, weights in RUNS:
            msg = f"\n===== START {run_id} weights={weights} =====\n"
            print(msg, end="", flush=True)
            log.write(msg)
            log.flush()
            cmd = [
                py,
                str(ROOT / "scripts" / "rodenas_optimisation.py"),
                "--weights",
                str(weights[0]),
                str(weights[1]),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            done = f"===== END {run_id} exit={proc.returncode} =====\n"
            print(done, end="", flush=True)
            log.write(done)
            log.flush()
            if proc.returncode != 0:
                sys.exit(proc.returncode)
    print(f"Both runs finished. Log: {LOG}")


if __name__ == "__main__":
    main()
