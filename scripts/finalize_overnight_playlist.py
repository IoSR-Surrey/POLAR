"""After overnight_anchor_sweep finishes, rewrite play_filtered_config.json
with new run folders after a blank separator line.
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAY = ROOT / "config" / "play_filtered_config.json"
LOG = ROOT / "outputs" / "cagrad" / "overnight_anchor_sweep_log.json"

# Paths already in the playlist before tonight's overnight plan.
BASELINE = [
    "outputs/cagrad/2026-08-03_193016",
    "outputs/cagrad/2026-08-03_194802",
    "outputs/cagrad/2026-08-03_201820",
    "outputs/cagrad/2026-08-03_224647",
    "outputs/cagrad/2026-08-03_235020",
    "outputs/cagrad/2026-08-04_005407",
    "outputs/cagrad/2026-08-04_011408",
    "outputs/cagrad/2026-08-04_012143",
    "outputs/cagrad/2026-08-04_092544",
    "outputs/cagrad/2026-08-04_094909",
    "outputs/cagrad/2026-08-04_105222",
    "outputs/cagrad/2026-08-04_111530",
    "outputs/cagrad/2026-08-04_151631",
]


def write_play(old_folders, new_folders):
    seen = set()
    old = []
    for p in old_folders:
        if p and p not in seen:
            old.append(p)
            seen.add(p)
    new = []
    for p in new_folders:
        if p and p not in seen:
            new.append(p)
            seen.add(p)

    lines = ["{", '    "folders": [']
    for i, path in enumerate(old):
        trailing = "," if (i < len(old) - 1 or new) else ""
        lines.append(f'        "{path}"{trailing}')
    if new:
        if old:
            lines.append("")  # blank line separator
        for i, path in enumerate(new):
            trailing = "," if i < len(new) - 1 else ""
            lines.append(f'        "{path}"{trailing}')
    lines.append("    ]")
    lines.append("}")
    lines.append("")
    PLAY.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {PLAY}: {len(old)} baseline + {len(new)} new (blank line between)")


def overnight_running():
    # crude: log finished_at present
    if not LOG.exists():
        return True
    try:
        data = json.loads(LOG.read_text(encoding="utf-8"))
    except Exception:
        return True
    return "finished_at" not in data


def main():
    print("Waiting for overnight sweep to finish...", flush=True)
    while overnight_running():
        time.sleep(60)
    data = json.loads(LOG.read_text(encoding="utf-8"))
    new_paths = [
        r["path"]
        for r in data.get("results", [])
        if r.get("status") == "success" and r.get("path")
    ]
    # Preserve full current playlist as baseline if denser than BASELINE
    try:
        current = json.loads(PLAY.read_text(encoding="utf-8")).get("folders") or []
    except Exception:
        current = []
    # Prefer fixed pre-overnight list so blank line always separates tonight from earlier work
    old = list(BASELINE)
    # If any current folders were only in tonight's runs, ignore for old
    write_play(old, new_paths)
    print("New paths:", new_paths, flush=True)
    print("Skipped:", data.get("skipped"), flush=True)


if __name__ == "__main__":
    main()
