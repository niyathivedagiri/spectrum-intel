"""Reproducible experiment runs.

Each run gets its own folder that is never overwritten:

    results/experiments/<name>/<YYYYmmdd-HHMMSS>-seed<seed>/
        config.json    parameters, seed, software versions
        metrics.json   numbers produced by the run
        *.csv          tables (one row per measurement point)
        *.png          plots

The validated results of the original project (results/*.png, results/metrics.json,
frozen copy in results/validated/) are left untouched.
"""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "results" / "experiments"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    return str(o)


def new_run(name: str, config: dict, seed: int, base: Path | None = None) -> Path:
    """Create a fresh run folder and write config.json into it."""
    base = base or EXPERIMENTS
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = base / name / f"{stamp}-seed{seed}"
    k = 1
    while run.exists():                         # never overwrite, even within one second
        run = base / name / f"{stamp}-seed{seed}-{k}"
        k += 1
    run.mkdir(parents=True)
    meta = {"experiment": name, "seed": seed, "created": stamp, "config": config,
            "versions": {"python": platform.python_version(), "numpy": np.__version__}}
    (run / "config.json").write_text(json.dumps(meta, indent=2, default=_jsonable))
    return run


def save_metrics(run: Path, metrics: dict) -> Path:
    path = run / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=_jsonable))
    return path


def save_csv(run: Path, filename: str, rows: list[dict]) -> Path:
    path = run / filename
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: _jsonable(v) if isinstance(v, (np.generic, np.ndarray, bytes)) else v
                        for k, v in r.items()})
    return path


def latest_run(name: str, base: Path | None = None) -> Path | None:
    folder = (base or EXPERIMENTS) / name
    runs = sorted(p for p in folder.glob("*") if p.is_dir()) if folder.exists() else []
    return runs[-1] if runs else None
