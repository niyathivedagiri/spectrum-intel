"""Labelled dataset of I/Q examples for training and testing the classifier.

Each example:  X[i] has shape (2, n_samples) float32  ->  row 0 = I, row 1 = Q
               y[i] is the class index into CLASSES
               snr[i] is the SNR in dB it was generated at

Every example is scaled to unit power (like a receiver's automatic gain
control), so the classifier cannot cheat by looking at loudness.

Command line:
    python -m spectrum_intel.dataset --per-class 1000 --out data/generated/dataset.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectrum_intel.dsp import normalize_power
from spectrum_intel.signals import CLASSES, generate

DEFAULT_SNRS = tuple(range(-10, 21, 2))   # -10, -8, ..., +20 dB


def make_dataset(per_class: int = 500, n_samples: int = 1024,
                 snrs: tuple[int, ...] = DEFAULT_SNRS, classes: tuple[str, ...] = CLASSES,
                 seed: int = 0):
    """Return (X, y, snr), shuffled, with the same number of examples per class."""
    rng = np.random.default_rng(seed)
    n = per_class * len(classes)
    X = np.empty((n, 2, n_samples), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    snr = np.empty(n, dtype=np.float32)

    i = 0
    for label_idx, label in enumerate(classes):
        for _ in range(per_class):
            s = float(rng.choice(snrs))
            x = normalize_power(generate(label, n_samples, s, rng))
            X[i, 0], X[i, 1] = x.real, x.imag
            y[i], snr[i] = label_idx, s
            i += 1

    order = rng.permutation(n)
    return X[order], y[order], snr[order]


def save_dataset(path: str | Path, X, y, snr, classes=CLASSES) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, snr=snr, classes=np.array(classes))
    return path


def load_dataset(path: str | Path):
    """Return (X, y, snr, classes)."""
    d = np.load(path)
    return d["X"], d["y"], d["snr"], tuple(str(c) for c in d["classes"])


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the I/Q classification dataset.")
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--samples", type=int, default=1024, help="I/Q samples per example")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/generated/dataset.npz")
    a = p.parse_args(argv)

    X, y, snr = make_dataset(a.per_class, a.samples, seed=a.seed)
    path = save_dataset(a.out, X, y, snr)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved {len(y)} examples x {a.samples} samples to {path} ({size_mb:.1f} MB)")
    print("Classes:", ", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)))
    print(f"SNR range: {snr.min():.0f} to {snr.max():.0f} dB")


if __name__ == "__main__":
    main()
