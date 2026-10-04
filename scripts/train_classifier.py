"""Train the CNN signal classifier and measure it on an unseen test set.

    python scripts/train_classifier.py              # baseline model (~10 min on a laptop CPU)
    python scripts/train_classifier.py --quick      # small, fast run to check everything works
    python scripts/train_classifier.py --evaluate-only   # re-test the saved model, redraw figures

Saves models/cnn_baseline.pt and results/classifier_*.png, results/metrics.json.
"""
import argparse

import _common  # noqa: F401  (path setup)
import matplotlib.pyplot as plt
import numpy as np
from _common import BASELINE_MODEL, RESULTS, update_metrics

from spectrum_intel import classifier as cl
from spectrum_intel.dataset import make_dataset
from spectrum_intel.plots import BLUE, INK, MUTED, save
from spectrum_intel.signals import CLASSES


def plot_accuracy(acc_by_snr: dict, path, title):
    snrs = sorted(acc_by_snr)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(snrs, [100 * acc_by_snr[s] for s in snrs], "o-", color=BLUE, lw=2, ms=5)
    ax.axhline(100 / len(CLASSES), color=MUTED, ls="--", lw=1)
    ax.text(snrs[-1], 100 / len(CLASSES) + 2, "random guessing", ha="right", color=MUTED, fontsize=9)
    ax.set_ylim(0, 102)
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(title)
    save(fig, path)


def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.per_class, a.epochs = 200, 2

    print("Generating test data (different seed = never seen in training) ...")
    Xte, yte, ste = make_dataset(max(a.per_class // 4, 100), seed=1)
    if a.evaluate_only:
        model = cl.load_model(BASELINE_MODEL)
        history = _common.json.loads(_common.METRICS.read_text())["classifier"]["history"]
        n_train = a.per_class * len(CLASSES)
    else:
        print("Generating training data ...")
        X, y, _ = make_dataset(a.per_class, seed=0)
        n_train = len(y)
        print(f"Training on {n_train} examples, device = {cl.get_device()}")
        model, history = cl.train_model(X, y, epochs=a.epochs)
        cl.save_model(model, BASELINE_MODEL, {"per_class": a.per_class, "epochs": a.epochs, "cfo_max": 0.0})

    pred = cl.predict(model, Xte)
    acc = float(np.mean(pred == yte))
    by_snr = cl.accuracy_by_snr(yte, pred, ste)
    cm_all = cl.confusion_matrix(yte, pred)
    hi = ste >= 0
    cm_hi = cl.confusion_matrix(yte[hi], pred[hi])
    print(f"Test accuracy: {100 * acc:.1f}%  (SNR >= 0 dB: {100 * np.mean(pred[hi] == yte[hi]):.1f}%)")

    plot_accuracy(by_snr, RESULTS / "classifier_accuracy_vs_snr.png",
                  f"CNN classifier: {100 * np.mean(pred[hi] == yte[hi]):.1f}% accurate at SNR ≥ 0 dB")
    plot_confusion(cm_hi, RESULTS / "classifier_confusion.png", "Confusion matrix (%), test set, SNR ≥ 0 dB")
    update_metrics("classifier", {
        "train_examples": int(n_train), "test_examples": int(len(yte)), "epochs": a.epochs,
        "test_accuracy": acc, "test_accuracy_snr_ge_0": float(np.mean(pred[hi] == yte[hi])),
        "accuracy_by_snr": by_snr, "confusion_all_snr": cm_all.round(4).tolist(),
        "confusion_snr_ge_0": cm_hi.round(4).tolist(), "classes": list(CLASSES), "history": history,
    })


if __name__ == "__main__":
    main()
