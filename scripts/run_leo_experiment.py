"""LEO extension: the same classifier, listening from a satellite at 550 km.

    python scripts/run_leo_experiment.py            # trains the Doppler-aware model if missing
    python scripts/run_leo_experiment.py --retrain

1. Model one overhead pass: elevation, range, Doppler and link SNR over time.
2. Test the baseline CNN on Doppler-shifted signals -> it degrades.
3. Train a second CNN with Doppler (frequency offset) augmentation -> it recovers.
4. Replay the pass minute by minute and compare the two models.
Saves results/leo_*.png and the 'leo' section of results/metrics.json.
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _common import BASELINE_MODEL, DOPPLER_MODEL, RESULTS, update_metrics

from spectrum_intel import classifier as cl
from spectrum_intel import leo
from spectrum_intel.dataset import make_dataset
from spectrum_intel.dsp import normalize_power
from spectrum_intel.plots import BLUE, MUTED, ORANGE, save
from spectrum_intel.signals import CLASSES, generate

CARRIER_HZ = 2.0e9          # S-band, as used by 5G NTN
CHANNEL_RATE_HZ = 1.0e6     # native sample rate of one channel
ALTITUDE_M = 550e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()

    # ---- 1. The pass ------------------------------------------------------------
    p = leo.overhead_pass(ALTITUDE_M, min_elevation_deg=10, dt_s=1.0)
    fd = leo.doppler_hz(p.range_rate_mps, CARRIER_HZ)
    snr = leo.link_snr_db(p.range_m, CARRIER_HZ, bandwidth_hz=CHANNEL_RATE_HZ)
    cfo_max = float(np.max(np.abs(fd)) / CHANNEL_RATE_HZ)
    print(f"Pass: {p.time_s[-1] / 60:.1f} min, Doppler ±{np.max(np.abs(fd)) / 1e3:.1f} kHz "
          f"(= ±{cfo_max:.3f} of the sample rate), SNR {snr.min():.1f} to {snr.max():.1f} dB")

    # ---- 2. Doppler-aware model -----------------------------------------------------
    if a.retrain or not DOPPLER_MODEL.exists():
        print("Training the Doppler-aware model (frequency-offset augmentation) ...")
        X, y, _ = make_dataset(a.per_class, seed=0, cfo_max=1.1 * cfo_max)
        model_d, _ = cl.train_model(X, y, epochs=a.epochs)
        cl.save_model(model_d, DOPPLER_MODEL, {"per_class": a.per_class, "epochs": a.epochs,
                                               "cfo_max": 1.1 * cfo_max})
    model_b = cl.load_model(BASELINE_MODEL)
    model_d = cl.load_model(DOPPLER_MODEL)

    # ---- 3. Static test: no Doppler vs Doppler ------------------------------------------
    Xc, yc, sc = make_dataset(400, seed=1)
    Xd, yd, sd = make_dataset(400, seed=2, cfo_max=cfo_max)
    acc = {}
    for mname, m in [("baseline", model_b), ("doppler_trained", model_d)]:
        for tname, (X, y, s) in [("no_doppler", (Xc, yc, sc)), ("doppler", (Xd, yd, sd))]:
            pred = cl.predict(m, X)
            acc[f"{mname}/{tname}"] = {"overall": float(np.mean(pred == y)),
                                       "snr_ge_0": float(np.mean(pred[s >= 0] == y[s >= 0])),
                                       "by_snr": cl.accuracy_by_snr(y, pred, s)}
            print(f"  {mname:16s} on {tname:10s}: {100 * acc[f'{mname}/{tname}']['snr_ge_0']:.1f}% (SNR >= 0 dB)")

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    style = {"baseline/no_doppler": (MUTED, "-", "baseline CNN, no Doppler"),
             "baseline/doppler": (ORANGE, "-", "baseline CNN, with LEO Doppler"),
             "doppler_trained/doppler": (BLUE, "-", "Doppler-trained CNN, with LEO Doppler")}
    for key, (col, ls, lab) in style.items():
        d = acc[key]["by_snr"]
        ks = sorted(d)
        ax.plot(ks, [100 * d[k] for k in ks], "o" + ls, color=col, lw=2, ms=4, label=lab)
    ax.set_ylim(0, 102)
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Doppler from orbit breaks the baseline; training with it fixes it")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, RESULTS / "leo_accuracy_vs_snr.png")

    # ---- 4. Replay the pass ------------------------------------------------------------------
    rng = np.random.default_rng(5)
    steps = np.arange(0, len(p.time_s), 20)
    track = {"time_min": [], "doppler_khz": [], "snr_db": [], "baseline": [], "doppler_trained": []}
    for i in steps:
        cfo = fd[i] / CHANNEL_RATE_HZ
        s = float(np.clip(snr[i], -10, 20))
        X, y = [], []
        for k, label in enumerate(CLASSES):
            for _ in range(60):
                x = normalize_power(generate(label, 1024, s, rng, cfo=cfo))
                X.append(np.stack([x.real, x.imag]))
                y.append(k)
        X, y = np.array(X, dtype=np.float32), np.array(y)
        track["time_min"].append(float(p.time_s[i] / 60))
        track["doppler_khz"].append(float(fd[i] / 1e3))
        track["snr_db"].append(float(snr[i]))
        track["baseline"].append(float(np.mean(cl.predict(model_b, X) == y)))
        track["doppler_trained"].append(float(np.mean(cl.predict(model_d, X) == y)))

    fig, (a1, a3, a2) = plt.subplots(3, 1, figsize=(8.5, 7.4), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 0.8, 1.2]})
    a1.plot(p.time_s / 60, fd / 1e3, color=BLUE, lw=2)
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_ylabel("Doppler (kHz)")
    a1.set_title(f"One pass at {ALTITUDE_M / 1e3:.0f} km, {CARRIER_HZ / 1e9:.0f} GHz: the baseline CNN fails "
                 "except overhead; the Doppler-trained CNN holds")
    a3.plot(p.time_s / 60, snr, color=MUTED, lw=2)
    a3.set_ylabel("link SNR (dB)")
    a2.plot(track["time_min"], [100 * v for v in track["baseline"]], "o-", color=ORANGE, lw=2, ms=4,
            label="baseline CNN")
    a2.plot(track["time_min"], [100 * v for v in track["doppler_trained"]], "o-", color=BLUE, lw=2, ms=4,
            label="Doppler-trained CNN")
    a2.set_ylim(0, 102)
    a2.set_xlabel("time since the satellite rose above 10° (minutes)")
    a2.set_ylabel("accuracy (%)")
    a2.legend(loc="lower center", fontsize=9)
    save(fig, RESULTS / "leo_pass.png")

    update_metrics("leo", {
        "altitude_km": ALTITUDE_M / 1e3, "carrier_ghz": CARRIER_HZ / 1e9,
        "channel_rate_mhz": CHANNEL_RATE_HZ / 1e6, "orbital_speed_kms": leo.orbital_speed(ALTITUDE_M) / 1e3,
        "pass_minutes": float(p.time_s[-1] / 60), "max_doppler_khz": float(np.max(np.abs(fd)) / 1e3),
        "max_cfo_fraction_of_rate": cfo_max,
        "max_doppler_rate_hz_per_s": float(np.max(np.abs(np.diff(fd)))),
        "range_km": [float(p.range_m.min() / 1e3), float(p.range_m.max() / 1e3)],
        "fspl_db": [float(leo.fspl_db(p.range_m.min(), CARRIER_HZ)), float(leo.fspl_db(p.range_m.max(), CARRIER_HZ))],
        "link_snr_db": [float(snr.min()), float(snr.max())],
        "accuracy": acc, "pass_track": track,
    })


if __name__ == "__main__":
    main()
