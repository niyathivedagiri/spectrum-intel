"""Energy detector vs CNN: who notices a weak signal first?

    python scripts/evaluate_detection.py

Both look at one channel window (1024 samples) with noise power 1 and decide
"occupied" or "empty". The energy detector is set for 1% false alarms; the CNN
says "occupied" whenever its top class is not 'noise'.
Second test: the real noise level is uncertain by +/-1 dB (it always is in practice).
Saves results/detection_*.png and the 'detection' section of results/metrics.json.
"""
import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _common import BASELINE_MODEL, RESULTS, update_metrics

from spectrum_intel import classifier as cl
from spectrum_intel.detector import EnergyDetector, energy, theoretical_pd
from spectrum_intel.dsp import awgn
from spectrum_intel.plots import BLUE, MUTED, ORANGE, save
from spectrum_intel.signals import CLASSES, clean_signal

N = 1024
PFA = 0.01
SNRS = np.arange(-20, 7, 2)
SIGNAL_CLASSES = [c for c in CLASSES if c != "noise"]
NOISE_IDX = CLASSES.index("noise")


def windows(label, snr_db, m, rng, noise_spread_db=0.0):
    """m channel windows: signal at snr_db over noise of power ~1 (no AGC, like a real channel).

    noise_spread_db > 0: the true noise level of each window varies randomly by up to
    +/- that many dB (temperature, interference, gain drift) while the detector still
    assumes power 1. The signal level stays fixed, so the SNR refers to nominal noise.
    """
    out = np.empty((m, N), dtype=complex)
    for i in range(m):
        level = 10 ** (rng.uniform(-noise_spread_db, noise_spread_db) / 10)
        out[i] = awgn(N, level, rng)
        if label != "noise":
            phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
            out[i] += np.sqrt(10 ** (snr_db / 10)) * phase * clean_signal(label, N, rng)
    return out


def main(per_point=200):
    rng = np.random.default_rng(10)
    model = cl.load_model(BASELINE_MODEL)
    ed = EnergyDetector(noise_power=1.0, n_samples=N, pfa=PFA)

    noise = windows("noise", 0, 3000, rng)
    p_noise_cnn = cl.predict_proba(model, cl.complex_to_input(noise))
    pfa_ed = float(np.mean(ed.detect(noise)))
    pfa_cnn = float(np.mean(np.argmax(p_noise_cnn, 1) != NOISE_IDX))
    print(f"False alarms on empty channels: energy {100 * pfa_ed:.1f}%, CNN {100 * pfa_cnn:.1f}%")

    pd_ed, pd_cnn, roc_sig = {}, {}, {}
    for s in SNRS:
        sig = np.concatenate([windows(c, s, per_point, rng) for c in SIGNAL_CLASSES])
        probs = cl.predict_proba(model, cl.complex_to_input(sig))
        pd_ed[int(s)] = float(np.mean(ed.detect(sig)))
        pd_cnn[int(s)] = float(np.mean(np.argmax(probs, 1) != NOISE_IDX))
        if s == -12:
            roc_sig = {"energy": energy(sig), "cnn": 1 - probs[:, NOISE_IDX]}
        print(f"  SNR {s:4d} dB  Pd energy {pd_ed[int(s)]:.2f}   Pd CNN {pd_cnn[int(s)]:.2f}")

    # --- Pd vs SNR -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(SNRS, [100 * theoretical_pd(s, N, PFA) for s in SNRS], "--", color=MUTED, lw=1.2,
            label="energy detector, theory")
    ax.plot(SNRS, [100 * pd_ed[int(s)] for s in SNRS], "o-", color=ORANGE, lw=2, ms=5,
            label=f"energy detector (false alarms {100 * pfa_ed:.1f}%)")
    ax.plot(SNRS, [100 * pd_cnn[int(s)] for s in SNRS], "o-", color=BLUE, lw=2, ms=5,
            label=f"CNN (false alarms {100 * pfa_cnn:.1f}%)")
    ax.set_xlabel("SNR in the channel (dB)")
    ax.set_ylabel("signals detected (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Detecting a signal in one channel: energy detector vs CNN")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, RESULTS / "detection_pd_vs_snr.png")

    # --- ROC at -12 dB (threshold-free comparison) ----------------------------
    def roc(score_noise, score_sig):
        th = np.quantile(score_noise, np.linspace(0, 1, 400))
        return (np.array([np.mean(score_noise > t) for t in th]),
                np.array([np.mean(score_sig > t) for t in th]))

    fa_e, pd_e = roc(energy(noise), roc_sig["energy"])
    fa_c, pd_c = roc(1 - p_noise_cnn[:, NOISE_IDX], roc_sig["cnn"])
    auc_e, auc_c = float(np.trapezoid(pd_e[::-1], fa_e[::-1])), float(np.trapezoid(pd_c[::-1], fa_c[::-1]))
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(fa_e, pd_e, color=ORANGE, lw=2, label=f"energy detector (AUC {auc_e:.2f})")
    ax.plot(fa_c, pd_c, color=BLUE, lw=2, label=f"CNN (AUC {auc_c:.2f})")
    ax.plot([0, 1], [0, 1], ":", color=MUTED)
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("detection rate")
    ax.set_title("ROC at −12 dB SNR")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, RESULTS / "detection_roc.png")

    # --- Noise uncertainty +/-1 dB: the "SNR wall" ------------------------------
    U = 1.0
    ed_worst = EnergyDetector(noise_power=10 ** (U / 10), n_samples=N, pfa=PFA)   # safe threshold
    noise_u = windows("noise", 0, 3000, rng, U)
    pfa_u = {"energy_nominal_threshold": float(np.mean(ed.detect(noise_u))),
             "energy_worst_case_threshold": float(np.mean(ed_worst.detect(noise_u))),
             "cnn": float(np.mean(cl.predict(model, cl.complex_to_input(noise_u)) != NOISE_IDX))}
    pd_u_ed, pd_u_cnn = {}, {}
    for s in SNRS:
        sig = np.concatenate([windows(c, s, per_point // 2, rng, U) for c in SIGNAL_CLASSES])
        pd_u_ed[int(s)] = float(np.mean(ed_worst.detect(sig)))
        pd_u_cnn[int(s)] = float(np.mean(cl.predict(model, cl.complex_to_input(sig)) != NOISE_IDX))
    print(f"With +/-{U:g} dB noise uncertainty: energy false alarms jump to "
          f"{100 * pfa_u['energy_nominal_threshold']:.0f}% unless the threshold is raised; CNN {100 * pfa_u['cnn']:.1f}%")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(SNRS, [100 * pd_ed[int(s)] for s in SNRS], "--", color=ORANGE, lw=1.2, alpha=0.6,
            label="energy detector, exact noise level")
    ax.plot(SNRS, [100 * pd_u_ed[int(s)] for s in SNRS], "o-", color=ORANGE, lw=2, ms=5,
            label=f"energy detector, noise known to ±{U:g} dB")
    ax.plot(SNRS, [100 * pd_u_cnn[int(s)] for s in SNRS], "o-", color=BLUE, lw=2, ms=5,
            label=f"CNN, noise known to ±{U:g} dB")
    ax.set_xlabel("SNR in the channel (dB, relative to nominal noise)")
    ax.set_ylabel("signals detected (%)")
    ax.set_ylim(0, 102)
    ax.set_title("With a ±1 dB uncertain noise level, the energy detector needs about 6 dB more SNR")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, RESULTS / "detection_noise_uncertainty.png")

    def snr_for(pd, target=0.9):
        ok = [s for s in SNRS if pd[int(s)] >= target]
        return int(min(ok)) if ok else None

    update_metrics("detection", {
        "window_samples": N, "target_pfa": PFA, "pfa_energy": pfa_ed, "pfa_cnn": pfa_cnn,
        "pd_energy": pd_ed, "pd_cnn": pd_cnn,
        "snr_for_90pct_detection": {"energy": snr_for(pd_ed), "cnn": snr_for(pd_cnn)},
        "roc_auc_at_minus12db": {"energy": auc_e, "cnn": auc_c},
        "noise_uncertainty_db": U, "pfa_with_uncertainty": pfa_u,
        "pd_energy_uncertain": pd_u_ed, "pd_cnn_uncertain": pd_u_cnn,
        "snr_for_90pct_detection_uncertain": {"energy": snr_for(pd_u_ed), "cnn": snr_for(pd_u_cnn)},
    })


if __name__ == "__main__":
    main()
