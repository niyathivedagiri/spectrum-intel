"""Experiment: the link scorecard - every metric vs SNR, packet size, and classification of our own frames.

Part 1  BER, SER, packet error rate, throughput, goodput, spectral efficiency and
        latency for BPSK, QPSK and 16-QAM across SNR.
        Hypothesis: no single modulation maximises goodput at every SNR; the best
        one changes with SNR (this is the measured basis for adaptive modulation).
Part 2  Packet-size trade-off (QPSK).
        Hypothesis: long packets waste less airtime on fixed overhead but fail more
        often, so the goodput-optimal packet size grows with SNR.
Part 3  Can the spectrum classifier (trained in Part A of the project, never on
        these frames) recognise the link's own transmissions?

    python scripts/exp_link_metrics.py        # ~2 min
Output: results/experiments/link_metrics/<run>/
"""
import argparse
import json

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from _common import BASELINE_MODEL, RESULTS

from spectrum_intel import channel as ch
from spectrum_intel import classifier as cl
from spectrum_intel import experiments as ex
from spectrum_intel import link
from spectrum_intel import metrics as m
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import Packet
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, save
from spectrum_intel.signals import CLASSES
from spectrum_intel.transmitter import PREAMBLE, SPS, build_frame

COL = {"bpsk": BLUE, "qpsk": ORANGE, "qam16": AQUA}
FS = m.DEFAULT_SAMPLE_RATE_HZ


def part1(rng_seed, snrs, n_packets, trials):
    rows = []
    for mod in LINK_MODULATIONS:
        for snr in snrs:
            reps = []
            for t in range(trials):
                rng = np.random.default_rng([rng_seed, 1, t, int(10 * snr) + 1000, LINK_MODULATIONS.index(mod)])
                data = bytes(rng.integers(0, 256, 64 * n_packets, dtype=np.uint8))
                reps.append(m.link_report(link.send(data, mod, snr, rng), FS))
            bits = sum(r["bits_demodulated"] for r in reps)
            errs = sum(r["raw_bit_errors"] for r in reps)
            lo, hi = m.wilson_interval(errs, bits)
            row = {"mod": mod, "snr_db": float(snr), "packets": sum(r["packets"] for r in reps),
                   "ber": errs / bits if bits else float("nan"), "ber_ci_low": lo, "ber_ci_high": hi}
            for key in ["ser", "packet_error_rate", "packet_success_rate", "frame_detection_rate",
                        "header_success_rate", "throughput_bps", "goodput_bps", "spectral_efficiency_bps_hz",
                        "overhead_fraction", "latency_frame_ms", "latency_message_ms", "phy_rate_bps",
                        "snr_est_db_mean"]:
                row[key] = float(np.nanmean([r[key] for r in reps]))
            rows.append(row)
        print(f"  part 1: {mod} done")
    return rows


def part2(rng_seed, snrs, sizes):
    rows = []
    for size in sizes:
        n_pk = max(20, 1536 // size)
        for snr in snrs:
            rng = np.random.default_rng([rng_seed, 2, size, int(10 * snr) + 1000])
            data = bytes(rng.integers(0, 256, size * n_pk, dtype=np.uint8))
            r = m.link_report(link.send(data, "qpsk", snr, rng, max_payload=size), FS)
            rows.append({"payload_bytes": size, "snr_db": float(snr), "packets": n_pk,
                         "goodput_bps": r["goodput_bps"], "packet_success_rate": r["packet_success_rate"],
                         "overhead_fraction": r["overhead_fraction"], "latency_frame_ms": r["latency_frame_ms"]})
        print(f"  part 2: {size}-byte packets done")
    return rows


def part3(rng_seed, snrs, frames):
    """Classify a 1024-sample window taken from the payload of real link frames."""
    model = cl.load_model(BASELINE_MODEL)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for mod in LINK_MODULATIONS:
        true_idx = CLASSES.index(mod)
        for snr in snrs:
            X = []
            for _ in range(frames):
                block = Packet(0, 1, bytes(rng.integers(0, 256, 64, dtype=np.uint8))).to_bytes()
                f = build_frame(block, mod)
                rx, info = ch.awgn_link(f.iq, snr, rng)
                start = info.delay + (len(PREAMBLE) + 40) * SPS          # inside the payload
                X.append(rx[start:start + 1024])
            pred = cl.predict(model, cl.complex_to_input(np.array(X)))
            rows.append({"mod": mod, "snr_db": float(snr), "accuracy": float(np.mean(pred == true_idx)),
                         "frames": frames})
        print(f"  part 3: {mod} done")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--packets", type=int, default=24)
    ap.add_argument("--trials", type=int, default=2)
    a = ap.parse_args()
    snrs1 = np.arange(-6, 17, 1.0)
    snrs2 = np.arange(-2, 13, 1.0)
    snrs3 = np.arange(-10, 21, 2.0)
    sizes = [16, 64, 256, 1024]
    run = ex.new_run("link_metrics", {"sample_rate_hz": FS, "symbol_rate_baud": FS / SPS,
                                      "occupied_bandwidth_hz": m.occupied_bandwidth_hz(FS),
                                      "part1": {"snr_db": snrs1.tolist(), "packets_per_trial": a.packets,
                                                "trials": a.trials, "payload_bytes": 64},
                                      "part2": {"snr_db": snrs2.tolist(), "payload_sizes": sizes, "mod": "qpsk"},
                                      "part3": {"snr_db": snrs3.tolist(), "frames_per_point": 60,
                                                "model": "models/cnn_baseline.pt"}}, a.seed)
    r1, r2, r3 = part1(a.seed, snrs1, a.packets, a.trials), part2(a.seed, snrs2, sizes), part3(a.seed, snrs3, 60)
    ex.save_csv(run, "scorecard.csv", r1)
    ex.save_csv(run, "packet_size.csv", r2)
    ex.save_csv(run, "classify_link_frames.csv", r3)

    sel = lambda rows, mod, key: np.array([r[key] for r in rows if r["mod"] == mod])  # noqa: E731

    # ---- Figure 1: error rates ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for mod in LINK_MODULATIONS:
        s = sel(r1, mod, "snr_db")
        ber, ser, per = sel(r1, mod, "ber"), sel(r1, mod, "ser"), sel(r1, mod, "packet_error_rate")
        axes[0].semilogy(s[ber > 0], ber[ber > 0], "o-", color=COL[mod], ms=3, label=mod.upper())
        axes[1].semilogy(s[ser > 0], ser[ser > 0], "o-", color=COL[mod], ms=3, label=mod.upper())
        axes[2].plot(s, 100 * per, "o-", color=COL[mod], ms=3, label=mod.upper())
    for ax, t in zip(axes, ["Bit error rate", "Symbol error rate", "Packet error rate (%)"]):
        ax.set_title(t)
        ax.set_xlabel("SNR per sample (dB)")
        ax.legend(fontsize=8)
    axes[0].set_ylim(1e-5, 0.6)
    axes[1].set_ylim(1e-5, 1)
    save(fig, run / "error_rates_vs_snr.png")

    # ---- Figure 2: goodput and spectral efficiency (the adaptive-modulation argument) ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    best = None
    for mod in LINK_MODULATIONS:
        s, g = sel(r1, mod, "snr_db"), sel(r1, mod, "goodput_bps") / 1e3
        thr = sel(r1, mod, "throughput_bps")[0] / 1e3
        axes[0].plot(s, g, "o-", color=COL[mod], ms=3, label=f"{mod.upper()} goodput")
        axes[0].axhline(thr, color=COL[mod], ls=":", lw=1)
        axes[0].text(s[0], thr + 4, f"{mod.upper()} throughput ceiling {thr:.0f} kbit/s", fontsize=7, color=COL[mod])
        best = g if best is None else np.maximum(best, g)
        axes[1].plot(s, sel(r1, mod, "spectral_efficiency_bps_hz"), "o-", color=COL[mod], ms=3, label=mod.upper())
    axes[0].plot(s, best, "-", color=INK, lw=2.5, alpha=0.35, label="best of the three at each SNR")
    axes[0].set_xlabel("SNR per sample (dB)")
    axes[0].set_ylabel("kbit/s delivered correctly")
    axes[0].set_title(f"Goodput at {FS / 1e6:g} MS/s ({FS / SPS / 1e3:g} kBd): the best modulation changes with SNR")
    axes[0].legend(fontsize=8, loc="center right")
    axes[1].set_xlabel("SNR per sample (dB)")
    axes[1].set_ylabel("bit/s/Hz")
    axes[1].set_title(f"Spectral efficiency (goodput / {m.occupied_bandwidth_hz(FS) / 1e3:.0f} kHz occupied)")
    axes[1].legend(fontsize=8)
    save(fig, run / "goodput_vs_snr.png")

    # crossover points: where the goodput-best modulation changes
    best_mod = [max(LINK_MODULATIONS, key=lambda md: sel(r1, md, "goodput_bps")[i]) if best[i] > 0 else None
                for i in range(len(s))]
    switches = [(float(s[i]), best_mod[i - 1], best_mod[i]) for i in range(1, len(s))
                if best_mod[i] != best_mod[i - 1] and best_mod[i - 1] is not None]

    # ---- Figure 3: packet size --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    shades = [MUTED, YELLOW, ORANGE, BLUE]
    for size, c in zip(sizes, shades):
        r = [x for x in r2 if x["payload_bytes"] == size]
        ax.plot([x["snr_db"] for x in r], [x["goodput_bps"] / 1e3 for x in r], "o-", ms=3, color=c,
                label=f"{size} B packets (overhead {100 * r[-1]['overhead_fraction']:.0f}%)")
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("QPSK goodput (kbit/s)")
    ax.set_title("Packet size trade-off: small packets survive low SNR, large ones win at high SNR")
    ax.legend(fontsize=8)
    save(fig, run / "packet_size_tradeoff.png")

    # ---- Figure 4: classifier on our own frames vs validated test set ---------------------------
    val = json.loads((RESULTS / "validated" / "metrics_v1.json").read_text())["classifier"]["accuracy_by_snr"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    vs = sorted(val, key=float)
    ax.plot([float(v) for v in vs], [100 * val[v] for v in vs], "--", color=MUTED, lw=1.5,
            label="validated test set, all 6 classes (existing result)")
    for mod in LINK_MODULATIONS:
        r = [x for x in r3 if x["mod"] == mod]
        ax.plot([x["snr_db"] for x in r], [100 * x["accuracy"] for x in r], "o-", ms=4, color=COL[mod],
                label=f"link frames sent as {mod.upper()}")
    ax.set_ylim(0, 102)
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("classified correctly (%)")
    ax.set_title("The spectrum classifier recognises the link's own transmissions")
    ax.legend(fontsize=8)
    save(fig, run / "classifier_on_link_frames.png")

    def first(rows, key, mod, thr, above=True):
        v, s_ = sel(rows, mod, key), sel(rows, mod, "snr_db")
        idx = np.where(v >= thr if above else v <= thr)[0]
        return float(s_[idx[0]]) if len(idx) else None

    summary = {
        "goodput_best_modulation_switches": [{"at_snr_db": x[0], "from": x[1], "to": x[2]} for x in switches],
        "peak_goodput_kbps": {md: float(sel(r1, md, "goodput_bps").max() / 1e3) for md in LINK_MODULATIONS},
        "overhead_fraction_64B": {md: float(sel(r1, md, "overhead_fraction")[0]) for md in LINK_MODULATIONS},
        "latency_frame_ms_64B": {md: float(sel(r1, md, "latency_frame_ms")[0]) for md in LINK_MODULATIONS},
        "snr_for_per_below_10pct": {md: first(r1, "packet_error_rate", md, 0.1, above=False)
                                    for md in LINK_MODULATIONS},
        "snr_for_ber_below_1e-3": {md: first(r1, "ber", md, 1e-3, above=False) for md in LINK_MODULATIONS},
        "packet_size_best_by_snr": {f"{snr:g}": max(sizes, key=lambda z: next(
            x["goodput_bps"] for x in r2 if x["payload_bytes"] == z and x["snr_db"] == snr)) for snr in snrs2},
        "classifier_accuracy_on_link_frames_snr_ge_0": {
            md: float(np.mean([x["accuracy"] for x in r3 if x["mod"] == md and x["snr_db"] >= 0]))
            for md in LINK_MODULATIONS},
    }
    ex.save_metrics(run, summary)
    print(json.dumps(summary, indent=1))
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
