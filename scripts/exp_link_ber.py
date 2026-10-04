"""Experiment: does the end-to-end receiver reach textbook BER?

Hypothesis: with synchronisation, frequency correction and pilot-based phase
tracking all estimated from the received signal (nothing known in advance except
the preamble/pilots), the measured bit error rate of BPSK, QPSK and 16-QAM in
AWGN stays within ~1 dB of the theoretical curves for coherent detection.

    python scripts/exp_link_ber.py               # ~40 s
    python scripts/exp_link_ber.py --frames 300  # tighter error bars

Output: results/experiments/link_ber/<run>/  (config, metrics.json, ber.csv, plots)
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel import experiments as ex
from spectrum_intel.metrics import (bit_errors, ebn0_from_snr, esn0_from_snr, theory_ber, theory_ser,
                                    wilson_interval)
from spectrum_intel.modulation import LINK_MODULATIONS, bits_per_symbol, nearest_index, symbol_indices
from spectrum_intel.packet import Packet
from spectrum_intel.payload import bytes_to_bits
from spectrum_intel.plots import BLUE, MUTED, ORANGE, AQUA, save
from spectrum_intel.receiver import receive
from spectrum_intel.transmitter import HEADER_BITS, build_frame

COLOURS = {"bpsk": BLUE, "qpsk": ORANGE, "qam16": AQUA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100, help="frames per SNR point")
    ap.add_argument("--payload", type=int, default=64, help="data bytes per packet")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    snrs = np.arange(-8, 13, 1.0)
    run = ex.new_run("link_ber", {"frames_per_point": a.frames, "payload_bytes": a.payload,
                                  "snr_per_sample_db": snrs.tolist(), "channel": "AWGN, random phase+delay"},
                     a.seed)
    rng = np.random.default_rng(a.seed)
    rows = []
    for mod in LINK_MODULATIONS:
        k = bits_per_symbol(mod)
        for snr in snrs:
            be = nb = se = ns = det = hdr = crc = 0
            est = []
            for _ in range(a.frames):
                data = bytes(rng.integers(0, 256, a.payload, dtype=np.uint8))
                block = Packet(0, 1, data).to_bytes()
                frame = build_frame(block, mod)
                rx, _ = ch.awgn_link(frame.iq, snr, rng)
                res = receive(rx)
                det += res.detected
                if res.detected:
                    est.append(res.snr_est_db)
                if not res.header_ok:
                    continue
                hdr += 1
                tx_bits = bytes_to_bits(block)
                be += bit_errors(tx_bits, res.payload_bits)
                nb += len(tx_bits)
                rx_pay = res.data_symbols[HEADER_BITS:]
                se += int(np.sum(nearest_index(rx_pay, mod) != symbol_indices(tx_bits, mod)))
                ns += len(rx_pay)
                crc += Packet.from_bytes(res.payload)[1]
            lo, hi = wilson_interval(be, nb)
            rows.append({
                "mod": mod, "snr_db": float(snr), "esn0_db": float(esn0_from_snr(snr)),
                "ebn0_db": float(ebn0_from_snr(snr, k)), "frames": a.frames,
                "ber": be / nb if nb else float("nan"), "ber_ci_low": lo, "ber_ci_high": hi,
                "ber_theory": float(theory_ber(mod, ebn0_from_snr(snr, k))),
                "ser": se / ns if ns else float("nan"),
                "ser_theory": float(theory_ser(mod, esn0_from_snr(snr))),
                "bits": nb, "bit_errors": be, "detect_rate": det / a.frames, "header_rate": hdr / a.frames,
                "packet_success": crc / a.frames,
                "snr_est_mean": float(np.mean(est)) if est else float("nan"),
                "snr_est_std": float(np.std(est)) if est else float("nan"),
            })
            print(f"  {mod:5s} SNR {snr:5.1f} dB  BER {rows[-1]['ber']:.2e} (theory {rows[-1]['ber_theory']:.2e})"
                  f"  packets ok {100 * rows[-1]['packet_success']:5.1f}%")
    ex.save_csv(run, "ber.csv", rows)

    # ---- implementation loss at BER = 1e-3 -------------------------------------------------
    def ebn0_at(ber_vals, ebn0, target=1e-3):
        b = np.array(ber_vals)
        ok = np.where((b[:-1] > target) & (b[1:] <= target))[0]
        if not len(ok):
            return None
        i = ok[0]
        x0, x1, y0, y1 = ebn0[i], ebn0[i + 1], np.log10(b[i]), np.log10(max(b[i + 1], 1e-12))
        return float(x0 + (np.log10(target) - y0) * (x1 - x0) / (y1 - y0))

    loss = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod in LINK_MODULATIONS:
        r = [x for x in rows if x["mod"] == mod]
        eb = np.array([x["ebn0_db"] for x in r])
        ber = np.array([x["ber"] for x in r])
        lo = np.array([x["ber_ci_low"] for x in r])
        hi = np.array([x["ber_ci_high"] for x in r])
        fine = np.linspace(eb.min(), eb.max(), 200)
        ax.semilogy(fine, theory_ber(mod, fine), "-", color=COLOURS[mod], lw=1, alpha=0.6)
        m = ber > 0
        ax.errorbar(eb[m], ber[m], yerr=[ber[m] - lo[m], hi[m] - ber[m]], fmt="o", ms=4,
                    color=COLOURS[mod], label=f"{mod.upper()} measured", capsize=2)
        meas, th = ebn0_at(ber, eb), ebn0_at(theory_ber(mod, fine), fine)
        loss[mod] = {"ebn0_at_1e-3_measured": meas, "ebn0_at_1e-3_theory": th,
                     "implementation_loss_db": (meas - th) if meas is not None and th is not None else None}
    ax.plot([], [], "-", color=MUTED, label="theory (lines)")
    ax.set_ylim(1e-5, 0.5)
    ax.set_xlim(-6, 15)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("bit error rate")
    ax.set_title("End-to-end link BER vs theory (AWGN, all sync estimated)")
    ax.legend(fontsize=8)
    save(fig, run / "ber_vs_ebn0.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for mod in LINK_MODULATIONS:
        r = [x for x in rows if x["mod"] == mod]
        ax.plot([x["snr_db"] for x in r], [100 * x["packet_success"] for x in r], "o-", color=COLOURS[mod],
                ms=4, label=f"{mod.upper()} ({a.payload} B packets)")
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("packets passing CRC (%)")
    ax.set_title("Packet success vs SNR")
    ax.legend(fontsize=8)
    save(fig, run / "packet_success_vs_snr.png")

    est = [(x["snr_db"], x["snr_est_mean"]) for x in rows if x["mod"] == "qpsk"]
    ex.save_metrics(run, {"implementation_loss": loss,
                          "snr_estimator_bias_db_qpsk": {f"{s:g}": e - s for s, e in est}})
    print("Implementation loss at BER 1e-3:", {m: v["implementation_loss_db"] for m, v in loss.items()})
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
