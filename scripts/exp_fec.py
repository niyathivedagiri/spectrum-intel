"""Experiment: what does forward error correction buy, what does it cost, and when does it fail?

Code: convolutional, K = 7, generators 133/171 (the Wi-Fi code), rate 1/2, punctured to 2/3 and 3/4,
Viterbi decoding with soft (LLR) or hard inputs, pseudo-random interleaver over each packet.
Everything goes through the real transmitter / channel / receiver (64-byte packets, random phase
and arrival time). The PHY header stays uncoded BPSK, as before.

Hypotheses (written before running)
  H1  Coding gain. In AWGN, rate 1/2 with soft decoding lowers the SNR needed for 10% packet
      errors by several dB for every modulation (theory for this code: up to 7 dB at very low
      error rates, less at a 10% packet error rate).
  H2  Soft beats hard decoding by about 2 dB (textbook value for Viterbi decoding).
  H3  Coding costs airtime: rate 1/2 halves the data rate. The best goodput at each SNR comes from
      a different modulation + code pair; coded schemes win at low and medium SNR, uncoded 16-QAM
      at high SNR. Higher rates (2/3, 3/4) fill the gap between them.
  H4  Bursty errors. When errors come in bursts (fast fading inside a packet, a pulsed jammer) the
      interleaver is essential: without it the code gains little; with it, coding gains more than
      in AWGN. When the whole packet sits in one fade (slow fading) there is nothing to spread and
      the gain shrinks.

Added after the first run (reported as such): the PHY header is uncoded, so below some SNR the
header, not the payload, is lost. Every point therefore also reports the payload-only packet error
rate (failures among packets whose header was received) = what the code itself achieves. And the
first jammer (50 us bursts every 500 us) turned out too short to test the interleaver, so a
long-burst jammer (500 us every 5 ms, one burst per frame) was added.
The first run also showed soft decoding failing badly under jammer bursts (worse than hard):
the LLRs used one noise level measured on the preamble, so symbols hit by a burst looked like
very confident (wrong) bits. The receiver now also estimates the noise locally around each symbol
(fec.local_noise) and uses the larger of the two. The first version is kept as "soft_fixed" and
shown beside it here.

    python scripts/exp_fec.py          # ~5 min on 2 cores
Output: results/experiments/fec/<run>/
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel.channel import ChannelConfig
from spectrum_intel.fec import CODES
from spectrum_intel.link import transfer_packet
from spectrum_intel.metrics import link_report
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import segment
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, save

MODNAME = {"bpsk": "BPSK", "qpsk": "QPSK", "qam16": "16-QAM"}
CODECOL = {"none": INK, "1/2": BLUE, "2/3": AQUA, "3/4": ORANGE}
CODENAME = {"none": "uncoded", "1/2": "rate 1/2", "2/3": "rate 2/3", "3/4": "rate 3/4"}
PARTS = ("fast", "slow", "jam", "jam_long")
BURST = [("uncoded", "none", "soft", True), ("rate 1/2, no interleaver", "1/2", "soft", False),
         ("rate 1/2, hard, interleaved", "1/2", "hard", True),
         ("rate 1/2, soft (fixed noise), interleaved", "1/2", "soft_fixed", True),
         ("rate 1/2, soft, interleaved", "1/2", "soft", True)]
BURSTCOL = {"uncoded": INK, "rate 1/2, no interleaver": YELLOW, "rate 1/2, hard, interleaved": MUTED,
            "rate 1/2, soft (fixed noise), interleaved": ORANGE, "rate 1/2, soft, interleaved": BLUE}


MIN_HEADERS = 30            # payload-only PER is reported only where at least this many headers arrived


class _Run:
    def __init__(self, recs):
        self.packets = recs


def run_point(job):
    """One measurement point: n packets of one scheme at one SNR (and channel). Seeded per point."""
    key, seed, n, mod, code, decoding, inter, snr, chan, pw = job
    rng = np.random.default_rng(seed)
    pk = segment(rng.integers(0, 256, 64 * 16, dtype=np.uint8).tobytes(), 64)
    cfg = None if chan is None else ChannelConfig(**chan)
    recs = [transfer_packet(pk[i % 16], 16, mod, snr, rng, channel=cfg, code=code, decoding=decoding,
                            interleave=inter, pilot_window=pw).record for i in range(n)]
    r = link_report(_Run(recs))
    got = r["packets"] - r["packets_lost"]
    return key, {"per": r["packet_error_rate"], "header_rate": r["header_success_rate"],
                 # payload-only error rate: only meaningful when enough headers arrived
                 "per_payload": r["packets_crc_failed"] / got if got >= MIN_HEADERS else float("nan"),
                 "goodput_kbps": r["goodput_bps"] / 1e3, "ber_raw": r["ber"], "ber_decoded": r["ber_decoded"]}


def snr_at(snrs, per, target=0.1):
    """First SNR where the (monotone-cleaned) PER curve drops to the target, by linear interpolation.

    Points without a measurement (NaN) count as failing. If the curve is already below the target
    at its first measured point, that point is returned: the true value is at or below it.
    """
    p = np.minimum.accumulate(np.nan_to_num(np.asarray(per, float), nan=1.0))
    idx = np.where(p <= target)[0]
    if not len(idx):
        return None
    i = idx[0]
    if i == 0 or np.isnan(per[i - 1]):
        return float(snrs[i])
    x0, x1, y0, y1 = snrs[i - 1], snrs[i], p[i - 1], p[i]
    return float(x0 + (y0 - target) / (y0 - y1) * (x1 - x0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--n", type=int, default=120, help="packets per point (AWGN part)")
    ap.add_argument("--nb", type=int, default=150, help="packets per point (burst part)")
    ap.add_argument("--cache", type=Path, default=None, help="pickle of raw points: reused if it exists, else written")
    a = ap.parse_args()
    S = a.seed
    snrs = np.arange(-10, 17, 1.0)
    fast = {"fading": "rayleigh", "doppler_hz": 500.0}
    slow = {"fading": "rayleigh", "doppler_hz": 0.0}
    fade_snrs = np.arange(0, 25, 3.0)
    sirs = np.arange(-10, 10.1, 2.5)
    run = ex.new_run("fec", {"code": "K=7 (133,171) rate 1/2, punctured 2/3 and 3/4 (802.11 patterns)",
                             "packet_bytes": 64, "header": "uncoded BPSK", "packets_per_point": a.n,
                             "packets_per_point_bursts": a.nb, "fast_fading": fast, "slow_fading": slow,
                             "fading_pilot_window": 3, "jammer": "pulsed, duty 0.1, QPSK at SNR 12 dB"}, S)

    # ---- jobs ----------------------------------------------------------------------------------
    schemes = [(m, c, "soft") for m in LINK_MODULATIONS for c in CODES] + [(m, "1/2", "hard") for m in LINK_MODULATIONS]
    jobs = []
    for si, (m, c, d) in enumerate(schemes):
        for i, s in enumerate(snrs):
            jobs.append((("awgn", m, c, d, i), [S, 1, si, i], a.n, m, c, d, True, float(s), None, 7))
    for part, chan, xs in [("fast", fast, fade_snrs), ("slow", slow, fade_snrs), ("jam", None, sirs),
                           ("jam_long", None, sirs)]:
        for bi, (name, c, d, inter) in enumerate(BURST):
            for i, x in enumerate(xs):
                if part.startswith("jam"):
                    job_chan, snr = {"jammer": "pulsed", "duty": 0.1, "sir_db": float(x),
                                     "jam_period_us": 500.0 if part == "jam" else 5000.0}, 12.0
                else:
                    job_chan, snr = chan, float(x)
                jobs.append(((part, name, i), [S, 2, PARTS.index(part), bi, i], a.nb, "qpsk", c, d,
                             inter, snr, job_chan, 7 if part.startswith("jam") else 3))
    if a.cache and a.cache.exists():
        out = pickle.loads(a.cache.read_bytes())
    else:
        with Pool(2) as pool:
            out = dict(pool.map(run_point, jobs, chunksize=4))
        if a.cache:
            a.cache.write_bytes(pickle.dumps(out))
    rows, summary = [], {}

    # ---- H1/H2: PER vs SNR in AWGN -----------------------------------------------------------------
    curves = {(m, c, d): [out[("awgn", m, c, d, i)] for i in range(len(snrs))] for m, c, d in schemes}
    for (m, c, d), v in curves.items():
        for s, r in zip(snrs, v):
            rows.append({"part": "awgn", "mod": m, "code": c, "decoding": d, "snr_db": float(s), **r})
    need = {f"{m} {c} {d}": snr_at(snrs, [r["per"] for r in v]) for (m, c, d), v in curves.items()}
    need_pay = {f"{m} {c} {d}": snr_at(snrs, [r["per_payload"] for r in v]) for (m, c, d), v in curves.items()}
    gain_pay = {m: {"soft_1/2": need_pay[f"{m} none soft"] - need_pay[f"{m} 1/2 soft"],
                    "soft_minus_hard": need_pay[f"{m} 1/2 hard"] - need_pay[f"{m} 1/2 soft"]} for m in LINK_MODULATIONS}
    summary["snr_for_per_10pct_payload_only"] = need_pay
    summary["coding_gain_db_payload_only"] = gain_pay
    gain = {m: {"soft_1/2": need[f"{m} none soft"] - need[f"{m} 1/2 soft"],
                "hard_1/2": need[f"{m} none soft"] - need[f"{m} 1/2 hard"],
                "soft_minus_hard": need[f"{m} 1/2 hard"] - need[f"{m} 1/2 soft"]} for m in LINK_MODULATIONS}
    summary["snr_for_per_10pct"] = need
    summary["coding_gain_db"] = gain
    for m in LINK_MODULATIONS:
        print(f"  {m:5s} 10% PER at " + "  ".join(f"{c}:{need[f'{m} {c} soft']:+.1f}" for c in CODES) +
              f"  hard 1/2:{need[f'{m} 1/2 hard']:+.1f}  -> gain soft {gain[m]['soft_1/2']:.1f} dB, "
              f"hard {gain[m]['hard_1/2']:.1f} dB")
        print(f"        payload only: " + "  ".join(f"{c}:{need_pay[f'{m} {c} soft']:+.1f}" for c in CODES) +
              f"  hard 1/2:{need_pay[f'{m} 1/2 hard']:+.1f}  -> gain soft {gain_pay[m]['soft_1/2']:.1f} dB, "
              f"soft-hard {gain_pay[m]['soft_minus_hard']:.1f} dB")
    hdr = [r["header_rate"] for r in curves[("bpsk", "1/2", "soft")]]
    summary["bpsk_header_success"] = dict(zip(snrs.tolist(), hdr))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, m in zip(axes, LINK_MODULATIONS):
        for c in CODES:
            ax.semilogy(snrs, [max(r["per"], 1 / (2 * a.n)) for r in curves[(m, c, "soft")]], "o-", ms=3,
                        color=CODECOL[c], label=f"{CODENAME[c]}{'' if c == 'none' else ' (soft)'}")
        ax.semilogy(snrs, [max(r["per"], 1 / (2 * a.n)) for r in curves[(m, "1/2", "hard")]], "s--", ms=3,
                    color=BLUE, alpha=0.6, label="rate 1/2 (hard)")
        ax.semilogy(snrs, [max(r["per_payload"], 1 / (2 * a.n)) for r in curves[(m, "1/2", "soft")]], ":",
                    color=BLUE, lw=1.6, label="rate 1/2 soft, payload only")
        ax.semilogy(snrs, [max(1 - r["header_rate"], 1 / (2 * a.n)) for r in curves[(m, "1/2", "soft")]], "-",
                    color=MUTED, lw=4, alpha=0.35, label="frame or PHY header lost (uncoded)")
        ax.axhline(0.1, color=MUTED, ls="--", lw=1)
        ax.set_title(f"{MODNAME[m]}: rate 1/2 soft gains {gain[m]['soft_1/2']:.1f} dB "
                     f"({gain_pay[m]['soft_1/2']:.1f} dB payload only)", fontsize=10)
        ax.set_xlabel("SNR per sample (dB)")
    axes[0].set_ylabel("packet error rate (floor = no errors seen)")
    axes[0].legend(fontsize=7, loc="lower left")
    save(fig, run / "per_vs_snr.png")

    # ---- H3: goodput -------------------------------------------------------------------------------
    gp = {(m, c): np.array([r["goodput_kbps"] for r in curves[(m, c, "soft")]]) for m in LINK_MODULATIONS for c in CODES}
    unc = np.max([gp[(m, "none")] for m in LINK_MODULATIONS], axis=0)
    cod_keys = [(m, c) for m in LINK_MODULATIONS for c in CODES if c != "none"]
    cod = np.max([gp[k] for k in cod_keys], axis=0)
    best_all = [max(gp, key=lambda k: gp[k][i]) for i in range(len(snrs))]
    summary["goodput"] = {"snr_db": snrs.tolist(), "best_uncoded_kbps": unc.tolist(), "best_coded_kbps": cod.tolist(),
                          "best_scheme": [f"{m} {c}" for m, c in best_all]}
    for i, s in enumerate(snrs):
        rows.append({"part": "goodput_envelope", "snr_db": float(s), "best_uncoded_kbps": unc[i],
                     "best_coded_kbps": cod[i], "best_scheme": " ".join(best_all[i])})
    print("  best scheme by SNR: " + ", ".join(f"{s:g}:{m}/{c}" for s, (m, c) in zip(snrs, best_all)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ls = {"bpsk": ":", "qpsk": "--", "qam16": "-"}
    for (m, c), v in gp.items():
        ax.plot(snrs, v, ls[m], color=CODECOL[c], lw=1, alpha=0.75)
    ax.plot(snrs, unc, color=INK, lw=2.6, label="best uncoded")
    ax.plot(snrs, np.maximum(unc, cod), color=BLUE, lw=2.6, alpha=0.55, label="best with coding allowed")
    for c in CODES:
        ax.plot([], [], color=CODECOL[c], label=CODENAME[c])
    for m in LINK_MODULATIONS:
        ax.plot([], [], ls[m], color=MUTED, label=MODNAME[m])
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("goodput (kbps)")
    ax.set_title("Every modulation + code pair (thin); the best of each family (bold)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    ratio = np.where(unc > 0, np.maximum(unc, cod) / np.maximum(unc, 1e-9), np.nan)
    ax.plot(snrs, np.minimum(ratio, 5), "o-", ms=3, color=BLUE)
    for x, r in zip(snrs, ratio):
        if r > 5:
            ax.annotate(f"{r:.0f}x", (x, 5), textcoords="offset points", xytext=(6, -4), fontsize=8, color=BLUE)
    ax.set_ylim(0, 5.3)
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("best goodput with coding / without")
    ax.set_title("What coding adds (blank where uncoded delivers nothing)", fontsize=10)
    save(fig, run / "goodput.png")

    # ---- H4: bursts --------------------------------------------------------------------------------
    burst = {}
    for part, xs in [("fast", fade_snrs), ("slow", fade_snrs), ("jam", sirs), ("jam_long", sirs)]:
        burst[part] = {name: [out[(part, name, i)] for i in range(len(xs))] for name, *_ in BURST}
        for name, v in burst[part].items():
            for x, r in zip(xs, v):
                rows.append({"part": f"burst_{part}", "scheme": name, "x": float(x), **r})
        need_b = {name: snr_at(xs, [r["per"] for r in v]) for name, v in burst[part].items()}
        need_bp = {name: snr_at(xs, [r["per_payload"] for r in v]) for name, v in burst[part].items()}
        summary[f"burst_{part}"] = {"x": xs.tolist(), "x_for_per_10pct": need_b, "x_for_per_10pct_payload_only": need_bp,
                                    "per": {n: [r["per"] for r in v] for n, v in burst[part].items()}}
        print(f"  {part:5s} 10% PER at " + "  ".join(f"{n}: {('%.1f' % x) if x is not None else 'never'}"
                                                   for n, x in need_b.items()))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
    titles = {"fast": "Fast fading inside each packet (Doppler 500 Hz)", "slow": "Slow fading: one fade per packet",
              "jam": "Short jammer bursts (50 us every 500 us)", "jam_long": "Long jammer bursts (500 us every 5 ms)"}
    for ax, (part, xs) in zip(axes, [("fast", fade_snrs), ("slow", fade_snrs), ("jam", sirs), ("jam_long", sirs)]):
        for name, v in burst[part].items():
            ax.semilogy(xs, [max(r["per_payload"], 1 / (2 * a.nb)) if r["per_payload"] == r["per_payload"] else np.nan
                             for r in v], "o-", ms=3, color=BURSTCOL[name],
                        lw=2 if name == "rate 1/2, soft, interleaved" else 1.3, label=name)
        hl = burst[part]["rate 1/2, soft, interleaved"]
        ax.semilogy(xs, [max(1 - r["header_rate"], 1 / (2 * a.nb)) for r in hl], "-", color=MUTED, lw=5, alpha=0.3,
                    label="frame or header lost (not counted left)")
        ax.axhline(0.1, color=MUTED, ls="--", lw=1)
        ax.set_title(titles[part], fontsize=10)
        ax.set_xlabel("signal-to-interference ratio (dB), SNR 12 dB" if part.startswith("jam")
                      else "mean SNR per sample (dB)")
    axes[0].set_ylabel("payload error rate, packets whose header arrived (QPSK)")
    axes[0].legend(fontsize=7, loc="lower left")
    save(fig, run / "bursts.png")

    for part in dict.fromkeys(r["part"] for r in rows):
        ex.save_csv(run, f"{part}.csv", [r for r in rows if r["part"] == part])
    ex.save_metrics(run, summary)
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
