"""Experiment: what fading, multipath and interference do to the link, and what the receiver can do about it.

Baseline for every part: the same link over plain AWGN (same average SNR, same seeds).

Hypotheses
  H1  Block Rayleigh fading turns the packet-success "cliff" into a slow slope: success
      follows the outage law exp(-10^((T - SNR)/10)), T = the AWGN 50% threshold, so every
      10x drop in failures costs ~10 dB more SNR (diversity order 1). A strong line of sight
      (Rician, high K) brings the link back towards AWGN.
  H2  When the channel changes during a frame (Doppler), averaging many pilots smooths away
      the changes; averaging fewer pilots tracks them but adds noise on a slow channel.
  H3  An echo of a symbol or more causes intersymbol interference that the one-tap pilot
      correction cannot remove; a preamble-trained equaliser (LS) removes it, and longer
      echoes need more taps. On a clean channel the equaliser costs some noise.
  H4  A pulsed jammer with the same average power as a tone hurts packets differently:
      it destroys whatever it overlaps, so shorter packets survive it more often.
  H5  An echo that also changes during the frame (Doppler + multipath) defeats an
      equaliser trained once on the preamble; shorter packets suffer less.

    python scripts/exp_channel.py        # ~3 min
Output: results/experiments/channel/<run>/
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel import experiments as ex
from spectrum_intel.link import transfer_packet
from spectrum_intel.packet import segment
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, save


def success(cfg, snr, n, seed, *, mod="qpsk", size=64, eq="none", taps=7, pw=7):
    """Share of packets passing CRC over n packets (fixed seed per point)."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size * 16, dtype=np.uint8).tobytes()
    pk = segment(data, size)
    ok = 0
    for i in range(n):
        t = transfer_packet(pk[i % len(pk)], len(pk), mod, snr, rng, max_payload=size, channel=cfg,
                            equalizer=eq, eq_taps=taps, pilot_window=pw)
        ok += t.record.crc_ok
    return ok / n


def crossing(x, y, level):
    """First x where y rises through `level` (linear interpolation), None if never."""
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            return float(x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="packets per point")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    n, S = a.n, a.seed
    C = ch.ChannelConfig
    run = ex.new_run("channel", {"packets_per_point": n, "modulation": "qpsk", "packet_bytes": 64,
                                 "sample_rate_hz": 1e6}, S)
    rows, summary = [], {}

    # ---- H1: fading vs AWGN ------------------------------------------------------------
    snrs = np.arange(-4, 31, 2.0)
    fading = {"AWGN": None, "Rician K=10 dB": C(fading="rician", k_factor_db=10),
              "Rician K=3 dB": C(fading="rician", k_factor_db=3), "Rayleigh": C(fading="rayleigh")}
    curves = {}
    for name, cfg in fading.items():
        curves[name] = [success(cfg, s, n, [S, 1, int(s * 10) + 500, len(name)]) for s in snrs]
        rows += [{"part": "fading", "channel": name, "snr_db": float(s), "success": v} for s, v in zip(snrs, curves[name])]
        print(f"  H1 {name:15s} " + " ".join(f"{v:.2f}" for v in curves[name]))
    T = crossing(snrs, curves["AWGN"], 0.5)
    outage = np.exp(-10 ** ((T - snrs) / 10))
    s90 = {k: crossing(snrs, v, 0.9) for k, v in curves.items()}
    s99 = {k: crossing(snrs, v, 0.99) for k, v in curves.items()}
    summary["H1_fading"] = {"awgn_50pct_threshold_db": T, "snr_for_90pct": s90, "snr_for_99pct": s99,
                            "rayleigh_vs_outage_theory_max_abs_diff": float(np.max(np.abs(np.array(curves["Rayleigh"]) - outage)))}
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for (name, v), col in zip(curves.items(), [INK, AQUA, YELLOW, ORANGE]):
        ax.plot(snrs, 100 * np.array(v), "o-", ms=3.5, color=col, label=name)
    ax.plot(snrs, 100 * outage, "--", color=ORANGE, lw=1, alpha=0.7, label="Rayleigh outage theory")
    ax.set_xlabel("average SNR per sample (dB)")
    ax.set_ylabel("packets passing CRC (%)")
    ax.set_title(f"Fading turns the cliff into a slope: 99% of packets need {s99['AWGN']:.0f} dB in AWGN, "
                 f"{s99['Rayleigh']:.0f} dB in Rayleigh", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, run / "fading_vs_snr.png")

    # ---- H2: Doppler vs pilot averaging -------------------------------------------------
    dops = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    track = {}
    for pw in (7, 3, 1):
        track[pw] = [success(C(fading="rayleigh", doppler_hz=d), 25.0, n, [S, 2, d, pw], pw=pw) for d in dops]
        rows += [{"part": "doppler", "pilot_window": pw, "doppler_hz": d, "success": v} for d, v in zip(dops, track[pw])]
        print(f"  H2 window {pw}: " + " ".join(f"{v:.2f}" for v in track[pw]))
    awgn_cost = {pw: success(None, 1.0, 4 * n, [S, 3, pw], pw=pw) for pw in (7, 3, 1)}
    summary["H2_doppler"] = {"rayleigh_25dB_success": {str(pw): dict(zip(map(str, dops), v)) for pw, v in track.items()},
                             "awgn_1dB_success_by_window": {str(k): v for k, v in awgn_cost.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    for pw, col in zip((7, 3, 1), [BLUE, AQUA, ORANGE]):
        axes[0].plot(dops, 100 * np.array(track[pw]), "o-", ms=3.5, color=col, label=f"average {pw} pilot{'s' if pw > 1 else ''}")
    axes[0].set_xlabel("Doppler spread (Hz)")
    axes[0].set_ylabel("packets passing CRC (%)")
    axes[0].set_title("Rayleigh fading at 25 dB: fewer pilots averaged = follows fast fading", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_axisbelow(True)
    axes[1].bar([str(k) for k in awgn_cost], [100 * v for v in awgn_cost.values()], color=[BLUE, AQUA, ORANGE])
    axes[1].set_xlabel("pilots averaged")
    axes[1].set_ylabel("packets passing CRC (%)")
    axes[1].set_title("...but costs noise on a still channel\n(AWGN, 1 dB)", fontsize=10)
    save(fig, run / "doppler_tracking.png")

    # ---- H3: multipath vs equaliser -------------------------------------------------------
    delays = [0, 2, 4, 8, 12, 16, 24, 32]
    eqs = [("none", 7, "no equaliser"), ("ls", 7, "LS 7 taps"), ("ls", 15, "LS 15 taps"), ("ls", 21, "LS 21 taps")]
    mp = {}
    for eq, taps, lab in eqs:
        mp[lab] = [success(None if d == 0 else C(multipath="two_ray", delay_spread_us=d, echo_db=-3), 15.0, n,
                           [S, 4, d, taps], eq=eq, taps=taps) for d in delays]
        rows += [{"part": "multipath", "receiver": lab, "delay_us": d, "success": v} for d, v in zip(delays, mp[lab])]
        print(f"  H3 {lab:13s} " + " ".join(f"{v:.2f}" for v in mp[lab]))
    eq_cost = {lab: success(None, 1.0, 4 * n, [S, 5, taps], eq=eq, taps=taps) for eq, taps, lab in eqs}
    summary["H3_multipath"] = {"two_ray_minus3dB_15dB_success": {k: dict(zip(map(str, delays), v)) for k, v in mp.items()},
                               "awgn_1dB_success": eq_cost}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    for (lab, v), col in zip(mp.items(), [INK, BLUE, AQUA, ORANGE]):
        axes[0].plot(delays, 100 * np.array(v), "o-", ms=3.5, color=col, label=lab)
    axes[0].axvline(8, color=MUTED, ls=":", lw=1)
    axes[0].text(8.4, 50, "1 symbol = 8 µs", color=MUTED, fontsize=8)
    axes[0].set_xlabel("echo delay (µs), echo 3 dB weaker")
    axes[0].set_ylabel("packets passing CRC (%)")
    axes[0].set_title("Two-ray multipath at 15 dB: the equaliser removes echoes, longer echoes need more taps", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_axisbelow(True)
    axes[1].bar(range(len(eq_cost)), [100 * v for v in eq_cost.values()], color=[INK, BLUE, AQUA, ORANGE])
    axes[1].set_xticks(range(len(eq_cost)), ["none", "LS7", "LS15", "LS21"])
    axes[1].set_ylabel("packets passing CRC (%)")
    axes[1].set_title("...and costs noise on a clean channel\n(AWGN, 1 dB)", fontsize=10)
    save(fig, run / "multipath_equaliser.png")

    # ---- H4: interference ---------------------------------------------------------------
    sirs = np.arange(-10, 31, 2.5)
    jam = {}
    for name, cfg in [("tone, 20 kHz", lambda s: C(jammer="tone", sir_db=s)),
                      ("pulsed, 10% duty", lambda s: C(jammer="pulsed", sir_db=s))]:
        jam[name] = [success(cfg(s), 15.0, n, [S, 6, int(s * 10) + 500, len(name)]) for s in sirs]
        rows += [{"part": "interference", "jammer": name, "sir_db": float(s), "success": v} for s, v in zip(sirs, jam[name])]
        print(f"  H4 {name:17s} " + " ".join(f"{v:.2f}" for v in jam[name]))
    sizes = [16, 64, 256]
    pulsed_size = {sz: success(C(jammer="pulsed", sir_db=10.0), 15.0, n, [S, 7, sz], size=sz) for sz in sizes}
    rows += [{"part": "pulsed_vs_size", "packet_bytes": sz, "success": v} for sz, v in pulsed_size.items()]
    summary["H4_interference"] = {"sir_for_90pct": {k: crossing(sirs, v, 0.9) for k, v in jam.items()},
                                  "pulsed_sir10_success_by_packet_bytes": {str(k): v for k, v in pulsed_size.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    for (name, v), col in zip(jam.items(), [BLUE, ORANGE]):
        axes[0].plot(sirs, 100 * np.array(v), "o-", ms=3.5, color=col, label=name)
    axes[0].set_xlabel("signal-to-interference ratio (dB), average power")
    axes[0].set_ylabel("packets passing CRC (%)")
    axes[0].set_title("Interference at SNR 15 dB: same average power, different damage", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_axisbelow(True)
    axes[1].bar([f"{s} B" for s in sizes], [100 * v for v in pulsed_size.values()], color=[AQUA, BLUE, ORANGE])
    axes[1].set_ylabel("packets passing CRC (%)")
    axes[1].set_title("Pulsed jammer, SIR 10 dB:\nshort packets slip between bursts", fontsize=10)
    save(fig, run / "interference.png")

    # ---- H5: time-varying multipath ------------------------------------------------------
    dops5 = [0, 50, 100, 200, 300, 500]
    tv = {}
    for sz in (16, 64):
        tv[sz] = [success(C(fading="rayleigh", doppler_hz=d, multipath="two_ray", delay_spread_us=8, echo_db=-3),
                          25.0, n, [S, 8, d, sz], size=sz, eq="ls", taps=15, pw=3) for d in dops5]
        rows += [{"part": "time_varying_multipath", "packet_bytes": sz, "doppler_hz": d, "success": v}
                 for d, v in zip(dops5, tv[sz])]
        print(f"  H5 {sz:3d} B " + " ".join(f"{v:.2f}" for v in tv[sz]))
    summary["H5_time_varying_multipath"] = {f"{sz}B": dict(zip(map(str, dops5), v)) for sz, v in tv.items()}
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for sz, col in zip((16, 64), [AQUA, BLUE]):
        ax.plot(dops5, 100 * np.array(tv[sz]), "o-", ms=3.5, color=col, label=f"{sz}-byte packets")
    ax.set_xlabel("Doppler spread (Hz)")
    ax.set_ylabel("packets passing CRC (%)")
    ax.set_title("Echo (8 µs) that also fades: an equaliser trained once on the preamble goes stale", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, run / "time_varying_multipath.png")

    for part in dict.fromkeys(r["part"] for r in rows):             # one table per part (different columns)
        ex.save_csv(run, f"{part}.csv", [r for r in rows if r["part"] == part])
    ex.save_metrics(run, summary)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
