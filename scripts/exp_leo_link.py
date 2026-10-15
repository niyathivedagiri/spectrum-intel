"""Experiment: the whole packet link through a LEO satellite pass. How much data does one pass deliver?

Ground transmitter (23 dBm) -> satellite at 550 km (30 dBi antenna), 2 GHz, 1 MHz channel, 64-byte
packets through the real transmitter / channel / receiver (spectrum_intel/leo_link.py). Every 4 s
of the pass a burst of 8 packets is sent; each burst stands for 4 s of transmission.

Hypotheses (written before running)
  H1  Without Doppler handling the link works only near the top of the pass, where Doppler passes
      through zero: the receiver alone tolerates only a few kHz of offset, Doppler reaches +-46 kHz.
  H2  Predicting Doppler from the orbit delivers as much as a blind frequency search, for a fraction
      of the processing. With a poorer oscillator (+-5 ppm = +-10 kHz) prediction alone fails at
      times, and prediction plus a narrow search fixes it.
  H3  Over a pass, adaptive modulation AND coding delivers more data than any fixed scheme and more
      than adaptive modulation alone (uncoded).
  H4  Lower passes deliver much less data (shorter, longer range); with a 10 dB weaker transmitter
      coding matters more: the gain of MCS over uncoded adaptation grows.

    python scripts/exp_leo_link.py      # ~5 min on 2 cores
Output: results/experiments/leo_link/<run>/
"""
import argparse
import time
from dataclasses import replace
from multiprocessing import Pool

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel.leo_link import LEOLinkConfig, run_pass, timeline
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, save

FIXED = ["bpsk/1/2", "qpsk/1/2", "qpsk/none", "qam16/3/4", "qam16/none"]
SCHEMES = FIXED + ["adaptive", "mcs"]
NAME = {"bpsk/1/2": "BPSK 1/2", "qpsk/1/2": "QPSK 1/2", "qpsk/none": "QPSK", "qam16/3/4": "16-QAM 3/4",
        "qam16/none": "16-QAM", "adaptive": "adaptive modulation", "mcs": "adaptive mod + coding"}
COL = {"bpsk/1/2": MUTED, "qpsk/1/2": YELLOW, "qpsk/none": ORANGE, "qam16/3/4": AQUA, "qam16/none": "#9b59b6",
       "adaptive": INK, "mcs": BLUE}
DCOL = {"none": MUTED, "search": ORANGE, "predicted": BLUE, "predicted+search": AQUA}


def job(cfg):
    t0 = time.process_time()
    r = run_pass(cfg)
    r["cpu_s"] = time.process_time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    base = LEOLinkConfig()
    seeds = list(range(a.seeds))
    run = ex.new_run("leo_link", {"base": vars(base), "seeds": seeds}, 0)

    cfgs = {}
    for d in ("none", "search", "predicted", "predicted+search"):          # part A
        for ppm in (2.0, 5.0):
            cfgs[("doppler", d, ppm)] = replace(base, doppler=d, osc_ppm=ppm, seed=0)
    for s in SCHEMES:                                                      # part B
        for sd in seeds:
            cfgs[("scheme", s, sd)] = replace(base, scheme=s, seed=sd)
    for e in (60.0, 30.0, 15.0):                                           # part C (90 = part B)
        for s in ("qpsk/1/2", "adaptive", "mcs"):
            for sd in seeds:
                cfgs[("elev", s, e, sd)] = replace(base, scheme=s, max_elevation_deg=e, seed=sd)
    for s in SCHEMES:                                                      # part D: 10 dB weaker transmitter
        cfgs[("weak", s)] = replace(base, scheme=s, eirp_dbm=13.0, seed=0)
    keys = list(cfgs)
    with Pool(2) as pool:
        res = dict(zip(keys, pool.map(job, [cfgs[k] for k in keys], chunksize=1)))
    rows, summary = [], {}
    tl = timeline(base)
    summary["pass"] = {"minutes": float(tl["t_s"][-1] / 60), "snr_db": [float(tl["snr_db"].min()), float(tl["snr_db"].max())],
                       "max_doppler_khz": float(np.max(np.abs(tl["doppler_hz"])) / 1e3),
                       "max_doppler_rate_hz_s": float(np.max(np.abs(tl["doppler_rate_hz_s"]))),
                       "doppler_change_in_one_packet_hz": float(np.max(np.abs(tl["doppler_rate_hz_s"])) * 0.006)}

    # ---- A: Doppler handling ------------------------------------------------------------------------
    A = {}
    for (part, d, ppm), r in [(k, v) for k, v in res.items() if k[0] == "doppler"]:
        n_pk = len(r["steps"]) * base.packets_per_step
        A[f"{d} @ {ppm:g} ppm"] = {**r["summary"], "cpu_ms_per_packet": 1e3 * r["cpu_s"] / n_pk}
        rows.append({"part": "doppler", "mode": d, "osc_ppm": ppm, **r["summary"], "cpu_ms_per_packet": 1e3 * r["cpu_s"] / n_pk})
        print(f"  A {d:17s} {ppm:g} ppm  {r['summary']['delivered_MB']:6.2f} MB  link up {100 * r['summary']['link_up_fraction']:5.1f}%  "
              f"residual <= {r['summary']['max_residual_cfo_khz']:5.1f} kHz  CPU {1e3 * r['cpu_s'] / n_pk:5.1f} ms/packet")
    summary["doppler"] = A
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    t = np.array(tl["t_s"]) / 60
    ax.plot(t, tl["doppler_hz"] / 1e3, color=MUTED, lw=1, ls="--", label="Doppler (kHz)")
    for d in ("none", "predicted", "search"):
        st = res[("doppler", d, 2.0)]["steps"]
        ax.plot(t, [s["goodput_kbps"] / 4 for s in st], color=DCOL[d], lw=1.6,
                label=f"goodput / 4, {d}")
    ax.axhspan(-6, 6, color=MUTED, alpha=0.12)
    ax.text(t[2], 7.5, "receiver alone tolerates about +-6 kHz", fontsize=8, color=MUTED)
    ax.set_xlabel("time in pass (min)")
    ax.set_ylabel("kHz  /  kbps / 4")
    ax.set_title("Without Doppler handling the link only works near the top of the pass", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    labels = [f"{d}\n{ppm:g} ppm" for (_, d, ppm) in [k for k in keys if k[0] == "doppler"]]
    mb = [res[k]["summary"]["delivered_MB"] for k in keys if k[0] == "doppler"]
    cpu = [A[f"{k[1]} @ {k[2]:g} ppm"]["cpu_ms_per_packet"] for k in keys if k[0] == "doppler"]
    cols = [DCOL[k[1]] for k in keys if k[0] == "doppler"]
    ax.barh(range(len(mb)), mb, color=cols)
    for i, (m, c) in enumerate(zip(mb, cpu)):
        ax.text(m + 0.15, i, f"{m:.1f} MB, {c:.0f} ms CPU/packet", va="center", fontsize=7)
    ax.set_yticks(range(len(mb)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(mb) * 1.7)
    ax.set_xlabel("data delivered in one overhead pass (MB)")
    ax.set_title("Doppler handling: data and processing cost", fontsize=10)
    save(fig, run / "doppler.png")

    # ---- B: schemes over the pass ---------------------------------------------------------------------
    B = {}
    for s in SCHEMES:
        mbs = [res[("scheme", s, sd)]["summary"]["delivered_MB"] for sd in seeds]
        up = [res[("scheme", s, sd)]["summary"]["link_up_fraction"] for sd in seeds]
        B[s] = {"delivered_MB": [float(np.mean(mbs)), float(np.std(mbs))], "link_up_fraction": float(np.mean(up))}
        rows.append({"part": "schemes", "scheme": s, "delivered_MB": np.mean(mbs), "std": np.std(mbs),
                     "link_up_fraction": np.mean(up)})
        print(f"  B {NAME[s]:22s} {np.mean(mbs):6.2f} +- {np.std(mbs):.2f} MB  link up {100 * np.mean(up):5.1f}%")
    summary["schemes"] = B
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.plot(t, tl["snr_db"], color=MUTED, ls=":", lw=1)
    ax2.set_ylabel("link SNR (dB, dotted)", color=MUTED)
    for s in ("bpsk/1/2", "qpsk/none", "qam16/none", "adaptive", "mcs"):
        st = res[("scheme", s, 0)]["steps"]
        ax.plot(t, [x["goodput_kbps"] for x in st], color=COL[s], lw=2.2 if s == "mcs" else 1.2, label=NAME[s])
    ax.set_xlabel("time in pass (min)")
    ax.set_ylabel("goodput (kbps)")
    ax.set_title("Overhead pass: fixed schemes vs adaptation (seed 0)", fontsize=10)
    ax.legend(fontsize=7, loc="lower center")
    ax = axes[1]
    v = [B[s]["delivered_MB"][0] for s in SCHEMES]
    e = [B[s]["delivered_MB"][1] for s in SCHEMES]
    ax.barh(range(len(SCHEMES)), v, xerr=e, color=[COL[s] for s in SCHEMES])
    for i, x in enumerate(v):
        ax.text(x + 0.2, i, f"{x:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(SCHEMES)), [NAME[s] for s in SCHEMES], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"data per pass (MB, mean of {len(seeds)} seeds)")
    ax.set_title("Data delivered in one overhead pass", fontsize=10)
    save(fig, run / "schemes.png")

    # ---- C + D: pass elevation, weaker transmitter --------------------------------------------------------
    C = {}
    elevs = [90.0, 60.0, 30.0, 15.0]
    for s in ("qpsk/1/2", "adaptive", "mcs"):
        C[s] = []
        for e in elevs:
            k = ("scheme", s) if e == 90.0 else ("elev", s, e)
            mbs = [res[k + (sd,)]["summary"]["delivered_MB"] for sd in seeds]
            C[s].append(float(np.mean(mbs)))
            rows.append({"part": "elevation", "scheme": s, "max_elevation_deg": e, "delivered_MB": np.mean(mbs)})
    summary["elevation"] = {"max_elevation_deg": elevs, "delivered_MB": C,
                            "pass_minutes": [float(timeline(replace(base, max_elevation_deg=e))["t_s"][-1] / 60) for e in elevs]}
    print("  C " + "  ".join(f"{s}: " + "/".join(f"{x:.2f}" for x in C[s]) for s in C))
    D = {s: res[("weak", s)]["summary"]["delivered_MB"] for s in SCHEMES}
    summary["weak_transmitter_13dBm"] = {"delivered_MB": D, "snr_db": [float(tl["snr_db"].min() - 10), float(tl["snr_db"].max() - 10)]}
    for s in SCHEMES:
        rows.append({"part": "weak", "scheme": s, "delivered_MB": D[s]})
    print("  D " + "  ".join(f"{NAME[s]} {D[s]:.2f}" for s in SCHEMES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    for s in ("qpsk/1/2", "adaptive", "mcs"):
        ax.plot(elevs, C[s], "o-", color=COL[s], lw=2 if s == "mcs" else 1.3, label=NAME[s])
    ax.set_xlabel("highest elevation of the pass (deg)")
    ax.set_ylabel("data per pass (MB)")
    ax.set_title("Lower passes are shorter and weaker", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    w = 0.38
    strong = [B[s]["delivered_MB"][0] for s in SCHEMES]
    ax.bar(np.arange(len(SCHEMES)) - w / 2, strong, w, color=MUTED, label="23 dBm transmitter")
    ax.bar(np.arange(len(SCHEMES)) + w / 2, [D[s] for s in SCHEMES], w, color=BLUE, label="13 dBm (10 dB weaker)")
    ax.set_xticks(range(len(SCHEMES)), [NAME[s].replace(" + ", "\n+ ").replace("adaptive ", "adaptive\n") for s in SCHEMES],
                  fontsize=7)
    ax.set_ylabel("data per overhead pass (MB)")
    ax.set_title("A weaker transmitter: coding decides what gets through", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, run / "elevation_power.png")

    for part in dict.fromkeys(r["part"] for r in rows):
        ex.save_csv(run, f"{part}.csv", [r for r in rows if r["part"] == part])
    ex.save_metrics(run, summary)
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
