"""Experiment: can the Python link keep up with real time, and where does the time go?

Hypotheses
  H1  Before optimisation the per-packet processing (transmitter + channel + receiver)
      is slower than the frame's airtime at 1 MS/s (real-time factor > 1), because the
      profiler showed ~60% of the time recomputing the same RRC filter taps every frame.
  H2  Computing the taps once (cache) removes that cost without changing any output,
      bringing the real-time factor close to or below 1.
Baseline: identical code with the cache switched off (signals.CACHE_TAPS = False).
Also checked: the real-time engine gives bit-identical packets to link.send.

Timings depend on the computer: rerun on yours. Packets per point: --n (default 40),
first packet of each point excluded (warm-up).

    python scripts/exp_realtime.py      # ~1 min
Output: results/experiments/realtime/<run>/
"""
import argparse
import os
import platform
import time

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel import experiments as ex
from spectrum_intel import link, signals
from spectrum_intel.engine import EngineConfig, LinkEngine
from spectrum_intel.metrics import DEFAULT_SAMPLE_RATE_HZ
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import segment
from spectrum_intel.plots import AQUA, BLUE, MUTED, ORANGE, save
from spectrum_intel.receiver import receive
from spectrum_intel.transmitter import build_frame

SIZES = [16, 64, 256, 1024]
SNR_DB = 12.0          # high enough that every packet is demodulated (same work per packet)


def time_point(mod, size, n, seed):
    rng = np.random.default_rng([seed, size, LINK_MODULATIONS.index(mod)])
    pk = segment(rng.integers(0, 256, size * (n + 1), dtype=np.uint8).tobytes(), size)
    out = []
    for i, p in enumerate(pk[:n + 1]):
        t0 = time.perf_counter()
        frame = build_frame(p.to_bytes(), mod)
        t1 = time.perf_counter()
        rx, _ = ch.awgn_link(frame.iq, SNR_DB, rng)
        t2 = time.perf_counter()
        res = receive(rx)
        t3 = time.perf_counter()
        if i:                                            # skip warm-up packet
            out.append({"tx_ms": 1e3 * (t1 - t0), "channel_ms": 1e3 * (t2 - t1), "rx_ms": 1e3 * (t3 - t2),
                        "total_ms": 1e3 * (t3 - t0), "airtime_ms": 1e3 * len(frame.iq) / DEFAULT_SAMPLE_RATE_HZ,
                        "header_ok": res.header_ok})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    machine = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
               "cpu_count": os.cpu_count()}
    run = ex.new_run("realtime", {"packets_per_point": a.n, "sizes": SIZES, "snr_db": SNR_DB,
                                  "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ} | machine, a.seed)
    rows = []
    for cached in (False, True):
        signals.CACHE_TAPS = cached
        signals._TAP_CACHE.clear()
        for mod in LINK_MODULATIONS:
            for size in SIZES:
                pts = time_point(mod, size, a.n, a.seed)
                med = {k: float(np.median([p[k] for p in pts])) for k in ("tx_ms", "channel_ms", "rx_ms", "total_ms")}
                air = pts[0]["airtime_ms"]
                rows.append({"taps_cached": cached, "mod": mod, "packet_bytes": size, **med, "airtime_ms": air,
                             "realtime_factor": med["total_ms"] / air,
                             "max_realtime_rate_msps": DEFAULT_SAMPLE_RATE_HZ / 1e6 * air / med["total_ms"],
                             "all_demodulated": all(p["header_ok"] for p in pts)})
                r = rows[-1]
                print(f"  cache={cached!s:5s} {mod:5s} {size:5d} B  cpu {r['total_ms']:6.2f} ms  "
                      f"air {air:7.2f} ms  RTF {r['realtime_factor']:.2f}")
    signals.CACHE_TAPS = True
    ex.save_csv(run, "timing.csv", rows)

    # engine == link.send for the same seed (bit-identical) + dashboard overhead per update
    cfg = EngineConfig(snr_db=2.0, seed=a.seed)
    eng = LinkEngine(cfg)
    t_state = []
    while eng.step() is not None:
        t0 = time.perf_counter()
        eng.state()
        t_state.append(1e3 * (time.perf_counter() - t0))
    ref = link.send(eng.data, cfg.mod, cfg.snr_db, np.random.default_rng(cfg.seed))
    identical = eng.received()[0] == ref.received and [vars(x) for x in eng.records] == [vars(x) for x in ref.packets]

    # ---- figure 1: real-time factor, before vs after -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    x = np.arange(len(SIZES))
    for ax, mod in zip(axes, LINK_MODULATIONS):
        for off, cached, col, lab in [(-0.2, False, ORANGE, "taps recomputed (before)"),
                                      (0.2, True, BLUE, "taps cached (after)")]:
            v = [next(r["realtime_factor"] for r in rows if r["mod"] == mod and r["packet_bytes"] == s
                      and r["taps_cached"] == cached) for s in SIZES]
            ax.bar(x + off, v, 0.38, color=col, label=lab)
        ax.axhline(1, color=MUTED, lw=1, ls="--")
        ax.set_xticks(x, [f"{s} B" for s in SIZES])
        ax.set_title(mod.upper().replace("QAM16", "16-QAM"), fontsize=10)
        ax.set_xlabel("packet size")
    axes[0].set_ylabel("CPU time ÷ airtime at 1 MS/s")
    axes[-1].text(len(SIZES) - 0.55, 1.06, "real time", color=MUTED, fontsize=8, ha="right")
    axes[0].legend(fontsize=8)
    fig.suptitle("Below the dashed line the link is processed faster than it is transmitted", fontsize=11)
    save(fig, run / "realtime_factor.png")

    # ---- figure 2: where the time goes (QPSK, 64 B) -----------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for yi, cached in enumerate([False, True]):
        r = next(r for r in rows if r["mod"] == "qpsk" and r["packet_bytes"] == 64 and r["taps_cached"] == cached)
        left = 0
        for k, col, lab in [("tx_ms", BLUE, "transmitter"), ("channel_ms", AQUA, "channel"), ("rx_ms", ORANGE, "receiver")]:
            ax.barh(yi, r[k], left=left, color=col, label=lab if yi == 0 else None)
            if r[k] > 0.8:
                ax.text(left + r[k] / 2, yi, f"{r[k]:.1f}", ha="center", va="center", color="white", fontsize=8)
            left += r[k]
        air = r["airtime_ms"]
    ax.axvline(air, color=MUTED, ls="--", lw=1)
    ax.text(air, 1.5, f" airtime {air:.1f} ms", color=MUTED, fontsize=8, va="center")
    ax.set_yticks([0, 1], ["before", "after"])
    ax.set_ylim(-0.6, 1.7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("median CPU time per 64-byte QPSK packet (ms)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.set_title("Computing the filter taps once removes most of the transmitter time and half the receiver time", fontsize=10)
    save(fig, run / "time_breakdown.png")

    def pick(cached, mod="qpsk", size=64):
        return next(r for r in rows if r["mod"] == mod and r["packet_bytes"] == size and r["taps_cached"] == cached)

    summary = {
        "machine": machine,
        "qpsk_64B_cpu_ms": {"before": pick(False)["total_ms"], "after": pick(True)["total_ms"]},
        "qpsk_64B_realtime_factor": {"before": pick(False)["realtime_factor"], "after": pick(True)["realtime_factor"]},
        "speedup_qpsk_64B": pick(False)["total_ms"] / pick(True)["total_ms"],
        "realtime_factor_after": {f"{r['mod']}_{r['packet_bytes']}B": r["realtime_factor"] for r in rows if r["taps_cached"]},
        "max_realtime_sample_rate_msps_after": {f"{r['mod']}_{r['packet_bytes']}B": r["max_realtime_rate_msps"]
                                                for r in rows if r["taps_cached"]},
        "engine_identical_to_link_send": bool(identical),
        "dashboard_state_ms_median": float(np.median(t_state)),
    }
    ex.save_metrics(run, summary)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
