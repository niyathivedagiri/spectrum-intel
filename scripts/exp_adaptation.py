"""Experiment: does choosing the modulation per packet beat any fixed modulation, and when does it stop helping?

Policies compared on the SAME channel realisation (common random numbers: packet k always
sees the same noise draws, whatever modulation is chosen):
  fixed BPSK / QPSK / 16-QAM      baselines
  threshold                       table lookup from the receiver's last SNR report
  olla                            threshold + outer loop learned from ACK/NACK (LTE-style)
  goodput                         highest expected goodput from the table (no error target)
  genie                           threshold using the TRUE current SNR (an upper bound, not realisable)

Hypotheses
  H1  With a steady SNR, adaptation reaches the goodput of the best fixed modulation at
      every SNR (it can only lose by its feedback being one packet old).
  H2  When the SNR changes over time (a user moving), adaptation beats every fixed choice,
      because no single modulation is best throughout.
  H3  Under fading that is correlated from packet to packet, adaptation's gain is largest
      when the channel changes slowly and shrinks as it changes faster, because the
      one-packet-old report goes stale. Table-only thresholds then overshoot the target
      packet error rate; the outer loop pulls it back towards the target.

    python scripts/exp_adaptation.py      # ~2 min
Output: results/experiments/adaptation/<run>/
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel.adaptation import LinkAdapter, slow_fading_snr
from spectrum_intel.link import transfer_packet
from spectrum_intel.metrics import link_report
from spectrum_intel.packet import segment
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, save

FIXED = {"BPSK": "bpsk", "QPSK": "qpsk", "16-QAM": "qam16"}
COL = {"BPSK": BLUE, "QPSK": ORANGE, "16-QAM": AQUA, "threshold": YELLOW, "olla": "#9b59b6", "goodput": INK,
       "genie": MUTED}
MODCOL = {"bpsk": BLUE, "qpsk": ORANGE, "qam16": AQUA}


class Run:
    def __init__(self, records):
        self.packets = records


def run_policy(policy, snrs, seed, part):
    """Send len(snrs) 64-byte packets; packet k sees SNR snrs[k]. Returns (report, records)."""
    data = np.random.default_rng([seed, part]).integers(0, 256, 64 * 16, dtype=np.uint8).tobytes()
    pk = segment(data, 64)
    adapter = None if policy in FIXED else LinkAdapter("threshold" if policy == "genie" else policy, 0.1)
    recs = []
    for k, snr in enumerate(snrs):
        if policy in FIXED:
            mod = FIXED[policy]
        elif policy == "genie":
            mod = adapter.choose(snr_override=snr)
        else:
            mod = adapter.choose()
        rng = np.random.default_rng([seed, part, k])                 # same channel draws for every policy
        r = transfer_packet(pk[k % 16], 16, mod, float(snr), rng).record
        if adapter is not None:
            adapter.update(r)
        recs.append(r)
    return link_report(Run(recs)), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--n", type=int, default=60, help="packets per point in part 1")
    a = ap.parse_args()
    S = a.seed
    run = ex.new_run("adaptation", {"packet_bytes": 64, "target_per": 0.1, "feedback_delay_packets": 1,
                                    "table": "spectrum_intel/link_table.json"}, S)
    rows, summary = [], {}

    # ---- H1: steady SNR --------------------------------------------------------------------
    snr_grid = np.arange(-4, 21, 2.0)
    pols1 = ["BPSK", "QPSK", "16-QAM", "threshold", "olla", "goodput"]
    gp = {p: [] for p in pols1}
    for i, s in enumerate(snr_grid):
        for p in pols1:
            rep, _ = run_policy(p, np.full(a.n, s), S, 100 + i)
            gp[p].append(rep["goodput_bps"] / 1e3)
            rows.append({"part": "steady", "policy": p, "snr_db": float(s), "goodput_kbps": rep["goodput_bps"] / 1e3,
                         "per": rep["packet_error_rate"]})
        print(f"  H1 {s:5.1f} dB  " + "  ".join(f"{p} {gp[p][-1]:6.1f}" for p in pols1))
    best_fixed = np.max([gp[p] for p in FIXED], axis=0)
    live = best_fixed > 0                                   # skip SNRs where nothing gets through
    adapt = ("threshold", "olla", "goodput")
    ratio = {p: float(np.mean(np.array(gp[p])[live] / best_fixed[live])) for p in adapt}
    summary["H1_steady"] = {"mean_ratio_to_best_fixed": ratio,
                            "worst_ratio_to_best_fixed": {p: float(np.min(np.array(gp[p])[live] / best_fixed[live]))
                                                          for p in adapt},
                            "snr_db": snr_grid.tolist(), "goodput_kbps": gp}
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for p in pols1:
        ax.plot(snr_grid, gp[p], "o-" if p in FIXED else "s--", ms=3.5, lw=2.2 if p == "goodput" else 1.4,
                color=COL[p], label=p if p in FIXED else f"adaptive ({p})")
    ax.set_xlabel("SNR per sample (dB), steady")
    ax.set_ylabel("goodput (kbps)")
    ax.set_title(f"Steady SNR: the goodput policy follows the best fixed modulation "
                 f"({100 * ratio['goodput']:.0f}% of it on average)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, run / "steady_snr.png")

    # ---- H2: a user moving (SNR swings slowly) -----------------------------------------------
    k = np.arange(600)
    traj = 8 + 10 * np.sin(2 * np.pi * k / 200)
    pols2 = ["BPSK", "QPSK", "16-QAM", "threshold", "olla", "goodput", "genie"]
    res2, recs2 = {}, {}
    for p in pols2:
        rep, recs = run_policy(p, traj, S, 2)
        res2[p] = {"goodput_kbps": rep["goodput_bps"] / 1e3, "per": rep["packet_error_rate"]}
        recs2[p] = recs
        rows.append({"part": "moving", "policy": p, **res2[p]})
        print(f"  H2 {p:9s} goodput {res2[p]['goodput_kbps']:6.1f} kbps  PER {res2[p]['per']:.3f}")
    summary["H2_moving_user"] = res2
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    ax.plot(k, traj, color=MUTED, lw=1, label="true SNR")
    mods = [r.mod for r in recs2["goodput"]]
    for m, c in MODCOL.items():
        idx = [i for i, x in enumerate(mods) if x == m]
        ax.scatter(idx, traj[idx], s=6, color=c, label=f"chose {m.upper().replace('QAM16', '16-QAM')}")
    fails = [i for i, r in enumerate(recs2["goodput"]) if not r.crc_ok]
    ax.scatter(fails, traj[fails] + 1.2, marker="x", s=14, color="#d64545", label="packet failed")
    ax.set_xlabel("packet number")
    ax.set_ylabel("SNR per sample (dB)")
    ax.set_title("Moving user, goodput policy: the modulation follows the SNR one packet late", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    axes[1].set_axisbelow(True)
    axes[1].bar(pols2, [res2[p]["goodput_kbps"] for p in pols2], color=[COL[p] for p in pols2])
    axes[1].set_ylabel("average goodput (kbps)")
    axes[1].set_title("Same channel, every policy", fontsize=10)
    axes[1].tick_params(axis="x", labelrotation=30)
    save(fig, run / "moving_user.png")

    # ---- H3: correlated fading, slow to fast ------------------------------------------------
    dops = [1, 5, 20, 50, 100, 200]
    slot = 0.005                                        # one packet every 5 ms
    pols3 = ["BPSK", "QPSK", "16-QAM", "threshold", "olla", "goodput", "genie"]
    mean3 = 10.0                                        # between the QPSK and 16-QAM thresholds
    res3 = {p: {"goodput": [], "per": []} for p in pols3}
    for j, fd in enumerate(dops):
        snrs = slow_fading_snr(400, mean3, fd, slot, np.random.default_rng([S, 3, j]))
        for p in pols3:
            rep, _ = run_policy(p, snrs, S, 300 + j)
            res3[p]["goodput"].append(rep["goodput_bps"] / 1e3)
            res3[p]["per"].append(rep["packet_error_rate"])
            rows.append({"part": "fading", "policy": p, "doppler_hz": fd, "fd_times_slot": fd * slot,
                         "goodput_kbps": rep["goodput_bps"] / 1e3, "per": rep["packet_error_rate"]})
        print(f"  H3 fd {fd:4d} Hz (fd*T {fd * slot:.3f})  " +
              "  ".join(f"{p} {res3[p]['goodput'][-1]:5.1f}/{res3[p]['per'][-1]:.2f}" for p in pols3))
    summary["H3_fading"] = {"doppler_hz": dops, "fd_times_slot": [d * slot for d in dops], "mean_snr_db": mean3,
                            "slot_ms": 1e3 * slot, "goodput_kbps": {p: v["goodput"] for p, v in res3.items()},
                            "per": {p: v["per"] for p, v in res3.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = [d * slot for d in dops]
    for p in pols3:
        sty = "o-" if p in FIXED else ("s--" if p != "genie" else ":")
        axes[0].semilogx(x, res3[p]["goodput"], sty, ms=3.5, color=COL[p], lw=2.2 if p == "goodput" else 1.4,
                         label=p if p in FIXED else f"adaptive ({p})")
        if p not in FIXED:
            axes[1].semilogx(x, res3[p]["per"], sty, ms=3.5, color=COL[p], lw=2.2 if p == "goodput" else 1.4, label=p)
    axes[1].axhline(0.1, color=MUTED, ls="--", lw=1)
    axes[1].text(x[0], 0.11, "10% target", color=MUTED, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Doppler x packet interval (how much the channel changes per packet)")
    axes[0].set_ylabel("goodput (kbps)")
    axes[1].set_ylabel("packet error rate")
    axes[0].set_title(f"Rayleigh fading, mean {mean3:g} dB: adaptation helps most when fading is slow", fontsize=10)
    axes[1].set_title("Table-only thresholds overshoot the target; the outer loop pulls back", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    save(fig, run / "fading_speed.png")

    for part in dict.fromkeys(r["part"] for r in rows):
        ex.save_csv(run, f"{part}.csv", [r for r in rows if r["part"] == part])
    ex.save_metrics(run, summary)
    print(f"  summary: {summary['H1_steady']}")
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
