"""Experiment: which channel should the cognitive radio use next, and what does choosing well buy?

Setup: the same eight licensed users as the original spectrum demo (two-state Markov on/off),
our own link quality differing per channel (SNR = base + offset), one 64-byte packet per slot
through the real transmitter / channel / receiver. If the licensed user is on, its signal is
added as interference at its own SNR (a collision: it harms them, and usually loses our packet).
Sensing errors use the CNN's measured rates (detects 100% of busy channels, 1.5% false alarms).
Every policy sees exactly the same licensed users, sensing errors and noise (common random numbers).
The first 100 slots (learning) are left out of the main numbers and shown in the learning curve.

Hypotheses
  H1  Learning each channel's on/off statistics (predict) lowers collisions below the
      original stay-or-hop rule, because it prefers channels whose owners rarely return.
  H2  Also learning our link quality per channel (quality) raises our goodput well above
      both, at no extra harm to licensed users; the gain is largest when channels differ
      most in usefulness (low and medium SNR) and shrinks when every channel is good.
  H3  A hop margin cuts channel changes; when a hop costs time, a moderate margin gives
      the best net goodput.
  H4  The ranking survives realistic sensing errors; poor sensing raises collisions for all.
  H5  (added after H2 and H4 showed the goodput-seeking policy colliding as often as the
      original, and far more under poor sensing) Penalising each channel's predicted chance
      of collision (harm weight) trades a little of our goodput for much less harm, most of
      all when sensing is poor.

    python scripts/exp_channel_selection.py      # ~5 min
Output: results/experiments/channel_selection/<run>/
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel import selection as se
from spectrum_intel.packet import segment
from spectrum_intel.plots import AQUA, BLUE, INK, MUTED, ORANGE, YELLOW, save

POL = ["random", "hop", "predict", "quality", "oracle"]
LABEL = {"random": "random (no sensing)", "hop": "stay-or-hop (original)", "predict": "predict (learns on/off)",
         "quality": "quality (learns on/off + link)", "oracle": "oracle (true statistics)"}
COL = {"random": MUTED, "hop": BLUE, "predict": ORANGE, "quality": AQUA, "oracle": INK}
SKIP = 100


def run(policy, slots, seed, base_snr=6.0, pd=se.CNN_SENSING[0], pfa=se.CNN_SENSING[1], margin=None, harm=0.0):
    band = se.Band(pd=pd, pfa=pfa, seed=seed)
    sel = se.make_selector(policy, band.n_ch, np.random.default_rng([seed, 9]), users=band.users,
                           link_snr_db=base_snr + band.offsets, margin=margin, harm_weight=harm)
    link = se.BandLink(sel, band)
    data = np.random.default_rng([seed, 3]).integers(0, 256, 64 * 16, dtype=np.uint8).tobytes()
    pk = segment(data, 64)
    for t in range(slots):
        link.step(pk[t % 16], 16, base_snr, np.random.default_rng([seed, 7, t]))
    return link.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=8)
    a = ap.parse_args()
    S, N = a.seed, a.slots
    users = se.DEFAULT_USERS
    run_dir = ex.new_run("channel_selection", {
        "slots": N, "learning_slots_excluded": SKIP, "packet_bytes": 64, "sensing_cnn": se.CNN_SENSING,
        "link_offsets_db": se.LINK_OFFSETS_DB, "hop_cost_ms": 1.0,
        "users": [{"signal": u.label, "snr_db": u.snr_db, "p_on": u.p_on, "p_off": u.p_off} for u in users]}, S)
    rows, summary = [], {}

    # ---- H1 + H2: main comparison, 3 seeds -----------------------------------------------------
    seeds = [S, S + 1, S + 2]
    main_res = {p: [] for p in POL}
    logs0 = {}
    for p in POL:
        for sd in seeds:
            log = run(p, N, sd)
            if sd == S:
                logs0[p] = log
            r = se.summarise(log, skip=SKIP)
            main_res[p].append(r)
            rows.append({"part": "main", "policy": p, "seed": sd, **{k: v for k, v in r.items() if k != "channel_use"}})
        m = {k: float(np.mean([r[k] for r in main_res[p]])) for k in ("collision_rate", "goodput_kbps", "delivery_rate", "hops_per_100")}
        print(f"  main {p:8s} collisions {100 * m['collision_rate']:5.1f}%  goodput {m['goodput_kbps']:6.1f} kbps  "
              f"delivered {100 * m['delivery_rate']:5.1f}%  hops/100 {m['hops_per_100']:5.1f}")
    summary["main"] = {p: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))]
                           for k in ("collision_rate", "goodput_kbps", "delivery_rate", "hops_per_100")}
                       for p, v in main_res.items()}
    summary["main_channel_use_seed0"] = {p: se.summarise(logs0[p], skip=SKIP)["channel_use"] for p in POL}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(len(POL))
    for ax, key, scale, lab in [(axes[0], "collision_rate", 100, "collisions with licensed users (% of slots)"),
                                (axes[1], "goodput_kbps", 1, "our goodput (kbps)")]:
        mu = [scale * summary["main"][p][key][0] for p in POL]
        sd = [scale * summary["main"][p][key][1] for p in POL]
        ax.set_axisbelow(True)
        ax.bar(x, mu, yerr=sd, color=[COL[p] for p in POL], capsize=3)
        for i, v in enumerate(mu):
            ax.text(i, v + max(mu) * 0.02, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x, [LABEL[p].split(" (")[0] for p in POL])
        ax.set_ylabel(lab)
    axes[0].set_title("Harm to licensed users (lower is better)", fontsize=10)
    axes[1].set_title("What we deliver (higher is better)", fontsize=10)
    fig.suptitle(f"Base SNR 6 dB, CNN sensing, mean of {len(seeds)} runs x {N - SKIP} slots", fontsize=10)
    save(fig, run_dir / "policies.png")

    # learning curve (seed S)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    w = 100
    for p in POL:
        coll = np.array([e["collided"] for e in logs0[p]], float)
        bits = np.array([8 * e["data_bytes"] * e["ok"] for e in logs0[p]], float)
        air = np.array([e["airtime_s"] + 0.001 * e["hop"] for e in logs0[p]])
        k = np.ones(w)
        axes[0].plot(np.arange(w - 1, N), 100 * np.convolve(coll, k, "valid") / w, color=COL[p], lw=1.3, label=LABEL[p])
        axes[1].plot(np.arange(w - 1, N), np.convolve(bits, k, "valid") / np.convolve(air, k, "valid") / 1e3, color=COL[p], lw=1.3)
    axes[0].set_ylabel(f"collisions, % (last {w} slots)")
    axes[1].set_ylabel(f"goodput, kbps (last {w} slots)")
    for ax in axes:
        ax.set_xlabel("slot")
        ax.axvline(SKIP, color=MUTED, ls=":", lw=1)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Learning: the learned policies settle within ~100-200 slots", fontsize=10)
    axes[1].set_title("...and then stay close to the oracle", fontsize=10)
    save(fig, run_dir / "learning_curve.png")

    # ---- H2b: base SNR ----------------------------------------------------------------------
    bases = [0.0, 6.0, 12.0]
    pol2 = ["hop", "predict", "quality", "oracle"]
    by_base = {p: [] for p in pol2}
    for b in bases:
        for p in pol2:
            rs = [se.summarise(run(p, 1000, sd, base_snr=b), skip=SKIP) for sd in (S, S + 1)]
            g = float(np.mean([r["goodput_kbps"] for r in rs]))
            c = float(np.mean([r["collision_rate"] for r in rs]))
            by_base[p].append(g)
            rows.append({"part": "base_snr", "policy": p, "base_snr_db": b, "goodput_kbps": g, "collision_rate": c})
        print(f"  base {b:4.1f} dB  " + "  ".join(f"{p} {by_base[p][-1]:6.1f}" for p in pol2))
    summary["by_base_snr"] = {"base_snr_db": bases, "goodput_kbps": by_base}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    wd = 0.2
    for i, p in enumerate(pol2):
        ax.bar(np.arange(len(bases)) + (i - 1.5) * wd, by_base[p], wd, color=COL[p], label=LABEL[p])
    ax.set_xticks(range(len(bases)), [f"base SNR {b:g} dB" for b in bases])
    ax.set_ylabel("our goodput (kbps)")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)
    ax.set_title("Knowing link quality matters most when only some channels are good", fontsize=10)
    save(fig, run_dir / "base_snr.png")

    # ---- H3: hop margin and hop cost ----------------------------------------------------------
    margins = [0.0, 0.05, 0.2, 0.5, 1.0]
    costs_ms = [0.0, 1.0, 5.0]
    hop_res = {}
    for mg in margins:
        log = run("quality", N, S, margin=mg)
        hop_res[mg] = {c: se.summarise(log, hop_s=c / 1e3, skip=SKIP) for c in costs_ms}
        for c in costs_ms:
            r = hop_res[mg][c]
            rows.append({"part": "hop_margin", "margin": mg, "hop_cost_ms": c, "goodput_kbps": r["goodput_kbps"],
                         "hops_per_100": r["hops_per_100"], "collision_rate": r["collision_rate"]})
        print(f"  margin {mg:4.2f}  hops/100 {hop_res[mg][0.0]['hops_per_100']:5.1f}  " +
              "  ".join(f"cost {c:g} ms: {hop_res[mg][c]['goodput_kbps']:6.1f}" for c in costs_ms))
    summary["hop_margin"] = {str(mg): {str(c): {"goodput_kbps": hop_res[mg][c]["goodput_kbps"],
                                                "hops_per_100": hop_res[mg][c]["hops_per_100"],
                                                "collision_rate": hop_res[mg][c]["collision_rate"]} for c in costs_ms}
                             for mg in margins}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(margins, [hop_res[m][0.0]["hops_per_100"] for m in margins], "o-", color=INK)
    axes[0].set_ylabel("channel changes per 100 slots")
    for c, col in zip(costs_ms, [AQUA, BLUE, ORANGE]):
        axes[1].plot(margins, [hop_res[m][c]["goodput_kbps"] for m in margins], "o-", color=col, label=f"hop costs {c:g} ms")
    axes[1].set_ylabel("goodput (kbps)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("hop margin (best channel must score this much better to switch)")
    axes[0].set_title("A margin stops needless hopping", fontsize=10)
    axes[1].set_title("When hops cost time, a moderate margin pays", fontsize=10)
    save(fig, run_dir / "hop_margin.png")

    # ---- H4: sensing quality ------------------------------------------------------------------
    sensing = [("perfect", 1.0, 0.0), ("CNN (measured)", *se.CNN_SENSING), ("degraded", 0.9, 0.10), ("poor", 0.7, 0.25)]
    pol4 = ["hop", "predict", "quality"]
    sens_res = {p: [] for p in pol4}
    for name, pd, pfa in sensing:
        for p in pol4:
            r = se.summarise(run(p, 1000, S, pd=pd, pfa=pfa), skip=SKIP)
            sens_res[p].append(r)
            rows.append({"part": "sensing", "sensing": name, "pd": pd, "pfa": pfa, "policy": p,
                         "collision_rate": r["collision_rate"], "goodput_kbps": r["goodput_kbps"]})
        print(f"  sensing {name:15s} " + "  ".join(
            f"{p} {100 * sens_res[p][-1]['collision_rate']:4.1f}%/{sens_res[p][-1]['goodput_kbps']:5.1f}" for p in pol4))
    summary["sensing"] = {"levels": [s[0] for s in sensing],
                          "collision_rate": {p: [r["collision_rate"] for r in v] for p, v in sens_res.items()},
                          "goodput_kbps": {p: [r["goodput_kbps"] for r in v] for p, v in sens_res.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(sensing))
    for p in pol4:
        axes[0].plot(xs, [100 * r["collision_rate"] for r in sens_res[p]], "o-", color=COL[p], label=LABEL[p])
        axes[1].plot(xs, [r["goodput_kbps"] for r in sens_res[p]], "o-", color=COL[p])
    for ax in axes:
        ax.set_xticks(xs, [f"{n}\n(Pd {pd:g}, Pfa {pfa:g})" for n, pd, pfa in sensing], fontsize=7)
    axes[0].set_ylabel("collisions (%)")
    axes[1].set_ylabel("goodput (kbps)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Missed detections turn straight into collisions", fontsize=10)
    axes[1].set_title("The quality-aware policy stays ahead at every sensing level", fontsize=10)
    save(fig, run_dir / "sensing.png")

    # ---- H5: harm weight -----------------------------------------------------------------------
    weights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    harm_res = {}
    for name, pd, pfa in [("CNN (measured)", *se.CNN_SENSING), ("poor", 0.7, 0.25)]:
        harm_res[name] = [se.summarise(run("quality", 1000, S, pd=pd, pfa=pfa, harm=wt), skip=SKIP) for wt in weights]
        for wt, r in zip(weights, harm_res[name]):
            rows.append({"part": "harm_weight", "sensing": name, "harm_weight": wt,
                         "collision_rate": r["collision_rate"], "goodput_kbps": r["goodput_kbps"]})
        print(f"  harm {name:15s} " + "  ".join(f"w{wt:g}: {100 * r['collision_rate']:4.1f}%/{r['goodput_kbps']:5.1f}"
                                               for wt, r in zip(weights, harm_res[name])))
    summary["harm_weight"] = {"weights": weights, **{n: {"collision_rate": [r["collision_rate"] for r in v],
                                                         "goodput_kbps": [r["goodput_kbps"] for r in v]}
                                                     for n, v in harm_res.items()}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (name, idx) in zip(axes, [("CNN (measured)", 1), ("poor", 3)]):
        v = harm_res[name]
        ax.plot([100 * r["collision_rate"] for r in v], [r["goodput_kbps"] for r in v], "o-", color=AQUA,
                label="quality, harm weight 0 to 4")
        for wt, r in zip(weights, v):
            ax.annotate(f"{wt:g}", (100 * r["collision_rate"], r["goodput_kbps"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=INK)
        for p in ("hop", "predict"):
            r = sens_res[p][idx]
            ax.plot(100 * r["collision_rate"], r["goodput_kbps"], "s", color=COL[p], ms=7, label=LABEL[p])
        ax.set_xlabel("collisions with licensed users (%)")
        ax.set_title(f"Sensing: {name}", fontsize=10)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("our goodput (kbps)")
    fig.suptitle("Trading our goodput for less harm: the harm weight moves along the curve (labels = weight)", fontsize=10)
    save(fig, run_dir / "harm_tradeoff.png")

    for part in dict.fromkeys(r["part"] for r in rows):
        ex.save_csv(run_dir, f"{part}.csv", [r for r in rows if r["part"] == part])
    ex.save_metrics(run_dir, summary)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
