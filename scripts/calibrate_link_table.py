"""Measure packet error rate vs SNR for each modulation (AWGN) -> the table link adaptation uses.

    python scripts/calibrate_link_table.py          # ~1 min, rewrites spectrum_intel/link_table.json

The table is measured with the real transmitter, channel and receiver (64-byte packets,
random phase and arrival time), so the adaptation decides from measured behaviour, not
from textbook formulas. The run is also saved under results/experiments/link_table/.
"""
import argparse
import json

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel.adaptation import TABLE_PATH, required_snr
from spectrum_intel.link import transfer_packet
from spectrum_intel.modulation import LINK_MODULATIONS
from spectrum_intel.packet import segment
from spectrum_intel.plots import AQUA, BLUE, MUTED, ORANGE, save

COL = {"bpsk": BLUE, "qpsk": ORANGE, "qam16": AQUA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="packets per point")
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    snrs = np.arange(-6, 17, 1.0)
    run = ex.new_run("link_table", {"packets_per_point": a.n, "packet_bytes": 64, "snr_db": snrs.tolist(),
                                    "channel": "AWGN, random phase and timing"}, a.seed)
    per = {}
    for mod in LINK_MODULATIONS:
        vals = []
        for s in snrs:
            rng = np.random.default_rng([a.seed, LINK_MODULATIONS.index(mod), int(s * 10) + 500])
            pk = segment(rng.integers(0, 256, 64 * 16, dtype=np.uint8).tobytes(), 64)
            fails = sum(not transfer_packet(pk[i % 16], 16, mod, s, rng).record.crc_ok for i in range(a.n))
            vals.append(fails / a.n)
        per[mod] = np.minimum.accumulate(vals).round(4).tolist()       # PER can only fall with SNR
        print(f"  {mod:5s} " + " ".join(f"{v:.2f}" for v in per[mod]))
    table = {"snr_db": snrs.tolist(), "per": per,
             "meta": {"packet_bytes": 64, "packets_per_point": a.n, "seed": a.seed, "channel": "AWGN",
                      "source_run": str(run.relative_to(ex.ROOT))}}
    TABLE_PATH.write_text(json.dumps(table, indent=1))
    ex.save_metrics(run, table | {"snr_for_per_10pct": {m: required_snr(table, m, 0.1) for m in LINK_MODULATIONS}})

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m in LINK_MODULATIONS:
        ax.semilogy(snrs, np.maximum(per[m], 1 / (2 * a.n)), "o-", ms=3.5, color=COL[m], label=m.upper())
    ax.axhline(0.1, color=MUTED, ls="--", lw=1)
    ax.text(snrs[0], 0.115, "10% target", color=MUTED, fontsize=8)
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("packet error rate (points at the floor = no errors seen)")
    ax.set_title("Calibration table for link adaptation (AWGN, 64-byte packets)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, run / "per_table.png")
    print(f"Table written to {TABLE_PATH}; run saved to {run}")


if __name__ == "__main__":
    main()
