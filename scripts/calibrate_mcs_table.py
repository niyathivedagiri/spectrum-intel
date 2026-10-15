"""Build the adaptive modulation-and-coding table from the FEC experiment's measurements.

    python scripts/calibrate_mcs_table.py            # uses the latest results/experiments/fec run

No new measurement: it copies the packet error rates (soft decoding, AWGN, 64-byte packets,
real transmitter/receiver incl. the uncoded header) that exp_fec.py measured for all 12
modulation + code pairs, makes each curve non-increasing with SNR, and writes
spectrum_intel/mcs_table.json with the source run recorded in it.
"""
import argparse
import csv
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from spectrum_intel import experiments as ex
from spectrum_intel.adaptation import MCS_TABLE_PATH, scheme_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None, help="FEC run folder (default: latest)")
    a = ap.parse_args()
    run = a.run or ex.latest_run("fec")
    rows = [r for r in csv.DictReader(open(run / "awgn.csv")) if r["decoding"] == "soft"]
    snrs = sorted({float(r["snr_db"]) for r in rows})
    per = {}
    for r in rows:
        per.setdefault(scheme_name(r["mod"], r["code"]), {})[float(r["snr_db"])] = float(r["per"])
    table = {"snr_db": snrs,
             "per": {k: np.minimum.accumulate([v[s] for s in snrs]).round(4).tolist() for k, v in sorted(per.items())},
             "meta": {"packet_bytes": 64, "decoding": "soft", "channel": "AWGN",
                      "source_run": str(Path(run).resolve().relative_to(ex.ROOT)),
                      "packets_per_point": json.loads((run / "config.json").read_text())
                      .get("config", {}).get("packets_per_point")}}
    MCS_TABLE_PATH.write_text(json.dumps(table, indent=1))
    print(f"{len(table['per'])} schemes x {len(snrs)} SNRs from {run} -> {MCS_TABLE_PATH}")


if __name__ == "__main__":
    main()
