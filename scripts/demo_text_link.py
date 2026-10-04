"""Watch a text message become a radio signal and come back.

    python scripts/demo_text_link.py
    python scripts/demo_text_link.py --text "Hello satellite" --mod qam16 --snr 4 --seed 3

Prints every stage (characters -> bytes -> bits -> packet -> symbols -> samples ->
channel -> synchronisation -> demodulation -> CRC -> text) and saves a figure.
Output: results/experiments/text_link_demo/<run>/
"""
import argparse

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel import experiments as ex
from spectrum_intel.metrics import bit_errors, ebn0_from_snr, esn0_from_snr
from spectrum_intel.modulation import bits_per_symbol, nearest_index, symbol_indices
from spectrum_intel.packet import Packet, segment
from spectrum_intel.payload import bytes_to_bits, bytes_to_text, describe_bytes, text_to_bytes
from spectrum_intel.plots import BLUE, INK, MUTED, ORANGE, save
from spectrum_intel.receiver import receive
from spectrum_intel.signals import constellation
from spectrum_intel.transmitter import HEADER_BITS, PILOT_SPACING, PREAMBLE, SPS, build_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="HELLO")
    ap.add_argument("--mod", default="qpsk", choices=["bpsk", "qpsk", "qam16"])
    ap.add_argument("--snr", type=float, default=3.0, help="SNR per sample, dB")
    ap.add_argument("--cfo", type=float, default=0.0, help="frequency offset, cycles/sample")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    run = ex.new_run("text_link_demo", vars(a), a.seed)
    k = bits_per_symbol(a.mod)

    print(f"\n1. TEXT            {a.text!r}")
    data = text_to_bytes(a.text)
    print(f"2. UTF-8 BYTES     {len(data)} bytes")
    for r in describe_bytes(data, 8):
        print(f"                     {r['char']!r:>4}  {r['hex']}  {r['bits']}")
    bits = bytes_to_bits(data)
    print(f"3. BITS            {len(bits)} bits: {''.join(map(str, bits[:48]))}{'...' if len(bits) > 48 else ''}")
    pkt = segment(data, 64)[0]
    block = pkt.to_bytes()
    print(f"4. PACKET          seq {pkt.seq}, {len(block)} bytes = 6 header + {len(pkt.data)} data + 2 CRC-16 "
          f"(CRC 0x{block[-2:].hex().upper()})")
    frame = build_frame(block, a.mod)
    pay_syms = frame.data_symbols[HEADER_BITS:]
    print(f"5. SYMBOLS         {a.mod.upper()}, {k} bit(s)/symbol -> {len(pay_syms)} payload symbols; "
          f"first: {np.round(pay_syms[:3], 3)}")
    print(f"6. FRAME           {len(PREAMBLE)} preamble + 32 header (BPSK) + pilots every 16 -> "
          f"{frame.info['n_symbols']} symbols")
    print(f"7. WAVEFORM        root-raised-cosine, {SPS} samples/symbol -> {len(frame.iq)} complex I/Q samples")

    rx, info = ch.awgn_link(frame.iq, a.snr, rng, cfo=a.cfo)
    print(f"8. CHANNEL         AWGN at SNR {a.snr:g} dB per sample (Es/N0 {float(esn0_from_snr(a.snr)):.1f} dB, "
          f"Eb/N0 {float(ebn0_from_snr(a.snr, k)):.1f} dB), random phase {np.degrees(info.phase):.0f} deg, "
          f"arrives after {info.delay} samples, frequency offset {a.cfo:g}")
    res = receive(rx, cfo_search=0.05 if abs(a.cfo) > 0.005 else 0.0)
    print(f"9. SYNC            frame found: {res.detected} (preamble match {res.sync_metric:.2f}), "
          f"estimated SNR {res.snr_est_db:.1f} dB, frequency offset {res.cfo_est:.5f}")
    if not res.header_ok:
        print("10. HEADER         CRC failed -> frame lost (nothing is invented)")
        ex.save_metrics(run, {"detected": res.detected, "header_ok": False})
        return
    errs = bit_errors(bytes_to_bits(block), res.payload_bits)
    parsed, crc_ok = Packet.from_bytes(res.payload)
    rx_pay = res.data_symbols[HEADER_BITS:]
    sym_err = int(np.sum(nearest_index(rx_pay, a.mod) != symbol_indices(bytes_to_bits(block), a.mod)))
    print(f"10. HEADER         modulation {res.mod.upper()}, {res.info['n_bytes']} bytes")
    print(f"11. DEMODULATION   {sym_err} symbol errors / {len(rx_pay)}, {errs} bit errors / {len(block) * 8}")
    print(f"12. CRC            {'PASS' if crc_ok else 'FAIL -> packet rejected'}")
    recovered = bytes_to_text(res.payload[6:6 + len(data)])
    print(f"13. RECOVERED      {recovered!r}  {'(exact)' if recovered == a.text else '(contains errors)'}\n")

    # ---- figure ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    nb = min(64, len(bits))
    ax.step(np.arange(nb), bits[:nb], where="post", color=BLUE)
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 1])
    ax.set_title(f"Bits of {a.text[:12]!r} (first {nb})")
    ax.set_xlabel("bit index")
    ax = fig.add_subplot(gs[0, 1])
    pts = constellation(a.mod)
    ax.plot(pay_syms.real, pay_syms.imag, "o", color=BLUE, ms=7)
    labels = ((np.arange(2 ** k)[:, None] >> np.arange(k)[::-1]) & 1)
    for p, lab in zip(pts, labels):
        ax.annotate("".join(map(str, lab)), (p.real, p.imag), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"{a.mod.upper()} symbols sent (bits -> points)")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    j = HEADER_BITS                                              # first payload symbol (data index)
    first = len(PREAMBLE) + (j // PILOT_SPACING) * (PILOT_SPACING + 1) + 1 + j % PILOT_SPACING
    seg = slice(first * SPS, first * SPS + 40 * SPS)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(frame.iq[seg].real, color=BLUE, lw=1, label="I")
    ax.plot(frame.iq[seg].imag, color=ORANGE, lw=1, label="Q")
    ax.set_title("Transmitted payload I/Q waveform (40 symbols)")
    ax.set_xlabel("sample")
    ax.legend(loc="upper right")
    ax = fig.add_subplot(gs[1, 0])
    rseg = slice(info.delay + seg.start, info.delay + seg.stop)
    ax.plot(rx[rseg].real, color=BLUE, lw=0.8)
    ax.plot(rx[rseg].imag, color=ORANGE, lw=0.8)
    ax.set_title(f"Received waveform (noise + unknown phase), SNR {a.snr:g} dB")
    ax.set_xlabel("sample")
    ax = fig.add_subplot(gs[1, 1])
    ok = nearest_index(rx_pay, a.mod) == symbol_indices(bytes_to_bits(block), a.mod)
    ax.plot(rx_pay[ok].real, rx_pay[ok].imag, ".", color=BLUE, ms=4, label="decided correctly")
    ax.plot(rx_pay[~ok].real, rx_pay[~ok].imag, "x", color=ORANGE, ms=6, label="symbol error")
    ax.plot(pts.real, pts.imag, "+", color=INK, ms=10, mew=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title("Received symbols after sync + equalisation")
    ax.legend(loc="lower right", fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = [f"Sent:       {a.text!r}", f"Received:   {recovered!r}", "",
             f"Bit errors: {errs} / {len(block) * 8}", f"Symbol errors: {sym_err} / {len(rx_pay)}",
             f"CRC: {'pass' if crc_ok else 'FAIL'}", f"Estimated SNR: {res.snr_est_db:.1f} dB (true {a.snr:g})"]
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=10, color=INK)
    fig.suptitle(f"Text -> {a.mod.upper()} -> AWGN channel -> receiver", fontsize=12)
    save(fig, run / "text_link_stages.png")
    ex.save_metrics(run, {"text": a.text, "recovered": recovered, "exact": recovered == a.text,
                          "bit_errors": errs, "bits": len(block) * 8, "symbol_errors": sym_err,
                          "symbols": len(rx_pay), "crc_ok": crc_ok, "snr_est_db": res.snr_est_db,
                          "cfo_est": res.cfo_est})
    print(f"Saved to {run}")


if __name__ == "__main__":
    main()
