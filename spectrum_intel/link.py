"""End-to-end link: bytes -> packets -> frames -> channel -> receiver -> packets -> bytes.

One packet is carried per frame (burst). Every stage is recorded so that bit
errors, packet failures and missing data are reported, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel.metrics import bit_errors
from spectrum_intel.packet import Packet, parse_unchecked, reassemble, segment
from spectrum_intel.payload import bytes_to_bits
from spectrum_intel.receiver import receive
from spectrum_intel.transmitter import build_frame


@dataclass
class PacketRecord:
    seq: int
    mod: str
    n_bits: int                 # bits in the MAC packet (header + data + CRC)
    detected: bool
    header_ok: bool
    crc_ok: bool
    used_corrupted: bool        # accepted despite a CRC failure (error-tolerant mode only)
    bit_errors: int | None      # raw bit errors in the packet (None if the frame was lost)
    snr_est_db: float
    cfo_est: float
    n_samples: int


@dataclass
class LinkResult:
    sent: bytes
    received: bytes
    missing: list[int]
    packets: list[PacketRecord]
    corrupted: list[int] = field(default_factory=list)   # seq numbers accepted with known errors
    frames: list = field(default_factory=list)        # (TxFrame, rx samples, RxResult) if kept

    @property
    def packet_success_rate(self) -> float:
        return float(np.mean([p.crc_ok for p in self.packets]))

    @property
    def exact(self) -> bool:
        return self.received == self.sent

    def raw_ber(self) -> float:
        """Raw bit error rate over packets whose frame was demodulated."""
        got = [p for p in self.packets if p.bit_errors is not None]
        n = sum(p.n_bits for p in got)
        return sum(p.bit_errors for p in got) / n if n else float("nan")


def send(data: bytes, mod: str, snr_db: float, rng: np.random.Generator, *,
         max_payload: int = 64, cfo: float = 0.0, cfo_search: float = 0.0,
         keep_frames: bool = False, keep_corrupted: bool = False, fill: int = 0x00) -> LinkResult:
    """Send data over an AWGN link with random phase/timing (and optional frequency offset).

    keep_corrupted=False (default): packets failing the CRC are dropped (missing).
    keep_corrupted=True: error-tolerant mode for media - a packet failing the CRC is
    still used if its header fields are self-consistent, and is listed in `corrupted`.
    fill: byte value used for missing packets (e.g. 128 = mid-grey for images).
    """
    packets = segment(data, max_payload)
    records, good, frames, corrupted, rough_pkts = [], [], [], [], []
    for pkt in packets:
        block = pkt.to_bytes()
        frame = build_frame(block, mod)
        rx, _ = ch.awgn_link(frame.iq, snr_db, rng, cfo=cfo)
        res = receive(rx, cfo_search=cfo_search)
        crc_ok, errs, used = False, None, False
        if res.header_ok:
            errs = bit_errors(bytes_to_bits(block), res.payload_bits)
            parsed, crc_ok = Packet.from_bytes(res.payload)
            if crc_ok:
                good.append(parsed)
            elif keep_corrupted:
                rough = parse_unchecked(res.payload, len(packets), max_payload)
                if rough is not None:
                    rough_pkts.append(rough)
                    used = True
        records.append(PacketRecord(pkt.seq, mod, len(block) * 8, res.detected, res.header_ok, crc_ok,
                                    used, errs, res.snr_est_db, res.cfo_est, len(frame.iq)))
        if keep_frames:
            frames.append((frame, rx, res))
    # CRC-valid packets always win; a damaged packet only fills a gap nobody else filled
    have = {p.seq for p in good}
    for p in rough_pkts:
        if p.seq not in have:
            good.append(p)
            have.add(p.seq)
            corrupted.append(p.seq)
    received, missing = reassemble(good, len(packets), fill=fill, chunk=max_payload, expected_len=len(data))
    return LinkResult(data, received, missing, records, corrupted=sorted(corrupted), frames=frames)
