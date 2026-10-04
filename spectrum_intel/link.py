"""End-to-end link: bytes -> packets -> frames -> channel -> receiver -> packets -> bytes.

One packet is carried per frame (burst). Every stage is recorded so that bit
errors, packet failures and missing data are reported, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel.metrics import bit_errors
from spectrum_intel.packet import Packet, reassemble, segment
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
         keep_frames: bool = False) -> LinkResult:
    """Send data over an AWGN link with random phase/timing (and optional frequency offset)."""
    packets = segment(data, max_payload)
    records, good, frames = [], [], []
    for pkt in packets:
        block = pkt.to_bytes()
        frame = build_frame(block, mod)
        rx, _ = ch.awgn_link(frame.iq, snr_db, rng, cfo=cfo)
        res = receive(rx, cfo_search=cfo_search)
        crc_ok, errs = False, None
        if res.header_ok:
            errs = bit_errors(bytes_to_bits(block), res.payload_bits)
            parsed, crc_ok = Packet.from_bytes(res.payload)
            if crc_ok:
                good.append(parsed)
        records.append(PacketRecord(pkt.seq, mod, len(block) * 8, res.detected, res.header_ok, crc_ok,
                                    errs, res.snr_est_db, res.cfo_est, len(frame.iq)))
        if keep_frames:
            frames.append((frame, rx, res))
    received, missing = reassemble(good, len(packets), chunk=max_payload, expected_len=len(data))
    return LinkResult(data, received, missing, records, frames)
