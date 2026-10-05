"""End-to-end link: bytes -> packets -> frames -> channel -> receiver -> packets -> bytes.

One packet is carried per frame (burst). Every stage is recorded so that bit
errors, packet failures and missing data are reported, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spectrum_intel import channel as ch
from spectrum_intel.metrics import bit_errors
from spectrum_intel.modulation import nearest_index, symbol_indices
from spectrum_intel.packet import Packet, parse_unchecked, reassemble, segment
from spectrum_intel.payload import bytes_to_bits
from spectrum_intel.receiver import receive
from spectrum_intel.transmitter import HEADER_BITS, build_frame


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
    n_samples: int              # airtime of the frame in samples
    data_bytes: int = 0         # user data bytes carried by this packet
    symbol_errors: int | None = None
    n_symbols: int = 0          # payload symbols in the frame


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


@dataclass
class PacketTransfer:
    """Everything that happened to one packet: what was sent, received and decided."""
    record: PacketRecord
    frame: object               # TxFrame
    rx: np.ndarray              # received samples (after the channel)
    res: object                 # RxResult
    good: Packet | None         # packet that passed the CRC
    rough: Packet | None        # damaged packet kept in error-tolerant mode


def transfer_packet(pkt: Packet, n_packets: int, mod: str, snr_db: float, rng: np.random.Generator, *,
                    max_payload: int = 64, cfo: float = 0.0, cfo_search: float = 0.0,
                    keep_corrupted: bool = False) -> PacketTransfer:
    """Send ONE packet through transmitter -> channel -> receiver and score it.

    Shared by `send` (whole messages) and the real-time engine (one packet per step),
    so both produce identical results for the same random generator.
    """
    block = pkt.to_bytes()
    frame = build_frame(block, mod)
    rx, _ = ch.awgn_link(frame.iq, snr_db, rng, cfo=cfo)
    res = receive(rx, cfo_search=cfo_search)
    crc_ok, errs, sym_err, good, rough = False, None, None, None, None
    tx_bits = bytes_to_bits(block)
    if res.header_ok:
        errs = bit_errors(tx_bits, res.payload_bits)
        rx_pay = res.data_symbols[HEADER_BITS:]
        sym_err = int(np.sum(nearest_index(rx_pay, mod) != symbol_indices(tx_bits, mod)[:len(rx_pay)]))
        parsed, crc_ok = Packet.from_bytes(res.payload)
        if crc_ok:
            good = parsed
        elif keep_corrupted:
            rough = parse_unchecked(res.payload, n_packets, max_payload)
    record = PacketRecord(pkt.seq, mod, len(block) * 8, res.detected, res.header_ok, crc_ok,
                          rough is not None, errs, res.snr_est_db, res.cfo_est, len(frame.iq),
                          data_bytes=len(pkt.data), symbol_errors=sym_err,
                          n_symbols=frame.n_payload_symbols)
    return PacketTransfer(record, frame, rx, res, good, rough)


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
        t = transfer_packet(pkt, len(packets), mod, snr_db, rng, max_payload=max_payload,
                            cfo=cfo, cfo_search=cfo_search, keep_corrupted=keep_corrupted)
        records.append(t.record)
        if t.good is not None:
            good.append(t.good)
        if t.rough is not None:
            rough_pkts.append(t.rough)
        if keep_frames:
            frames.append((t.frame, t.rx, t.res))
    # CRC-valid packets always win; a damaged packet only fills a gap nobody else filled
    have = {p.seq for p in good}
    for p in rough_pkts:
        if p.seq not in have:
            good.append(p)
            have.add(p.seq)
            corrupted.append(p.seq)
    received, missing = reassemble(good, len(packets), fill=fill, chunk=max_payload, expected_len=len(data))
    return LinkResult(data, received, missing, records, corrupted=sorted(corrupted), frames=frames)
